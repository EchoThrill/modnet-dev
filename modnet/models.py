import pickle
from typing import List, Tuple, Dict, Optional, Callable, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"

import tensorflow_probability as tfp
from modnet.preprocessing import MODData
from modnet.matbench.benchmark import matbench_kfold_splits
from modnet.utils import LOG
from modnet import __version__

from functools import partial
import multiprocessing
import tqdm

__all__ = ("MODNetModel",)


class MODNetModel:
    """Container class for the underlying tf.keras `Model`, that handles
    setting up the architecture, activations, training and learning curve.

    Attributes:
        n_feat: The number of features used in the model.
        weights: The relative loss weights for each target.
        optimal_descriptors: The list of column names used
            in training the model.
        model: The `tf.keras.model.Model` of the network itself.
        target_names: The list of targets names that the model
            was trained for.

    """

    def __init__(
        self,
        targets: List,
        weights: Dict[str, float],
        num_neurons=([64], [32], [16], [16]),
        num_classes: Optional[Dict[str, int]] = None,
        n_feat: Optional[int] = 64,
        act="relu",
    ):
        """Initialise the model on the passed targets with the desired
        architecture, feature count and loss functions and activation functions.

        Parameters:
            targets: A nested list of targets names that defines the hierarchy
                of the output layers.
            weights: The relative loss weights to apply for each target.
            num_classes: Dictionary defining the target types (classification or regression).
                Should be constructed as follows: key: string giving the target name; value: integer n,
                 with n=0 for regression and n>=2 for classification with n the number of classes.
            num_neurons: A specification of the model layers, as a 4-tuple
                of lists of integers. Hidden layers are split into four
                blocks of `tf.keras.layers.Dense`, with neuron count specified
                by the elements of the `num_neurons` argument.
            n_feat: The number of features to use as model inputs.
            act: A string defining a tf.keras activation function to pass to use
                in the `tf.keras.layers.Dense` layers.

        """

        self.__modnet_version__ = __version__

        if n_feat is None:
            n_feat = 64
        self.n_feat = n_feat
        self.weights = weights
        self.num_classes = num_classes
        self.num_neurons = num_neurons
        self.act = act

        self._scaler = None
        self.optimal_descriptors = None
        self.target_names = None
        self.targets = targets
        self.model = None

        f_temp = [x for subl in targets for x in subl]
        self.targets_flatten = [x for subl in f_temp for x in subl]
        self.num_classes = {name: 0 for name in self.targets_flatten}
        if num_classes is not None:
            self.num_classes.update(num_classes)
        self._multi_target = len(self.targets_flatten) > 1

        self.model = self.build_model(
            targets, n_feat, num_neurons, act=act, num_classes=self.num_classes
        )

    def build_model(
        self,
        targets: List,
        n_feat: int,
        num_neurons: Tuple[List[int], List[int], List[int], List[int]],
        num_classes: Optional[Dict[str, int]] = None,
        act: str = "relu",
    ):
        """Builds the tf.keras model and sets the `self.model` attribute.

        Parameters:
            targets: A nested list of targets names that defines the hierarchy
                of the output layers.
            n_feat: The number of features to use as model inputs.
            num_neurons: A specification of the model layers, as a 4-tuple
                of lists of integers. Hidden layers are split into four
                blocks of `tf.keras.layers.Dense`, with neuron count specified
                by the elements of the `num_neurons` argument.
            num_classes: Dictionary defining the target types (classification or regression).
                Should be constructed as follows: key: string giving the target name; value: integer n,
                with n=0 for regression and n>=2 for classification with n the number of classes.
            act: A string defining a tf.keras activation function to pass to use
                in the `tf.keras.layers.Dense` layers.

        """

        num_layers = [len(x) for x in num_neurons]

        # Build first common block
        f_input = tf.keras.layers.Input(shape=(n_feat,))
        previous_layer = f_input
        for i in range(num_layers[0]):
            previous_layer = tf.keras.layers.Dense(num_neurons[0][i], activation=act)(
                previous_layer
            )
            if self._multi_target:
                previous_layer = tf.keras.layers.BatchNormalization()(previous_layer)
        common_out = previous_layer

        # Build intermediate representations
        intermediate_models_out = []
        for _ in range(len(targets)):
            previous_layer = common_out
            for j in range(num_layers[1]):
                previous_layer = tf.keras.layers.Dense(num_neurons[1][j], activation=act)(
                    previous_layer
                )
                if self._multi_target:
                    previous_layer = tf.keras.layers.BatchNormalization()(previous_layer)
            intermediate_models_out.append(previous_layer)

        # Build outputs
        final_out = []
        for group_idx, group in enumerate(targets):
            for prop_idx in range(len(group)):
                previous_layer = intermediate_models_out[group_idx]
                for k in range(num_layers[2]):
                    previous_layer = tf.keras.layers.Dense(
                        num_neurons[2][k], activation=act
                    )(previous_layer)
                    if self._multi_target:
                        previous_layer = tf.keras.layers.BatchNormalization()(
                            previous_layer
                        )
                clayer = previous_layer
                for pi in range(len(group[prop_idx])):
                    previous_layer = clayer
                    for li in range(num_layers[3]):
                        previous_layer = tf.keras.layers.Dense(num_neurons[3][li])(
                            previous_layer
                        )
                    n = num_classes[group[prop_idx][pi]]
                    if n >= 2:
                        out = tf.keras.layers.Dense(
                            n, activation="softmax", name=group[prop_idx][pi]
                        )(previous_layer)
                    else:
                        out = tf.keras.layers.Dense(
                            1, activation="linear", name=group[prop_idx][pi]
                        )(previous_layer)
                    final_out.append(out)

        return tf.keras.models.Model(inputs=f_input, outputs=final_out)

    def fit(
        self,
        training_data: MODData,
        val_fraction: float = 0.0,
        val_key: Optional[str] = None,
        val_data: Optional[MODData] = None,
        lr: float = 0.001,
        epochs: int = 200,
        batch_size: int = 128,
        xscale: Optional[str] = "minmax",
        metrics: List[str] = ["mae"],
        callbacks: List[Callable] = None,
        verbose: int = 0,
        loss: str = "mse",
        **fit_params,
    ) -> None:
        """Train the model on the passed training `MODData` object.

        Paramters:
            training_data: A `MODData` that has been featurized and
                feature selected. The first `self.n_feat` entries in
                `training_data.get_optimal_descriptors()` will be used
                for training.
            val_fraction: The fraction of the training data to use as a
                validation set for tracking model performance during
                training.
            val_key: The target name to track on the validation set
                during training, if performing multi-target learning.
            lr: The learning rate.
            epochs: The maximum number of epochs to train for.
            batch_size: The batch size to use for training.
            xscale: The feature scaler to use, either `None`,
                `'minmax'` or `'standard'`.
            metrics: A list of tf.keras metrics to pass to `compile(...)`.
            loss: The built-in tf.keras loss to pass to `compile(...)`.
            fit_params: Any additional parameters to pass to `fit(...)`,
                these will be overwritten by the explicit keyword
                arguments above.

        """

        if self.n_feat > len(training_data.get_optimal_descriptors()):
            raise RuntimeError(
                "The model requires more features than computed in data. "
                f"Please reduce n_feat below or equal to {len(training_data.get_optimal_descriptors())}"
            )

        self.xscale = xscale
        self.target_names = list(self.weights.keys())
        self.optimal_descriptors = training_data.get_optimal_descriptors()

        x = training_data.get_featurized_df()[
            self.optimal_descriptors[: self.n_feat]
        ].values

        # For compatibility with MODNet 0.1.7; if there is only one target in the training data,
        # use that for the name of the target too.
        if len(self.targets_flatten) == 1 and len(training_data.df_targets.columns) == 1:
            self.targets_flatten = list(training_data.df_targets.columns)

        y = []
        for targ in self.targets_flatten:
            if self.num_classes[targ] >= 2:  # Classification
                y_inner = tf.keras.utils.to_categorical(
                    training_data.df_targets[targ].values,
                    num_classes=self.num_classes[targ],
                )
                loss = "categorical_crossentropy"
            else:
                y_inner = training_data.df_targets[targ].values.astype(
                    np.float, copy=False
                )
            y.append(y_inner)

        # Scale the input features:
        # x = np.nan_to_num(x)
        if self.xscale == "minmax":
            self._scaler = MinMaxScaler(feature_range=(-0.5, 0.5))

        elif self.xscale == "standard":
            self._scaler = StandardScaler()

        x = self._scaler.fit_transform(x)
        x = np.nan_to_num(x,nan=-1)

        if val_data is not None:
            val_x = val_data.get_featurized_df()[
                self.optimal_descriptors[: self.n_feat]
            ].values
            #val_x = np.nan_to_num(val_x)
            val_x = self._scaler.transform(val_x)
            val_x = np.nan_to_num(val_x,nan=-1)
            try:
                val_y = list(
                    val_data.get_target_df()[self.targets_flatten].values.astype(np.float, copy=False).transpose()
                )
            except Exception:
                val_y = list(
                    val_data.get_target_df().values.astype(np.float, copy=False).transpose()
                )
            validation_data = (val_x, val_y)
        else:
            validation_data = None

        # Optionally set up print callback
        if verbose:
            if val_fraction > 0 or validation_data:
                if self._multi_target and val_key is not None:
                    val_metric_key = f"val_{val_key}_mae"
                else:
                    val_metric_key = "val_mae"
                print_callback = tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: print(
                        f"epoch {epoch}: loss: {logs['loss']:.3f}, "
                        f"val_loss:{logs['val_loss']:.3f} {val_metric_key}:{logs[val_metric_key]:.3f}"
                    )
                )

            else:
                print_callback = tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: print(
                        f"epoch {epoch}: loss: {logs['loss']:.3f}"
                    )
                )

                if callbacks is None:
                    callbacks = [print_callback]
                else:
                    callbacks.append(print_callback)

        fit_params = {
            "x": x,
            "y": y,
            "epochs": epochs,
            "batch_size": batch_size,
            "verbose": verbose,
            "validation_split": val_fraction,
            "validation_data": validation_data,
            "callbacks": callbacks,
        }

        self.model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(lr=lr),
            metrics=metrics,
            loss_weights=self.weights,
        )
        self.history = self.model.fit(**fit_params)

    def fit_preset(
        self,
        data: MODData,
        presets: List[Dict[str, Any]] = None,
        val_fraction: float = 0.15,
        verbose: int = 0,
        classification: bool = False,
        refit: bool = True,
        fast: bool = False,
        nested: int = 5,
        callbacks: List[Any] = None,
        n_jobs=None,
        ) -> Tuple[List[List[Any]],
                   np.ndarray,
                   Optional[List[float]],
                   List[List[float]],
                   Dict[str, Any]
        ]:
        """Chooses an optimal hyper-parametered MODNet model from different presets.

        This function implements the "inner loop" of a cross-validation workflow. By
        modifying the `nested` argument, it can be run in full nested mode (i.e.
        train n_fold * n_preset models) or just with a simple random hold-out set.

        The data is first fitted on several well working MODNet presets
        with a validation set (10% of the furnished data by default).

        Sets the `self.model` attribute to the model with the lowest mean validation loss across
        all folds.

        Args:
            data: MODData object contain training and validation samples.
            presets: A list of dictionaries containing custom presets.
            verbose: The verbosity level to pass to tf.keras
            val_fraction: The fraction of the data to use for validation.
            classification: Whether or not we are performing classification.
            refit: Whether or not to refit the final model for each fold with
                the best-performing settings.
            fast: Used for debugging. If `True`, only fit the first 2 presets and
                reduce the number of epochs.
            nested: integer specifying whether or not to perform a full nested CV. If 0,
                a simple validation split is performed based on val_fraction argument.
                If an integer, use this number of inner CV folds, ignoring the `val_fraction` argument.
                Note: If set to 1, the value will be overwritten to a default of 5 folds.
            n_jobs: number of jobs for multiprocessing

        Returns:
            - A list of length num_outer_folds containing lists of MODNet models of length num_inner_folds.
            - A list of validation losses achieved by the best model for each fold during validation (excluding refit).
            - The learning curve of the final (refitted) model (or `None` if `refit` is `False`)
            - A nested list of learning curves for each trained model of lengths (num_outer_folds,  num_inner folds).
            - The settings of the best-performing preset.

        """

        if callbacks is None:
            es = tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                min_delta=0.001,
                patience=100,
                verbose=verbose,
                mode="auto",
                baseline=None,
                restore_best_weights=False,
            )
            callbacks = [es]

        if presets is None:
            from modnet.model_presets import gen_presets
            presets = gen_presets(self.n_feat, len(data.df_targets), classification=classification)

        if fast and len(presets) >= 2:
            presets = presets[:2]
            for k, _ in enumerate(presets):
                presets[k]["epochs"] = 100

        val_losses = 1e20 * np.ones((len(presets),))

        num_nested_folds = 5
        if nested:
            num_nested_folds = nested
        if num_nested_folds <= 1:
            num_nested_folds = 5

        # create tasks
        tasks = []
        for i, params in enumerate(presets):
            n_feat = min(len(data.get_optimal_descriptors()), params["n_feat"])

            splits = matbench_kfold_splits(data, n_splits=num_nested_folds, classification=classification)
            if not nested:
                splits = [train_test_split(range(len(data.df_featurized)),test_size=val_fraction)]
                n_splits = 1
            else:
                n_splits = num_nested_folds

            for ind, (train, val) in enumerate(splits):
                val_params = {}
                train_data, val_data = data.split((train, val))
                val_params["val_data"] = val_data

                tasks += [{'train_data' : train_data,
                   'targets' : self.targets,
                   'weights' : self.weights,
                   'num_classes' : self.num_classes,
                   'n_feat' : n_feat,
                   'num_neurons' : params["num_neurons"],
                   'lr' : params["lr"],
                   'batch_size' : params["batch_size"],
                   'epochs' : params["epochs"],
                   'loss' : params["loss"],
                   'act' : params["act"],
                   'callbacks' : callbacks,
                   'preset_id' : i,
                   'fold_id' : ind,
                   'verbose' : verbose,
                   **val_params,
                }]

        val_losses = np.zeros((len(presets),n_splits))
        learning_curves = [[None]*n_splits]*len(presets)
        models = [[None]*n_splits]*len(presets)

        ctx = multiprocessing.get_context('spawn')
        pool = ctx.Pool(processes=n_jobs, initializer=init_worker)
        LOG.info(f'Multiprocessing on {n_jobs} cores. Total of {multiprocessing.cpu_count()} cores available.')

        for res in tqdm.tqdm(pool.imap_unordered(map_validate_model, tasks, chunksize=1), total=len(tasks)):
            val_loss, learning_curve, model, preset_id, fold_id = res

            # reload model
            model, model_json, weights = model
            model.model = tf.keras.models.model_from_json(model_json)
            model.model.set_weights(weights)

            val_losses[preset_id,fold_id] = val_loss
            learning_curves[preset_id][fold_id] = learning_curve
            models[preset_id][fold_id] = model
        pool.close()
        pool.join()

        val_loss_per_preset = np.mean(val_losses,axis=1)
        best_preset_idx = int(np.argmin(val_loss_per_preset))
        best_model_idx =int(np.argmin(val_losses[best_preset_idx,:]))
        best_preset = presets[best_preset_idx]
        best_learning_curve = learning_curves[best_preset_idx][best_model_idx]
        best_model = models[best_preset_idx][best_model_idx]

        LOG.info(
            "Preset #{} resulted in lowest validation loss with params {}".format(
                best_preset_idx + 1, tasks[n_splits*best_preset_idx+best_model_idx]
            )
        )

        if refit:
            LOG.info("Refitting with all data and parameters: {}".format(best_preset))
            # Building final model

            n_feat = min(len(data.get_optimal_descriptors()), best_preset['n_feat'])
            self.model = MODNetModel(
                self.targets,
                self.weights,
                num_neurons=best_preset['num_neurons'],
                n_feat=n_feat,
                act=best_preset['act'],
                num_classes=self.num_classes).model
            self.n_feat = n_feat
            self.fit(
                data,
                val_fraction=0,
                lr=best_preset['lr'],
                epochs=best_preset['epochs'],
                batch_size=best_preset['batch_size'],
                loss=best_preset['loss'],
                callbacks=callbacks,
                verbose=verbose)
        else:
            self.n_feat = best_model.n_feat
            self.model = best_model.model
            self._scaler = best_model._scaler

        return models, val_losses, best_learning_curve, learning_curves, best_preset

    def predict(self, test_data: MODData, return_prob=False) -> pd.DataFrame:
        """Predict the target values for the passed MODData.

        Parameters:
            test_data: A featurized and feature-selected `MODData`
                object containing the descriptors used in training.
            return_prob: For a classification tasks only: whether to return the probability of each
                class OR only return the most probable class.

        Returns:
            A `pandas.DataFrame` containing the predicted values of the targets.


        """
        # prevents Nan predictions if some features are inf
        x = test_data.get_featurized_df().replace([np.inf, -np.inf, np.nan], 0)[
            self.optimal_descriptors[:self.n_feat]
        ].values

        # Scale the input features:
        x = np.nan_to_num(x)
        if self._scaler is not None:
            x = self._scaler.transform(x)
            x = np.nan_to_num(x,nan=-1)

        p = np.array(self.model.predict(x))
        if len(p.shape) == 2:
            p = np.array([p])
        p_dic = {}
        for i, name in enumerate(self.targets_flatten):
            if self.num_classes[name] >= 2:
                if return_prob:
                    temp = p[i, :, :] / (p[i, :, :].sum(axis=1)).reshape((-1, 1))
                    for j in range(temp.shape[-1]):
                        p_dic['{}_prob_{}'.format(name, j)] = temp[:, j]
                else:
                    p_dic[name] = np.argmax(p[i, :, :], axis=1)
            else:
                p_dic[name] = p[i, :, 0]
        predictions = pd.DataFrame(p_dic)
        predictions.index = test_data.structure_ids

        return predictions

    def evaluate(self, test_data: MODData) -> pd.DataFrame:
        """Evaluates the target values for the passed MODData by returning the corresponding loss.

        Parameters:
            test_data: A featurized and feature-selected `MODData`
                object containing the descriptors used in training.


        Returns:
            Loss score
        """
        # prevents Nan predictions if some features are inf
        x = test_data.get_featurized_df().replace([np.inf, -np.inf, np.nan], 0)[
            self.optimal_descriptors[:self.n_feat]
        ].values

        # Scale the input features:
        x = np.nan_to_num(x)
        if self._scaler is not None:
            x = self._scaler.transform(x)
            x = np.nan_to_num(x,nan=-1)

        y = []
        for targ in self.targets_flatten:
            if self.num_classes[targ] >= 2:  # Classification
                y_inner = tf.keras.utils.to_categorical(
                    test_data.df_targets[targ].values,
                    num_classes=self.num_classes[targ],
                )
                loss = "categorical_crossentropy"
            else:
                y_inner = test_data.df_targets[targ].values.astype(
                    np.float, copy=False
                )
            y.append(y_inner)

        return self.model.evaluate(x,y)[0]





    #############

    def save(self, filename: str):
        """Save the `MODNetModel` across 3 files with the same base
        filename:

        * <filename>.json contains the tf.keras model JSON dump.
        * <filename>.pkl contains the `MODNetModel` object, excluding the
          tf.keras model.
        * <filename>.h5 contains the model weights.

        Parameters:
            filename: The base filename to save to.

        """

        LOG.info("Saving model...")
        model_json = self.model.to_json()
        with open(f"{filename}.json", "w") as f:
            f.write(model_json)
        self.model.save_weights(f"{filename}.h5")
        try:
            with open(f"{filename}.history.pkl", "wb") as f:
                pickle.dump(self.model.history.history, f)
        except Exception:
            import traceback
            traceback.print_exc()
            LOG.info("Failed to save model history.")

        model = self.model
        self.model = None
        self.history = None
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(self, f)
        self.model = model
        LOG.info("Saved model to {}(.json/.h5/.pkl)".format(filename))

    @staticmethod
    def load(filename: str):
        """Load the `MODNetModel` from 3 files with the same base
        filename:

        * <filename>.json contains the tf.keras model JSON dump.
        * <filename>.pkl contains the `MODNetModel` object, excluding the
          tf.keras model.
        * <filename>.h5 contains the model weights.

        Returns:
            The loaded `MODNetModel` object.

        """

        LOG.info("Loading model from {}(.json/.h5/.pkl)".format(filename))

        with open(f"{filename}.pkl", "rb") as f:
            mod = pickle.load(f)

        if not isinstance(mod, MODNetModel):
            raise RuntimeError(
                "Pickled data in {filename}.pkl did not contain a `MODNetModel`."
            )

        with open(f"{filename}.json", "r") as f:
            model_json = f.read()

        mod.model = tf.keras.models.model_from_json(model_json)
        mod.model.load_weights(f"{filename}.h5")

        if not hasattr(mod, "__modnet_version__"):
            mod.__modnet_version__ = "<=0.1.7"

        LOG.info(
            "Loaded `MODNetModel` created with modnet version {}.".format(
                mod.__modnet_version__
            )
        )

        return mod


def init_worker():
    '''
    Add KeyboardInterrupt exception to mutliprocessing workers "
    '''
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def validate_model(train_data = None,
                   val_data = None,
                   targets = None,
                   weights = None,
                   num_classes=None,
                   n_feat = 100,
                   num_neurons = [[8],[8],[8],[8]],
                   lr=0.1,
                   batch_size = 64,
                   epochs = 100,
                   loss='mse',
                   act = 'relu',
                   xscale='minmax',
                   callbacks = [],
                   preset_id = None,
                   fold_id = None,
                   verbose = 0,
            ):

    #deprecated
    # data type can get messed up when passed to new process
    #train_data.df_featurized = train_data.df_featurized.apply(pd.to_numeric)
    #train_data.df_targets = train_data.df_targets.apply(pd.to_numeric)
    #if val_data is not None:
    #    val_data.df_featurized = val_data.df_featurized.apply(pd.to_numeric)
    #    val_data.df_targets = val_data.df_targets.apply(pd.to_numeric)
    #verbose=1

    model = MODNetModel(
        targets,
        weights,
        num_neurons=num_neurons,
        n_feat=n_feat,
        act=act,
        num_classes=num_classes
    )

    model.fit(
        train_data,
        lr = lr,
        epochs = epochs,
        batch_size = batch_size,
        loss = loss,
        xscale=xscale,
        callbacks = callbacks,
        verbose = verbose,
        val_fraction = 0,
        val_data = val_data,
    )


    learning_curve = model.model.history.history["val_loss"]

    val_loss = model.evaluate(val_data)

    #save model
    model_json = model.model.to_json()
    model_weights = model.model.get_weights()
    model.model = None
    model.history = None
    model = (model, model_json, model_weights)

    return val_loss, learning_curve, model, preset_id, fold_id


def map_validate_model(kwargs):
    return validate_model(**kwargs)


#### Probabilistoc models####

class Bayesian_MODNetModel(MODNetModel):
    """Container class for the underlying Probabilistic Bayesian Neural Network, that handles
    setting up the architecture, activations, training and learning curve. Only epistemic uncertainty is taken into account.

    Attributes:
        n_feat: The number of features used in the model.
        weights: The relative loss weights for each target.
        optimal_descriptors: The list of column names used
            in training the model.
        model: The `keras.model.Model` of the network itself.
        target_names: The list of targets names that the model
            was trained for.

    """


    def __init__(
        self,
        targets: List,
        weights: Dict[str, float],
        num_neurons=([64], [32], [16], [16]),
        bayesian_layers=None,
        prior=None,
        posterior=None,
        kl_weight=None,
        num_classes: Optional[Dict[str, int]] = None,
        n_feat: Optional[int] = 64,
        act="relu",
    ):
        """Initialise the model on the passed targets with the desired
        architecture, feature count and loss functions and activation functions.

        Parameters:
            targets: A nested list of targets names that defines the hierarchy
                of the output layers.
            weights: The relative loss weights to apply for each target.
            num_neurons: A specification of the model layers, as a 4-tuple
                of lists of integers. Hidden layers are split into four
                blocks of `tf.keras.layers.Dense`, with neuron count specified
                by the elements of the `num_neurons` argument.
            bayesian_layers: Same shape as num_neurons, with True for a Bayesian DenseVariational layer,
                False for a normal Dense layer. Default is None and will only set last layer as Bayesian.
            prior: Prior to use for the DenseVariational layers, default is independent normal with learnable mean.
            posterior: Posterior to use for the DenseVariational layers, default is indepent normal with learnable mean and variance.
            kl_weight: Amount by which to scale the KL divergence loss between prior and posterior.
            num_classes: Dictionary defining the target types (classification or regression).
                Should be constructed as follows: key: string giving the target name; value: integer n,
                 with n=0 for regression and n>=2 for classification with n the number of classes.
            n_feat: The number of features to use as model inputs.
            act: A string defining a tf.keras activation function to pass to use
                in the `tf.keras.layers.Dense` layers.

        """

        self.__modnet_version__ = __version__

        if n_feat is None:
            n_feat = 64
        self.n_feat = n_feat
        self.weights = weights
        self.num_classes = num_classes
        self.num_neurons = num_neurons
        self.act = act

        self._scaler = None
        self.optimal_descriptors = None
        self.target_names = None
        self.targets = targets
        self.model = None

        f_temp = [x for subl in targets for x in subl]
        self.targets_flatten = [x for subl in f_temp for x in subl]
        self.num_classes = {name: 0 for name in self.targets_flatten}
        if num_classes is not None:
            self.num_classes.update(num_classes)
        self._multi_target = len(self.targets_flatten) > 1

        self.model = self.build_model(
            targets, n_feat, num_neurons,
            bayesian_layers=bayesian_layers, prior=prior, posterior=posterior, kl_weight=kl_weight,
            act=act, num_classes=self.num_classes
        )

    def build_model(
        self,
        targets: List,
        n_feat: int,
        num_neurons: Tuple[List[int], List[int], List[int], List[int]],
        bayesian_layers=None,
        prior=None,
        posterior=None,
        kl_weight=None,
        num_classes: Optional[Dict[str, int]] = None,
        act: str = "relu",
    ):
        """Builds the Bayesian Neural Network and sets the `self.model` attribute.

        Parameters:
            targets: A nested list of targets names that defines the hierarchy
                of the output layers.
            n_feat: The number of features to use as model inputs.
            num_neurons: A specification of the model layers, as a 4-tuple
                of lists of integers. Hidden layers are split into four
                blocks of `keras.layers.Dense`, with neuron count specified
                by the elements of the `num_neurons` argument.
            num_classes: Dictionary defining the target types (classification or regression).
                Should be constructed as follows: key: string giving the target name; value: integer n,
                with n=0 for regression and n>=2 for classification with n the number of classes.
            act: A string defining a Keras activation function to pass to use
                in the `keras.layers.Dense` layers.

        """

        num_layers = [len(x) for x in num_neurons]

        # define probabilistic layers
        tfd = tfp.distributions

        if bayesian_layers is None:
            bayesian_layers = [[False]*nl for nl in num_layers]

        if posterior is None:
            def posterior(kernel_size, bias_size=0, dtype=None):
                n = kernel_size + bias_size
                c = np.log(np.expm1(1.))
                return tf.keras.Sequential([
                    tfp.layers.VariableLayer(2 * n, dtype=dtype),
                    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                        tfd.Normal(loc=t[..., :n],
                                    scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                        reinterpreted_batch_ndims=1)),
                ])

        if prior is None:
            def prior(kernel_size, bias_size=0, dtype=None):
                n = kernel_size + bias_size
                return tf.keras.Sequential([
                    tfp.layers.VariableLayer(n, dtype=dtype),
                    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                        tfd.Normal(loc=t, scale=1),
                        reinterpreted_batch_ndims=1)),
                ])


        bayesian_layer = partial(tfp.layers.DenseVariational,
                                 make_posterior_fn=posterior,
                                 make_prior_fn=prior,
                                 kl_weight=1/3619,
                                 activation=act)
        dense_layer = partial(tf.keras.layers.Dense, activation=act)

        # Build first common block
        f_input = tf.keras.layers.Input(shape=(n_feat,))
        previous_layer = f_input
        for i in range(num_layers[0]):
            if bayesian_layers[0][i]:
                previous_layer = bayesian_layer(num_neurons[0][i])(previous_layer)
            else:
                previous_layer = dense_layer(num_neurons[0][i])(previous_layer)
            if self._multi_target:
                previous_layer = tf.keras.layers.BatchNormalization()(previous_layer)
        common_out = previous_layer

        # Build intermediate representations
        intermediate_models_out = []
        for _ in range(len(targets)):
            previous_layer = common_out
            for j in range(num_layers[1]):
                if bayesian_layers[1][j]:
                    previous_layer = bayesian_layer(num_neurons[1][j])(previous_layer)
                else:
                    previous_layer = dense_layer(num_neurons[1][j])(previous_layer)
                if self._multi_target:
                    previous_layer = tf.keras.layers.BatchNormalization()(previous_layer)
            intermediate_models_out.append(previous_layer)

        # Build outputs
        final_out = []
        for group_idx, group in enumerate(targets):
            for prop_idx in range(len(group)):
                previous_layer = intermediate_models_out[group_idx]
                for k in range(num_layers[2]):
                    if bayesian_layers[2][k]:
                        previous_layer = bayesian_layer(num_neurons[2][k])(previous_layer)
                    else:
                        previous_layer = dense_layer(num_neurons[2][k])(previous_layer)
                    if self._multi_target:
                        previous_layer = tf.keras.layers.BatchNormalization()(
                            previous_layer
                        )
                clayer = previous_layer
                for pi in range(len(group[prop_idx])):
                    previous_layer = clayer
                    for li in range(num_layers[3]):
                        if bayesian_layers[3][li]:
                            previous_layer = bayesian_layer(num_neurons[3][li])(previous_layer)
                        else:
                            previous_layer = dense_layer(num_neurons[3][li])(previous_layer)
                    n = num_classes[group[prop_idx][pi]]
                    if n >= 2:
                        out = tfp.layers.DenseVariational(
                            n,
                            make_posterior_fn=posterior, make_prior_fn=prior, kl_weight=kl_weight,
                            activation="softmax", name=group[prop_idx][pi]
                        )(previous_layer)
                    else:
                        out = tfp.layers.DenseVariational(
                            1,
                            make_posterior_fn=posterior, make_prior_fn=prior, kl_weight=kl_weight,
                            activation="linear", name=group[prop_idx][pi]
                        )(previous_layer)
                    final_out.append(out)

        return tf.keras.models.Model(inputs=f_input, outputs=final_out)

    def predict(self, test_data: MODData, return_prob=False, return_unc=False) -> pd.DataFrame:
        """Predict the target values for the passed MODData.

        Parameters:
            test_data: A featurized and feature-selected `MODData`
                object containing the descriptors used in training.
            return_prob: For a classification tasks only: whether to return the probability of each
                class OR only return the most probable class.
            return_unc: whether to return the standard deviation as a second dataframe

        Returns:
            A `pandas.DataFrame` containing the predicted values of the targets.
            If return_unc=True, two `pandas.DataFrame` : (predictions,std) containing the predicted values of the targets and
             the standard deviations of the epistemic uncertainty.


        """
        # prevents Nan predictions if some features are inf
        x = test_data.get_featurized_df().replace([np.inf, -np.inf, np.nan], 0)[
            self.optimal_descriptors[:self.n_feat]
        ].values

        # Scale the input features:
        x = np.nan_to_num(x)
        if self._scaler is not None:
            x = self._scaler.transform(x)
            x = np.nan_to_num(x)

        all_predictions = []

        for i in range(1000):
            p = self.model.predict(x)
            if len(self.targets_flatten) == 1:
                p = np.array([p])
            all_predictions.append(p)

        p_dic = {}
        unc_dic = {}
        for i, name in enumerate(self.targets_flatten):
            if self.num_classes[name] >= 2:
                if return_prob:
                    preds = np.array([pred[i] for pred in all_predictions])
                    probs = preds/(preds.sum(axis=-1)).reshape((-1, 1))
                    mean_prob = probs.mean()
                    std_prob = probs.std()
                    for j in range(mean_prob.shape[-1]):
                        p_dic['{}_prob_{}'.format(name, j)] = mean_prob[:,j]
                        unc_dic['{}_prob_{}'.format(name, j)] = std_prob[:,j]
                else:
                    p_dic[name] = np.argmax(np.array([pred[i] for pred in all_predictions]).mean(axis=0), axis=1)
                    unc_dic[name] = np.max(np.array([pred[i] for pred in all_predictions]).mean(axis=0), axis=1)
            else:
                mean_p = np.array([pred[i] for pred in all_predictions]).mean(axis=0)
                std_p = np.array([pred[i] for pred in all_predictions]).std(axis=0)
                p_dic[name] = mean_p[:,0]
                unc_dic[name] = std_p[:,0]

        predictions = pd.DataFrame(p_dic)
        unc = pd.DataFrame(unc_dic)

        predictions.index = test_data.structure_ids
        unc.index = test_data.structure_ids

        if return_unc:
            return predictions, unc
        else:
            return predictions

    def fit_preset(*args,**kwargs):
        '''Deprecated, use the autofit_preset class instead'''

        raise RuntimeError(
            "Deprecated, use the autofit_preset class instead"
        )



class Bootstrap_MODNetModel(MODNetModel):
    """Container class for 100 Bootstrap Keras `Model`, that handles
    setting up the architecture, activations, training and learning curve.

    Attributes:
        n_feat: The number of features used in the model.
        weights: The relative loss weights for each target.
        optimal_descriptors: The list of column names used
            in training the model.
        model: The `keras.model.Model` of the network itself.
        target_names: The list of targets names that the model
            was trained for.

    """

    def __init__(self,*args,n_models=100,**kwargs):
        super().__init__(*args,**kwargs)
        self.n_models = n_models
        self.model = []
        for i in range(self.n_models):
            self.model.append(
                self.build_model(
                        self.targets, self.n_feat, self.num_neurons, act=self.act, num_classes=self.num_classes
            ))

    def fit(
        self,
        training_data: MODData,
        val_fraction: float = 0.0,
        val_key: Optional[str] = None,
        val_data: Optional[MODData] = None,
        lr: float = 0.001,
        epochs: int = 200,
        batch_size: int = 128,
        xscale: Optional[str] = "minmax",
        metrics: List[str] = ["mae"],
        callbacks: List[Callable] = None,
        verbose: int = 0,
        loss: str = "mse",
        **fit_params,
    ) -> None:
        """Train the model on the passed training `MODData` object.

        Paramters:
            training_data: A `MODData` that has been featurized and
                feature selected. The first `self.n_feat` entries in
                `training_data.get_optimal_descriptors()` will be used
                for training.
            val_fraction: The fraction of the training data to use as a
                validation set for tracking model performance during
                training.
            val_key: The target name to track on the validation set
                during training, if performing multi-target learning.
            lr: The learning rate.
            epochs: The maximum number of epochs to train for.
            batch_size: The batch size to use for training.
            xscale: The feature scaler to use, either `None`,
                `'minmax'` or `'standard'`.
            metrics: A list of Keras metrics to pass to `compile(...)`.
            loss: The built-in Keras loss to pass to `compile(...)`.
            fit_params: Any additional parameters to pass to `fit(...)`,
                these will be overwritten by the explicit keyword
                arguments above.

        """

        if self.n_feat > len(training_data.get_optimal_descriptors()):
            raise RuntimeError(
                "The model requires more features than computed in data. "
                f"Please reduce n_feat below or equal to {len(training_data.get_optimal_descriptors())}"
            )

        self.xscale = xscale
        self.target_names = list(self.weights.keys())
        self.optimal_descriptors = training_data.get_optimal_descriptors()

        x = training_data.get_featurized_df()[
            self.optimal_descriptors[: self.n_feat]
        ].values

        # For compatibility with MODNet 0.1.7; if there is only one target in the training data,
        # use that for the name of the target too.
        if len(self.targets_flatten) == 1 and len(training_data.df_targets.columns) == 1:
            self.targets_flatten = list(training_data.df_targets.columns)

        y = []
        for targ in self.targets_flatten:
            if self.num_classes[targ] >= 2:  # Classification
                y_inner = tf.keras.utils.to_categorical(
                    training_data.df_targets[targ].values,
                    num_classes=self.num_classes[targ],
                )
                loss = "categorical_crossentropy"
            else:
                y_inner = training_data.df_targets[targ].values.astype(
                    np.float, copy=False
                )
            y.append(y_inner)

        # Scale the input features:
        x = np.nan_to_num(x)
        if self.xscale == "minmax":
            self._scaler = MinMaxScaler(feature_range=(-0.5, 0.5))

        elif self.xscale == "standard":
            self._scaler = StandardScaler()

        x = self._scaler.fit_transform(x)

        if val_data is not None:
            val_x = val_data.get_featurized_df()[
                self.optimal_descriptors[: self.n_feat]
            ].values
            val_x = np.nan_to_num(val_x)
            val_x = self._scaler.transform(val_x)
            try:
                val_y = list(
                    val_data.get_target_df()[self.targets_flatten].values.astype(np.float, copy=False).transpose()
                )
            except Exception:
                val_y = list(
                    val_data.get_target_df().values.astype(np.float, copy=False).transpose()
                )
            validation_data = (val_x, val_y)
        else:
            validation_data = None

        # Optionally set up print callback
        if verbose:
            if val_fraction > 0 or validation_data:
                if self._multi_target and val_key is not None:
                    val_metric_key = f"val_{val_key}_mae"
                else:
                    val_metric_key = "val_mae"
                print_callback = tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: print(
                        f"epoch {epoch}: loss: {logs['loss']:.3f}, "
                        f"val_loss:{logs['val_loss']:.3f} {val_metric_key}:{logs[val_metric_key]:.3f}"
                    )
                )

            else:
                print_callback = tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: print(
                        f"epoch {epoch}: loss: {logs['loss']:.3f}"
                    )
                )

                if callbacks is None:
                    callbacks = [print_callback]
                else:
                    callbacks.append(print_callback)
        self.history =[]
        ### Resampling self.n_models times and fitting####
        for i in range(self.n_models):
            LOG.info(f"Fitting model #{i+1}/{self.n_models}")
            idxs = resample(np.arange(len(x)), replace=True, random_state=2943)
            x_bootstrap = x[idxs,...]
            y_bootstrap = [y_inner[idxs,...] for y_inner in y]

            fit_params = {
                "x": x_bootstrap,
                "y": y_bootstrap,
                "epochs": epochs,
                "batch_size": batch_size,
                "verbose": verbose,
                "validation_split": val_fraction,
                "validation_data": validation_data,
                "callbacks": callbacks,
            }

            self.model[i].compile(
                loss=loss,
                optimizer=tf.keras.optimizers.Adam(lr=lr),
                metrics=metrics,
                loss_weights=self.weights,
            )

            history = self.model[i].fit(**fit_params)
            self.history.append(history)
            model_summary=""
            for k in history.history.keys():
                model_summary+="{}: {:.4f}\t".format(k,history.history[k][-1])
            LOG.info(model_summary)

    def predict(self, test_data: MODData, return_unc=False, return_prob=False) -> pd.DataFrame:
        """Predict the target values for the passed MODData.

        Parameters:
            test_data: A featurized and feature-selected `MODData`
                object containing the descriptors used in training.
            return_prob: For a classification tasks only: whether to return the probability of each
                class OR only return the most probable class.

        Returns:
            A `pandas.DataFrame` containing the predicted values of the targets.


        """
        # prevents Nan predictions if some features are inf
        x = test_data.get_featurized_df().replace([np.inf, -np.inf, np.nan], 0)[
            self.optimal_descriptors[:self.n_feat]
        ].values

        # Scale the input features:
        x = np.nan_to_num(x)
        if self._scaler is not None:
            x = self._scaler.transform(x)
            x = np.nan_to_num(x)

        all_predictions = []
        for i in range(self.n_models):
            p = self.model[i].predict(x)
            if len(self.targets_flatten) ==1:
                p = np.array([p])
            all_predictions.append(p)

        p_dic = {}
        unc_dic = {}
        for i, name in enumerate(self.targets_flatten):
            if self.num_classes[name] >= 2:
                if return_prob:
                    preds = np.array([pred[i] for pred in all_predictions])
                    probs = preds/(preds.sum(axis=-1)).reshape((-1, 1))
                    mean_prob = probs.mean()
                    std_prob = probs.std()
                    for j in range(mean_prob.shape[-1]):
                        p_dic['{}_prob_{}'.format(name, j)] = mean_prob[:,j]
                        unc_dic['{}_prob_{}'.format(name, j)] = std_prob[:,j]
                else:
                    p_dic[name] = np.argmax(np.array([pred[i] for pred in all_predictions]).mean(axis=0), axis=1)
                    unc_dic[name] = np.max(np.array([pred[i] for pred in all_predictions]).mean(axis=0), axis=1)
            else:
                mean_p = np.array([pred[i] for pred in all_predictions]).mean(axis=0)
                std_p = np.array([pred[i] for pred in all_predictions]).std(axis=0)
                p_dic[name] = mean_p[:,0]
                unc_dic[name] = std_p[:,0]

        predictions = pd.DataFrame(p_dic)
        unc = pd.DataFrame(unc_dic)

        predictions.index = test_data.structure_ids
        unc.index = test_data.structure_ids

        if return_unc:
            return predictions, unc
        else:
            return predictions

    def fit_preset(*args,**kwargs):
        '''Deprecated, use the autofit_preset class instead'''

        raise RuntimeError(
            "Deprecated, use the autofit_preset class instead"
        )
