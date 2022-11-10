"""
Aaron Berndsen:
A Conformal Neural Network using Theano for computation and structure,
but built to obey sklearn's basic 'fit' 'predict' functionality

*code largely motivated from deeplearning.net examples
and Graham Taylor's "Vanilla RNN" (https://github.com/gwtaylor/theano-rnn/blob/master/rnn.py)

You'll require theano and libblas-dev

tips/tricks/notes:
* if training set is large (>O(100)) and redundant, use stochastic gradient descent (batch_size=1), otherwise use conjugate descent (batch_size > 1)
*
"""
import datetime
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from tensorflow.keras import layers, models


class CNN(BaseEstimator):
    """
    Conformal Neural Network,
    backend by TensorFlow, but compliant with sklearn interface.
    We determine the image input size (assumed square images) and
    the number of outputs in .fit from the training data


    Parameters
    ----------
    learning_rate : float, optional
    n_ephocs : int, optional
    batch_size: int
        number of samples in each training batch. Default 200.
    activation_funct: optional, str
        Chooses activation function, defaut 'tanh'
    nkerns: list of ints
        number of kernels on each layer
    kernel_size: list of ints, or 2-tuples
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
        Can be a single integer to specify the same value for all spatial dimensions.
    filters: int
        Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    pool_size: list of 2-tuples
        maxpooling in convolution layer (index-0),and direction x or y (index-1)
    loss_type : string, {'binary_crossentropy', 'mse',...}
        Type of loss function. This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
        Can be any from https://www.tensorflow.org/api_docs/python/tf/keras/losses
    optimizer : string, {'adam'}
        This is how the model is updated based on the data it sees and its loss function.
        Can be any from https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    metrics : list of strings, {['accuracy']}
        List of metrics to be evaluated by the model during training and testing. Used to monitor the training and testing steps.
    L1_reg : float, optional
    L2_reg : float, optional
    use_symbolic_softmax : bool, optional
    n_in : width (or length) of input image (assumed square)
    n_out : number of class labels
    """

    def __init__(self, **kwargs):
        self.activation = kwargs.get("activation", "tanh")
        self.loss_type = kwargs.get("loss_type", "binary_crossentropy")
        self.optimizer = kwargs.get("optimizer", "adam")
        self.metrics = kwargs.get("metrics", ["accuracy"])
        self.nkerns = kwargs.get("nkerns", [20, 45])
        self.n_hidden = kwargs.get("n_hidden", 500)
        self.kernel_size = kwargs.get("kernel_size", [15, 7])
        self.pool_size = kwargs.get("pool_size", [(3, 3), (2, 2)])
        self.n_epochs = kwargs.get("n_epochs", 60)
        self.batch_size = kwargs.get("batch_size", 25)
        self.L1_reg = kwargs.get("L1_reg", 0.00)
        self.L2_reg = kwargs.get("L2_reg", 0.00)
        # Note, n_in and n_out are actually set in
        # .fit, they are here to help pickle
        # self.n_in = kwargs.get("n_in", 50)
        # self.n_out = kwargs.get("n_out", 2)

        # self.setup_model()

    def setup_model(self):
        """
        Setup the model in TF for the CNN

        Notes
        -----
        This function constructs the actual layers,

        There are three layers:
        layer0 : a convolutional filter making kernel_size[0] shifted copies,
                 then downsampled by max pooling in grids of pool_size[0]
                 (N, 1, nx, ny)
                 --> (N, nkerns[0], nx1, ny1)  (nx1 = nx - kernel_size[0][0] + 1)
                                      (ny1 = ny - kernel_size[0][1] + 1)
                 --> (N, nkerns[0], nx1/pool_size[0][1], ny1/pool_size[0][1])
        layer1 : a convolutional filter making kernel_size[1] shifted copies,
                 then downsampled by max pooling in grids of pool_size[1]
                 (N, nkerns[0], nx1/2, ny1/2)
                 --> (N, nkerns[1], nx2, ny2) (nx2 = nx1 - kernel_size[1][0] + 1)
                 --> (N, nkerns[1], nx3, ny3) (nx3 = nx2/pool_size[1][0], ny3=ny2/pool_size[1][1])
        layer2 : hidden layer of nkerns[1]*nx3*ny3 input features and n_hidden hidden neurons
        layer3 : final LR layer with n_hidden neural inputs and n_out outputs/classes
        """

        # shape of input images
        nx, ny = self.n_in, self.n_in

        # Reshape matrix of rasterized images of shape (batch_size, nx*ny)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # layer0_input = input_tensor.reshape((self.batch_size, 1, nx, ny))
        # Need to change above for TF?

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (nx-filx+1,ny-fily+1)
        # maxpooling reduces this further to (nx/poosize[0][0],ny/pool_size[0][1])
        # 4D output tensor is thus of shape (batch_size,nkerns[0],xx,yy)
        nim = self.kernel_size[0]
        if isinstance(nim, int):
            fil1x = nim
            fil1y = nim
        else:
            fil1x = nim[0]
            fil1y = nim[1]

        # Don't think we need this
        # rng = np.random.RandomState(23455)

        # Model init
        self.model = models.Sequential()

        ####################################
        # I think I have to change this to TF
        ####################################
        # filters? kernel_size? activation?

        # input_shape_1 = (self.batch_size, nx, ny, 1)
        input_shape_1 = (nx, ny, 1)
        kernel_size_1 = (fil1x, fil1y)

        # print(vars(self))
        # print("input_shape:", input_shape_1)
        # print("batch_size:", self.batch_size)
        # print("kernel_size:", self.kernel_size)
        # print("kernel_size_1:", kernel_size_1)
        # print("poolsize:", self.pool_size)
        # print("nkerns:", self.nkerns)

        self.model.add(
            layers.Conv2D(
                filters=self.nkerns[0],
                # self.batch_size,
                # kernel_size=self.kernel_size,
                kernel_size=kernel_size_1,
                activation=self.activation,
                input_shape=input_shape_1,
                # data_format="channels_first",
                padding="same",
            )
        )
        # print("1")
        # print(self.model.summary())
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_size[0],
                # data_format="channels_first"
            )
        )
        """
        self.layer0 = LeNetConvPoolLayer(
            rng,
            input_layer=layer0_input,
            image_shape=(batch_size, 1, nx, ny),
            filter_shape=(nkerns[0], 1, fil1x, fil1y),
            pool_size=pool_size[0],
        )
        """
        ####################################

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (nbin-nim+1,nbin-nim+1) = x
        # maxpooling reduces this further to (x/2,x/2) = y
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],y,y)
        poox = (nx - fil1x + 1) / self.pool_size[0][0]
        pooy = (ny - fil1y + 1) / self.pool_size[0][1]
        nconf = self.kernel_size[1]
        if isinstance(nconf, int):
            fil2x = nconf
            fil2y = nconf
        else:
            fil2x = nconf[0]
            fil2y = nconf[1]

        input_shape_2 = (poox, pooy, 1)
        kernel_size_2 = (fil2x, fil2y)
        ####################################
        # I think I have to change this to TF
        ####################################
        # filters? kernel_size? activation?
        self.model.add(
            layers.Conv2D(
                filters=self.nkerns[1],
                kernel_size=kernel_size_2,
                activation=self.activation,
                input_shape=input_shape_2,
                padding="same",
            )
        )
        self.model.add(layers.MaxPooling2D(pool_size=self.pool_size[1]))

        # print("2")
        # print(self.model.summary())

        """
        self.layer1 = LeNetConvPoolLayer(
            rng,
            input_layer=self.layer0.output,
            image_shape=(batch_size, nkerns[0], poox, pooy),
            filter_shape=(nkerns[1], nkerns[0], fil2x, fil2y),
            pool_size=pool_size[1],
        )
        """
        ####################################

        # the TanhLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)
        self.model.add(layers.Flatten())
        # layer2_input = self.layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        poo2x = (poox - fil2x + 1) / self.pool_size[1][0]
        poo2y = (pooy - fil2y + 1) / self.pool_size[1][1]

        # units : Positive integer, dimensionality of the output space.
        self.model.add(
            layers.Dense(self.nkerns[1] * poo2x * poo2y, activation=self.activation)
        )
        """
        self.layer2 = HiddenLayer(
            rng,
            input_layer=layer2_input,
            n_in=nkerns[1] * poo2x * poo2y,
            n_out=n_hidden,
            activation=T.tanh,
        )
        """

        # classify the values of the fully-connected sigmoidal layer

        # Not sure if this really is a logistic regression in TF
        self.model.add(layers.Dense(units=1))
        """
        self.layer3 = LogisticRegression(
            input_layer=self.layer2.output, n_in=n_hidden, n_out=n_out
        )
        """

        ##############################################
        # Not sure what all of this does.
        """
        # CNN regularization
        self.L1 = self.layer3.L1
        self.L2_sqr = self.layer3.L2_sqr

        # create a list of all model parameters to be fit by gradient descent
        self.params = (
            self.layer3.params
            + self.layer2.params
            + self.layer1.params
            + self.layer0.params
        )

        self.y_pred = self.layer3.y_pred
        self.p_y_given_x = self.layer3.p_y_given_x
        """

        # optimizer?, loss?, metrics?
        # String (name of optimizer) or optimizer instance. See tf.keras.optimizers.
        # Loss function. Maybe be a string (name of loss function), or a tf.keras.losses.Loss instance.
        # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance.
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss_type, metrics=self.metrics
        )

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training set.

        y : array-like, shape = [n_samples]
            Labels for X.

        Returns
        -------
        z : float

        """
        return np.mean(self.model.predict(X) == y)

    def fit(
        self,
        train_data,
        train_labels,
        test_data=None,
        test_labels=None,
        n_epochs=None,
    ):
        """Fit model

        Pass in test_data, test_labels to compute test error and report during
        training.

        train_data : ndarray (N_pfds x n_in)
        train_labels : ndarray (N_pfds x n_out)
        test_data : ndarray (N_pfds x n_in)
        test_labels : ndarray (N_pfds x n_out)

        n_epochs : int, used to override self.n_epochs from init.
        """
        # prepare the CNN
        if not isinstance(train_data, np.ndarray):
            train_data = np.asarray(train_data)
        else:
            print(type(train_data))

        print("data shape", np.shape(train_data))
        print("data", train_data)
        print("sub data shape", np.shape(train_data[0]))
        print("label shape", np.shape(train_labels))
        print("labels", train_labels)
        self.n_in = int(np.sqrt(train_data.shape[1]))
        self.n_out = len(np.unique(train_labels))

        print("n_in", self.n_in, "n_out", self.n_out)
        reshaped_train_data = np.reshape(
            train_data, (train_data.shape[0], self.n_in, self.n_in)
        )
        print(np.shape(reshaped_train_data))

        self.setup_model()

        if n_epochs:
            self.n_epochs = n_epochs

        if test_data and test_labels:
            self.trained_model = self.model.fit(
                reshaped_train_data,
                train_labels,
                epochs=self.n_epochs,
                validation_data=(test_data, test_labels),
            )
        else:
            self.trained_model = self.model.fit(
                reshaped_train_data,
                train_labels,
                epochs=self.n_epochs,
            )

    def predict(self, data):
        """
        the CNN expects inputs with Nsamples = self.batch_size.
        In order to run 'predict' on an arbitrary number of samples we
        pad as necessary.

        """
        """
        if isinstance(data, list):
            data = np.array(data)
        if data.ndim == 1:
            data = np.array([data])

        nsamples = data.shape[0]
        n_batches = nsamples // self.batch_size
        n_rem = nsamples % self.batch_size
        if n_batches > 0:
            preds = [
                list(
                    self.model.predict(
                        data[i * self.batch_size : (i + 1) * self.batch_size]
                    )
                )
                for i in range(n_batches)
            ]
        else:
            preds = []
        if n_rem > 0:
            z = np.zeros((self.batch_size, self.n_in * self.n_in))
            z[0:n_rem] = data[
                n_batches * self.batch_size : n_batches * self.batch_size + n_rem
            ]
            preds.append(self.model.predict(z)[0:n_rem])

        return np.hstack(preds).flatten()
        """
        print("Not implemented")
        return None

    def predict_proba(self, data):
        """
        the CNN expects inputs with Nsamples = self.batch_size.
        In order to run 'predict_proba' on an arbitrary number of samples we
        pad as necessary.

        """
        """
        if isinstance(data, list):
            data = np.array(data)
        if data.ndim == 1:
            data = np.array([data])

        nsamples = data.shape[0]
        n_batches = nsamples // self.batch_size
        n_rem = nsamples % self.batch_size
        if n_batches > 0:
            preds = [
                list(
                    self.model.predict(
                        data[i * self.batch_size : (i + 1) * self.batch_size]
                    )
                )
                for i in range(n_batches)
            ]
        else:
            preds = []
        if n_rem > 0:
            z = np.zeros((self.batch_size, self.n_in * self.n_in))
            z[0:n_rem] = data[
                n_batches * self.batch_size : n_batches * self.batch_size + n_rem
            ]
            preds.append(self.model.predict(z)[0:n_rem])

        return np.vstack(preds)
        """
        print("Not implemented")
        return None

    def errors(self, y):
        """Return a float representing the number of errors in the sequence
        over the total number of examples in the sequence ; zero one
        loss over the size of the sequence

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_out.ndim:
            raise TypeError(
                "y should have the same shape as self.y_out",
                ("y", y.type, "y_pred", self.y_pred.type),
            )
        """
        print("Not implemented")
        return None

    def __getstate__(self):
        """Return state sequence."""
        """
        # check if we're using ubc_AI.classifier wrapper,
        # adding it's params to the state
        if hasattr(self, "orig_class"):
            superparams = self.get_params()
            # now switch to orig. class (MetaCNN)
            oc = self.orig_class
            cc = self.__class__
            self.__class__ = oc
            params = self.get_params()
            for k, v in superparams.items():
                params[k] = v
            self.__class__ = cc
        else:
            params = self.get_params()  # sklearn.BaseEstimator
        if hasattr(self, "cnn"):
            weights = [p.get_value() for p in self.cnn.params]
        else:
            weights = []
        state = (params, weights)
        return state
        """
        print("Not implemented")
        return None

    def _set_weights(self, weights):
        """Set fittable parameters from weights sequence.

        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        """
        i = iter(weights)
        if hasattr(self, "cnn"):
            for param in self.cnn.params:
                param.set_value(i.next())
        """
        print("Not implemented")
        return None

    def __setstate__(self, state):
        """Set parameters from state sequence.

        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        """
        params, weights = state
        # we may have several classes or superclasses
        for k in ["n_comp", "use_pca", "feature"]:
            if k in params:
                self.set_params(**{k: params[k]})
                params.pop(k)

        # now switch to MetaCNN if necessary
        if hasattr(self, "orig_class"):
            cc = self.__class__
            oc = self.orig_class
            self.__class__ = oc
            self.set_params(**params)
            self.ready()
            if len(weights) > 0:
                self._set_weights(weights)
            self.__class__ = cc
        else:
            self.set_params(**params)
            self.ready()
            self._set_weights(weights)
        """
        print("Not implemented")
        return None

    def save(self, fpath=".", fname=None):
        """Save a pickled representation of Model state."""
        """
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == ".pkl":
            # User supplied an absolute path to a pickle file
            fpath, fname = os.path.split(fpath)

        elif fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime("%Y-%m-%d-%H:%M:%S")
            class_name = self.__class__.__name__
            fname = "%s.%s.pkl" % (class_name, date_str)

        fabspath = os.path.join(fpath, fname)

        # logger.info("Saving to %s ..." % fabspath)
        file = open(fabspath, "wb")
        state = self.__getstate__()
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()
        """
        print("Not implemented")
        return None

    def load(self, path):
        """Load model parameters from path."""
        # logger.info("Loading from %s ..." % path)
        """
        file = open(path, "rb")
        state = pickle.load(file)
        self.__setstate__(state)
        file.close()
        """
        print("Not implemented")
        return None
