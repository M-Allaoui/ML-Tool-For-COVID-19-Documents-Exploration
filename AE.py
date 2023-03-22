import pandas as pd
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.initializers import VarianceScaling
from sklearn.utils.linear_assignment_ import linear_assignment

def autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape=(dims[0],), name='input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')
x = pd.read_csv("CORD19.csv")
x=np.array(x)
print(np.shape(x))
dims = [x.shape[-1], 500, 500, 2000, 100]
init = VarianceScaling(scale=1. / 3., mode='fan_in',
                           distribution='uniform')
pretrain_optimizer = SGD(lr=0.01, momentum=0.9)
pretrain_epochs = 50
batch_size = 50

autoencoder, encoder = autoencoder(dims, init=init)

path = F"C:/Users/GOOD DAY/PycharmProjects/umap_clustering/umap"
autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
#autoencoder.compile(optimizer='adam', loss='mse')#mse

autoencoder.load_weights('ae_weights_CORD19.h5')
#autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs) #, callbacks=cb)

#autoencoder.save_weights(path + '/AE_weights/ae_weights_cifar10_vgg16.h5')
#autoencoder.load_weights(path + '/ae_weights_reuters2.h5')
data=encoder.predict(x)

np.savetxt('CORD19_sae.csv', data, delimiter=',', fmt='%f')