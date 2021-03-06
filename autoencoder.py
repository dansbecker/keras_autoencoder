import numpy as np
from keras.models import Model
from keras.layers import Input, Dense

def autoencode(data):
    n_obs = data.shape[0]
    input_size = data.shape[1]
    encoding_size = 50

    x = Input(shape=(input_size,))
    intermed = Dense(encoding_size * 2, activation='sigmoid')(x)
    z = Dense(encoding_size, activation='sigmoid', name='z')(intermed)
    intermed = Dense(encoding_size * 2, activation='sigmoid')(z)
    x_reconstruction = Dense(input_size, activation='sigmoid', name='x_reconstruction')(intermed)

    model = Model(input=[x], output=[z, x_reconstruction])
    model.compile(optimizer='Adam',
                  loss = 'mse',
                  loss_weights = [0, 1]
                  )

    model.fit(x = data,
              y = {'z': np.zeros([n_obs, encoding_size]),
                   'x_reconstruction': data},
              nb_epoch=20)

    encoded = model.predict(data)[0]
    return encoded

if __name__ == "__main__":
    data = np.load('unencoded_data.npy')
    encoded_data = autoencode(data)
