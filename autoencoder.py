import numpy as np
from keras.models import Model
from keras.layers import Input, Dense

data = np.load('/Users/dan/Downloads/datarobot_text_features/datarobot_home_depot_search_term_features.npy')
n_obs = data.shape[0]
input_size = data.shape[1]
encoding_size = 50

x = Input(shape=(input_size,))
z = Dense(encoding_size, activation='sigmoid', name='z')(x)
x_reconstruction = Dense(input_size, activation='sigmoid', name='x_reconstruction')(z)

model = Model(input=[x], output=[z, x_reconstruction])
model.compile(optimizer='Adam',
              loss = 'mse',
              loss_weights = [1, 0]
              )

model.fit(x = data,
          y = {'x_reconstruction': data, 'z': np.zeros([n_obs, encoding_size])},
          nb_epoch=5)

encoded = model.predict(data)[0]
