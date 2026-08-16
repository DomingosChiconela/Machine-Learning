from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np


model = keras.Sequential([
    layers.Dense(units=1, input_shape=[1])
])


# Configurando o optimizador e a funcao de perda
model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)


# Dados
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


# Configurando a paragem antecipada
early_stopping = EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True
)


hist = model.fit(
    xs,
    ys,
    epochs=500,
    callbacks=[early_stopping]
)




print(model.predict(np.array([10.0])))