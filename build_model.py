import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

def build_model(input_dim, output_dim):
    """
    Constrói uma rede neural para simular o comportamento dos protozoários.

    Parâmetros:
    - input_dim (int): Dimensão das características de entrada.
    - output_dim (int): Número de ações possíveis (dimensão da saída).

    Retorna:
    - model (tf.keras.Model): Modelo de rede neural compilado.
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
