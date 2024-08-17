import numpy as np
import tensorflow as tf

def generate_data(num_samples=1000, input_dim=5, output_dim=3):
    """
    Gera dados fictícios para treinamento e teste do modelo.

    Parâmetros:
    - num_samples (int): Número de amostras a serem geradas.
    - input_dim (int): Dimensão das características de entrada.
    - output_dim (int): Número de ações possíveis (dimensão da saída).

    Retorna:
    - X (numpy.ndarray): Dados de entrada com formato (num_samples, input_dim).
    - y (numpy.ndarray): Dados de saída codificados em one-hot com formato (num_samples, output_dim).
    """
    np.random.seed(0)

    # Dados de entrada fictícios (ex: características ambientais)
    X = np.random.rand(num_samples, input_dim)

    # Dados de saída fictícios (ex: ações)
    y = np.random.randint(0, output_dim, size=(num_samples,))

    # Converter y para uma codificação one-hot
    y = tf.keras.utils.to_categorical(y, num_classes=output_dim)
    
    return X, y
