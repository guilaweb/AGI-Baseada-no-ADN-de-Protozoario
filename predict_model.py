import numpy as np
import tensorflow as tf

def load_model():
    """
    Carrega o modelo treinado a partir do arquivo.

    Retorna:
    - model (tf.keras.Model): Modelo de rede neural carregado.
    """
    return tf.keras.models.load_model('protozoario_model.h5')

def predict_actions(model, X_test):
    """
    Faz previsões sobre as ações com base nas características de entrada.

    Parâmetros:
    - model (tf.keras.Model): Modelo de rede neural treinado.
    - X_test (numpy.ndarray): Dados de entrada para teste.

    Retorna:
    - actions (numpy.ndarray): Ações previstas pelo modelo.
    """
    predictions = model.predict(X_test)
    actions = np.argmax(predictions, axis=1)
    return actions

def main():
    """
    Executa o processo de previsão com o modelo treinado e exibe os resultados.
    """
    input_dim = 5
    output_dim = 3

    # Gerar novos dados fictícios para teste
    X_test = np.random.rand(10, input_dim)
    
    # Carregar o modelo treinado
    model = load_model()
    
    # Fazer previsões
    actions = predict_actions(model, X_test)
    
    # Exibir resultados
    for i, action in enumerate(actions):
        if action == 0:
            print(f'Amostra {i+1}: Mover para frente')
        elif action == 1:
            print(f'Amostra {i+1}: Reproduzir')
        elif action == 2:
            print(f'Amostra {i+1}: Ficar parado')

if __name__ == "__main__":
    main()
