import numpy as np
from generate_data import generate_data
from build_model import build_model

def train_model():
    """
    Treina o modelo usando dados fictícios e salva o modelo treinado em um arquivo.
    """
    num_samples = 1000
    input_dim = 5
    output_dim = 3
    
    # Gerar dados fictícios
    X, y = generate_data(num_samples, input_dim, output_dim)
    
    # Construir o modelo
    model = build_model(input_dim, output_dim)
    
    # Treinar o modelo
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    
    # Salvar o modelo treinado
    model.save('protozoario_model.h5')
    print('Modelo treinado e salvo como "protozoario_model.h5"')

if __name__ == "__main__":
    train_model()
