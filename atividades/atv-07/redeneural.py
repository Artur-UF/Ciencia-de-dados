# Rede nerual para reconhecimento de digitos
# Importação de bibliotecas
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Carregar o conjunto de dados MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar os dados (valores de pixel entre 0 e 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Transformar os rótulos em vetores categóricos (one-hot encoding)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Construção da Rede Neural
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Camada de entrada: transforma 28x28 em vetor de 784 entradas
    Dense(128, activation='relu'),  # Camada oculta com 128 neurônios e ativação ReLU
    Dense(64, activation='relu'),   # Segunda camada oculta com 64 neurônios
    Dense(10, activation='softmax') # Camada de saída com 10 neurônios (1 para cada dígito) e ativação Softmax
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento do modelo
print("Treinando a Rede Neural...")
history = model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.2)

# Avaliação no conjunto de teste
print("\nAvaliando no conjunto de teste...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\nAcurácia no conjunto de teste: {test_accuracy * 100:.2f}%")

# Fazer previsões em novas imagens
predictions = model.predict(x_test[:5])  # Testar nas primeiras 5 imagens do conjunto de teste

# Mostrar os resultados
import matplotlib.pyplot as plt
import numpy as np

for i in range(5):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Verdadeiro: {np.argmax(y_test[i])}, Previsto: {np.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()

