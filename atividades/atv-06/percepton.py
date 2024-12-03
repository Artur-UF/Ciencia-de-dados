import numpy as np
import matplotlib.pyplot as plt

# Dados de entrada (X) e rótulos (y)
X = np.array([[2, 1], [1, -1], [-1, -1], [-2, 1], [0, 1], [0, -1], [1, 2], [2, -1]])
y = np.array([1, 0, 0, 1, 1, 0, 1, 0])

# Inicialização de parâmetros
w = np.zeros(X.shape[1])  # Pesos (w1, w2)
b = 0                     # Bias
learning_rate = 1         # Taxa de aprendizado
epochs = 10               # Número de épocas

# Treinamento
for epoch in range(epochs):
    for idx, x_i in enumerate(X):
        # Saída prevista
        z = np.dot(x_i, w) + b
        y_pred = 1 if z >= 0 else 0

        # Atualização de pesos e bias
        error = y[idx] - y_pred
        w += learning_rate * error * x_i
        b += learning_rate * error

        #print(f'{w} | {b} | {error}')

# Resultado
print(f"Pesos finais: {w}")
print(f"Bias final: {b}")


# Teste do modelo
def predict(X):
    return [1 if np.dot(x, w) + b >= 0 else 0 for x in X]


print(f'Alvos originais: {y}')
print(f"Predições:       {predict(X)}")

xlim = (-3, 3)
ylim = (-3, 3)

x = np.linspace(-3, 3)
y = (-w[0]*x + b)/w[1]


plt.figure(1, layout='constrained')
plt.scatter(X[:, 0], X[:, 1], zorder=3)
plt.plot(x, y, 'r', linewidth=0.8)

plt.vlines(0, ylim[0], ylim[1], 'k', linewidth=0.5)
plt.hlines(0, xlim[0], xlim[1], 'k', linewidth=0.5)

plt.xlim(xlim[0], xlim[1])
plt.ylim(ylim[0], ylim[1])
plt.xlabel('x')
plt.ylabel('y')

plt.savefig('plot.png', dpi=400)

