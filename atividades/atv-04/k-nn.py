'''
1) IRIS
2) Regularizar dados
3) 
4) mudar o 'k' e comparar acurácia
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()

## Check the type of the object
#print(type(iris))  # <class 'sklearn.utils.Bunch'>
#
## Access the data, target, and other attributes
#print("Data shape:", iris.data.shape)               # (150, 4)
#print("Target shape:", iris.target.shape)           # (150,)
#print("Target names:", iris.target_names)           # ['setosa' 'versicolor' 'virginica']
#print("Feature names:", iris.feature_names)         # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
#print("Description:", iris.DESCR)                   # Description of the dataset


test_size = np.linspace(0.1, 0.9, 149)
nn = np.arange(1, 150, 1)
#print(test_size)
#print(nn)

def test_accuracy(kn, size):
    x, y = iris.data, iris.target

    # Divide o dataset para treinamento e teste
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size)
    #print(x_train)


    # Cria o classificador usando o número de features (neighbors)
    knn = KNeighborsClassifier(n_neighbors=kn)
    #print(knn)

    # Treina o classificador
    knn.fit(x_train, y_train)
    #print(knn.fit)

    # Previsões
    y_pred = knn.predict(x_test)


    # Avaliar acurácia
    return accuracy_score(y_test, y_pred)

fig, ax = plt.subplots(1, 1, figsize=(10, 10), layout='constrained')

phase_diagram = np.zeros((len(test_size), len(nn)))

samples = 50

for i in range(samples):
    for s in range(len(test_size)):
        for n in range(len(nn)):
            if nn[n] + 1 > 150 - test_size[s]*150:
                break
            phase_diagram[s][n] += test_accuracy(nn[n], test_size[s])
phase_diagram /= samples


plt.imshow(phase_diagram, origin='lower')
plt.colorbar()
plt.xticks(np.arange(0, len(nn), 1)[::10], nn[::10])
plt.yticks(np.arange(0, len(test_size), 1)[::10], list(f'{i:.2f}' for i in test_size[::10]))
plt.xlabel('k-neighbors')
plt.ylabel('test_size')

plt.savefig('phase_diagram.png', dpi=400)


