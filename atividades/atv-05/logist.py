import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

# Load the Breast Cancer dataset
data = load_breast_cancer()

# Access the data, target, and other attributes
print("Data shape:",    data.data.shape)               # (150, 4)
print("Target shape:",  data.target.shape)           # (150,)
print("Target names:",  data.target_names)           # ['setosa' 'versicolor' 'virginica']
print("Feature names:", data.feature_names)         # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print("Description:",   data.DESCR)                   # Description of the dataset





X = data.data  # Features
y = data.target  # Binary target (0 = benign, 1 = malignant)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X = scaler.fit_transform(X)

def logistic(x, y, t_size, rand):
    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=t_size, random_state=rand)

    # Initialize and train logistic regression model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

samples = 100

t_sizes = np.linspace(.1, .9, 20)

accs = np.zeros(len(t_sizes))

rand = 666

for j in range(samples):
    for i in range(len(t_sizes)):
        accs[i] += logistic(X, y, t_sizes[i], rand)
    rand += 1

accs /= samples

plt.plot(t_sizes, accs)


plt.savefig('plot-logistic.png', dpi=400)

