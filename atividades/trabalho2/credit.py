import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp
import numpy as np

# Clasificador logistico
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

pd.set_option('future.no_silent_downcasting', True)

path = 'credit_risk_dataset.csv'

df = pd.read_csv(path)

print(df.info())

'''

*-*-*-*-*-* Descrição Geral *-*-*-*-*-*-*-*-

O dataset contêm dados referentes a clientes que tomaram empréstimos

Dataset composto por 11 colunas:
- person_age: idade
- person_income: renda anual
- person_home_ownership: propriedade da casa
- person_emp_length: tempo empregado (em anos)
- loan_intent: intenção do empréstimo
- loan_grade: categoria do empréstimo
- loan_amnt: valor total do empréstimo
- loan_int_rate: taxa de juros do empréstimo
- loan_status: status do empréstimo (0 pagamento em dia, 1 está devendo)
- loan_percent_income: razão empréstimo/renda
- cb_person_default_on_file: histórico de dívidas
- cb_person_cred_hist_length: tamanho do histórico de crédito (em anos)

O objetivo é utilizar a coluna de loan_status como alvo e como essa é uma categoria binária
vamos usar uma Regressão logística para testar e prever a categoria.


*-*-*-*-*-* Análise Estatística *-*-*-*-*-*-

Como os dados das colunas loan_intent, loan_grade, cb_person_default_on_file, são textuais
não usamos eles para realizar a regressão.
'''

x = df.drop(['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'], axis=1)

x = x[x['person_age'] < 100]
x = x[x['person_emp_length'] < 100]
x = x[pd.notna(x['loan_int_rate'])]

y = x['loan_status'].values
x = x.drop('loan_status', axis=1)


print(10*'*-*-*-')

print(x.info())

colunas = x.columns

for i in range(len(colunas)):
    plt.figure(i, layout='constrained')
    data = x[colunas[i]]
    media = np.mean(data)
    mediana = np.median(data)
    moda = sp.mode(data)[0]
    plt.hist(data)
    plt.vlines(media, 0, 32000, colors='r', label='Média')
    plt.vlines(mediana, 0, 32000, colors='g', label='Mediana')
    plt.vlines(moda, 0, 32000, colors='b', label='Moda')
    plt.title(colunas[i])
    plt.yscale('log')
    plt.legend()
    plt.savefig('credplot-'+colunas[i]+'.png', dpi=400)


tamanhos_teste = np.linspace(0.05, 0.99, 100)

def LogReg(x, y, tamanhos_teste):
    '''
    Faz a regressão logística para uma amostra separada em dados e alvo
    '''
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=tamanhos_teste)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def mede_acuracia(x, y, n_acur):

    acuracia = np.zeros(len(tamanhos_teste))

    for i in range(len(tamanhos_teste)):
        for j in range(n_acur):
            acuracia[i] += LogReg(x, y, tamanhos_teste[i])

    acuracia /= n_acur
    return acuracia


n_acur = 50


atributos = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

plt.figure(20, layout='constrained')
plt.plot(tamanhos_teste, mede_acuracia(x, y, n_acur), c='k', label='Todos atributos', zorder=10)
for i in range(len(atributos)):
    plt.plot(tamanhos_teste, mede_acuracia(x.drop(atributos[i], axis=1), y, n_acur), linestyle='dashed', label='Menos '+atributos[i])
plt.ylabel('Acurácia')
plt.xlabel('Tamanho da amostra teste')
plt.ylim(0.8170, 0.8350)
plt.legend()
plt.savefig('acc-percent-income.png', dpi=400)



