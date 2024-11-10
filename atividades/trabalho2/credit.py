import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp
import numpy as np
pd.set_option('future.no_silent_downcasting', True)

path = 'credit_risk_dataset.csv'

df = pd.read_csv(path)

print(10*'*-*-*-')


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
- loan_status: status do empréstimo (0 em dia, 1 está devendo)
- loan_percent_income: razão empréstimo/renda
- cb_person_default_on_file: histórico de dívidas
- cb_person_cred_hist_length: tamanho do histórico de crédito (em anos)

O objetivo é utilizar a coluna de loan_status como alvo e como essa é uma categoria binária
vamos usar uma Regressão logística para testar e prever a categoria.


*-*-*-*-*-* Análise Estatística *-*-*-*-*-*-

Como os dados das colunas loan_intent, loan_grade, cb_person_default_on_file, são textuais
não usamos eles para realizar a regressão.
'''

df = df.replace({'loan_grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7},
                'person_home_ownership': {'OWN': 1, 'RENT': 2, 'MORTGAGE': 3}, 
                'loan_intent': {'PERSONAL': 1,'EDUCATION': 2,'MEDICAL': 3,'VENTURE': 4,'HOMEIMPROVEMENT': 5,'DEBTCONSOLIDATION': 6}})


y = df['loan_status']
x = df.drop(['loan_status', 'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'], axis=1)

x = x[x['person_age'] < 100]
x = x[x['person_emp_length'] < 100]
x = x[pd.notna(x['loan_int_rate'])]


print(10*'*-*-*-')

print(x.info())

colunas = x.columns

#print(x[pd.notna(x['loan_int_rate'])])

#print(np.median(x['loan_int_rate']))
#print(sp.mode(x['loan_int_rate']))


for i in range(len(colunas)):
    plt.figure(i, layout='constrained')
    data = x[colunas[i]]
    mean = np.mean(data)
    median = np.median(data)
    mode = sp.mode(data)[0]
    plt.hist(data)
    plt.vlines(mean, 0, 32000, colors='r', label='Média')
    plt.vlines(median, 0, 32000, colors='g', label='Mediana')
    plt.vlines(mode, 0, 32000, colors='b', label='Moda')
    plt.title(colunas[i])
    plt.yscale('log')
    plt.legend()
    plt.savefig('credplot-'+colunas[i]+'.png', dpi=400)


