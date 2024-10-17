import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp

#with pd.ExcelFile('BRA.xlsx') as file:
#    print(file.sheet_names)

df = pd.read_excel('BRA.xlsx', 'Subnational 1 tree cover loss')

limiar = 75

df_filter = df[df['threshold'] == limiar]

cols = list(col for col in df_filter.columns if col.startswith('tc_loss_ha_') or col == 'subnational1' or col == 'area_ha')

years = np.asarray(list(int(y.split('_')[-1]) for y in cols[2:]))


df_filter = df_filter[cols]


# Funções de ajuste
def func1(x, a, b):
    return a + b*x

def func2(x, a, b):
    return a*x**b


fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')

index_cut_year = 8

states = ['Acre', 'Amazonas', 'Amapá', 'Roraima', 'Pará']

colors = ['g', 'r', 'b', 'purple', 'orange']

for i in range(len(states)):
    values = df_filter[df_filter['subnational1'] == states[i]].values.flatten()[1:]
    area = values[1]
    values = values[1:]
    values_cumul = np.cumsum(values)
    values_cumul = values_cumul/values_cumul[0]

    plt.subplot(121)
    plt.plot(years, values/area, colors[i], label=states[i])
    plt.subplot(122)
    plt.plot(years, values_cumul, colors[i], label=states[i]) #+f' | y(t) = {pr[0]:.2f}t^({pr[1]:.2f})')

    if states[i] == 'Amazonas':
        pr1, pcov1 = sp.curve_fit(func1, years[:index_cut_year-1], values_cumul[:index_cut_year-1])

        x1 = np.linspace(2002, 2018, 1000)
        y1 = func1(x1, pr1[0], pr1[1])

        plt.plot(x1, y1, 'k', linestyle='-.')

        log_years = np.log10(abs(years[-index_cut_year+1:]-years[index_cut_year]))
        log_values = np.log10(list(values_cumul)[-index_cut_year+1:])

        pr2, pcov2 = sp.curve_fit(func1, log_years, log_values)

        x2 = np.linspace(2010, 2024, 1000)
        y2 = func2(x2, pr2[0], pr2[1])

        plt.plot(x2, y2, 'k', linestyle='--')


plt.subplot(121)
plt.legend()
plt.xlabel('Anos')
plt.ylabel('Perda de cobertura proporcional')

plt.subplot(122)
plt.legend()
plt.xlabel('Anos')
plt.ylabel('Perda Cumulativa')

plt.savefig('new-comp.png', dpi=400)

