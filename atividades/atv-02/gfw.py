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

years = list(int(y.split('_')[-1]) for y in cols[2:])


df_filter = df_filter[cols]

def func(x, a, b):
    return a + b*x

fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')


index_cut_year = 8

states = ['Acre', 'Amazonas', 'Amapá', 'Roraima', 'Pará']

for name in states:
    values = df_filter[df_filter['subnational1'] == name].values.flatten()[1:]
    area = values[1]
    values = values[1:]
    values_cumul = np.cumsum(values)
    values_cumul = values_cumul/values_cumul[0]

    log_years = np.log10(years)
    log_values = np.log10(list(values_cumul))

    pr, pcov = sp.curve_fit(func, years[-index_cut_year:], log_values[-index_cut_year:])

    x = np.linspace(2013, 2024, 1000)
    y = func(x, pr[0], pr[1])

    plt.subplot(121)
    plt.plot(years, values/area, label=name)
    plt.subplot(122)
    plt.plot(years, log_values, label=name)
    plt.plot(x, y)


plt.subplot(121)
plt.legend()
plt.xlabel('Anos')
plt.ylabel('Perda de cobertura proporcional')

plt.subplot(122)
plt.legend()
plt.xlabel('Anos')
plt.ylabel('Perda Cumulativa')

plt.savefig('comp.png', dpi=400)

