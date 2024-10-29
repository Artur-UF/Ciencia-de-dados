import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv', encoding='utf-8')

df_filtered = df.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Country_Region', 'Lat', 'Long_', 'Combined_Key'], axis=1)

# Extrai datas e cria contagem de dias
values_columns = df_filtered.drop(['Admin2', 'Province_State'], axis=1).columns

date_begin = values_columns[0]
date_end = values_columns[-1]

t = np.arange(0, len(values_columns), 1)

fig, ax = plt.subplots(1, 1, figsize=(10, 10), layout='constrained')

cities = ['New York', 'Los Angeles', 'Washington']

for i in range(len(cities)):
    plt.plot(t, df_filtered[df_filtered['Admin2'] == cities[i]].values.flatten()[2:], label=cities[i])

plt.xlabel(date_begin+f'-'+date_end)
plt.ylabel('Cases')
plt.legend()
plt.savefig('plot_covid.png', dpi=400)

