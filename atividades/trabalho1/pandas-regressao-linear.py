import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dt = pd.read_csv('dados-trabalho1.csv')

x = dt['x']
y = dt['y']
deltay = dt['y_err']
n = len(dt.index)

# Definição das quantidades ponderadas
w = deltay**(-2)
wm = w.mean()
xwm = (x*w).mean()
x2wm = ((x**2)*w).mean()
ywm = (y*w).mean()
xywm = (x*y*w).mean()
deltapond = wm*x2wm - xwm**2

# Coeficientes ponderados: a = angular || b = linear
apond = (wm*xywm - xwm*ywm)/deltapond
bpond = (ywm*x2wm - xywm*xwm)/deltapond

# Ajuste ponderado
xpond = np.linspace(0, 20, 100)
ypond = apond*xpond + bpond

# Erros dos coeficientes
deltaa2 = wm/deltapond
deltab2 = x2wm/deltapond

fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout='constrained')

plt.errorbar(x, y, yerr=deltay, ecolor='r', fmt='.k', elinewidth=1, capsize=2, capthick=0.4, zorder=3)
plt.plot(xpond, ypond, 'b', linewidth=0.7, label='Ajuste ponderado', zorder=1)
plt.plot()
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste por Mínimos Quadrados usando pandas\n'+f'a = {apond:.2f}, b = {bpond:.2f}\n'+r'$(\Delta a)^2 = $'+f'{deltaa2:.5f}, '+r'$(\Delta b)^2 = $'+f'{deltab2:.5f}')
plt.savefig('ajuste-pandas.png', dpi=400)

