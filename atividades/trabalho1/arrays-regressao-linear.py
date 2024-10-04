import matplotlib.pyplot as plt
import numpy as np
np.random.seed(666)

n = 20
erroy = np.random.normal(0, 0.9, n)

x = np.arange(0, n, 1)
y = x + erroy
deltay = abs(erroy/4)

# Definição das quantidades ponderadas
w = deltay**(-2)
wm = sum(w)/n
xwm = sum(x*w)/n
x2wm = sum((x**2)*w)/n
ywm = sum(y*w)/n
xywm = sum(x*y*w)/n
deltapond = wm*x2wm - xwm**2

# Coeficientes ponderados: a = angular || b = linear
apond = (wm*xywm - xwm*ywm)/deltapond
bpond = (ywm*x2wm - xywm*xwm)/deltapond

# Ajuste ponderado
xpond = np.linspace(0, 20, 100)
ypond = apond*xpond + bpond

# Erro dos coeficientes
deltaa2 = wm/deltapond
deltab2 = x2wm/deltapond

fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout='constrained')

plt.errorbar(x, y, yerr=deltay, ecolor='r', fmt='.k', elinewidth=1, capsize=2, capthick=0.4, zorder=3)
plt.plot(xpond, ypond, 'b', linewidth=0.7, label='Ajuste ponderado', zorder=1)
plt.plot()
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste por Mínimos Quadrados usando arrays\n'+f'a = {apond:.2f}, b = {bpond:.2f}\n'+r'$(\Delta a)^2 = $'+f'{deltaa2:.5f}, '+r'$(\Delta b)^2 = $'+f'{deltab2:.5f}')
plt.savefig('ajuste-arrays.png', dpi=400)

