import numpy as np
import matplotlib.pyplot as plt
np.random.seed(666)

n = 20
erroy = list(np.random.normal(0, 0.9, n))

# Dados
x = list(range(0, n))
y = list(i + erroy[i] for i in x)
deltay = list(abs(erroy[i])/4 for i in range(n))

# Definição de quantidades ponderadas
w = list(deltay[i]**(-2) for i in range(n))
wm = sum(w)/n
xwm = sum(list(x[i]*w[i] for i in range(n)))/n
x2wm = sum(list((x[i]**2)*w[i] for i in range(n)))/n
ywm = sum(list(y[i]*w[i] for i in range(n)))/n
xywm = sum(list(x[i]*y[i]*w[i] for i in range(n)))/n
deltapond = wm*x2wm - xwm**2

# Coeficientes do ajuste ponderado: a = angular || b = linear
apond = (wm*xywm - xwm*ywm)/deltapond
bpond = (ywm*x2wm - xywm*xwm)/deltapond

# Ajuste ponderado
xpond = list(0 + (0.1*i) for i in range(201))
ypond = list(apond*xpond[i] + bpond for i in range(len(xpond)))

# Erro dos coeficientes
deltaa2 = wm/deltapond
deltab2 = x2wm/deltapond

fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout='constrained')

plt.errorbar(x, y, yerr=deltay, ecolor='r', fmt='.k', elinewidth=1, capsize=2, capthick=0.4, zorder=3)
plt.plot(xpond, ypond, 'b', linewidth=0.7, label='Ajuste ponderado', zorder=2)
plt.plot()
plt.legend()
plt.title('Ajuste por Mínimos Quadrados usando listas\n'+f'a = {apond:.2f}, b = {bpond:.2f}\n'+r'$(\Delta a)^2 = $'+f'{deltaa2:.5f}, '+r'$(\Delta b)^2 = $'+f'{deltab2:.5f}')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('ajuste-listas.png', dpi=400)

f = open('dados-trabalho1.csv', 'w')
f.write('x,y,y_err\n')
for i in range(n):
    f.write(f'{x[i]},{y[i]},{deltay[i]}\n')
f.close()

