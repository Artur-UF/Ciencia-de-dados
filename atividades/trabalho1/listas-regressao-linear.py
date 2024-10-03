import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
erroy = list(np.random.normal(0, 0.9, 20))

x = list(range(0, 20))
y = list(i + erroy[i] for i in x)
n = len(x)

# O 'm' no nome significa a média dessas grandezas
xm = sum(x)/n
ym = sum(y)/n
x2m = sum(list(x[i]**2 for i in range(n)))/n
xym = sum(list(x[i]*y[i] for i in range(n)))/n
delta = (x2m*n) - (xm*xm)

# a = coeficiente angular
a = ((xym*n) - (xm*ym))/delta
# b = coeficiente linear
b = ((x2m*ym) - (xym*xm))/delta

# Chi^2 e as quantidades ponderadas
w = list((y[i] - a*x[i] - b)**(-2) for i in range(n))
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

deltaa2 = wm/deltapond
deltab2 = x2wm/deltapond

fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout='constrained')

plt.scatter(x, y, s=3, c='k', zorder=3)
plt.plot(xpond, ypond, 'b', linewidth=0.7, label='Ajuste ponderado', zorder=2)
plt.plot()
plt.legend()
plt.title('Ajuste por Mínimos Quadrados usando listas\n'+f'a = {apond:.2f}, b = {bpond:.2f}\n'+r'$(\Delta a)^2 = $'+f'{deltaa2:2f}, '+r'$(\Delta b)^2 = $'+f'{deltab2:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('ajuste-listas.png', dpi=400)

f = open('dados-trabalho1.csv', 'w')
f.write('x,y,w_i\n')
for i in range(n):
    f.write(f'{x[i]},{y[i]},{w[i]}\n')
f.close()

