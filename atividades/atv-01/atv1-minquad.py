import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"text.usetex" : True, "font.family" : "serif", "font.serif" : ["Computer Modern Serif"], "font.size" : 12})
np.random.seed(95623)


errox = np.random.normal(0, 0.4, 20)
erroy = np.random.normal(0, 0.9, 20)


x = np.arange(0, 20, 1) + errox
y = x + 2 + erroy
n = len(x)

xm = sum(x)/n
ym = sum(y)/n
x2m = sum(x**2)/n
xym = sum(x*y)/n
delta = (x2m*n) - (xm*xm)

a = ((xym*n) - (xm*ym))/delta
b = ((x2m*ym) - (xym*xm))/delta

w = (y - a*x - b)**(-2)
wm = sum(w)/n
xwm = sum(x*w)/n
x2wm = sum((x**2)*w)/n
ywm = sum(y*w)/n
xywm = sum(x*y*w)/n
deltapond = wm*x2wm - xwm**2

apond = (wm*xywm - xwm*ywm)/deltapond
bpond = (ywm*x2wm - xywm*xwm)/deltapond

# Ajuste não ponderado
xaj = np.linspace(0, 20, 100)
yaj = a*xaj + b

# Ajuste ponderado
xpond = np.linspace(0, 20, 100)
ypond = apond*xaj + bpond

deltab2 = x2wm/deltapond

fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout='constrained')

plt.scatter(x, y, s=3, c='k')
plt.plot(xaj, yaj, 'r', linewidth=0.7, label='Ajuste não ponderado')
plt.plot(xpond, ypond, 'b', linewidth=0.7, label='Ajuste ponderado')
plt.plot()
plt.grid()
plt.legend()
plt.title('Ajuste por Mínimos Quadrados\n'+f'NP: a = {a:.2f}, b = {b:.2f}\nP: a = {apond:.2f}, b = {bpond:.2f}, '+r'$(\Delta b)^2 = $'+f'{deltab2:.2f}')
plt.savefig('minquad-ajuste.png', dpi=400)

f = open('dados-atv1.csv', 'w')
for i in range(n):
    f.write(f'{x[i]},{y[i]}\n')
f.close()

