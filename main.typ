
#import "@preview/basic-report:0.3.1": *
#import "@preview/dashy-todo:0.1.2": todo

#set math.equation(numbering: "(1)")

#show: it => basic-report(
  doc-category: "Redes neuronales",
  doc-title: "Trabajo práctico 4: \nMemorias asociativas",
  author: "María Luz Stewart Harris",
  affiliation: "Instituto Balseiro",
  logo: image("assets/balseiro.png", width: 5cm),
  language: "es",
  compact-mode: true,
  it
)
// #image("assets/balseiro.png", width: 5cm)

= Modelo de Hopfield sin ruido<sec:sin_ruido>

== Creación de patrones
Se creó un modelo de Hopfield de $N$ neuronas con $p$ patrones, donde $1$ y $-1$ son los posibles estados de una neurona y son equiprobables: 
```python
def create_random_memories(N,p):
    return np.random.choice(a=[-1, 1], size=(p,N))
```
== Evaluación de matriz de conexiones
A partir de los patrones generados, se evaluó la matriz de conexiones:

```python
def get_weights(memories):
    p, N = np.shape(memories)
    w = memories.T @ memories
    w = w.astype(np.float32) / N
    np.fill_diagonal(w,0)
    return w
```

== Tiempo de convergencia

Tomando cada patrón generado como condición inicial, se iteró el sistema hasta que:
    + $s_i = s_(i-1)$, siendo $s_i$ el estado en la iteración $i$.
    + Se alcancen las 10 iteraciones.

En caso de que se hayan alcanzado las 10 iteraciones sin que se cumpla la condición $s_i = s_(i-1)$, se considera que el sistema no convergió.
Se encontró que en todos los casos donde no había convergencia, el sistema oscilaba entre 2 estados.

El calculo del estado $s_i$ a partir de $s_(i-1)$ y $w$ se realizó de dos formas diferentes, llamadas secuencial y paralela:

```python
def update_parallel(s, w):
    weighted_sum = np.matmul(s,w)
    s[:] = np.array([1 if sum_i > 0 else -1 for sum_i in weighted_sum])

def update_sequential(s, w):
    for i, _ in enumerate(s):
        weighted_sum = np.dot(w[i],s)
        s[i] = 1 if weighted_sum > 0 else -1

```

#figure(
  image("figs/conv_times_N_3000_alfa_0.1.png"),
  caption: "Tiempo de convergencia para modelo de Hopfield sin ruido con N=3000, α=0.1 (máximo 10 iteraciones de todo el sistema). Los casos marcados como \"No converge\" son aquellos en los que el sistema oscilaba entre dos estados."
)<fig:t_convergencia>

La #ref(<fig:t_convergencia>) muestra la distribución de tiempo de convergencia partiendo de cada uno de los patrones creados, tanto para el cálculo secuencial del estado como para el paralelo.
La media del tiempo de convergencia del cálculo secuencial es menor a la del cálculo paralelo (1,16 iteraciones y 1,45 iteraciones respectivamente).
Además, el modelo con cálculo secuencial converge para los 300 patrones utilizados como valores iniciales $s(t=0)$, mientras que al utilizar el cálculo paralelo el modelo no convergió en 3 casos.

== Cálculo de overlaps

Para todos los casos simulados con el cálculo secuencial, se obtuvo el overlap entre el estado inicial $s(t=0)$ y el estado al que convergió:
```python
def get_overlap(s,s0):
    N = len(s)
    return np.dot(s,s0) / N
```

#figure(
    grid(
      columns: 2,
      rows: 2,
      gutter: 0mm,
      image("figs/overlap_histogram_N_500.png"),
      image("figs/overlap_histogram_N_1000.png"),
      image("figs/overlap_histogram_N_2000.png"),
      image("figs/overlap_histogram_N_4000.png"),
    ),
  caption: "Histogramas del overlap entre el punto inicial y el punto de convergencia, tomando como puntos iniciales todos los patrones almacenados."
)<fig:overlaps_fixed_N>

La #ref(<fig:overlaps_fixed_N>) muestra histogramas de $m$ (los overlaps) para diferentes $alpha$ y $N$. 
Recordando que:
 - $m=1$ significa que el sistema convergió exactamente al patrón utilizado como estado inicial $s(t=0)$
 - $m=0$ significa que el sistema convergió a un estado no correlacionado a $s(t=0)$
 - $m=-1$ significa que el sistema convergió a un estado opuesto a $s(t=0)$
Se comprueba que mientras menor es $alpha$, $s(t=0)$ y el estado de convergencia son más similares, dado que el valor medio de la distribución de $m$ tiende a 1.

Para todo $N$, mientras menor es $alpha$ más lejano a 1 es el promedio de $m$ (es decir, el patrón recuperado y el utilizado como $(s(t=0)$ son más diferentes). Por lo tanto, para un dado $N$ existe una relación de compromiso entre la cantidad de patrones que se pueden recuperar ($N alpha$) y la capacidad del sistema de recuperar exactamente el patrón deseado.

= Modelo de Hopfield con ruido

Se creó un modelo de Hopfield similar al de la #ref(<sec:sin_ruido>), con la distinción de que la actualización del estado de una neurona tiene un componente aleatorio:
$ h_i (t) &= sum_(j=1)^N w_(i j) s_j (t) \
P(s_i (t+1) = 1)  &= e^(beta h_i(t))/(e^(beta h_i (t))+e^(-beta h_i (t))) \ 
P(s_i (t+1) = -1) &= 1 - P(s_i = 1) \
T &= 1/beta $ <eq:ruido>

```python
def update_sequential_noisy(s,w,beta):
    for i, _ in enumerate(s):
        exp_bh = np.exp(beta * np.dot(w[i],s))
        P_i = exp_bh/(exp_bh+1./exp_bh)
        s[i] = random.choices([-1, 1], weights=[1-P_i, P_i], k=1)[0]
```

De la #ref(<eq:ruido>):

$ lim_(T->0) P(s_i (t+1) = 1) &= lim_(beta->infinity) e^(beta h_i (t))/(e^(beta h_i (t))+e^(-beta h_i (t))) \
& = cases(
  0 "si" h_i (t) < 0,
  1/2 "si" h_i (t) = 0,
  1 "si" h_i (t) > 0
) $ <eq:ruido_T_0>

$ lim_(T->infinity) P(s_i (t+1) = 1) &= lim_(beta->0) e^(beta h_i (t))/(e^(beta h_i (t))+e^(-beta h_i (t))) =  1/2 forall h_i (t) $ <eq:ruido_T_infinity>

La #ref(<eq:ruido_T_0>) muestra que para $T=0$ el sistema se comporta igual que al no tener ruido mientras que la #ref(<eq:ruido_T_infinity>) muestra que para $T$ muy grande el estado #box[$s(t+1)$] deja de depender del estado $s(t)$ y de la matriz de pesos y pasa a tomar un valor aleatorio. Concordando con esto, la simulación de la #ref(<fig:m_vs_T>) muestra que mientras mayor es $T$, menos correlacionados está el estado $s(t=10)$ con el patrón inicial utilizado como $s(t=0)$, promediado para todos los patrones. La #ref(<fig:m_vs_tiempo>) muestra como se va alejando el estado $s(t)$ del patrón inicial utilizado como $s(t=0)$ dependiendo de T (promediado para todos los patrones). Nuevamente se observa que mientras mayor es $T$, menor es la correlación entre $s(t)$ y el estado inicial.

#figure(
  image("figs/overlaps_vs_T_N_4000_alfa_0.01.png"),
  caption: [Overlap promedio entre s(t=0) y s(t=10) en función de T.]
)<fig:m_vs_T>



#figure(
  image("figs/overlaps_evolution_N_4000_alfa_0.01.png"),
  caption: [Evolución del overlap promedio entre $s(t=0)$ y $s(t=i)$.]
)<fig:m_vs_tiempo>

= Anexo
El repostorio de código utilizado para simular y graficar se encuentra en: #box[#link( "https://github.com/malustewart/redes-neuronales-tp4" )]
