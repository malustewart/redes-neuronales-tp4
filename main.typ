
#import "@preview/basic-report:0.3.1": *
#import "@preview/dashy-todo:0.1.2": todo

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
#image("assets/balseiro.png", width: 5cm)

= Modelo de Hopfield sin ruido

== Creación de patrones
Se creó un modelo de Hopfield de $N$ neuronas con $p$ patrones: 
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

El sistema puede iterarse de forma secuencial o paralela:

```python
def update_parallel(s, w):
    weighted_sum = np.matmul(s,w)
    s[:] = np.array([1 if sum_i > 0 else -1 for sum_i in weighted_sum])

def update_sequential(s, w):
    for i, _ in enumerate(s):
        weighted_sum = np.dot(w[i],s)
        s[i] = 1 if weighted_sum > 0 else -1

```

Tomando cada patrón generado como condición inicial, se itera el sistema hasta que:
    + $s_i = s_(i-1)$, siendo $s_i$ el estado en la iteración $i$.
    + Se alcancen las 10 iteraciones.

En caso de que se alcancen las 10 iteraciones sin que se cumpla que $s_i = s_(i-1)$, se considera que el sistema no convergió.
Se encontró que en todos los casos donde no había convergencia, el sistema oscilaba entre al menos 2 estados.

#figure(image("figs/conv_times_N_3000_alfa_0.1.png"),
caption: "Tiempo de convergencia para modelo de Hopfield sin ruido con N=3000, α=0.1 (máximo 10 iteraciones de todo el sistema). Los casos marcados como \"No converge\" son aquellos en los que el sistema oscilaba entre dos estados.")

#todo(position: "inline")[comentar algo sobre el grafico]

== Cálculo de overlaps

#figure(image("figs/overlap_histogram_N_500.png"),
caption: "Histograma del overlap entre el punto inicial y el punto fijo de convergencia, tomando como puntos iniciales todos los patrones almacenados (N=500)."
)


#todo(position: "inline")[Poner mas graficos de overlap para Ns mas grandes y poner que Se observa que al aumentar α empeora la capacidad para recuperar recuerdos]

#todo(position: "inline")[Poner mas graficos de overlap de alfas fijos y variando N y hacer algun comentario]


= Modelo de Hopfield con ruido

#todo(position: "inline")[figura de overlap en funcion de T, y agregar comentario que explica que va bajando el overlap porque se pierde la correlacion entre el punto inicial y el final]

#todo(position: "inline")[poner grafico de evolucion de overlap promedio en varias iteraciones para distintos alfas, y mencionar como es que al agregar mucho recuerdos se deteriora la capacidad de recordar esos recuerdos]

= Anexo

#todo(position: "inline")[poner link al repositorio]
