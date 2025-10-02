
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

== Tiempo de convergencia

```python
def update_parallel(s, w):
    weighted_sum = np.matmul(s,w)
    s[:] = np.array([1 if sum_i > 0 else -1 for sum_i in weighted_sum])

def update_sequential(s, w):
    for i, _ in enumerate(s):
        weighted_sum = np.dot(w[i],s)
        s[i] = 1 if weighted_sum > 0 else -1

```


#todo(position: "inline")[codigo de actualizacion de s secuencial]
#todo(position: "inline")[codigo de actualizacion de s paralelo]

#figure(image("figs/conv_times_N_3000_alfa_0.1.png"),
caption: "Tiempo de convergencia para modelo de Hopfield sin ruido con N=3000, α=0.1 (máximo 10 iteraciones de todo el sistema). Los casos marcados como \"No converge\" son aquellos en los que el sistema oscilaba entre dos estados.")


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
