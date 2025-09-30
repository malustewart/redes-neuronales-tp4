import numpy as np
import time

def create_random_memories(N,p):
    # returns p random memories
    # does not return the opposite patterns, which are also memories
    return np.random.choice(a=[-1, 1], size=(p,N))

def get_weights(memories):
    t0 = time.perf_counter()
    p, N = np.shape(memories)
    w = np.matmul(memories.transpose(), memories) / N
    np.fill_diagonal(w,0) # w_ii=0 for all i < N
    t1 = time.perf_counter()
    print(f"get_weights: {t1-t0}")
    return w

def get_weights_but_fast(memories):
    t0 = time.perf_counter()
    p, N = np.shape(memories)
    w = memories.T @ memories
    w = w.astype(np.float32) / N
    np.fill_diagonal(w,0) # w_ii=0 for all i < N
    t1 = time.perf_counter()
    print(f"get_weights_but_fast: {t1-t0} (N={N}, p={p})")
    return w

def create_random_state(N):
    return np.random.choice(a=[-1, 1],size=(N))

def state_to_str(state):
    return "".join(["-" if s < 0 else "+" for s in state])

def states_to_str(states):
    return "\n".join([state_to_str(state) for state in states])

def print_state(state, prefix):
    print(prefix, state_to_str(state))

def print_states(states, prefix=""):
    print(prefix, states_to_str(states), sep="\n")

def update_parallel(s, w):
    #updates s in-place in parallel
    weighted_sum = np.matmul(s,w)
    s[:] = np.array([1 if sum_i > 0 else -1 for sum_i in weighted_sum])

def update_sequential(s, w):
    #updates s in-place sequentially
    for i, _ in enumerate(s):
        weighted_sum = np.dot(w[i],s)
        s[i] = 1 if weighted_sum > 0 else -1

def update(s,w,mode="sequential"):
    return update_parallel(s,w) if mode=="parallel" else update_sequential(s,w)

def get_conv_time_and_point(s0, w, max_iters, mode="sequential"):
    s = s0.copy()
    prev_s = np.empty_like(s)
    for k in range(max_iters):
        prev_s[:] = s
        update(s, w, mode)
        print_state(prev_s, f"p{k}")
        print_state(s, f"s{k}")
        if np.equal(s,prev_s).all():
            return [s, k]
    return [s, np.inf]

def punto_1_3():
    N = 30
    alfa = 0.10
    p = int(alfa*N)

    X = create_random_memories(N, p)
    w = get_weights_but_fast(memories=X)

    print_states(X, "Memories:")

    t0 = time.perf_counter()
    print("PARALLEL")
    conv_points, conv_time = map(np.array, zip(*[ get_conv_time_and_point(x, w, 20, mode="parallel") for x in X ]))
    print(conv_time)
    print_states(conv_points)
    t1 = time.perf_counter()
    print("SEQUENTIAL")
    conv_points, conv_time = map(np.array, zip(*[ get_conv_time_and_point(x, w, 20, mode="sequential") for x in X ]))
    print(conv_time)
    print_states(conv_points)
    t2 = time.perf_counter()

    print(f"convergence total runtime parallel: {t1-t0}")
    print(f"convergence total runtime sequential: {t2-t1}")
    # print(f"parallel convergence iterations: \n{conv_parallel[:,1]}")
    # print(f"sequential convergence iterations: \n{conv_sequential[:][1]}")


if __name__ == "__main__":
    # N = 1000  # Cantidad de neuronas
    # alfa = 0.18
    # p = int(alfa*N)   # Cant de patrones

    np.random.seed(12345)

    # X = create_random_memories(N, p)
    # w = get_weights(memories=X)
    # w = get_weights_but_fast(memories=X)

    punto_1_3()

