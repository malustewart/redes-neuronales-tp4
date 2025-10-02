import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

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

def update_parallel_noisy(s, w, beta):
    exp_bh = np.exp(beta * np.matmul(w,s))
    P = exp_bh/(exp_bh+1./exp_bh)
    s[:] = np.array([random.choices([-1, 1], weights=[1-p, p], k=1)[0] for p in P])

def update_sequential_noisy(s,w,beta):
    for i, _ in enumerate(s):
        exp_bh = np.exp(beta * np.dot(w[i],s))
        P_i = exp_bh/(exp_bh+1./exp_bh)
        s[i] = random.choices([-1, 1], weights=[1-P_i, P_i], k=1)[0]

def update_noisy(s, w, beta, mode="sequential"):
    return update_parallel_noisy(s,w, beta) if mode=="parallel" else update_sequential_noisy(s,w, beta)

def get_next_state_noisy(s, w, beta, overlaps, iters=10, mode="sequential"):
    for i in range(iters):
        update_noisy(s, w, beta, mode)

def get_overlap(s,x):
    N = len(s)
    return np.dot(s,x) / N

def get_conv_time_and_point(s0, w, max_iters, mode="sequential"):
    s = s0.copy()
    prev_s = np.empty_like(s)
    for k in range(max_iters):
        prev_s[:] = s
        update(s, w, mode)
        if np.equal(s,prev_s).all():
            return [s, k]
    return [s, np.inf]

def punto_1_3_calc(Ns, alfas):
    print("******** PUNTO 1.3 CALCULO *********")
    for N in Ns:
        for alfa in alfas:
            p = int(alfa*N)
            print(f"N={N} - alfa = {alfa} - p={p}")

            X = create_random_memories(N, p)
            w = get_weights_but_fast(memories=X)

            conv_points_p, conv_time_p = map(np.array, zip(*[ 
                get_conv_time_and_point(x, w, 20, mode="parallel") 
                for x in tqdm(X, desc="Parallel calculation") 
            ]))
            conv_points_s, conv_time_s = map(np.array, zip(*[ 
                get_conv_time_and_point(x, w, 20, mode="sequential") 
                for x in tqdm(X, desc="Sequential calculation")
            ]))
            filename = f".\data\tp4_1_3_N_{N}_p_{p}.npz"
            np.savez(filename,
                     N=N,
                     p=p,
                     X=X,
                     w=w,
                     conv_points_p=conv_points_p,
                     conv_time_p=conv_time_p,
                     conv_points_s=conv_points_s,
                     conv_time_s=conv_time_s,
                     )

def punto_1_4_calc(Ns, alfas):
    print("******** PUNTO 1.4 CALCULO *********")
    for N in Ns:
        for alfa in alfas:
            p = int(alfa*N)
            print(f"N={N} - alfa = {alfa} - p={p}")

            X = create_random_memories(N, p)
            w = get_weights_but_fast(memories=X)

            conv_points_s, conv_time_s = map(np.array, zip(*[ 
                get_conv_time_and_point(x, w, 20, mode="sequential") 
                for x in tqdm(X, desc="Sequential calculation")
            ]))
            overlaps = [get_overlap(s,x) for s,x,t in zip(conv_points_s, X, conv_time_s) if t < np.inf]
            filename = f".\data\tp4_1_4_N_{N}_p_{p}.npz"
            np.savez(filename,
                     N=N,
                     p=p,
                     X=X,
                     w=w,
                     conv_points_s=conv_points_s,
                     conv_time_s=conv_time_s,
                     overlaps=overlaps,
                     )

def punto_2_calc(Ns, alfas, Ts, iters=10):
    print("******** PUNTO 1.4 CALCULO *********")

    for N in Ns:
        for alfa in alfas:
            p = int(N*alfa)
            overlaps = np.ndarray((len(Ts), p, iters), np.float32)

            for i, T in tqdm(enumerate(Ts), f"N={N} - alfa={alfa}"):
                print(f"\nT={T}\n")
                beta = 1/T
                X = create_random_memories(N, p)
                w = get_weights_but_fast(memories=X)
                for j, x in enumerate(X):
                    s = x.copy()
                    for k in range(iters):
                        update_noisy(s, w, beta, mode="sequential")
                        overlaps[i][j][k] = get_overlap(s,x)

            T_str = "-".join(f"{T:.3f}" for T in Ts)
            filename = f".\\data\\tp4_2_overlap_2_N_{N}_alfa_{alfa:.3f}_T_{T_str}_iters_{iters}.npz"
            np.savez(filename,
                    T=Ts,
                    N=N,
                    alfa=alfa,
                    overlaps=overlaps
                    )

if __name__ == "__main__":
    np.random.seed(12345)


