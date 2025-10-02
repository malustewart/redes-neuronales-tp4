import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import os

def plot_overlap_histograms(overlap_data, bins=20, save_folder=".\\figs"):
    """
    Plot one histogram per N-alfa combination.

    overlap_data: dict of the form
        overlap_data[N][alfa] = overlaps (array)
    """
    for N, alfa_dict in overlap_data.items():
        for alfa, overlaps in alfa_dict.items():
            plt.figure()
            plt.hist(overlaps, bins=bins, edgecolor='black', alpha=0.7)
            plt.title(f"Overlaps: N={N}, α={alfa}")
            plt.xlabel("Overlap m")
            plt.ylabel("Frecuencia")
            filename = os.path.join(save_folder, f"overlap_histogram_N_{N}_alfa_{alfa}.png")
            plt.savefig(filename)
            plt.close()


def plot_histograms_per_N(overlap_data, bins=20, save_folder=".\\figs"):
    """
    For each N, plot all α histograms overlapped.
    """
    for N, alfa_dict in overlap_data.items():
        plt.figure()
        for alfa, overlaps in alfa_dict.items():
            plt.hist(overlaps, bins=bins, alpha=0.5, label=f"α={alfa}", edgecolor='black')
        plt.title(f"Overlaps para N={N}")
        plt.xlabel("Overlap m")
        plt.ylabel("Frecuencia")
        plt.legend()
        filename = os.path.join(save_folder, f"overlap_histogram_N_{N}.png")
        plt.savefig(filename)
        plt.close()

def plot_histograms_per_alfa(overlap_data, bins=20, save_folder=".\\figs"):
    """
    For each α, plot all N histograms overlapped.
    """
    # First, find all α values
    alfas = set()
    for alfa_dict in overlap_data.values():
        alfas.update(alfa_dict.keys())
    alfas = sorted(list(alfas))

    for alfa in alfas:
        plt.figure()
        for N, alfa_dict in overlap_data.items():
            if alfa in alfa_dict:
                overlaps = alfa_dict[alfa]
                plt.hist(overlaps, bins=bins, alpha=0.5, label=f"N={N}", edgecolor='black')
        plt.title(f"Overlaps para α={alfa}")
        plt.xlabel("Overlap m")
        plt.ylabel("Frecuencia")
        plt.legend()
        filename = os.path.join(save_folder, f"overlap_histogram_alfa_{alfa}.png")
        plt.savefig(filename)
        plt.close()


def plot_conv_times(conv_times:dict, save_folder=".\\figs"):
    for N in conv_times.keys():
        for alfa in conv_times[N].keys():
            t_s, count_s = conv_times[N][alfa]["sequential"]
            t_p, count_p = conv_times[N][alfa]["parallel"]
            t_s = [str(int(t)) if t < np.inf else "No converge" for t in t_s]
            t_p = [str(int(t)) if t < np.inf else "No converge" for t in t_p]
            plt.figure()
            plt.bar(t_s, count_s, label="Secuencial", alpha=0.3, color='C0')
            plt.bar(t_p, count_p, label="Paralelo", alpha=0.3, color='C1')

            plt.title(f"Iteraciones hasta convergencia: N={N}, α={alfa}")
            plt.xlabel("Iteraciones hasta convergencia")
            plt.ylabel("Frecuencia")
            plt.legend()

            # Save plot
            filename = os.path.join(save_folder, f"conv_times_N_{N}_alfa_{alfa}.png")
            plt.savefig(filename)
            plt.close()
            print(f"Saved {filename}")

def plot_overlaps_vs_T(data, save_folder=".\\figs"):
    overlaps = data["overlaps"]
    Ts = data["Ts"]
    N = data["N"]
    alfa = data["alfa"]
    _, _, iters = np.shape(overlaps)
    last_iter = overlaps[:, :, -1]         # shape (len(Ts), p)
    mean_overlap = last_iter.mean(axis=1)  # average over patterns

    plt.figure()
    plt.plot(Ts, mean_overlap, marker="o")
    plt.xlabel("T")
    plt.ylabel("Overlap promedio")
    plt.title(f"Overlap promedio vs T (N={N}, alfa={alfa:.3f}, iteraciones={iters})")
    plt.grid(True)
    
    # Save plot
    filename = os.path.join(save_folder, f"overlaps_vs_T_N_{N}_alfa_{alfa}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_overlaps_evolution(data, save_folder=".\\figs"):
    # Plot the evolution of the first pattern overlap as system evolved
    # One plot for every T in Ts
    overlaps = data["overlaps"]
    Ts = data["Ts"]
    N = data["N"]
    alfa = data["alfa"]
    _, _, iters = np.shape(overlaps)

    plt.figure()
    for T, overlap in zip(Ts[::4],overlaps[::4]):
        plt.plot(range(iters + 1), np.concatenate([[1],np.mean(overlap, axis=0)]), marker="o", label=f"T={T}")
    plt.xlabel("Iteraciones")
    plt.ylabel("Overlap promedio")
    plt.title(f"Evolucion del overlap promedio (N={N}, alfa={alfa:.3f}, iteraciones={iters})")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    filename = os.path.join(save_folder, f"overlaps_evolution_N_{N}_alfa_{alfa}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def load_all_overlaps_ej_1(Ns, alfas):
    overlap_data = {}
    for N in Ns:
        overlap_data[N] = {}
        for alfa in alfas:
            p = int(alfa*N)
            filename = f".\\data\\tp4_1_4_N_{N}_p_{p}.npz"
            with np.load(filename) as data:
                overlaps = data["overlaps"]
                overlap_data[N][alfa] = overlaps
    return overlap_data

def load_all_data_ej_2(N, alfa, Ts, iters):
        T_str = "-".join(f"{T:.3f}" for T in Ts)
        filename = f".\\data\\tp4_2_overlap_2_N_{N}_alfa_{alfa:.3f}_T_{T_str}_iters_{iters}.npz"
        data = {}
        with np.load(filename) as d:
            data["Ts"] = d["T"]
            data["overlaps"] = d["overlaps"]  # shape (len(Ts), p, iters)
            data["N"] = d["N"].item()
            data["alfa"] = d["alfa"].item()
            data["iters"] = d["overlaps"].shape[2]
        return data

def get_conv_times(Ns, alfas):
    conv_data = {}
    for N in Ns:
        conv_data[N] = {}
        for alfa in alfas:
            p = int(alfa*N)
            conv_data[N][alfa] = {}
            filename = f".\\data\\tp4_1_3_N_{N}_p_{p}.npz"
            with np.load(filename) as data:
                conv_data[N][alfa]["sequential"] = np.unique(data["conv_time_sequential"], return_counts=True)
                conv_data[N][alfa]["parallel"] = np.unique(data["conv_time_parallel"], return_counts=True)
    return conv_data

def punto_1_3_process(Ns, alfas):
    print("******** PUNTO 1.3 PROCESAMIENTO *********")
    conv_times = get_conv_times(Ns, alfas)
    plot_conv_times(conv_times)

def punto_1_4_process(Ns, alfas):
    print("******** PUNTO 1.4 PROCESAMIENTO *********")

    overlap_data = load_all_overlaps_ej_1(Ns, alfas)

    bins = np.linspace(0.5,1.001,30)
    plot_overlap_histograms(overlap_data, bins=bins)
    plot_histograms_per_N(overlap_data, bins=bins)
    plot_histograms_per_alfa(overlap_data, bins=bins)

def punto_2_process(Ns, alfas, Ts, iters):
    print("******** PUNTO 2 PROCESAMIENTO *********")
    for N in Ns:
        for alfa in alfas:
            data = load_all_data_ej_2(N, alfa, Ts, iters)
            plot_overlaps_vs_T(data)
            plot_overlaps_evolution(data)

if __name__ == "__main__":
    # punto_1_3_process([3000],[0.1])
    # punto_1_4_process([500, 1000, 2000, 4000],[0.12, 0.14, 0.16, 0.18])
    punto_2_process([4000],[0.01],[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], iters=10)
