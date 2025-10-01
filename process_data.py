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
            plt.figure()
            plt.bar(t_s, count_s, label="Secuencial", alpha=0.7, color='C0')
            plt.bar(t_p, count_p, label="Paralelo", alpha=0.7, color='C1')

            count_non_conv_s = count_s[np.where(t_s == np.inf)]
            count_non_conv_p = count_p[np.where(t_p == np.inf)]
            # plt.bar([-1], count_non_conv_s, alpha=0.7,color="C0")
            # plt.bar([-1], count_non_conv_p, alpha=0.7,color="C1", hatch="///")



            # # --- Sequential ---
            # for t, c in zip(t_s, count_s):
            #     if np.isinf(t):
            #         # Special bar for non-convergence
            #         plt.bar("No converge", c, label="Secuencial (no converge)", 
            #                 alpha=0.7, hatch="///", color="C0", edgecolor="black")
            #     else:
            #         plt.bar(t, c, label="Secuencial", alpha=0.7, color="C0", edgecolor="black")

            # # --- Parallel ---
            # for t, c in zip(t_p, count_p):
            #     if np.isinf(t):
            #         plt.bar("No converge", c, label="Paralelo (no converge)", 
            #                 alpha=0.7, hatch="\\\\\\", color="C1", edgecolor="black")
            #     else:
            #         plt.bar(t, c, label="Paralelo", alpha=0.7, color="C1", edgecolor="black")


            plt.title(f"Iteraciones hasta convergencia: N={N}, α={alfa}")
            plt.xlabel("Iteraciones hasta convergencia")
            plt.ylabel("Frecuencia")
            plt.legend()

            # Save plot
            filename = os.path.join(save_folder, f"conv_times_N_{N}_alfa_{alfa}.png")
            plt.savefig(filename)
            plt.close()
            print(f"Saved {filename}")

def load_all_overlaps(Ns, alfas):
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

    overlap_data = load_all_overlaps(Ns, alfas)

    bins = np.linspace(0.5,1.001,30)
    plot_overlap_histograms(overlap_data, bins=bins)
    plot_histograms_per_N(overlap_data, bins=bins)
    plot_histograms_per_alfa(overlap_data, bins=bins)

if __name__ == "__main__":
    punto_1_3_process([3000],[0.1])
    # punto_1_4_process([500, 1000, 2000, 4000],[0.12, 0.14, 0.16, 0.18])
