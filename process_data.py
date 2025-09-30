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
            filename = os.path.join(save_folder, f"histogram_N_{N}_alfa_{alfa}.png")
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
        filename = os.path.join(save_folder, f"histogram_N_{N}.png")
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
        filename = os.path.join(save_folder, f"histogram_alfa_{alfa}.png")
        plt.savefig(filename)
        plt.close()


def load_all_overlaps(Ns, alfas):
    overlap_data = {}
    for N in Ns:
        overlap_data[N] = {}
        for alfa in alfas:
            p = int(alfa*N)
            filename = f"tp4_1_4_N_{N}_p_{p}.npz"
            with np.load(filename) as data:
                overlaps = data["overlaps"]
                overlap_data[N][alfa] = overlaps
    return overlap_data


def punto_1_4_process(Ns, alfas):
    print("******** PUNTO 1.4 PROCESAMIENTO *********")
    for N in Ns:
        for alfa in alfas:
            overlap_data = load_all_overlaps(Ns, alfas)

            p = int(alfa*N)
            print(f"N={N} - alfa={alfa} - p={p}")

            filename = f"tp4_1_4_N_{N}_p_{p}.npz"
            with np.load(filename) as data:
                N = data["N"]
                p = data["p"]
                X = data["X"]
                w = data["w"]
                conv_points_s = data["conv_points_s"]
                conv_time_s = data["conv_time_s"]
                overlaps = data["overlaps"]

    bins = np.linspace(0.5,1.001,30)
    plot_overlap_histograms(overlap_data, bins=bins)
    plot_histograms_per_N(overlap_data, bins=bins)
    plot_histograms_per_alfa(overlap_data, bins=bins)

if __name__ == "__main__":
    punto_1_4_process([500, 1000, 2000],[0.12, 0.14])
    plt.show()
