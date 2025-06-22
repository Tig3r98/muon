import numpy as np
import matplotlib.pyplot as plt
from numba import njit

#L correction

dist = [0.208, 0.517, 0.885, 1.115, 1.703]
N = 100_000_000
dim_x = 0.8
dim_y = 0.3

def sturges (N_eventi):
	return int( np.ceil ( 1 + 3.322 * np.log(N_eventi)))

def stdev_mean(x,bessel=True):
    s = np.std(x)
    return s/np.sqrt(len(x))

@njit
def sample_theta(n): #n is the number of events
    result = np.empty(n)
    for i in range(n):
        while True:
            theta_toy = np.random.uniform(0, np.pi / 2)
            y = np.random.uniform(0, 1)  # massimo di cos^2(theta) è 1
            if y < np.cos(theta_toy)**2:
                result[i] = theta_toy
                break
    return result

@njit
def generate_muon(z):
    theta = sample_theta(N)
    phi = np.random.uniform(0, 2 * np.pi, N)

    x = x_0 + np.abs(z)* np.tan(theta) * np.cos(phi)
    y = y_0 + np.abs(z)* np.tan(theta) * np.sin(phi)

    return x, y

def generate(L):
    x_muon, y_muon = generate_muon (L) #genera i muoni su S2 al variare di L
    L_corr = []
    count = 0
    for i in range(N):
        if (0 < x_muon[i] and x_muon[i]< dim_x and 0 < y_muon[i] and y_muon[i]< dim_y):
            count = count + 1
            dx = x_0[i] - x_muon[i]
            dy = y_0[i] - y_muon[i]
            dz = L
            L_corr.append( np.sqrt(dx**2 + dy**2 + dz**2) )
    return L_corr

#string util
import math
def format_with_uncertainty(value, error, sig_figs=2):
    if error == 0:
        return f"{value}", f"0"
    #exponent of the error
    exponent = math.floor(math.log10(abs(error)))
    #round error to desired sig figs
    err_rounded = round(error, -exponent + (sig_figs - 1))
    #round value to match the error's decimal place
    value_rounded = round(value, -exponent + (sig_figs - 1))
    #format strings
    fmt = f".{-exponent + (sig_figs - 1)}f" if exponent < sig_figs else ".0f"
    return f"{value_rounded:{fmt}}", f"{err_rounded:{fmt}}"

#genera i muoni su S1
print("Generating muon origins...")
x_0 = np.random.uniform(0, dim_x, N)
y_0 = np.random.uniform(0, dim_y, N)
for L in dist:
    print("Generating muons for L "+str(L)+"...")
    L_corr_array = generate(L)
    print('L = '+str(L)+'m:')
    L_corr = np.mean(L_corr_array)
    L_corr_err = stdev_mean(L_corr_array)
    L_corr_fmt, L_corr_err_fmt = format_with_uncertainty(L_corr, L_corr_err)
    print(f'L_corr({L}m): ({L_corr_fmt} ± {L_corr_err_fmt})m')
    #Plot
    xMax = np.max(L_corr_array)
    xMin = np.min(L_corr_array)
    Nbin = sturges(len(L_corr_array))
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.hist (L_corr_array, 
            bins = np.linspace(xMin, xMax, Nbin),
            color = 'skyblue',
            histtype = 'step'
            )
    ax.axvline(x = L_corr, color = 'r', label = 'L_corr')
    ax.set_title(f"L = {L}m corretto: ({L_corr_fmt} ± {L_corr_err_fmt})m", size=11)
    ax.set_xlabel ('Distanza percorsa (m)')
    ax.set_ylabel ('Conteggi')
    ax.set_yscale ('log')
    plt.savefig(str(L)+".png")