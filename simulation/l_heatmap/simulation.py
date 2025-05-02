import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from numba import njit
import time

# Variables
n_event = 100_000_000
dim_x = 0.8 #meters
dim_y = 0.3
#dim_z = 0.02 unused
bins_per_centimeter = 2

sim_start = 0.22
sim_step = 0.01
steps = 179

@njit # Much faster
def sample_theta(n):
    result = np.empty(n)
    for i in range(n):
        while True:
            theta_toy = np.random.uniform(0, np.pi / 2)
            y = np.random.uniform(0, 1)  # massimo di cos^2(theta) è 1
            if y < np.cos(theta_toy)**2:
                result[i] = theta_toy
                break
    return result

def sample_phi(n):
    return np.random.uniform(0, 2 * np.pi, n)

def simulate_events(args):
    l, n = args  # unpack the tuple
    print("[THREAD] Eseguendo simulazione con L="+str(round(l, 2))+" su "+str(n)+" eventi")
    
    # Generate x_0 y_0 as intersection point of muon on S1, so z_0=0
    x_0 = np.random.uniform(0, dim_x, n)
    y_0 = np.random.uniform(0, dim_y, n)
    # Generate muon directions
    phi = sample_phi(n)
    theta = sample_theta(n)

    # Calculate S2 plane coordinates
    tan_theta = np.tan(theta)
    x_1 = x_0 + tan_theta * l * np.cos(phi)
    y_1 = y_0 + tan_theta * l * np.sin(phi)
    
    # Coincidence condition
    mask = (0 < x_1) & (x_1 < dim_x) & (0 < y_1) & (y_1 < dim_y)
    
    return x_0[mask].tolist(), y_0[mask].tolist() #x_0s, y_0s with coincidence    

def simulate(z_1):
    
    l = np.abs(z_1)
    # Run parallel simulations
    n_processes = multiprocessing.cpu_count()
    events_per_proc = n_event // n_processes
    print("Eseguendo "+str(events_per_proc)+" eventi su "+str(n_processes)+" processori")
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        args_list = [(l, events_per_proc)] * n_processes
        results = list(executor.map(simulate_events, args_list))
    print("Simulazione terminata. Elaborazione risultati...")
    
    # Combine results
    x_values = np.concatenate([np.array(x) for x, _ in results])
    y_values = np.concatenate([np.array(y) for _, y in results])
    n_coinc = x_values.size
    
    perc = "0%"
    if n_coinc == 0:
        print("Nessuna coincidenza!")
        exit()
    else:
        perc = str(round((n_coinc/n_event)*100, 2))+"%"
        print("Coincidenze: "+str(n_coinc)+" su "+str(n_event)+" - "+perc)
    
    print("Calcolo istogramma...")
    
    heatmap, xedges, yedges = np.histogram2d(x_values, y_values, bins=(round(dim_x*100)*bins_per_centimeter, round(dim_y*100)*bins_per_centimeter))
    
    # Plot the heatmap
    plt.figure(figsize=(dim_x*10, dim_y*10+1))
    plt.imshow(
        heatmap.T,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin='lower',
        cmap='plasma',
        aspect='auto',
        #vmax=11000,
        #vmin=0
        #norm=mpl.colors.LogNorm(vmin=10, vmax=50)
    )
    plt.colorbar(label='Density')
    l_string = '{:.2f}'.format(round(l, 2))
    plt.title("L="+l_string+", "+str(n_event//1000000)+"Mevents, "+perc+" Aeff")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(False)
    plt.savefig(l_string+".png")
    print("Elaborazione completata, salvato file: "+l_string+".png")

if __name__ == "__main__":
    start = int(time.time())
    print("Eseguendo "+str(steps)+" simulazioni a partire da "+str(sim_start)+"m con step "+str(sim_step)+" e "+str(n_event)+" eventi per simulazioni")
    for i in range(steps):
        l = sim_start + (i) * sim_step
        simulate(l)
    stop = int(time.time())
    print("Programma terminato in "+str(stop-start)+"s.")