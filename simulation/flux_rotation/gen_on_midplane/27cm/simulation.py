import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from numba import njit
import time

# Variables
n_event = 100_000_000
dim_x = 0.8 #meters (long axis)
dim_y = 0.3
#dim_z = 0.02 unused
bins_per_centimeter = 1 #2

sim_start = 0
sim_step = 5
steps = 19

@njit # Much faster
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

def sample_phi(n):
    return np.random.uniform(0, 2 * np.pi, n)

#rodriguez
#rot_axis = np.array([1, 0, 0])  # X-axis
#dirs = np.stack((dx, dy, dz), axis=1)  # Shape: (n, 3)
#rotated_dirs = np.array([rotate_vector(v, rot_axis, rot) for v in dirs])
#dx_rot, dy_rot, dz_rot = rotated_dirs[:, 0], rotated_dirs[:, 1], rotated_dirs[:, 2]
#def rotate_vector(v, axis, angle):
#    axis = axis / np.linalg.norm(axis)
#    cos_a = np.cos(angle)
#    sin_a = np.sin(angle)
#    cross = np.cross(axis, v)
#    dot = np.dot(v, axis)
#    return v * cos_a + cross * sin_a + axis * dot * (1 - cos_a)

@njit
def to_cart(theta, phi):
    '''Compute direction vectors from polar angles'''
    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)
    return dx, dy, dz

@njit
def rot_x(dx, dy, dz, rot):
    '''Rotate vector components around the x axis by rot radians'''
    # x-component remains unchanged
    dy_f = dy * np.cos(rot) - dz * np.sin(rot)
    dz_f = dy * np.sin(rot) + dz * np.cos(rot)
    return dx, dy_f, dz_f

from math import sqrt
@njit
def normalize(dx, dy, dz):
    '''Normalize a vector'''
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    dx /= norm
    dy /= norm
    dz /= norm
    return dx, dy, dz


def simulate_events(args):
    l, rot_deg, n = args  # unpack the tuple (lenght, rotation, number of events)
    rot = np.deg2rad(rot_deg)
    print("[THREAD] Eseguendo simulazione con L="+str(round(l, 2))+" e t="+str(rot_deg)+" su "+str(n)+" eventi")
    
    # Generate x_0 y_0 as intersection point of muon on S1, so z_0=0
    x_0 = np.random.uniform(0, dim_x, n)
    y_0 = np.random.uniform(0, dim_y, n)
    
    # Generate muon directions
    phi = sample_phi(n)
    theta = sample_theta(n)
    
    # Simple way of applying a rotation around x-axis:
    # Convert to direction vectors
    dx, dy, dz = to_cart(theta, phi)
    # Rotate direction by rot around x-axis (simulate detector rotation) by applying Rx(theta) matrix
    dx, dy, dz = rot_x(dx, dy, dz, rot)
    # Normalize rotated direction vector
    dx, dy, dz = normalize(dx, dy, dz)
    
    # Return to polar coordinates
    phi = np.arctan2(dy, dx) #Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
    theta = np.arccos(dz)

    # Calculate S2 plane coordinates
    tan_theta = np.tan(theta)
    x_1 = x_0 + tan_theta * l * np.cos(phi)
    y_1 = y_0 + tan_theta * l * np.sin(phi)
    
    # Coincidence condition
    mask = (0 < x_1) & (x_1 < dim_x) & (0 < y_1) & (y_1 < dim_y)
    
    return x_0[mask].tolist(), y_0[mask].tolist() #x_0s, y_0s with coincidence    

def simulate(z_1, theta):
    
    l = np.abs(z_1) #distance between detectors
    #theta is the azimuthal angle    
    
    # Run parallel simulations
    n_processes = multiprocessing.cpu_count()
    events_per_proc = n_event // n_processes
    print("Eseguendo "+str(events_per_proc)+" eventi su "+str(n_processes)+" processori")
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        args_list = [(l, theta, events_per_proc)] * n_processes
        results = list(executor.map(simulate_events, args_list))
    print("Simulazione terminata. Elaborazione risultati...")
    
    # Combine results
    x_values = np.concatenate([np.array(x) for x, _ in results])
    y_values = np.concatenate([np.array(y) for _, y in results])
    n_coinc = x_values.size
    
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
        #vmax=40000,
        #vmin=0
        #norm=mpl.colors.LogNorm(vmin=10, vmax=50)
    )
    plt.colorbar(label='Density')
    plt.title("theta="+str(theta)+", "+str(n_event//1000000)+"Mevents, "+str(n_coinc//1000)+"k coinc.")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(False)
    plt.savefig(str(theta)+".png")
    print("Elaborazione completata, salvato file: "+str(theta)+".png")

if __name__ == "__main__":
    start = int(time.time())
    print("Eseguendo "+str(steps)+" simulazioni a partire da "+str(sim_start)+"° con step "+str(sim_step)+"° e "+str(n_event)+" eventi per simulazioni")
    for i in range(steps):
        theta = sim_start + (i) * sim_step
        simulate(.27/2, theta)
    stop = int(time.time())
    print("Programma terminato in "+str(stop-start)+"s.")