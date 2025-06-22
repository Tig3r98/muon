import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import skew
from scipy.stats import kurtosis
import time
import sys
import random 
import math
from scipy.stats import chi2
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import expon, norm
from iminuit.cost import ExtendedBinnedNLL
from mpl_toolkits.mplot3d import Axes3D
from numba import njit
#threading
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

#generare N numeri psaudo casuli distriubuti con una distribuzione sferica uniforme
L = 0.2
N = 1_000_000
R = 10 #hemisphere radius

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

@njit
def generate_hemisphere(R, n):
    '''Generate n cartesian coordinates on a hemisphere'''
    # phi angle uniform between 0 and 2pi
    phi = np.random.uniform(0, 2 * np.pi, n)

    # cos(theta) uniform between 0 and 1
    #u = np.random.uniform(0, 1, n)
    #theta = np.arccos(u)  # theta between 0 and pi/2
    
    # fix for uniform theta
    u = np.random.uniform(0, 1, n) 
    theta = 0.5 * np.arccos(1 - 2 * u)
    
    # consider cos² distribution too - NOT WORKING
    #u = np.random.rand(n)
    #theta = np.arccos((1 - u)**(1/3))

    # Coordinate cartesiane
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    
    return x, y, z

@njit
def generate_plane(R, n):
    x = np.random.uniform(-R,R,n)
    y = np.random.uniform(-R,R,n)
    z = np.full(n, 10)
    
    return x, y, z

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
    
@njit
def project_to_plane(x, y, z, dx, dy, dz, z_plane):
    """
    Projects rays from (x, y, z) in direction (dx, dy, dz) onto z = z_plane
    """
    # Avoid division by zero: mask rays that go straight along z-plane
    valid = dz != 0
    t = (z_plane - z[valid]) / dz[valid]
    
    # Only keep forward-going rays
    forward = t < 0
    t = t[forward]
    
    x_proj = x[valid][forward] + t * dx[valid][forward]
    y_proj = y[valid][forward] + t * dy[valid][forward]
    
    return x_proj, y_proj
    
    # Also filter out points outside the sphere's cap circle (needed for flux visualization errors)
    #r2 = x_proj**2 + y_proj**2
    #r_max2 = R**2 - z_plane**2
    #inside = r2 <= r_max2
    
    #return x_proj[inside], y_proj[inside]

#FASTER!
#@njit
#def project_to_plane(x, y, z, dx, dy, dz, z_plane):
#    n = len(x)
#    
#    # Inizializza array temporanei al massimo della lunghezza possibile
#    x_proj = np.empty(n, dtype=np.float64)
#    y_proj = np.empty(n, dtype=np.float64)
#    
#    count = 0
#    for i in range(n):
#        # avoid division by zero
#        if dz[i] == 0.0:
#            continue  
#        # keep only backward-going rays (t < 0)
#        t = (z_plane - z[i]) / dz[i]
#        if t >= 0:
#            continue  
#        # project
#        x_t = x[i] + t * dx[i]
#        y_t = y[i] + t * dy[i]
#        r2 = x_t**2 + y_t**2
#        r_max2 = R**2 - z_plane**2
#        if r2 <= r_max2:
#            x_proj[count] = x_t
#            y_proj[count] = y_t
#            count += 1
#
#    return x_proj[:count], y_proj[:count]

# Fast binning
import numba
@njit(parallel=True)
def parallel_hist2d(x, y, bins, x_range, y_range):
    H = np.zeros((bins, bins), dtype=np.int64)
    dx = (x_range[1] - x_range[0]) / bins
    dy = (y_range[1] - y_range[0]) / bins

    for i in numba.prange(x.size):
        xi = int((x[i] - x_range[0]) / dx)
        yi = int((y[i] - y_range[0]) / dy)
        if 0 <= xi < bins and 0 <= yi < bins:
            H[xi, yi] += 1
    return H

from matplotlib.colors import LogNorm
def plot_flux_heatmaps(x, y, z, dx, dy, dz, R, name):
    z_planes = np.arange(9.5, 10.0, 0.10)  # z = 0, 0.5, ..., 2.0
    bins = 50 #100 # resolution of heatmap
    extent = [-R, R, -R, R]  # XY limits based on sphere

    rows = 1
    fig, axes = plt.subplots(nrows=rows, ncols=5, figsize=(20, rows*4))
    axes = axes.flatten()
    
    for i, z_plane in enumerate(z_planes):
        print("Elaborazione grafico "+str(i)+"...")
        x_proj, y_proj = project_to_plane(x, y, z, dx, dy, dz, z_plane)
        
        #H, xedges, yedges = np.histogram2d(
        #    x_proj, y_proj, bins=bins, range=[[extent[0], extent[1]], [extent[2], extent[3]]]
        #)
        # Parallelizza il binning
        H = parallel_hist2d(x_proj, y_proj, bins, [extent[0], extent[1]], [extent[2], extent[3]])
        
        ax = axes[i]
        im = ax.imshow(
            H.T,
            extent=extent,
            origin='lower',
            cmap='inferno',
            aspect='equal',
            vmax=40000,
            vmin=30000
            #norm=LogNorm(vmin=20, vmax=20000)
        )
        ax.set_title(f'z = {z_plane:.1f} m')
        ax.set_xlabel('x (m)')
        if i == 0:
            ax.set_ylabel('y (m)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    #plt.show()
    plt.savefig(name+".png")


@njit
def ray_plane_intersection(x0, y0, z0, dx, dy, dz, p0, pn): #
    """
    Calculate ray (muon) intersection https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection.html
    A ray is described by its origin and its incremental direction with equation l_0+l*t (t is the ray parameter)
    An (infinite) plane is defined by a point p0 and its normal n
    
    Args:
        x0, y0, z0: ray origin (l_0)
        dx, dy, dz: ray direction (l)
       
        p0: a point on the plane (numpy array of length 3 dtype=np.float64)
        pn: normal vector of the plane (numpy array of length 3 dtype=np.float64)
    
    Returns:
        mask: which rays hit the plane
        x_int, y_int, z_int: intersection points (unmasked)
        t: ray parameter (unmasked)
    """

    #dotproduct(pn,l)
    denom = dx * pn[0] + dy * pn[1] + dz * pn[2]
    #if the plane and ray are parallel they either perfectly coincide, offering an infinite number of solutions, or they do not intersect at all. In practice we indicate no intersection when the denominator is less than a very small threshold.
    valid = np.abs(denom) > 1e-6 

    #dotProduct(p0-l0, pn) / denom
    #t = ((p0[0] - x0)*pn[0] + (p0[1] - y0)*pn[1] + (p0[2] - z0)*pn[2]) / denom
    
    #initialize arrays
    t = np.full_like(x0, np.nan)
    x_int = np.full_like(x0, np.nan)
    y_int = np.full_like(x0, np.nan)
    z_int = np.full_like(x0, np.nan)
                
    
    t[valid] = ((p0[0] - x0[valid]) * pn[0] + 
                (p0[1] - y0[valid]) * pn[1] + 
                (p0[2] - z0[valid]) * pn[2]) / denom[valid]

    hit = valid & (t > 0)

    #intersection coordinates
    x_int[hit] = x0[hit] + t[hit] * dx[hit]
    y_int[hit] = y0[hit] + t[hit] * dy[hit]
    z_int[hit] = z0[hit] + t[hit] * dz[hit]    
    
    return hit, x_int, y_int, z_int, t

def get_coincidence_hits(x, y, z, dx, dy, dz, z_center, alpha, det_x=0.8, det_y=0.2, separation=0.25):
    """
    Check which rays intersect both detectors after a rotation around x-axis.
    
    Args:
        x, y, z       : ray origins (arrays)
        dx, dy, dz    : ray directions (arrays)
        z_center      : center of the rotation axis (y = 0)
        alpha         : rotation angle around x-axis (radians)
        det_x         : detector width in x (meters)
        det_y         : detector height in y (meters)
        separation    : separation between detectors in z (meters)

    Returns:
        coincidence_mask         : boolean mask of rays hitting both detectors
        hits1_local[:3, mask]    : coordinates of the hit on detector 1 in its local frame
    """
    
    # Detector centers
    z_s1 = z_center - separation/2
    z_s2 = z_center + separation/2

    # Rotation matrix around x-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha),  np.cos(alpha)]
    ])
    
    # Detector centers in global coordinates after rotation
    offset = np.array([0, 0, separation / 2])
    center_s1 = Rx @ (-offset) + np.array([0, 0, z_center])
    center_s2 = Rx @ (+offset) + np.array([0, 0, z_center])

    # Normal of detectors after rotation
    normal_rot = Rx @ np.array([0, 0, 1])
    
    #todo check that center_s1 center_s2 and normal_rot are numpy arrays
    
    # Perform ray-plane intersection
    hit1, x1, y1, z1, _ = ray_plane_intersection(x, y, z, dx, dy, dz, center_s1, normal_rot)
    hit2, x2, y2, z2, _ = ray_plane_intersection(x, y, z, dx, dy, dz, center_s2, normal_rot)
    
    # Filter rays that intersect both planes
    valid = hit1 & hit2

    # Rotate hit points into detector local frame
    R_inv = Rx.T  # Inverse of rotation matrix
    # Build global coordinates arrays (hit points in global frame)
    hits1_global = np.vstack((x1[valid], y1[valid], z1[valid])) #vstack: Stack arrays in sequence vertically (row wise)
    hits2_global = np.vstack((x2[valid], y2[valid], z2[valid]))
    # Build local coordinates arrays (hit points in detector frame)
    hits1_local = R_inv @ np.vstack((x1 - center_s1[0], y1 - center_s1[1], z1 - center_s1[2]))
    hits2_local = R_inv @ np.vstack((x2 - center_s2[0], y2 - center_s2[1], z2 - center_s2[2]))

    # Check bounds inside each detector (rectangle in local frame)
    half_x = det_x / 2
    half_y = det_y / 2

    in_d1 = (
        (np.abs(hits1_local[0]) <= half_x) &
        (np.abs(hits1_local[1]) <= half_y)
    )
    in_d2 = (
        (np.abs(hits2_local[0]) <= half_x) &
        (np.abs(hits2_local[1]) <= half_y)
    )

    #Compute mask
    coincidence_mask = np.zeros_like(x, dtype=bool)
    valid_idx = np.where(valid)[0]
    coincidence_mask[valid_idx] = in_d1 & in_d2

    # Return local coordinates of detector 1 hits for coincident rays
    final_hits = hits1_local[:, in_d1 & in_d2]

    return coincidence_mask, final_hits


def simulate_events(args):
    #todo python docs
    n, rot_deg = args  # unpack the tuple (number of events, rotation degrees)
    rot = np.deg2rad(rot_deg)
    print("[THREAD] Eseguendo simulazione su "+str(n)+" eventi")
    
    # Generate intersection points.
    #x, y, z = generate_hemisphere(R, n)
    x, y, z = generate_plane(R, n)
    
    # Generate muon directions
    phi = sample_phi(n)
    theta = sample_theta(n)
    
    # Simple way of applying a rotation around x-axis:
    # Convert to direction vectors
    dx, dy, dz = to_cart(theta, phi)
    # Rotate direction by rot around x-axis (simulate detector rotation) by applying Rx(theta) matrix
    #dx, dy, dz = rot_x(dx, dy, dz, rot)
    # Normalize rotated direction vector
    #dx, dy, dz = normalize(dx, dy, dz)
    
    # Return to polar coordinates
    #phi = np.arctan2(dy, dx) #Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
    #theta = np.arccos(dz)
    
    z_center = 3
    
    hits, hits_points = get_coincidence_hits(x, y, z, dx, dy, dz, z_center, rot)
    
    return hits, hits_points

def simulate(N, deg):
    
    n_processes = multiprocessing.cpu_count()
    events_per_proc = N // n_processes
    print("Eseguendo "+str(events_per_proc)+" eventi su "+str(n_processes)+" processori")
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        args_list = [(events_per_proc, deg)] * n_processes
        results = list(executor.map(simulate_events, args_list))
    print("Simulazione terminata. Elaborazione risultati...")
    
    # Plot flux heatmap
    #results_x, results_y, results_z, results_dx, results_dy, results_dz = zip(*results)
    # Combine results
    #x = np.concatenate([np.array(x) for x in results_x])
    #y = np.concatenate([np.array(y) for y in results_y])
    #z = np.concatenate([np.array(z) for z in results_z])
    #dx = np.concatenate([np.array(dx) for dx in results_dx])
    #dy = np.concatenate([np.array(dy) for dy in results_dy])
    #dz = np.concatenate([np.array(dz) for dz in results_dz])
    #plot_flux_heatmaps(x, y, z, dx, dy, dz, R, str(deg)) 
    
    
    # Combine results
    mask, hits = zip(*results)
    x_hits, y_hits = hits
    x_values = np.concatenate([np.array(x) for x in x_hits])
    y_values = np.concatenate([np.array(y) for y in y_hits])
    n_coinc = x_values.size
    
    
    # FINALLY plot this.
    
    
    perc = "0%"
    if n_coinc != 0:
        perc = str(round((n_coinc/n_event)*100, 2))+"%"
    
    print("Calcolo istogramma...")
    
    #Fast binning
    dim_x = 0.8
    dim_y = 0.2
    bins_per_centimeter = 1
    H = parallel_hist2d(x_values, y_values, bins=(round(dim_x*100)*bins_per_centimeter, round(dim_y*100)*bins_per_centimeter), x_range=dim_x, y_range=dim_y)
    
    # Plot the heatmap
    plt.figure(figsize=(dim_x*10, dim_y*10+1))
    plt.imshow(
        H.T,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin='lower',
        cmap='plasma',
        aspect='auto',
        #vmax=35000,
        #vmin=0
        #norm=mpl.colors.LogNorm(vmin=10, vmax=50)
    )
    plt.colorbar(label='Density')
    plt.title("theta="+str(deg)+", "+str(n_event//1000000)+"Mevents, "+perc+" Aeff")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(False)
    plt.savefig(str(deg)+".png")
    print("Elaborazione completata, salvato file: "+str(deg)+".png")


if __name__ == "__main__":
    start = int(time.time())
    for i in range(4):
        a = 0 + (i) * 30
        simulate(N, a)
    stop = int(time.time())
    print("Programma terminato in "+str(stop-start)+"s.")





    
    # DEBUG: plot points
    #fig = plt.figure(figsize=(6, 6))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(x, y, z, s=1)
    #ax.set_title("Distribuzione uniforme su calotta sferica")
    #plt.show()
    
    # DEBUG: plot directions
    #fig = plt.figure(figsize=(8, 8))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.quiver(x, y, z, dx, dy, dz, length=1.0, normalize=True, linewidth=0.5, color='blue')
    #ax.set_title("Rays from Hemisphere Surface with Given Angles")
    #ax.set_xlim([-R, R])
    #ax.set_ylim([-R, R])
    #ax.set_zlim([0, R])
    #plt.show()
    
    #
    ## Calculate S2 plane coordinates
    #tan_theta = np.tan(theta)
    #x_1 = x_0 + tan_theta * l * np.cos(phi)
    #y_1 = y_0 + tan_theta * l * np.sin(phi)
    #
    ## Coincidence condition
    #mask = (0 < x_1) & (x_1 < dim_x) & (0 < y_1) & (y_1 < dim_y)
    #
    #return x_0[mask].tolist(), y_0[mask].tolist() #x_0s, y_0s with coincidence  