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
from numba import njit # jit
from concurrent.futures import ProcessPoolExecutor #threading
import multiprocessing #threading

N = 10_000_000_000
sequential_batch = 100 #to surpass ram limits
SMT = 2 #divide threads by this count. can help with ram requirements.
R = 10 #hemisphere radius OR plane side and height OR uniform generation box half-side
z_center = 3.3 #detector height (height of the rotation axis)
separation = 0.25 #separation between detectors
dim_x = 0.8
dim_y = 0.3
bins_per_centimeter = 1

@njit
def sample_theta(n):
    """Generate n angles in radians with cos² distribution"""
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
    """Generate n angles in radians with uniform distribution"""
    return np.random.uniform(0, 2 * np.pi, n)

@njit
def generate_hemisphere(R, n):
    """Generate n cartesian coordinates on a hemisphere"""
    #phi angle uniform between 0 and 2pi
    phi = np.random.uniform(0, 2 * np.pi, n)

    u = np.random.uniform(0, 1, n)
    #theta between 0 and pi/2
    #theta = np.arccos(u)
    theta = 0.5 * np.arccos(1 - 2 * u) # Fix for uniform theta

    # Cartesian coords
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
def generate_uniform(n):
    '''
    Generate uniformly in a box 2 times the size of the detectors.
    WARNING: Requires removing constraints on ray direction for intersection.    
    '''
    x = np.random.uniform(-dim_x*R, +dim_x*R, n)
    y = np.random.uniform(-dim_y*R, +dim_y*R, n)
    z = np.random.uniform(-dim_y*R, dim_y*R, n)
    return x, y, z

@njit
def muon_to_cart(theta, phi):
    """Compute muon direction vectors from polar angles, flipping z to direct downwards"""
    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)
    return dx, dy, -dz

# Fast binning
import numba
@njit(parallel=True)
def parallel_hist2d(x, y, H, x_bins, y_bins, x_range, y_range):
    """
    Args:
        x, y              :
        H                 : ndarray of shape (x_bins, y_bins) (integers), initialized to 0
                            Use H = np.zeros((x_bins, y_bins), dtype=np.int32)
        x_bins, y_bins    : 
        x_range, y_range  : 
    Returns:
        Histogram
    """
    
    dx = (x_range[1] - x_range[0]) / x_bins
    dy = (y_range[1] - y_range[0]) / y_bins

    for i in numba.prange(x.size):
        xi = int((x[i] - x_range[0]) / dx)
        yi = int((y[i] - y_range[0]) / dy)
        if 0 <= xi < x_bins and 0 <= yi < y_bins:
            H[xi, yi] += 1
    return H

@njit
def ray_plane_intersection(x0, y0, z0, dx, dy, dz, p0, pn): #
    """
    Calculate ray (muon) intersection https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection.html
    A ray is described by its origin and its incremental direction with equation l_0+l*t (t is the ray parameter)
    An (infinite) plane is defined by a point p0 and its normal n
    
    Args:
        x0, y0, z0  : ray origin (l_0)
        dx, dy, dz  : ray direction (l)
       
        p0          : a point on the plane (numpy array of length 3 dtype=np.float32)
        pn          : normal vector of the plane (numpy array of length 3 dtype=np.float32)
    
    Returns:
        mask                 : which rays hit the plane
        x_int, y_int, z_int  : intersection points (unmasked, NaN when invalid)
        t                    : ray parameter (unmasked, NaN when invalid)
    """
    
    #dotproduct(pn,l)
    denom = dx * pn[0] + dy * pn[1] + dz * pn[2]
    
    # If the plane and ray are parallel they either perfectly coincide,
    # offering an infinite number of solutions, or they do not intersect at all.
    # In practice we indicate no intersection when the denominator is less
    # than a very small threshold.
    #valid = np.abs(denom) > 1e-6 
    valid = np.abs(denom) > 1e-10
    
    # Initialize arrays
    t = np.full_like(x0, np.nan)
    x_int = np.full_like(x0, np.nan)
    y_int = np.full_like(x0, np.nan)
    z_int = np.full_like(x0, np.nan)
    
    #dotProduct(p0-l0, pn) / denom
    #t = ((p0[0] - x0)*pn[0] + (p0[1] - y0)*pn[1] + (p0[2] - z0)*pn[2]) / denom
    t[valid] = ((p0[0] - x0[valid]) * pn[0] + 
                (p0[1] - y0[valid]) * pn[1] + 
                (p0[2] - z0[valid]) * pn[2]) / denom[valid]

    hit = valid & (t > 0) #REMOVE FOR UNIFORM MODE

    #intersection coordinates
    x_int[hit] = x0[hit] + t[hit] * dx[hit]
    y_int[hit] = y0[hit] + t[hit] * dy[hit]
    z_int[hit] = z0[hit] + t[hit] * dz[hit] 

    #print(str(len(z_int[hit]))+" coincidences over "+str(len(x0))+" rays") # Debug
    
    return hit, x_int, y_int, z_int, t

def get_coincidence_hits(x, y, z, dx, dy, dz, z_center, alpha, det_x, det_y, separation):
    """
    Check which rays intersect both detectors after a rotation around x-axis.
    
    Args:
        x, y, z     : ray origins (arrays)
        dx, dy, dz  : ray directions (arrays)
        z_center    : center of the rotation axis (y = 0)
        alpha       : rotation angle around x-axis (radians)
        det_x       : detector width in x (meters)
        det_y       : detector height in y (meters)
        separation  : separation between detectors in z (meters)

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
    R_inv = Rx.T  # Inverse of rotation matrix
    
    # Detector centers in global coordinates after rotation
    offset = np.array([0, 0, separation / 2])
    center_s1 = Rx @ (-offset) + np.array([0, 0, z_center])
    center_s2 = Rx @ (+offset) + np.array([0, 0, z_center])

    # Normal of detectors after rotation
    normal_rot = Rx @ np.array([0, 0, 1])
    
    # Perform ray-plane intersection
    hit1, x1, y1, z1, _ = ray_plane_intersection(x, y, z, dx, dy, dz, center_s1, normal_rot)
    hit2, x2, y2, z2, _ = ray_plane_intersection(x, y, z, dx, dy, dz, center_s2, normal_rot)
    
    # Filter rays that intersect both planes
    valid = hit1 & hit2

    # Rotate hit points into detector local frame
    # Build global coordinates arrays
    hits1_global = np.vstack((x1[valid], y1[valid], z1[valid])) - center_s1[:, None]
    hits2_global = np.vstack((x2[valid], y2[valid], z2[valid])) - center_s2[:, None]
    # Transform only valid hits to local (detector) frames
    hits1_local = R_inv @ hits1_global  # Shape (3, N_valid)
    hits2_local = R_inv @ hits2_global

    # Check bounds inside each detector (rectangle in local frame)
    half_x, half_y = dim_x / 2, dim_y / 2

    in_d1 = (
        (np.abs(hits1_local[0]) <= half_x) &
        (np.abs(hits1_local[1]) <= half_y)
    )
    in_d2 = (
        (np.abs(hits2_local[0]) <= half_x) &
        (np.abs(hits2_local[1]) <= half_y)
    )

    #Compute mask
    in_both = in_d1 & in_d2
    coincidence_mask = np.zeros_like(x, dtype=np.bool)
    valid_indices = np.where(valid)[0]
    coincidence_mask[valid_indices] = in_both
    num_coincident = np.sum(in_both)
    
    print(f"[THREAD] {num_coincident} coincidenze su {len(x)}")

    return coincidence_mask, hits1_local[:2, in_both]


def simulate_events(args):
    n, rot_deg = args  # unpack the tuple (number of events, rotation degrees)
    rot = np.deg2rad(rot_deg)
    print("[THREAD] Eseguendo simulazione su "+str(n)+" eventi")
    
    # Generate intersection points.
    x, y, z = generate_hemisphere(R, n)
    #x, y, z = generate_plane(R, n)
    #x, y, z = generate_uniform(n)
    
    # Generate muon directions
    phi = sample_phi(n)
    theta = sample_theta(n)
    
    # Convert to direction vectors
    dx, dy, dz = muon_to_cart(theta, phi)
    
    hits, hits_points = get_coincidence_hits(x, y, z, dx, dy, dz, z_center, rot, dim_x, dim_y, separation)
    
    del x, y, z, dx, dy, dz, phi, theta # memory optimization
    
    return hits, hits_points

def simulate(n_event, deg):
    
    n_processes = multiprocessing.cpu_count()
    n_processes //= SMT
    events_per_proc = n_event // n_processes
    events_per_proc //=sequential_batch
    
    #prepare binning
    x_bins = int(round(dim_x*100)*bins_per_centimeter)
    y_bins = int(round(dim_y*100)*bins_per_centimeter)
    x_range=(0-dim_x/2, dim_x-dim_x/2)
    y_range=(0-dim_y/2, dim_y-dim_y/2)
    H = np.zeros((x_bins, y_bins), dtype=np.int32)
    n_coinc = 0
    
    for i in range(sequential_batch): #sequential batching
        print("Batch #"+str(i+1)+"/"+str(sequential_batch))
        print("Eseguendo "+str(events_per_proc)+" eventi su "+str(n_processes)+" processori")
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            args_list = [(events_per_proc, deg)] * n_processes
            results = list(executor.map(simulate_events, args_list))
        print("Batch terminato, binning...")
        #Combine results
        masks, hits = zip(*results)  # List of boolean arrays, list of 2D arrays
        x_hits = [h[0] for h in hits]  # x coordinates from each 2D array
        y_hits = [h[1] for h in hits]  # y coordinates
        x_values = np.concatenate(x_hits)
        y_values = np.concatenate(y_hits)
        #batch
        H = parallel_hist2d(x_values, y_values, H, x_bins=x_bins, y_bins=y_bins, x_range=x_range, y_range=y_range)
        n_coinc += x_values.size
        del masks, hits, x_hits, y_hits, x_values, y_values
    
    print("Simulazione terminata. Elaborazione risultati...")
    
    print(str(n_coinc)+" coincidenze trovate")
    
    # Percentage
    perc = "0%"
    if n_coinc != 0:
        perc = str(round((n_coinc/n_event)*100, 2))+"%"
    
    # Plot
    plt.figure(figsize=(dim_x*10, dim_y*10+1))
    plt.imshow(
        H.T,
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        origin='lower',
        cmap='plasma',
        aspect='auto',
        #vmax=35000,
        #vmin=0
        #norm=mpl.colors.LogNorm(vmin=10, vmax=50)
    )
    plt.colorbar(label='Density')
    plt.title("theta="+str(deg)+", "+str(n_event//1_000_000_000)+"Gevents, "+str(n_coinc//1000)+"kCoinc.")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(False)
    plt.savefig(str(deg)+".png")
    print("Elaborazione completata, salvato file: "+str(deg)+".png")


if __name__ == "__main__":
    start = int(time.time())
    for i in range(3):
        a = 30 + (i) * 30
        simulate(N, a)
    stop = int(time.time())
    print("Programma terminato in "+str(stop-start)+"s.")