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

#generare N numeri psaudo casuli distriubuti con una distribuzione sferica uniforme
L = 0.2
N = 10_000_000
R = 2 #hemisphere radius

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
    u = np.random.uniform(0, 1, n)
    theta = np.arccos(u)  # theta between 0 and pi/2

    # Coordinate cartesiane
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    
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
    """projects rays from (x,y,z) in direction (dx,dy,dz) onto z=z_plane"""
    #avoid division by zero: mask rays that go straight along z-plane
    valid = dz != 0
    t = (z_plane - z[valid]) / dz[valid]
    
    #correction 1: only keep forward-going rays
    forward = t < 0
    t = t[forward]
    
    x_proj = x[valid][forward] + t * dx[valid][forward]
    y_proj = y[valid][forward] + t * dy[valid][forward]
    
    #return x_proj, y_proj
    
    #correction 2: Also filter out points outside the sphere's cap circle (needed for flux visualization errors)
    r2 = x_proj**2 + y_proj**2
    r_max2 = R**2 - z_plane**2
    inside = r2 <= r_max2
    return x_proj[inside], y_proj[inside]

from matplotlib.colors import LogNorm
def plot_flux_heatmaps(x, y, z, dx, dy, dz, R, name):
    z_planes = np.arange(0, 2.5, 0.5)  # z = 0, 0.5, ..., 2.0
    bins = 100  # resolution of heatmap
    extent = [-R, R, -R, R]  # XY limits based on sphere

    fig, axes = plt.subplots(1, len(z_planes), figsize=(20, 4))
    
    for i, z_plane in enumerate(z_planes):
        x_proj, y_proj = project_to_plane(x, y, z, dx, dy, dz, z_plane)
        
        H, xedges, yedges = np.histogram2d(
            x_proj, y_proj, bins=bins, range=[[extent[0], extent[1]], [extent[2], extent[3]]]
        )
        
        ax = axes[i]
        im = ax.imshow(
            H.T,
            extent=extent,
            origin='lower',
            cmap='inferno',
            aspect='equal',
            norm=LogNorm(vmin=20, vmax=2000)
        )
        ax.set_title(f'z = {z_plane:.1f} m')
        ax.set_xlabel('x (m)')
        if i == 0:
            ax.set_ylabel('y (m)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(name+".png")


def simulate_events(args):
    l, rot_deg, n = args  # unpack the tuple (lenght, rotation, number of events)
    rot = np.deg2rad(rot_deg)
    print("[THREAD] Eseguendo simulazione con L="+str(round(l, 2))+" e t="+str(rot_deg)+" su "+str(n)+" eventi")
    
    # Generate intersection points.
    x, y, z = generate_hemisphere(R, n)
    
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
    #phi = np.arctan2(dy, dx) #Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
    #theta = np.arccos(dz)
    
    plot_flux_heatmaps(x, y, z, dx, dy, dz, R, str(rot_deg))
    
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

for i in range(20):
    a = 0 + (i) * 5
    simulate_events((L, a, N))
