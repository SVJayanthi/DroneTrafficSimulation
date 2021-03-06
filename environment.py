# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:36:21 2019

@author: Sravan
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import scipy.integrate as integrate
import matplotlib.animation as animation


"""
Variables: Wind speed, Battery level, Air traffic (# of drones)
Fixed: Distance, Air Resistance, Gravity
Rules: Drone Speed (Air traffic, Wind speed, Battery level), Collisions (Drone position)
Study: Time, Speed
"""
class ParticleBox:
    """Orbits class
    
    init_state is an [N x 6] array, where N is the number of particles:
       [[xi1, yi1, zi1, xf1, yf1, zf1, v1, vy1, vz1],
        [xi2, yi2, zi2, xf2, yf2, zf2, vx2, vy2, vz2],
        ...               ]

    bounds is the size of the box: [xmin, xmax, ymin, ymax, zmin, zmax]
    """
    def __init__(self,
                 init_state,
                 bounds = [-2, 2, -2, 2, 0, 4],
                 size = 0.04,
                 M = 0.05,
                 G = 9.8):
        self.init_state = np.asarray(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.size = size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.G = G

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        
        # update positions
        self.state[:, :3] += dt * self.state[:, 3:]

        # find pairs of particles undergoing a collision
        D = squareform(pdist(self.state[:, :3], 'euclidean'))
        ind1, ind2 = np.where(D < 2 * self.size)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # update velocities of colliding pairs
        for i1, i2 in zip(ind1, ind2):
            # mass
            m1 = self.M[i1]
            m2 = self.M[i2]

            # location vector
            r1 = self.state[i1, :3]
            r2 = self.state[i2, :3]

            # velocity vector
            v1 = self.state[i1, 3:]
            v2 = self.state[i2, 3:]

            # relative location & velocity vectors
            r_rel = r1 - r2
            v_rel = v1 - v2

            # momentum vector of the center of mass
            v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)

            # collisions of spheres reflect v_rel over r_rel
            rr_rel = np.dot(r_rel, r_rel)
            vr_rel = np.dot(v_rel, r_rel)
            v_rel = 2 * r_rel * vr_rel / rr_rel - v_rel

            # assign new velocities
            self.state[i1, 3:] = v_cm + v_rel * m2 / (m1 + m2)
            self.state[i2, 3:] = v_cm - v_rel * m1 / (m1 + m2) 

        # check for crossing boundary
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size)
        crossed_z1 = (self.state[:, 2] < self.bounds[4] + self.size)
        crossed_z2 = (self.state[:, 2] > self.bounds[5] - self.size)

        self.state[crossed_x1, 0] = self.bounds[0] + self.size
        self.state[crossed_x2, 0] = self.bounds[1] - self.size

        self.state[crossed_y1, 1] = self.bounds[2] + self.size
        self.state[crossed_y2, 1] = self.bounds[3] - self.size

        self.state[crossed_z1, 1] = self.bounds[4] + self.size
        self.state[crossed_z2, 1] = self.bounds[5] - self.size

        self.state[crossed_x1 | crossed_x2, 3] *= -1
        self.state[crossed_y1 | crossed_y2, 4] *= -1
        self.state[crossed_z1 | crossed_z2, 5] *= -1

        # add gravity
        self.state[:, 5] -= self.M * self.G * dt


#------------------------------------------------------------
# set up initial state
np.random.seed(0)
init_state = np.random.random((50, 6))
init_state[:, :2] -= 0.5
init_state[:, :3] *= 4
init_state[:, 3:] *= 3.9

box = ParticleBox(init_state)
dt = 1. / 30 # 30fps


#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()

ax = p3.Axes3D(fig)
#ax = fig.add_subplot(111, projection='3d')
#ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
#                     xlim=(-3.2, 3.2), ylim=(-2.4, 2.4))

# particles holds the locations of the particles
# ms is marker size
particles, = ax.plot([], [], [], 'bo', ms=15)




def init():
    """initialize animation"""
    global box, particles
    
    # Setting the axes properties
    ax.set_xlim3d([-2.0, 2.0])
    ax.set_xlabel('X')
    
    ax.set_ylim3d([-2.0, 2.0])
    ax.set_ylabel('Y')
    
    ax.set_zlim3d([0.0, 4.0])
    ax.set_zlabel('Z')
    
    ax.set_title('3D Simulation')
    
    #particles = np.empty((3, 3))

    return particles

def animate(i):
    """perform animation step"""
    global box, dt, ax, fig
    box.step(dt)

    ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])
    
    #ax.scatter(box.state[:, 0], box.state[:, 1], box.state[:, 2])
    
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_3d_properties(box.state[:, 2])
    particles.set_markersize(10)
    
    return particles
    
    
ani = animation.FuncAnimation(fig, animate, frames=600, interval=10, init_func=init)

plt.show()