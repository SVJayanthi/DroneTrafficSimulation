# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 22:36:21 2019

@author: Sravan
"""
import csv
import numpy as np
from scipy.spatial.distance import pdist, squareform, euclidean, cdist

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import scipy.integrate as integrate
import matplotlib.animation as animation


"""
Variables: Wind speed, Air traffic (# of drones), Obstacles (Trees, Buildings) 
Fixed: Distance, Air Resistance, Gravity, Battery level
Rules: Drone Speed (Air traffic, Wind speed, Battery level), Collisions (Drone position)
Study: Time, Speed
Movement: v_air = sqrt(mg/(nAρ)), p = 1.22 kg m^-3, A = 1 m^2
½cρAv2 = mgtanθ, c = drag coefficient
P = ½ρnAv_air(v_air2 – v2sin2θ)
Collisions: Drone - Increase/Decrease Speed, 2) Change path- increasing elevation

https://www.research-drone.com/en/extreme_climb_rate.html
https://en.wikipedia.org/wiki/Amazon_Prime_Air
https://homepages.abdn.ac.uk/nph120/meteo/DroneFlight.pdf
"""
class ParticleBox:
    """Orbits class
    
    init_state is an [N x 6] array, where N is the number of particles:
       [[xi1, yi1, zi1, xf1, yf1, zf1, vx1, vy1, vz1, t1],
        [xi2, yi2, zi2, xf2, yf2, zf2, vx2, vy2, vz2, t2],
        ...               ]

    bounds is the size of the box: [xmin, xmax, ymin, ymax, zmin, zmax]
    """
    def __init__(self,
                 drones = 50,
                 wind = [0, 0, 0],
                 obstacles = 0,
                 bounds = [-32000, 32000, -32000, 32000, 0, 150],
                 size = 1.5,
                 max_height = 122,
                 max_speed = 22.34,
                 acc = 7,
                 M = 25.0,
                 G = 9.81):
        self.drones = drones
        self.wind = wind
        self.size = size
        self.G = G
        self.max_height = max_height
        self.max_speed = max_speed
        self.acc_vert = acc
        self.acc_vert_eff = acc + G
        self.acc_hor = acc
        self.obstacles = 0
        self.obstacles_size = 40
        self.time_elapsed = 0
        self.bounds = bounds
        
        np.random.seed(0)
        init_state = np.random.random((drones, 10))
        init_state[:, :2] -= 0.5
        init_state[:, :2] *= bounds[1]*2
        init_state[:, 2:] = 0.0
        for i in range(len(init_state)):
            vecs = [64000.0, 64000.0]
            while vecs[0] > bounds[1] or vecs[0] < bounds[0] or vecs[1] > bounds[3] or vecs[1] < bounds[2]:
                vecs = np.random.standard_normal(2)
                mags = np.linalg.norm(vecs)
                vecs /= mags
                vecs *= 16000
                vecs += init_state[i, :2]
            init_state[i, 3:5] =vecs
        
        if obstacles > 0:
            np.random.seed(1)
            obs_state = np.random.random((obstacles, 3))
            obs_state[:, :3] -= 0.5
            obs_state[:, :2] *= bounds[1]*2
            obs_state[:, 2] *= bounds[5]*2
        
        self.init_state = np.asarray(init_state, dtype=float)
        #self.obs_state = np.asarray(obs_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.state = self.init_state.copy()

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        
        # find distance to goal
        D = cdist(self.state[:, :3], self.state[:, 3:6], 'euclidean')
        ind, din = np.where(D > 122)
        uniqua = (ind == din)
        ind = ind[uniqua]
        
        # update velocities of individual drones
        for i in zip(ind):
            #velocity vector
            v = self.state[i, 6:9]
            v_avg = v[2]
            
            
            # unit vector
            r = self.state[i, 3:5] - self.state[i, :2]
            m = np.linalg.norm(r)
            u = r / m
            
            
            
            #crossing height boundary
            if (self.max_height - self.state[i, 2]) > self.state[i, 8]:
                if self.max_height - self.state[i, 2] > self.acc_vert - self.G:
                    self.state[i, 8] += self.acc_vert * dt
                else:
                    self.state[i, 8] += (self.max_height - self.state[i, 2]) * dt
            elif self.state[i, 2] < self.max_height:
                if self.state[i, 8] - (self.max_height - self.state[i, 2])/2 > self.acc_vert + self.G:
                    self.state[i, 8] -= self.acc_vert * dt
                elif self.state[i, 8] == self.max_height - self.state[i, 2]:
                    self.state[i, 8] += 0
                else:
                    self.state[i, 8] -= (self.state[i, 8] - (self.max_height - self.state[i, 2])/2) * dt
            elif self.state[i, 2] >= self.max_height:
                self.state[i, 8] -= (self.state[i, 8] + self.max_height - self.state[i, 2]) * dt
            
            #accelerating x direction
            if abs(self.state[i, 3] - self.state[i, 0]) > abs(self.state[i, 6]):
                if self.state[i, 3] - self.state[i, 0] > self.acc_hor:
                    self.state[i, 6] += self.acc_hor * dt
                elif self.state[i, 3] - self.state[i, 0] < -self.acc_hor:
                    self.state[i, 6] -= self.acc_hor * dt
                else:
                    self.state[i, 6] += (self.state[i, 3] - self.state[i, 0]) * dt
            elif abs(self.state[i, 3] - self.state[i, 0]) > 0:
                if self.state[i, 3] - self.state[i, 0] > self.acc_hor:
                    self.state[i, 6] -= self.acc_hor * dt
                elif self.state[i, 3] - self.state[i, 0] < -self.acc_hor:
                    self.state[i, 6] += self.acc_hor * dt
                elif self.state[i, 6] == self.state[i, 3] - self.state[i, 0]:
                    self.state[i, 6] += 0
                elif self.state[i, 3] - self.state[i, 0] > 0:
                    self.state[i, 6] -= (self.state[i, 6] - (self.state[i, 3] - self.state[i, 0])/2) * dt
                elif self.state[i, 3] - self.state[i, 0] < 0:
                    self.state[i, 6] -= (self.state[i, 6] - (self.state[i, 3] - self.state[i, 0])/2) * dt
            elif abs(self.state[i, 3] - self.state[i, 0]) == 0:
                self.state[i, 6] -= self.state[i, 6] * dt
            
            #accelerating y direction
            if abs(self.state[i, 4] - self.state[i, 1]) > abs(self.state[i, 7]):
                if self.state[i, 4] - self.state[i, 1] > self.acc_hor:
                    self.state[i, 7] += self.acc_hor * dt
                elif self.state[i, 4] - self.state[i, 1] < -self.acc_hor:
                    self.state[i, 7] -= self.acc_hor * dt
                else:
                    self.state[i, 7] += (self.state[i, 4] - self.state[i, 1]) * dt
            elif abs(self.state[i, 4] - self.state[i, 1]) > 0:
                if self.state[i, 4] - self.state[i, 1] > self.acc_hor:
                    self.state[i, 7] -= self.acc_hor * dt
                elif self.state[i, 4] - self.state[i, 1] < -self.acc_hor:
                    self.state[i, 7] += self.acc_hor * dt
                elif self.state[i, 7] == self.state[i, 4] - self.state[i, 1]:
                    self.state[i, 7] += 0
                elif self.state[i, 4] - self.state[i, 1] > 0:
                    self.state[i, 7] -= (self.state[i, 7] - (self.state[i, 4] - self.state[i, 1])/2) * dt
                elif self.state[i, 4] - self.state[i, 1] < 0:
                    self.state[i, 7] -= (self.state[i, 7] - (self.state[i, 4] - self.state[i, 1])/2) * dt
            elif abs(self.state[i, 4] - self.state[i, 1]) == 0:
                self.state[i, 7] -= self.state[i, 7] * dt

        #find drones hovering
        done = np.where(D <= 122)
        for d in zip(done):
            #accelerating negative z
            if self.state[i, 2] > self.state[i, 8]:
                if self.state[i, 2] > self.acc_vert + self.G:
                    self.state[i, 8] -= (self.acc_vert + self.G) * dt
                else:
                    self.state[i, 8] -= self.state[i, 2] * dt
            elif self.state[i, 2] > 0:
                if self.state[i, 8] - self.state[i, 2]/2 > self.acc_vert - self.G:
                    self.state[i, 8] += (self.acc_vert - self.G) * dt
                elif self.state[i, 8] == -self.state[i, 2]:
                    self.state[i, 8] += 0
                else:
                    self.state[i, 8] -= (self.state[i, 8] - self.state[i, 2]/2) * dt
            elif self.state[i, 2] == 0:
                if (abs(self.state[i, 8]) > 0):
                    self.state[i, 8] -= self.state[i, 8] * dt
                    self.state[i, 9] = self.time_elapsed
                
        
        E = squareform(pdist(self.state[:, :3], 'euclidean'))
        ind1, ind2 = np.where(E < (2 * self.size))
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]
        
        for i1, i2 in zip(ind1, ind2):
            if (self.state[i1, 2] > self.state[i2, 2]):
                self.state[i1, 8] += (self.acc_vert-self.G) * dt
                self.state[i2, 8] -= (self.acc_vert+self.G) * dt
            else:
                self.state[i1, 8] -= (self.acc_vert+self.G) * dt
                self.state[i2, 8] += (self.acc_vert-self.G) * dt
                    
        if self.obstacles > 0:
            DO = np.vstack([self.state[:, :3], self.obs_state])
            F = squareform(pdist(DO, 'euclidean'))
            d_rone, obs = np.where(F < (2 * self.obstacles_size))
            unique = (d_rone < obs and obs >= self.drones)
            d_rone = d_rone[unique]
            obs = obs[unique]
            
            for d, o in zip(d_rone, obs):
                if (self.obs_state[o-self.drones, 2] < 110 and self.state[d, 2] < self.obs_state[o-self.drones, 2]):
                    self.state[d, 8] += self.acc_vert * dt
                else:
                    r = self.state[d, 3:5] - self.state[d, :2]
                    ro = self.obs_state[o-self.drones, :2] - self.state[d, :2]
                    
                    r_rel = np.cross(r, ro)
                    if (r_rel[2] > 0):
                        self.state[d, 6] += self.acc_hor * dt
                        self.state[d, 7] += self.acc_hor * dt
                    else:
                        self.state[d, 6] -= self.acc_hor * dt
                        self.state[d, 7] -= self.acc_hor * dt
                    
        #update velocity        
        self.state[:, :6] += self.wind[0]
        self.state[:, :7] += self.wind[1]
        self.state[:, :8] += self.wind[2] - (dt * self.G)
        
        
        #restrict velocity
        np.clip(self.state[:, 6], -self.max_speed, self.max_speed)
        np.clip(self.state[:, 7], -self.max_speed, self.max_speed)
        
        # update positions
        self.state[:, :3] += dt * self.state[:, 6:9]


#------------------------------------------------------------
# set up initial state

box = ParticleBox()
dt = 1. # 1 fps
    
#ani = animation.FuncAnimation(fig, animate, frames=600, interval=10, init_func=init)
for i in range(100000):
    box.step(dt)

#final = np.hstack([box.init_state[:, :3], box.state[:, 3:]])

#with open('people.csv', 'w') as writeFile:
#    writer = csv.writer(writeFile)
#    writer.writerows(final) #2d list

"""with open('initial.csv', 'w') as writeInit:
    writer = csv.writer(writeInit)
    writer.writerows(box.init_state)
    
writeInit.close()
    """
    
with open('final.csv', 'w') as writeFin:
    writer = csv.writer(writeFin)
    writer.writerows(box.state)

writeFin.close()

print(box.state)