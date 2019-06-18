# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 23:30:41 2019

@author: Sravan
"""
import csv
import numpy as np
from scipy.spatial.distance import pdist, squareform, euclidean, cdist

import math as m
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
                 drones = 100,
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
        self.obstacles = obstacles
        self.obstacles_size = 100
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
            self.obs_state = np.random.random((obstacles, 3))
            self.obs_state[:, :3] -= 0.5
            self.obs_state[:, :2] *= bounds[1]*2
            self.obs_state[:, 2] *= bounds[5]*2
        
        self.init_state = np.asarray(init_state, dtype=float)
        #self.obs_state = np.asarray(obs_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.state = self.init_state.copy()
        
        #update velocity
        self.state[:, 6] = self.wind[0]
        self.state[:, 7] = self.wind[1]
        self.state[:, 8] = self.wind[2]

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
            v = self.state[i, 8]
            a_ver = self.acc_vert
            a_ver_eff = self.acc_vert_eff
            height = self.max_height - self.state[i, 2]
            if height > 0:
                n = 1
                if v > 0:
                    n = v / abs(v)
                v_target = self.accelerate(v, a_ver, height, dt, n)
                if (v_target[0] == 0):
                    self.state[i, 2] = self.max_height
                self.state[i, 2] += v_target[0]
                self.state[i, 8] = v_target[1]
            elif height < 0:
                n = v / abs(v)
                
                v_target = self.accelerate(v, a_ver_eff, height, dt, n)
                if (v_target[0] == 0):
                    self.state[i, 2] = self.max_height
                self.state[i, 2] += v_target[0]
                self.state[i, 8] = v_target[1]
            else:
                self.state[i, 8] += 0 * dt
            
            # unit vector
            r = self.state[i, 3:5] - self.state[i, :2]
            m = np.linalg.norm(r)
            u = r / m
            
            #accelearting horizontal
            a_hor = self.acc_hor
            v_hor = self.state[i, 6:8]
            h = np.linalg.norm(v_hor)
            
            n = 1
            if h != 0:
                n = h / abs(h)
            v_target = self.accelerate(h, a_hor, m, dt, n)
            if (v_target[0] == 0):
                self.state[i, :2] = self.state[i, 3:5]
            self.state[i, :2] += v_target[0] * u
            if (v_target[1] > self.max_speed):
                v_target[1] = self.max_speed
            self.state[i, 6:8] = v_target[1] * u
            
            if (self.state[i, 6] < -self.max_speed + self.wind[0]):
                self.state[i, 6] = -self.max_speed + self.wind[0]
            elif (self.state[i, 6] > self.max_speed + self.wind[0]):
                self.state[i, 6] = self.max_speed + self.wind[0]
                
            if (self.state[i, 7] < -self.max_speed + self.wind[1]):
                self.state[i, 7] = -self.max_speed + self.wind[1]
            elif (self.state[i, 7] > self.max_speed + self.wind[1]):
                self.state[i, 7] = self.max_speed + self.wind[1]

        #find drones hovering
        done, fund = np.where(D <= 122)
        uniquo = (done == fund)
        done = done[uniquo]
        for d in zip(done):
            if self.state[d, 2] != 0:
                #velocity vector
                v = self.state[d, 8]
                a_ver_eff = self.acc_vert_eff
                
                #accelerating negative z
                n = -1
                v_target = self.accelerate(v, a_ver_eff, -self.state[d, 2], dt, n)
                if (v_target[0] == 0):
                    self.state[d, 2] = 0
                    self.state[d, 9] = self.time_elapsed
                self.state[d, 2] += v_target[0]
                self.state[d, 8] = v_target[1]


        E = squareform(pdist(self.state[:, :3], 'euclidean'))
        ind1, ind2 = np.where(E < (20 * self.size))
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]
        
        for i1, i2 in zip(ind1, ind2):
            if (self.state[i1, 2] > self.state[i2, 2]):
                self.state[i1, 8] += (self.acc_vert) * dt
                self.state[i2, 8] -= (self.acc_vert_eff) * dt
            else:
                self.state[i1, 8] -= (self.acc_vert) * dt
                self.state[i2, 8] += (self.acc_vert_eff) * dt
                    
        if self.obstacles > 0:
            DO = np.vstack([self.state[:, :3], self.obs_state])
            F = squareform(pdist(DO, 'euclidean'))
            d_rone, obs = np.where(F < (2 * self.obstacles_size))
            unique = (d_rone < obs)
            uniqua = (d_rone < self.drones)
            uniqui = (obs >= self.drones)
            uni = unique & uniqua & uniqui
            d_rone = d_rone[uni]
            obs = obs[uni]
            
            for d, o in zip(d_rone, obs):
                print(d)
                print(o)
                r = self.state[d, 3:5] - self.state[d, :2]
                ro = self.state[d, :2] - self.obs_state[o-self.drones, :2]
                m = np.linalg.norm(r)
                v_m = np.linalg.norm(self.state[d, 6:8])
                if (self.obs_state[o-self.drones, 2] < (self.max_height - self.size * 2) and self.state[d, 2] < self.obs_state[o-self.drones, 2]):
                    self.state[d, 8] += self.acc_vert * dt
                    self.state[d, 2] = self.state[d, 8] * dt
                elif (m > 200 or v_m != 0):
                    u = r / m
                    diff = ro - (2 * np.dot(ro, u) * u)
                    diff_m = np.linalg.norm(diff)
                    diff_u = diff / diff_m
                    rem = diff_u - u
                    rem /= np.linalg.norm(rem)
                    
                    self.state[d, 6:8] += rem * self.acc_hor * dt
                    self.state[d, 0:2] += 0.5 * u * self.acc_hor * dt
    

    def accelerate(self, velocity, acceleration, target, dt, n):
        v = velocity
        v_avg = v
        stop = n * v**2/(2 * acceleration)
        t_end = abs(v / acceleration)
        
        b1 = (v**2 + t_end**2)**(0.5)
        b2 = ((v + n * acceleration * dt)**2 + (t_end + dt)**2)**(0.5)
        s1 = ((acceleration * dt)**2 + dt**2)**(0.5)
        s2 = dt * 2
        P = (b2 - b1) + s1 + s2
        t = ((P/2) * (P/2 - s1) * (P/2 - s2) * (P/2 - b2 + b1))**(0.5)
        h = 2 * t / (b2 - b1)
        area = n * (t + b1 * h)
        
        if (t_end <= dt and stop > n * (target - area)):
            v_avg = 0
            v_end = 0
        elif (stop > n * (target - area)):
            t_max = 0
            if stop < target:
                a = 2 * (acceleration)**2
                b = 4 * (acceleration) * v * n
                c = v**2 - 2 * acceleration * target
                t_max = (-b + (b**2 - 4 * a * c)**(0.5)) / (2 * a)
            v_max = v + n * acceleration * (t_max / dt)
            v_end = 2 * v_max - v - n * acceleration * dt
            v_avg = ((v_max + v) / 2) * (t_max / dt) + ((v_max + v_end) / 2) * ((dt - t_max) / dt)
        else:
            v_avg = v + n * acceleration * dt / 2
            v_end = v + n * acceleration * dt
        
        return [v_avg * dt, v_end]

#------------------------------------------------------------
# set up initial state
boxes = []
dt = 1. # 1 fps
for i in range(2000):
    boxes.append(ParticleBox(drones = i + 1))
    length = m.trunc(400 * m.log(i+10))
    for j in range(length):
        boxes[i].step(dt)
    with open('drones.csv', 'w') as writeFin:
        writer = csv.writer(writeFin)
            writer.writerow(box.state[i - 1, 9])

#final = np.hstack([box.init_state[:, :3], box.state[:, 3:]])

#with open('people.csv', 'w') as writeFile:
#    writer = csv.writer(writeFile)
#    writer.writerows(final) #2d list

"""with open('initial.csv', 'w') as writeInit:
    writer = csv.writer(writeInit)
    writer.writerows(box.init_state)
    
writeInit.close()
    """

with open('drones_fin.csv', 'w') as writeFin:
    writer = csv.writer(writeFin)
    for box in boxes:
        writer.writerow(box.state[:, 9])

writeFin.close()

print("completo")