# DroneTrafficSimulation

![mandelbrot.png](https://github.com/SVJayanthi/DroneTrafficSimulation/blob/master/Environment.png)

## Author
Sravan Jayanthi

## Description
The future of transportation and delivery lies in the air, where there is an order of magnitude more space available for human commerce and transportation. The world of tomorrow will see thousands if not millions of drones zipping through the air in order to serve the tens of billions of humans that reside on this planet and those afar. In order to gain a better understanding of that future, this simulations is designed to model the air traffic of many drones and flying vehicles traveling rapidly through the air space. The principle condition of the simulation is that drones will avoid collisions at nearly any cost to their performance and travel time. Then, a variety of different factors were studied including environmental, mechanical, and congestion related to understand what would have the greatest impact on drone speed and travel times.

## Code

### Implementation
The drones were randomly assigned starting locations and tasked with reaching a destination location in the minimal amount of time. The drones were programmed to perform their decision-making autonomously such that they would adjust to obstacles or other drones in their path to avoid collisions. 
#### Drone Location Initialization
```python
    vecs = [64000.0, 64000.0]
    while vecs[0] > bounds[1] or vecs[0] < bounds[0] or vecs[1] > bounds[3] or vecs[1] < bounds[2]:
        vecs = np.random.standard_normal(2)
        mags = np.linalg.norm(vecs)
        vecs /= mags
        vecs *= 16000
        vecs += init_state[i, :2]
    init_state[i, 3:5] =vecs
```
The mechanics of drone acceleration such as propulsion through opposite spinning rotors are modeled to represent the mechanical factors that are associated with drone travel. When the parameters of payload weight, wing span, motor revolutions per minute, and battery capacity are changed, the subsequent effects on the speed and time taken to travel can be acutely determined.
The programs were designed to simulate and render efficiently and each implemented unique high-performance optimization compilers. The namesake program was designed utilizing NUMBA AutoJIT high performance python compiler. The cuda extension was designed implementing parallel computation with NVIDIA CUDA GPU technology. 
### CUDA Enabled Drone Acceleration Computation
```python
    @autojit
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
```
Other factors that were studied are the environmental factors such as wind, inclement weather (snow/rain), and obstacles such as trees, hills, and buildings. These would require the drone to adjust their travel paths so that their pitch is stronger towards the wind or they accelerate upwards to avoid colliding into hills or buildings.
### Obstacle Avoidance Protocol
```python
        r = self.state[d, 3:5] - self.state[d, :2]
        ro = self.state[d, :2] - self.obs_state[o-self.drones, :2]
        m = np.linalg.norm(r)
        v_m = np.linalg.norm(self.state[d, 6:8])
        if (self.obs_state[o-self.drones, 2] < (self.max_height - self.size * 2) and self.state[d, 2] < self.obs_state[o-self.drones, 2]):
            self.state[d, 8] += self.acc_vert * dt
            self.state[d, 2] = self.state[d, 8] * dt
```
Drone congestion would be another interesting factor to study to see how rapidly flying objects would perform if the airspace is busy with others doing the same. This simulations was designed with assumption that communications technologies would be able to fit within a drone so that each drone in the air space can communicate with eachother. Thus, each drone is aware of the locations of every other drone and can adjust their flight path in order to avoid colliding with any of them.
### Flight Path Adjustment Routine
```python
        r = self.state[i1, 3:5] - self.state[i1, :2]
        ro = self.state[i1, :2] - self.state[i2, :2]
        m = np.linalg.norm(r)
        if (m < (self.size * 20)):
            u = r / m
            diff = ro - (2 * np.dot(ro, u) * u)
            diff_m = np.linalg.norm(diff)
            diff_u = diff / diff_m
            rem = diff_u - u
            rem /= np.linalg.norm(rem)
```

## Conclusions
This study of the importance of congestion, mechanical, and environmental factors on the capacity for drones to reach their target destination in the minimal amount of time has determined that, environmental factors including wind or obstacles have the most bearing followed by mechanical factors including battery life and payload weight, and then congestion. The output is located in the `data/` folder where travel times are documented for a given distance showing that a change in environmental factors influence the time taken the most for the many drones traveling in that region.

## License
MIT
