# DroneTrafficSimulation

![mandelbrot.png](https://github.com/SVJayanthi/DroneTrafficSimulation/blob/master/Environment.png)

## Author
Sravan Jayanthi

## Description
The future of transportation and delivery lies in the air, where there is an order of magnitude more space available for human commerce and transportation. The world of tomorrow will see thousands if not millions of drones zipping through the air in order to serve the tens of billions of humans that reside on this planet and those afar. In order to gain a better understanding of that future, this simulations is designed to model the air traffic of many drones and flying vehicles traveling rapidly through the air space. The principle condition of the simulation is that drones will avoid collisions at nearly any cost to their performance and travel time. Then, a variety of different factors were studied including environmental, mechanical, and congestion related to understand what would have the greatest impact on drone speed and travel times.

## Code

### Implementation
The drones were randomly assigned starting locations and tasked with reaching a destination location in the minimal amount of time. The drones were programmed to perform their decision-making autonomously such that they would adjust to obstacles or other drones in their path to avoid collisions. 
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
#### Drone Location Initialization
The drones 


The programs were designed to simulate and render efficiently and each implemented unique high-performance optimization compilers. The namesake program was designed utilizing NUMBA AutoJIT high performance python compiler. The cuda extension was designed implementing parrallel computation with NVIDIA CUDA GPU technology. 

### Drone Location Initialization

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

## License

MIT
