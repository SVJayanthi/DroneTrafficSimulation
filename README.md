# DroneTrafficSimulation

This is a simulation and rendering of the Mandelbrot Set,  named after mathematician Benoit Mandelbrot, that provides a visual for the set of complex numbers that remain bounded in absolute value from the function.

![mandelbrot.png](https://github.com/SVJayanthi/MandelbrotSet/blob/master/output/mandelbrot_1.png)

The function is

  `f_{n+1} = f_{n}^2 + c`

with the initial condition simply formed by taking the coordinates in the complex plane,

  `f_{0} = x + iy`

It produces an aesthetic depiction of a fractal that does not diverge with recursive details similar in design to other parts of the fractal.

## Code

### Implementation

The programs were designed to simulate and render efficiently and each implemented unique high-performance optimization compilers. The namesake program was designed utilizing NUMBA AutoJIT high performance python compiler. The cuda extension was designed implementing parrallel computation with NVIDIA CUDA GPU technology. 

### Dynamic

```
@autojit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
  height = image.shape[0]
  width = image.shape[1]

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height

  for x in range(width):
    real = min_x + x * pixel_size_x
    for y in range(height):
      imag = min_y + y * pixel_size_y 
      image[y, x] = mandel(real, imag, iters)

image = np.zeros((1024, 1536), dtype = np.uint8)
start = timer()
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20) 
dt = timer() - start
```

### CUDA

```
@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
  height = image.shape[0]
  width = image.shape[1]

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height

  startX, startY = cuda.grid(2)
  gridX = cuda.gridDim.x * cuda.blockDim.x;
  gridY = cuda.gridDim.y * cuda.blockDim.y;

  for x in range(startX, width, gridX):
    real = min_x + x * pixel_size_x
    for y in range(startY, height, gridY):
      imag = min_y + y * pixel_size_y 
      image[y, x] = mandel_gpu(real, imag, iters)

gimage = np.zeros((1024, 1536), dtype = np.uint8)
blockdim = (32, 8)
griddim = (32, 16)

start = timer()
d_image = cuda.to_device(gimage)
mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 20) 
d_image.to_host()
dt = timer() - start
```

## License

[MIT](LICENSE)
