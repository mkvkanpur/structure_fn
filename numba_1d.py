import numpy as np
from numba import cuda
import numpy
import math

# CUDA kernel
@cuda.jit
def matmul(A):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    (bx,by) = (cuda.blockIdx.x, cuda.blockIdx.y)
    local_r = (cuda.threadIdx.y, cuda.threadIdx.x)

    if row < A.shape[0] and col < A.shape[1]:
        print(by, bx, local_r[0], local_r[1], row, col, A[row, col])


N = (8,4)
# Initialize the data arrays
A = numpy.zeros(N) # matrix containing all 3's

x = np.linspace(0,N[0]-1,N[0])
y = np.linspace(0,N[1]-1,N[1])
ux, uy = np.meshgrid(x, y, indexing='ij')

# Copy the arrays to the device
A_global_mem = cuda.to_device(A)

# Configure the blocks
threadsperblock = (4, 2)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(A.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

print("blockspergrid = ", blockspergrid)
# Start the kernel
matmul[blockspergrid, threadsperblock](A_global_mem)

# Copy the result back to the host
A = A_global_mem.copy_to_host()

print(np.sum(A))