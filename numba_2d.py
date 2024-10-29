Mydatatype = "float64"
from numba import cuda, float64, int32
import numpy as np
import matplotlib.pyplot as plt
import math
import h5py as hp

TPB = (16,16)
TPB2 = (TPB[0]*2, TPB[1]*2)  # double elements along each dirn, for u'
normalization_const =  TPB[0]*TPB[1]
##CUDA kernel
@cuda.jit
def structfn(N, zpx, zpy, zmx, zmy, sf_p, sf_m):

    (i,j) = cuda.grid(2)

    loc_i, loc_j = cuda.threadIdx.x, cuda.threadIdx.y #(x,y) in C
    block_id = (cuda.blockIdx.x, cuda.blockIdx.y)
    global_r = (cuda.blockIdx.x*TPB[0]+cuda.threadIdx.x, \
              cuda.blockIdx.y*TPB[1]+cuda.threadIdx.y)
  
    zpx_loc = zpx[i,j]
    zpy_loc = zpy[i,j]
    zmx_loc = zmx[i,j]
    zmy_loc = zmy[i,j]
    cuda.syncthreads()
 
    for lx in range(N[0]//2):
        for ly in range(N[1]//2):
            l_norm = math.sqrt(lx**2+ly**2)
            l_ind = int(math.ceil(l_norm))
            dzp_vec = (zpx[i+lx,j+ly]- zpx_loc, zpy[i+lx,j+ly]- zpy_loc)
            dzm_vec = (zmx[i+lx,j+ly]- zmx_loc, zmy[i+lx,j+ly]- zmy_loc)

            if l_norm > 1e-5:
                S3_loc = (dzp_vec[0]**2+dzp_vec[1]**2)* \
                         (dzm_vec[0]*lx+dzm_vec[1]*ly)/l_norm
            cuda.syncthreads()

            cuda.atomic.add(sf_p,(0, l_ind),S3_loc)
            cuda.syncthreads()

            if l_norm > 1e-5:
                S3_loc = (dzm_vec[0]**2+dzm_vec[1]**2)* \
                         (dzp_vec[0]*lx+dzp_vec[1]*ly)/l_norm
            cuda.syncthreads()

            cuda.atomic.add(sf_m,(0, l_ind),S3_loc)
            cuda.syncthreads()


# MAIN
#cuda.select_device(0)  # Select the first GPU
#device = cuda.current_context().device
#print(f"Using GPU: {device.name}")
#print(f"Total shared memory per block: {device.MAX_SHARED_MEMORY_PER_BLOCK}")

N = (1024,1024)
lmax = math.ceil(np.linalg.norm(N))//2
count = np.zeros(lmax, dtype=int)
qmin = 2
qmax = 3
Qdiff_p1 = 1 #qmax-qmin+2
sf_p = np.zeros((Qdiff_p1, lmax))
sf_m = np.zeros((Qdiff_p1, lmax))

for i in range(N[0]//2):
    for j in range(N[1]//2):
        vec = np.array([i,j])
        ind = int(math.ceil(np.linalg.norm(vec)))
        count[ind] += 1


####### Read the data ##########
# zpx = np.ones(N, dtype=Mydatatype)
# zpy = np.ones(N, dtype=Mydatatype)
# zmx = np.ones(N, dtype=Mydatatype)
# zmy = np.ones(N, dtype=Mydatatype)

#File_handle = hp.File("/home/mkv/MHD_data/2D/256_square/field_200..h5",'r')
File_handle = hp.File("/home/mkv/MHD_data/2D/1024_square/field_51..h5",'r')
zpx = np.fft.irfft2(np.asarray(File_handle["zpkx"]))*N[0]*N[1]
zpy = np.fft.irfft2(np.asarray(File_handle["zpkz"]))*N[0]*N[1]
zmx = np.fft.irfft2(np.asarray(File_handle["zmkx"]))*N[0]*N[1]
zmy = np.fft.irfft2(np.asarray(File_handle["zmkz"]))*N[0]*N[1]

'''
print(zpx.shape)
dx = dy =  2*np.pi/
for i in range(N[0]):
    for j in range(N[1]):
        zpx[i,j] = zmx[i,j] = i*dx
        zpy[i,j] = zmy[i,j] = j*dy
'''
dx = dy = 2*np.pi/N[0]
xforsf = np.arange(lmax)*dx


import math
BPG_x = math.ceil(N[0]/(2*TPB[0]))
BPG_y = math.ceil(N[1]/(2*TPB[0]))
BPG = (BPG_x, BPG_y)
print(BPG)
#Copy the arrays to the device
zpx_global_mem = cuda.to_device(zpx)
zpy_global_mem = cuda.to_device(zpy)
sf_p_global_mem = cuda.to_device(sf_p)

zmx_global_mem = cuda.to_device(zmx)
zmy_global_mem = cuda.to_device(zmy)
sf_m_global_mem = cuda.to_device(sf_m)

#start the kernel
structfn[BPG,TPB](N, zpx_global_mem, zpy_global_mem, zmx_global_mem, zmy_global_mem, \
                    sf_p_global_mem, sf_m_global_mem)

# Copy the result back to the host
sf_p = sf_p_global_mem.copy_to_host()
sf_m = sf_m_global_mem.copy_to_host()
factor = N[0]*N[1]/4
#print("before = ",sf[2,:])
for l_loop in range(lmax):
    sf_p[:,l_loop] /=  (factor*count[l_loop])
    sf_m[:,l_loop] /=  (factor*count[l_loop])
#print(xforsf)
#print(sf[0,:])


plt.figure()
plt.semilogx(xforsf[1:], -0.5*sf_p[0,1:]/xforsf[1:], label='S_3p')
plt.semilogx(xforsf[1:], -0.5*sf_m[0,1:]/xforsf[1:], label='S_3m')
plt.axhline(y=0)
plt.axhline(y=0.1)
# for q in range(qmin,qmax+1):
#     plt.loglog(xforsf[1:], sf_p[q-qmin+1,1:], label='$q = %i$' %q)
#     plt.loglog(xforsf[1:], sf_m[q-qmin+1,1:], label='$q = %i$' %q)

plt.legend(loc = 'upper left')
plt.savefig('numba2_new.png')
print( sf_p[0,1:5],  sf_m[0,1:5])
# print( sf_p[1,:], sf_p[2,:], sf_m[1,:], sf_m[2,:])