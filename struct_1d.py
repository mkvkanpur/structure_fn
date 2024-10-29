import numpy as np
import matplotlib.pyplot as plt

def structfn(u, sf):
    for l in range(1,lmax):
        du = np.abs(u[l:N//2+l]-u[0:N//2])
        for q in range(qmin,qmax+1):
            sf[q-qmin,l] = np.mean(du**q)

N = 200
dx = 0.1
u = np.zeros(N)

lmax = N//2
qmin = 2
qmax = 8
sf = np.zeros((qmax-qmin+1, lmax))
xforsf = np.arange(lmax)*dx

for i in range(N):
    x = i*dx
    u[i] = x #np.tanh((x-10)/0.1)

structfn(u, sf)


plt.figure()
for q in range(qmin,qmax+1):
    plt.loglog(xforsf, sf[q-qmin,:], label='$d = %i$' %q)

plt.legend(loc = 'upper left')
plt.savefig('test.png')

