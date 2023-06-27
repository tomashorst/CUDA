#@title modelo SIR en numba

import numpy as np
from numba import cuda
import cupy as cp
import cupyx


@cuda.jit
def SIRkernel(S,I,R,beta,g,dt,N):
      i = cuda.threadIdx.x + cuda.blockIdx.x*cuda.blockDim.x;
      if i<N:
        oldS = S[i]
        oldI = I[i]
        oldR = R[i]
        b=beta[i]

        newS = oldS - dt * b * oldS * oldI;
        newI = oldI + dt * (b*oldS*oldI - g*oldI);
        newR = oldR + dt * (g*oldI);

        S[i]=newS;
        I[i]=newI;
        R[i]=newR;

N = 10 # nro de poblaciones
g = 0.1  # tasa de recuperacion
dt = 0.1  # paso de tiempo

# Declarar y Alocar memoria para los arrays de device S, I, R y beta usando CuPy
# ....
S = cp.zeros(N, dtype=cp.float32)
I = cp.zeros(N, dtype=cp.float32)
R = cp.zeros(N, dtype=cp.float32)
beta = cp.zeros(N, dtype=cp.float32)

# Inicializar S[i]=0.999, I[i]=0.001, R[i]=0, y beta[i]=0.02+i*0.02 usando CuPy
# ....
S.fill(cp.float32(0.999))
I.fill(cp.float32(0.001))
R.fill(cp.float32(0.0))
beta.fill(cp.float32(0.1))

beta = cp.arange(N, dtype=cp.float32)
beta = cp.float32(0.02)+cp.float32(0.02)*beta

print("S=",S, len(beta))
print("I=",I, len(beta))
print("R=",R, len(beta))
print("beta=",beta, len(beta))

ntot = 5000

f = open("data4.csv", "w")
    
h_I = np.zeros(N, dtype=np.float32)
#h_I=I.get()
#np.savetxt(f, h_I.reshape(1, -1), delimiter='\t', fmt='%f')

# loop de tiempo
for p in range(ntot):
  # imprimir I[] en columnas
    # ...
    h_I=I.get()
    
    #print(h_I[0],h_I[1])
    np.savetxt(f, h_I.reshape(1, -1), delimiter='\t', fmt='%f')

    block_size = 256
    grid_size = (N + block_size - 1)//block_size

    # Llamar al kernel de actualizacion de S[],I[],R[]
    SIRkernel[grid_size, block_size](S, I, R, beta, cp.float32(g), cp.float32(dt), N)

    #cp.cuda.Device().synchronize()
    
