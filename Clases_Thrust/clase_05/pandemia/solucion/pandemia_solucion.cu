/*
	Vamos a resolver el modelo SIR de epidemias
	
	dS/dt = -beta*S*I
	dI/dt = beta*S*I - gamma*I
	dR/dt = gamma*I

 	para N valores de beta
*/


// Poner los #include que hagan falta!
#include<thrust/device_vector.h>
#include<thrust/fill.h>
#include<thrust/sequence.h>

#define gamma	0.1  // tasa de recuperacion
#define Dt	0.1  // paso de tiempo

// escribir un Kernel "modeloSIR" que actualice los arrays S, I, R, cada elemento con diferente beta
// ....
// ....
__global__ void modeloSIR(float *S, float *I, float *R, float *beta, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	if(i<N){
		//dS/dt = -beta*S*I --> newS = S - Dt*beta*S*I
		float newS = S[i] - Dt*beta[i]*S[i]*I[i];

        	//dI/dt = beta*S*I - gamma*I
		float newI = I[i] + Dt*(beta[i]*S[i]*I[i]-gamma*I[i]);

        	//dR/dt = gamma*I
		float newR = R[i] + Dt*(gamma*I[i]);

		S[i]=newS;
		I[i]=newI;
		R[i]=newR;
	}
}

int main(void)
{
    int N=10;

    // Declarar y Alocar memoria para los arrays de device S, I, R y beta usando thrust 
    // ....
    thrust::device_vector<float> S(N);
    thrust::device_vector<float> I(N);
    thrust::device_vector<float> R(N);
    thrust::device_vector<float> beta(N);

    // Inicializar S[i]=0.999, I[i]=0.001, R[i]=0, y beta[i]=0.02+i*0.02 usando Thrust
    // ....
    thrust::fill(S.begin(),S.end(),0.999);
    thrust::fill(I.begin(),I.end(),0.001);
    thrust::fill(R.begin(),R.end(),0.0);

    thrust::sequence(beta.begin(),beta.end(),0.02,0.02);

    int ntot=5000;
	
    // loop de tiempo
    for(int n=0;n<ntot;n++){	

	// imprimir I[] en columnas
	// ...
	for(int i=0;i<N;i++){
		std::cout << I[i] << "\t";
	}
	std::cout << "\n";

	// extraer los punteros crudos Sraw, Iraw, Rraw, Betaraw de los vectores de thrust
	float *Sraw = thrust::raw_pointer_cast(&S[0]);   
	float *Iraw = thrust::raw_pointer_cast(&I[0]);   
	float *Rraw = thrust::raw_pointer_cast(&R[0]);   
	float *betaraw = thrust::raw_pointer_cast(&beta[0]);   

    	// Llamar al kernel de actualizacion de S[],I[],R[]
	modeloSIR<<< (N+32-1)/32,32>>>(Sraw,Iraw,Rraw,betaraw,N);
    }
}

// dibujar asi...
// gnuplot> plot [][1e-5:] for[i=1:10] 'zzz' u (column(i)) w lp t sprintf('beta=%f',(i*0.02+0.02))

