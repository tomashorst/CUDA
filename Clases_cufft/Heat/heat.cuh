/*
 * heat.cuh
 *
 *  Created on: May 16, 2012
 *      Author: eze, ale
 */

#pragma once
#include <stdint.h> /* uint8_t */
#include "cutil.h"
#include "ppmimages.cuh"

using namespace std;


//---------- DEVICE FUNCTIONS -----------//


__global__ void CUDAkernel_SetQxQy(float *qqmodule, int nx, int ny){
        // compute idx and idy, the location of the element in the original LX*LY array 
        int idx = blockIdx.x*blockDim.x+threadIdx.x;
        int idy = blockIdx.y*blockDim.y+threadIdx.y;
        if ( idx < nx && idy < ny){
                // Using cos(qx) cos(qy) boundary conditions are automatically determined by the function periodicity otherwhise modulo "%" operations are required.
                float qx=2.f*M_PI/(float)LX*idx;
                float qy=2.f*M_PI/(float)LY*idy;
                int index = idx*ny + idy;
                qqmodule[index]=-2.f*(cosf(qx)+cosf(qy)-2.f);
        }
}

__global__ void CUDAkernel_Real2RealScaled(cufftReal *a, int nx, int ny, float scale){
        // compute idx and idy, the location of the element in the original LX*LY array 
        int idx = blockIdx.x*blockDim.x+threadIdx.x;
        int idy = blockIdx.y*blockDim.y+threadIdx.y;
        if ( idx < nx && idy < ny){
                int index = idx*ny + idy;
                a[index]= scale*a[index];
        }       
}


// Paso de Euler en el espacio de Fourier 
__global__ void euler_step(cufftComplex *a, const cufftComplex *b, const float *qqmodule, int nx, int ny){
        // compute idx and idy, the location of the element in the original NxN array 
        int idx = blockIdx.x*blockDim.x+threadIdx.x;
        int idy = blockIdx.y*blockDim.y+threadIdx.y;
        if ( idx < nx && idy < ny){
                int index = idx*ny + idy;
                const float fac=-(qqmodule[index]);
                a[index].x = (1.f+C_BETA*fac*DT)*a[index].x + DT*b[index].x;
                a[index].y = (1.f+C_BETA*fac*DT)*a[index].y + DT*b[index].y;
        }       
}



//////////////////////////////////////////////////////////////////////
class heat_model
{
        private:
        // cuFFT plan
        cufftHandle Sr2c, Sc2r; 

        public:

        // real arrays to hold the scalar field phi and the force (fuentes)
        cufftReal *h_phi_r, *h_force_r;
        cufftReal *d_phi_r, *d_force_r;

        // complex arrays to hold the transformed scalar field F[phi] and force F[force]
        cufftComplex *d_phi, *d_force; 

        
        // float array for the Fourier modes weights (the q's)
        float *d_qmodule; 
                        
        // intializing the class
        heat_model(){

                h_phi_r = (cufftReal *)malloc(sizeof(cufftReal)*LX*LY);
                h_force_r = (cufftReal *)malloc(sizeof(cufftReal)*LX*LY);
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_phi_r, sizeof(cufftReal)*LX*LY));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_force_r, sizeof(cufftReal)*LX*LY));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_phi, sizeof(cufftComplex)*LX*(LY/2+1)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_force, sizeof(cufftComplex)*LX*(LY/2+1)));                
                CUDA_SAFE_CALL(cudaMalloc((void**)&d_qmodule, sizeof(float)*LX*(LY/2+1))); 


                /* Fill everything with 0s */

                for (unsigned int k=0; k<LX*LY; k++){
                        h_phi_r[k] = (cufftReal) 0;
                        h_force_r[k] = (cufftReal) 0;
                }

                CUDA_SAFE_CALL(cudaMemset(d_phi_r, 0, LX*LY*sizeof(cufftReal)));
                CUDA_SAFE_CALL(cudaMemset(d_force_r, 0, LX*LY*sizeof(cufftReal))); 
                CUDA_SAFE_CALL(cudaMemset(d_phi, 0, LX*(LY/2+1)*sizeof(cufftComplex)));
                CUDA_SAFE_CALL(cudaMemset(d_force, 0, LX*(LY/2+1)*sizeof(cufftComplex)));               
                CUDA_SAFE_CALL(cudaMemset(d_qmodule, 0, LX*(LY/2+1)*sizeof(float)));

                // cuFFT plans R2C and C2R
                CUFFT_SAFE_CALL(cufftPlan2d(&Sr2c,LX,LY,CUFFT_R2C));
                CUFFT_SAFE_CALL(cufftPlan2d(&Sc2r,LX,LY,CUFFT_C2R));
        }


////////// Initial Condition for the fields //////////

        void InitParticular(int quien, const int nx, const int ny, const int shape, const float x_offset, const float y_offset, const float width, const float value){
		cufftReal *h, *d_h;
		if(quien==0) {
			h=h_phi_r;
			d_h=d_phi_r;
		}
		else{
			h=h_force_r;
			d_h=d_force_r;
		}	 
                for(int i=0;i<nx;i++){
                        for(int j=0;j<ny;j++){

                                int k=i*ny+j;

                                // uniforme
                                if (shape==0) h[k] = value;

                                // band
                                if (shape==1) h[k] = (ny*(0.5-width*0.5+y_offset) < j && j < ny*(0.5+width*0.5+y_offset))?(value):(0.0);

                                // circle
                                if (shape==2){
                                const float a = i-nx*(0.5+x_offset);
                                const float b = j-ny*(0.5+y_offset);
                                const float c = ny*width*0.5;
                                h[k] = ( a*a + b*b < c*c )?(value):(0.0);
                                }

                                // square_with_hole
                                if (shape==3){
                                h[k]= (nx*0.35 < i && i < nx*0.65 && ny*0.35 < j && j < ny*0.65)?(value):(0.0);
                                h[k]= (nx*0.45 < i && i < nx*0.55 && ny*0.45 < j && j < ny*0.55)?(0.0):(h[k]);
                               }

                                // two circles
                                if (shape==4){
                                float a = i-nx*(0.5+x_offset);
                                float b = j-ny*(0.5+y_offset);
                                float c = ny*width*0.5;
                                h[k] = ( a*a + b*b < c*c )?(value):(0.);
                                a = i-nx*(0.5-x_offset);
                                b = j-ny*(0.5-y_offset);
                                c = ny*width*0.5;
                                h[k] = ( a*a + b*b < c*c )?(-value):(h[k]);
				}


                        }
                }
                CUDA_SAFE_CALL(cudaMemcpy(d_h,h, sizeof(cufftReal)*LX*LY, cudaMemcpyHostToDevice));
        };


        void SetQxQy(){
                dim3 dimBlock(TILE_X, TILE_Y);
                dim3 dimGrid(LX/TILE_X, LY/TILE_Y/2+1);
                assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
                assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
                
                CUDAkernel_SetQxQy<<<dimGrid, dimBlock>>>(d_qmodule,LX,LY/2+1);
        };

////////// Copy Functions //////////

        /* transfer from CPU to GPU memory */
        void CpyHostToDevice(){
                CUDA_SAFE_CALL(cudaMemcpy(d_phi_r,h_phi_r, sizeof(cufftReal)*LX*LY, cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(d_force_r,h_force_r, sizeof(cufftReal)*LX*LY, cudaMemcpyHostToDevice));
        }

        /* transfer from GPU to CPU memory */
        void CpyDeviceToHost(){
                CUDA_SAFE_CALL(cudaMemcpy(h_phi_r, d_phi_r, sizeof(cufftReal)*LX*LY, cudaMemcpyDeviceToHost));
        }

////////// Fourier Transform Functions //////////

        void TransformToFourierSpace(){
                CUFFT_SAFE_CALL(cufftExecR2C(Sr2c, d_phi_r, d_phi));
        };

        void TransformForceToFourierSpace(){
                CUFFT_SAFE_CALL(cufftExecR2C(Sr2c, d_force_r, d_force));
        };

        void AntitransformFromFourierSpace(){
                CUFFT_SAFE_CALL(cufftExecC2R(Sc2r, d_phi, d_phi_r));
        };

        /* apply scaling ( an FFT followed by iFFT will give you back the same array times the length of the transform) */
        void Normalize(void){
                dim3 dimBlock(TILE_X, TILE_Y);
                dim3 dimGrid(LX/TILE_X, LY/TILE_Y);
                assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
                assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);

                CUDAkernel_Real2RealScaled<<<dimGrid, dimBlock>>>(d_phi_r, LX, LY, 1.f / ((float) LX * (float) LY));
        }

////////// Evolution Functions //////////

        void EulerStep(){
                dim3 dimBlock(TILE_X, TILE_Y);
                dim3 dimGrid(LX/TILE_X, LY/TILE_Y/2+1);
                assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
                assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
                euler_step<<<dimGrid, dimBlock>>>(d_phi, d_force, d_qmodule, LX, (LY/2+1));
        };
        
////////// Print Functions //////////

        void PrintData(ofstream &fstr){
                for(int i=0;i<LX;i++){
                        for(int j=0;j<LY;j++){
                                int k=i*LY+j;
                                fstr << setprecision (4) << h_phi_r[k] << " ";
                        }
                        fstr << endl;
                }
                fstr << endl;
        };

        void PrintPicture(char *picturename, int x){
        	writePPMbinaryImage(picturename, h_phi_r);
        /*char cmd[200];  char cmd2[200];
        sprintf(cmd, "convert frame%d.ppm frame%d.jpg", 100000000+x, 100000000+x);
        system(cmd);
        sprintf(cmd2, "rm frame%d.ppm", 100000000+x);
        system(cmd2);*/
        };
        

/////////////// Cleaning Issues ////////////////

        ~heat_model(){
                cufftDestroy(Sr2c);
                cufftDestroy(Sc2r);
                cudaFree(d_phi_r);
                cudaFree(d_force_r);
                free(h_phi_r);
                free(h_force_r);
                cudaFree(d_phi);
                cudaFree(d_force);
                cudaFree(d_qmodule);
        };
};


