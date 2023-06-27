APPS=convolucion2d_cufft.solucion

FLAGS=-DPRINTCONVOLUTION -DPRINTSTRUCTUREFACTOR -DRANDOM -DFILTER 

#-DLAPLACIAN #-DFILTER #-DRANDOM # -DPRINTSTRUCTUREFACTOR  #-DCUADRADO

all: ${APPS}

%: %.cu
	nvcc -O2 -lcufft $(FLAGS) -I/share/apps/icnpg/common/ -o $@ $< 
clean:
	rm -f ${APPS}

run:	all
	./convolucion2d_cufft

submit:	all
	qsub jobGPU; watch qstat
