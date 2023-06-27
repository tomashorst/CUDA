APP?=pandemia

all: ${APP}

${APP}: ${APP}.cu
	nvcc -O2 -o $@ $< -Wno-deprecated-gpu-targets

clean:
	rm -f pandemia pandemia_solucion pandemia_thrust

run: ${APP}	
	./${APP}

submit: ${APP}
	qsub -N ${APP} jobGPU ./$(APP); 
	watch qstat

