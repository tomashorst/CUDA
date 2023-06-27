APP?=reducebykey

N?=10000

all: ${APP}

${APP}: ${APP}.cu
	nvcc -O2 -o $@ $< -Wno-deprecated-gpu-targets -std=c++11

clean:
	rm -f reducebykey

run: ${APP}	
	./${APP}

submit: ${APP}
	qsub -N ${APP} jobGPU "./$(APP) $(N)"; 
	watch qstat

