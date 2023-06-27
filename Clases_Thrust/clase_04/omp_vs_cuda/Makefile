APP?=ompvsgpu

all: ${APP}

${APP}: $(APP).cu
	nvcc -O2 -o $@ $< -Wno-deprecated-gpu-targets -Xcompiler -fopenmp	

clean:
	rm -f $(APP)

run: $(APP)	
	./$(APP)

submit:	$(APP)
	qsub -N $(APP) jobGPU ./$(APP) 
	#watch qstat

