APP?=test

all: $(APP)

${APP}: $(APP).cu
	nvcc -O2 -o $@ $< -Wno-deprecated-gpu-targets

clean:
	rm -f $(APP) 

run: $(APP)	
	./$(APP)

submit:	
	qsub -N $(APP) jobGPU ./$(APP); watch qstat

