APP?=zip_iterator

all: ${APP}

${APP}: ${APP}.cu
	nvcc -O2 -o $@ $< -Wno-deprecated-gpu-targets

clean:
	rm -f zip_iterator reduce_with_transform_iterator2 reduce_with_transform_iterator \
	zip_iterator2 tabulate_with_counting_iterator tabulate_with_counting_iterator2 permutation_iterator

run: ${APP}	
	./${APP}

submit:	
	qsub -N ${APP} jobGPU ./$(APP); 
	watch qstat

