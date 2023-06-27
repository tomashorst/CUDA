APPS=llenar_vector

all: ${APPS}

%: %.cu
	nvcc -o $@ $<
clean:
	rm -f ${APPS}

run:	all
	./llenar_vector

submit:	all
	qsub jobGPU; watch qstat
