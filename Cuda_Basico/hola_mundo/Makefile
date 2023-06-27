APPS=hello

all: ${APPS}

%: %.cu
	nvcc -o $@ $<
clean:
	rm -f ${APPS}

run:	all
	./hello

submit:	all
	qsub jobGPU

submitwatch:	all
	qsub jobGPU; watch qstat
