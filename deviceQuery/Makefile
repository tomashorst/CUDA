APPS=queplacasoy

all: ${APPS}

%: %.cu
	nvcc -o $@ $<
clean:
	rm -f ${APPS}

run:	all
	./queplacasoy

submit:	all
	qsub jobGPU; watch qstat
