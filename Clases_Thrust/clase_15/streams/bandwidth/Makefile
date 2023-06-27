CXX=nvcc
CPPFLAGS=-I..
CXXFLAGS=-O -arch=sm_20 -use_fast_math
LDFLAGS=

TARGETS=bandwidth


%.o: %.cu
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c $<

bandwidth: bandwidth.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

all: $(TARGETS)

.PHONY: clean all

clean:
	rm -f *.o $(TARGETS)
