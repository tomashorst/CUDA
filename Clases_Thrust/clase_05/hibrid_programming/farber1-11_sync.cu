#include <iostream>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include "simple_timer.h"

using namespace std;

int main(int argc, char **argv)
{
	int N;
	if(argc==1){
		N=10000000;
	}
	else{
		N=atoi(argv[1]);
	}

	gpu_timer crono;
	
	crono.tic();
	// task 1: create the array
	thrust::device_vector<int> a(N);

	// task 2: fill the array
	thrust::sequence(a.begin(), a.end(), 0);

	// task 3: calculate the sum of the array
	int sumA= thrust::reduce(a.begin(),a.end(), (unsigned long long)0, thrust::plus<unsigned long long>());

	// task 4: calculate the sum of 0 .. N−1, aunque todos sabemos que es N(N-1)/2
	int sumCheck=0;
	// con OMP 
	#pragma omp parallel for reduction(+ : sumCheck)
	for(int i=0; i < N; i++) sumCheck += i;

	// task 5: check the results agree
	if(sumA == sumCheck) cout << "Test Succeeded!" << endl;
	else { cerr << "Test FAILED!" << endl; return(1);}

	cout << "ms=" << crono.tac() << endl;
	
	return(0);
}
