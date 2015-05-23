/* 
 * @author Connie Shi
 * Lab 3:  Write a reduction program in CUDA that finds the maximum
 *         of an array of M integers.
 * Part 1: Write a CUDA version that does not take thread divergence
 *	       into account. Uses interleaved addressing.
 *
 * 		Should be run on cuda1 machine with 1024 max threads per block.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 1024
#define THREADS_PER_SM 2048

/** Function Declarations **/
void generate_random(int random[], int num_elements);
__global__ void maximum(int random[], int num_elements);


/* Generates M random numbers from 1 to 100000 and put in array 
 * Multiply by 100000 because rand()/RAND_MAX is [0, 100000]
 */
void generate_random(int random[], int num_elements) { 
	int i;
	time_t t;

	srand((unsigned)time(&t)); //randomizes seeds

	for (i = 0; i < num_elements; i++) {
		random[i] = (int)(((double)rand()/RAND_MAX)*100000);
	}
}

/* global function called from host and executed on device
 * to do the parallel max reduction, using a tree-like 
 * structure branching using nearest neighbors.
 * Does NOT avoid branch divergence.
 */
__global__
void maximum(int random[], int num_elements) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int stride;

	__syncthreads();

	for (stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		if (tid % (2 * stride) == 0 && tid + stride < num_elements) {
			if (random[tid] < random[tid + stride]) {
				random[tid] = random[tid + stride];
			}
		}
	}
}

/**************************************************************/

int main(int argc, char* argv[]) {
	int* h_random;
	int* d_random;
	clock_t start, end;

	if (argc != 2) {
		printf("Invalid number of commands: usage ./cudabasic M\n");
		exit(1);
	}

	int num_elements = atoi(argv[1]);

	// Create array of M random elements
	h_random = (int*) malloc(sizeof(int) * num_elements);
	generate_random(h_random, num_elements);

	// Work in finding max starts
	start = clock();

	// Allocate space on device and copy over elements
	cudaError_t err = cudaMalloc((void**)&d_random, sizeof(int) * num_elements);
	if (err != cudaSuccess) {
		printf("cudaMalloc failure\n");
	}
	
	err = cudaMemcpy(d_random, h_random, sizeof(int) * num_elements, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("cudaMemcpy failure\n");
	}

	// Calculation for device dimensions, one element per thread
	int n_blocks = (int)ceil((double)num_elements/THREADS_PER_BLOCK);
	int n_threads = (num_elements > THREADS_PER_BLOCK) ? THREADS_PER_BLOCK : num_elements;

	// Execute kernel using calculated dimensions
	maximum<<<n_blocks, n_threads>>>(d_random, num_elements);
	
	// Copy back reduction results
	cudaMemcpy(h_random, d_random, sizeof(int) * num_elements, cudaMemcpyDeviceToHost);

	// Reduction results are in random[blockIdx.x * n_threads] for each block
	// Iterate over first element per block to find the max
	int i; 
	int largest = h_random[0];
	for (i = 0; i < num_elements; i += n_threads) {
		if (largest < h_random[i]) {
			largest = h_random[i];
		}
	}

	end = clock();
	
	printf("Time to find max %f\n", (double)(end-start)/CLOCKS_PER_SEC);  
	printf("Largest: %d\n", largest);

	// Clean up resources
	cudaFree(d_random);
	free(h_random);
}
