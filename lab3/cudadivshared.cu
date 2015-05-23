/*
 * @author Connie Shi 
 * Lab 3:  Write a reduction program in CUDA that finds the maximum 
 *         of an array of M integers.
 * Part 3 (Improved): 
 * 		   Write a CUDA version that makes use of shared memory, 
 * 		   prefetching, and different granularities. Performs better
 * 		   than original cudashared.cu version, because it does not
 *		   divide the data into subsets to sequential search.
 * 
 * 		Should be run on cuda1 machine with 1024 max threads per block.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 1024
#define WARP 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* Function Declarations */
void generate_random(int random[], int num_elements);
__global__ void max_in_blocks(int random[], int num_elements);
__device__ void sequential(int random[], int num_elements);


/* Generates M random numbers from 1 to 100000*/
void generate_random(int random[], int num_elements) { 
	int i;
	time_t t;

	srand((unsigned)time(&t)); //randomizes seed
	
	for (i = 0; i < num_elements; i++) {
		random[i] = (int)(((double)rand()/RAND_MAX)*100000);
	}
}

/* global function called from host and executed on kernel
 * Uses a tree-like structure to do parallel max reduction.
 * Avoids branch diversion, uses prefetching and shared memory.
 */
__global__
void max_in_blocks(int random[], int num_elements) {

	__shared__ int sdata[THREADS_PER_BLOCK];
	unsigned int tid = threadIdx.x;
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride;
	
	// Take data from global memory to shared memory for faster access
	sdata[tid] = random[index];

	__syncthreads();

	for (stride = blockDim.x/2; stride >= 32; stride >>= 1) {
		if (tid < stride && tid + stride < num_elements) {
			int current = sdata[tid + stride]; 
			if (sdata[tid] < current) {
				sdata[tid] = current;
			}
		}
		__syncthreads();
	}

	// Prevents branch divergence because it will stop running
	// Through while loop when it reaches the size of the warp
	// At which point, the max is in the first 32 positions.
	// Sequential search 32 elements is very fast.
	if (tid < 32) {
		sequential(sdata, num_elements);	
		random[blockIdx.x] = sdata[0];
	}
}

/* Sequential searches through the first 32 positions of the block
 * to prevent further divvying up of the warp into different tasks.
 */
__device__
void sequential(int sdata[], int num_elements) {
	int i;
	int max = 0;
	int tid = threadIdx.x;

	for (i = tid; i < tid + WARP && i < num_elements; i++) {
		if (max < sdata[i]) {
			max = sdata[i];
		}
	}
	// Put in index position, first element of the block
	sdata[0] = max;
}

/**************************************************************/

int main(int argc, char*argv[]) {
	int* h_random;
	int* d_random;
	int i;
	int largest = 0;
	clock_t start, end;

	if (argc != 2) {
		printf("Invalid number of commands: usage ./cudadivshared M\n");
		exit(1);
	}

	// Generate array of random elements
	int num_elements = atoi(argv[1]);
	h_random = (int*)malloc(sizeof(int) * num_elements);
	generate_random(h_random, num_elements);

	start = clock();

	// Calculation for grid dimensions
	int leftover = num_elements % WARP;
	int d_elements = num_elements - leftover;
	int n_blocks = (int)ceil((double)d_elements/THREADS_PER_BLOCK);
	int n_threads = (d_elements > THREADS_PER_BLOCK) ? THREADS_PER_BLOCK : d_elements;

	// Allocate space on device and copy over elements
	cudaError_t err = cudaMalloc((void**)&d_random, sizeof(int) * d_elements);
	if (err != cudaSuccess) {
		printf("cudaMalloc failure\n");
	}
	err = cudaMemcpy(d_random, h_random, sizeof(int) * d_elements, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
		printf("cudaMemcpy failure\n");
	}

	// Execute kernel
	max_in_blocks<<<n_blocks, n_threads>>>(d_random, d_elements);

	// While kernel is executing, find the max in leftover elements
	for (i = d_elements; i < num_elements; i++) {
		if (largest < h_random[i]) {
			largest = h_random[i];
		}
	}

	// Retrieve reduction results, only the first n_blocks element
	cudaMemcpy(h_random, d_random, sizeof(int) * n_blocks, cudaMemcpyDeviceToHost);
	
	// Check through n_blocks elements for the max
	for (i = 0; i < n_blocks; i ++) {
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
