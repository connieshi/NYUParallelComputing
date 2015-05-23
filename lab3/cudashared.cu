/*
 * @author Connie Shi
 * Lab 3:  Write a reduction program in CUDA that finds the maximum of
 *         an array of M integers.
 * Part 3: Write a CUDA version that makes use of shared memory, 
 * 		   prefetching, coalesing and different granularities.
 * 
 * 		Should be run on cuda1 machine with 1024 max threads per block.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 1024
#define WARP 32

/* Function Declarations */
void generate_random(int random[], int num_elements);
__global__ void max_in_block(int random[], int num_elements);
__device__ void sequential(int random[], int sdata[], int n_els, int el_per_thread);
__device__ void check_warp(int sdata[], int num_elements);

/* Generate M random numbers from 1 to 100000*/
void generate_random(int random[], int num_elements){ 
	int i;
	time_t t;

	srand((unsigned)time(&t));
	
	for (i = 0; i < num_elements; i++) {
		random[i] = (int)(((double)rand()/(double)RAND_MAX)*100000);
	}
}

/* global function called from host and executed on kernel
 * Uses a tree-like structure that avoids branch divergence. Each thread
 * sequentially searches over its own assigned elements_per_thread
 * and puts the max of the subset in array stored in shared memory.
 */
__global__
void max_in_block(int random[], int num_elements, int elements_per_thread) {

	// Find max of subset and store in shared memory
	__shared__ int sdata[THREADS_PER_BLOCK];
	sequential(random, sdata, num_elements, elements_per_thread);
	__syncthreads();

	unsigned int tid = threadIdx.x;
	unsigned int stride;

	// Tree reduction on array in shared memory for max
	for (stride = blockDim.x/2; stride >= WARP; stride >>= 1) {
		if (tid < stride && tid + stride < blockDim.x) {
			int current = sdata[tid + stride];
			if (sdata[tid] < current) {
				sdata[tid] = current;
			}
		}
		__syncthreads();
	} 

	__syncthreads();

	// Check warp size for max and put in blockIdx.x
	if (tid == 0) {
		check_warp(sdata, num_elements);
		random[blockIdx.x] = sdata[0];
	}
}

/* Sequential searches through elements_per_thread for each thread
 * A subset of the data is assigned to each thread to check sequentially
 * Stores the result in shared memory array in threadIdx.x position
 */
__device__ 
void sequential(int random[], int sdata[], int num_elements, int elements_per_thread) {
	int i;
	int max = 0;
	unsigned int index = (threadIdx.x + blockIdx.x * blockDim.x) * elements_per_thread;

	for (i = index; i < index + elements_per_thread && i < num_elements; i++) {
		if (max < random[i]) {
			max = random[i];
		}
	}
	sdata[threadIdx.x] = max;
}

/* Check warp size for the max thread and put in sdata[0] position
 */
__device__ 
void check_warp(int sdata[], int num_elements) {
	int i;
	int max = 0;
	int tid = threadIdx.x;

	for (i = tid; i < tid + WARP && i < num_elements; i++) {
		if (max < sdata[i]) {
			max = sdata[i];
		}
	}
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
		printf("Invalid number of commands: usage ./cudashared M\n");
		exit(1);
	}

	// Generate array of random elements
	int num_elements = atoi(argv[1]);
	h_random = (int*)malloc(sizeof(int)*num_elements);
	generate_random(h_random, num_elements);

	start = clock();

	// Calculate grid dimensions
	int leftover = num_elements % WARP;
	int d_elements = num_elements - leftover;
	int elements_per_thread = THREADS_PER_BLOCK; 
	int n_threads = (int)ceil((double)d_elements/elements_per_thread);
	int n_blocks = (int)ceil(((double)n_threads/THREADS_PER_BLOCK));

	// Allocate space on device and copy over
	cudaError_t err = cudaMalloc((void**)&d_random, sizeof(int) * d_elements);
	if (err != cudaSuccess) {
		printf("cudaMalloc failure\n");
	}
	err = cudaMemcpy(d_random, h_random, sizeof(int) * d_elements, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("cudaMemcpyfailure\n");
	}

	// Execute kernel
	max_in_block<<<n_blocks, THREADS_PER_BLOCK>>>(d_random, d_elements, elements_per_thread);

	// While kernel is executing, find the max in leftover elements
	for (i = d_elements; i < num_elements; i++) {
		if (largest < h_random[i]) {
			largest = h_random[i];
		}
	}
 
	// Retrieve reduction results, only the first n_blocks element
	cudaMemcpy(h_random, d_random, sizeof(int) * n_blocks, cudaMemcpyDeviceToHost);

	// Check through n_blocks elements for the max
	for (i = 0; i < n_blocks; i++) {
		if (h_random[i] > largest) {
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
