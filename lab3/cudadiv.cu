/*
 * @author Connie Shi 
 * Lab 3:  Write a reduction program in CUDA that finds the maximum 
 *         of an array of M integers.
 * Part 2: Write a CUDA version that DOES take thread divergence 
 * 		   into account. Uses sequential addressing.
 * 
 *   Should be run on cuda1 machine with 1024 max threads per block.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 1024
#define WARP 32

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
 * DOES avoid branch divergence. Uses coalescing.
 */
__global__
void max_in_blocks(int random[], int num_elements) {
	unsigned int tid = threadIdx.x;
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride;

	// Stop when warp (size 32) will have branch divergence
	for (stride = blockDim.x/2; stride >= 32; stride >>= 1) {
		if (tid < stride) {
			if (random[index] < random[index + stride]) {
				random[index] = random[index + stride];
			}
		}
		__syncthreads();
	}

	__syncthreads();

	// The max is in the first 32 positions
	// Sequential search 32 elements is very fast
	if (tid == 0) {
		sequential(random, num_elements);
	}
}

/* Sequential searches through the first 32 positions of the block
 * to prevent further divvying up of the warp into different branches
 */
__device__
void sequential(int random[], int num_elements) {
	int i;
	int max = 0;
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	for (i = tid; i < tid + WARP && i < num_elements; i++) {
		if (max < random[i]) {
			max = random[i];
		}
	}

	// Put in block index position
	random[blockIdx.x] = max;
}

/**************************************************************/

int main(int argc, char*argv[]) {
	int* h_random;
	int* d_random;
	int i;
	int largest = 0;
	clock_t start, end;

	if (argc != 2) {
		printf("Invalid number of commands: usage ./cudadiv M\n");
		exit(1);
	}

	// Generate array of random elements
	int num_elements = atoi(argv[1]);
	h_random = (int*)malloc(sizeof(int) * num_elements);
	generate_random(h_random, num_elements);

	// Work in finding max starts
	start = clock();

	// Calculation for grid dimensions to multiple of warp
	int leftover = num_elements % WARP;
	int d_elements = num_elements - leftover;
	int n_blocks = (int)ceil((double)d_elements/THREADS_PER_BLOCK);
	int n_threads = (d_elements > THREADS_PER_BLOCK) ? 
						THREADS_PER_BLOCK : d_elements;

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

	// Retrieve reduction results, only need n_blocks elements back
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
