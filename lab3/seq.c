/* @author Connie Shi
 * Sequential version of lab 3
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX(a,b) (a > b) ? a : b

int* generate_random(int random[], int num_elements){
	int i;
	srand(time(NULL));
	
	for (i = 0; i < num_elements; i++) {
		random[i] = (int)(((double)rand()/RAND_MAX)*100000);
	}
	
	return random;
}

int maximum(int random[], int num_elements) {
	int max = 0;
	int i;
	
	for (i = 0; i < num_elements; i++) {
		max = MAX(max, random[i]);
	}
	
	return max;
}

int main(int argc, char*argv[]) {
	clock_t start, end;

	if (argc != 2) {
		printf("Invalid number of commands: usage ./seq M\n");
		exit(1);
	}
	
	int num_elements = atoi(argv[1]);
	int* random = (int*)malloc(sizeof(int)*num_elements);
	generate_random(random, num_elements);
	
	start = clock();
	int max = maximum(random, num_elements);
	end = clock();

	printf("Time to find max %f\n", (double)(end-start)/CLOCKS_PER_SEC);
	printf("Largest: %d\n", max);
}
