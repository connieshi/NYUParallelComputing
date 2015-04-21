
/* 
 * NYU Parallel Computing Lab 2: OpenMP Modified Traveling Salesman
 * @author Connie Shi
 * 
 * You have a group of n cities (from 0 to n-1). The traveling salesman 
 * must start from city number 0, visit each city once, and does not have
 * to come back to the initial city. Find the shortest path.
 * 
 * Program uses branch and bound and greedy algorithm. Explores each
 * path starting at a different city using the greedy method, as soon as
 * the cost becomes larger than the previous calculated shortest path,
 * don't pursue and move to another branch.
 *
 * Utilizes OpenMP to run threads in parallel
 * 
 * NOTE: Has to be compiled with gcc -o tsm -fopenmp -lm tsm.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MOST_CITIES 1000 
#define MAX_LENGTH 100000
#define MAX_NUM 100000000

typedef struct Path {
	long visited[MOST_CITIES];	// Boolean city visited
	long cities_visited;		// Number of cities visited so far
	long path[MOST_CITIES];		// Path of cities
	long distance;			// Distance traveled
} Path;

typedef struct Graph {
	long** matrix;			// Input matrix
	Path** list; 			// List of paths
	Path* best_path;		// Shortest path overall
	long cities; 			// Number of cities total
} Graph;

/** Function Declarations **/
void create_one_path(Path* p);
void create_path_list();
void go_first_city(Path* p, long first_city);
void greedy_path(Path* p, long first_city);
long check_current_distance(Path* current, Path* best_path);
void read_input(char* file);
void print_path(Path* p);

/** Global Variables **/
Graph* graph;

/* 
 * Reads file from passed in argument
 * Initiates and creates necessary structs to represent graph
 */
void read_input(char* file) {
	FILE *fp;
	long i, j, k;
	long temp[MAX_LENGTH];
	float total_size = 0;
	
	// Open file to read
	fp = fopen(file, "r");
	if (!fp) {
	    	printf("Cannot open file %s\n", file);
		exit(1);
	}
	
	// Stores each number in temporary array
	// Necessary because we do not yet know the number of cities
	i = 0;
	while (fscanf(fp, "%d", &temp[i]) > 0) {
		i++;
		total_size++;
	}
	
	// Allocate graph and all necessary fields
	graph = (Graph*) malloc(sizeof(Graph));
	if (!graph) {
		printf("Cannot allocate graph\n");
		exit(1);
	}
	graph->cities = (long) sqrt(total_size); //Calculate number of cities
	graph->matrix = (long**) malloc(graph->cities * sizeof(long*));
	if (!graph->matrix) {
		printf("Cannot allocate matrix\n");
		exit(1);
	}
	graph->best_path = NULL;
	
	// Copy over values from temporary array to allocated matrix
	k = 0;
	for (i = 0; i < graph->cities; i++) {
		graph->matrix[i] = (long*) malloc(graph->cities * sizeof(long));
		if(!graph->matrix[i]) {
           		printf("Cannot allocate matrix[%d]\n", i);
            		exit(1);
        }
		for (j = 0; j < graph->cities; j++) {
			graph->matrix[i][j] = temp[k];
			k++;
		}
	}
	
	fclose(fp);
}

/*
 * Creates and initializes the list of paths
 * That will traverse through the graph with different starting cities
 * From 1 to n-1, because at any generated input, the distance from
 * 0 to any other city is the same
 */
void create_path_list() {
	long i;
	
	graph->list = (Path**) malloc(graph->cities * sizeof(Path));
	graph->list[0] = NULL; //0th location unused
	
	for (i = 1; i < graph->cities; i++) {
		graph->list[i] = (Path*) malloc(sizeof(Path));
		create_one_path(graph->list[i]);
	}
}

/*
 * Creates and initializes each individual path, starting at city 0
 */
void create_one_path(Path* p) {
	long i;
	
	for (i = 0; i < graph->cities; i++) {
		p->visited[i] = 0;
		p->path[i] = MAX_NUM; // Error catching
	}
	
	p->path[0] = 0; // Path always starts at 0
	p->cities_visited = 1;
	p->visited[0] = 1;
	p->distance += graph->matrix[0][0];
}

/*
 * Every path goes from 0 to a different city as starting point
 * Since the distance from 0 to any other city is the same.
 */
void go_first_city(Path* p, long first_city) {
	p->path[1] = first_city;
	p->visited[first_city] = 1;
	p->distance += graph->matrix[0][first_city];
	p->cities_visited++;
}

/*
 * Greedy algorithm to find the shortest path from city 0 to
 * Each city and then greedy from that point forward to find
 * The shortest path overall.
 */
void greedy_path(Path* p, long first_city) {
	go_first_city(p, first_city); // From 1 to n-1

	long i = p->cities_visited;
	long j;
	long current;
	long min_index;
	
	// Travel through every city once while calculating distance
	while (i < graph->cities) {
		current = p->path[i-1]; // Current city
		long min = MAX_NUM;
		
		// Check for next closest city ignoring cities that have already
		// Been visited or is on the diagonal (cannot travel from
		// One city to itself as a self-loop is illegal)
		for (j = 0; j < graph->cities; j++) {
			if (j == current || p->visited[j] == 1) {
				continue;
			} else if (min > graph->matrix[current][j]) {
				min = graph->matrix[current][j];
				min_index = j;
			}
		}
		
		// Go to the next city and update all information
		p->path[i] = min_index;
		p->visited[min_index] = 1;
		p->distance += min;
		p->cities_visited += 1;
		i++;
		
		// If the cost of going to this next city already exceeds the
		// Current shortest path, there is no point continuing further
		if (graph->best_path != NULL && 
		  check_current_distance(p, graph->best_path) == -1) {
			p = NULL;
			return;
		}
	}
	
	// If while loop finished without returning, we found the current 
	// Shortest path. Updating shared variable is a critical section
	#pragma omp critical
	if (graph->best_path == NULL 
	  || check_current_distance(p, graph->best_path) == 1) {
		graph->best_path = p;
	}
}

/*
 * Checks to see if the current path's distance traveled exceeds
 * the total distance of the best path. Return 1 to continue traveling,
 * and -1 to stop.
 */
long check_current_distance(Path* current, Path* best_path) {
	return (current->distance >= best_path->distance) ? -1 : 1;
}

/*
 * Prints the shortest path and the total distance traveled
 */
void print_path(Path* p) {
	long i;
	
	printf("Shortest path:\n");
	for (i = 0; i < graph->cities; i++) {
		printf("%d ", p->path[i]);
	}
	
	printf("\n");
	printf("Total Weight: %d\n", p->distance);
}

/********************************************************************/

int main(int argc, char *argv[]) { 
	long i; 
	
	if (argc != 2) {
		printf("Invalid number of commands: usage ./tsm filename.txt");
		exit(1);
	}
	
	read_input(argv[1]);
	create_path_list();
	
	// Assign optimal number of thread count based on 
	// crunchy4.cims.nyu.edu which is a 4 8-Core 2.4GHz AMD Opteron 6136
	// Optimal number of threads is found by testing using time command
	int thread_count = 0;
	int num_cities = graph->cities;
	if (num_cities > 1 && num_cities <= 16) {
		thread_count = 2;
	} else if (num_cities > 16 && num_cities <= 64) {
		thread_count = 4;
	} else if (num_cities > 64 && num_cities <= 128) {
		thread_count = 8;
	} else if (num_cities > 128) {
		thread_count = 16;
	}
	
	#pragma omp parallel for num_threads(thread_count) \
		default(none) private(i) shared(graph) \
		schedule(dynamic)
	for (i = 1; i < graph->cities; i++) {
		greedy_path(graph->list[i], i);
	}

	print_path(graph->best_path);
	
	// Clean up resources malloced
	for (i = 0; i < graph->cities; i++) {
		free(graph->list[i]);
		free(graph->matrix[i]);
	}
	free(graph->list);
	free(graph->matrix);
	free(graph);

	return 0;
}
