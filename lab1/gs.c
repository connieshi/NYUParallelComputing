#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
/* 
 * @author Connie Shi cs3313@nyu.edu
 * Parallel Computing Lab 1
 * 
 * Given a set of n equations with n unknowns (x1 to xn), program 
 * will calculate the values of x1 to xn within an error margin of e%.
 */
 
/***** Globals ******/
float **a;    /* The coefficients */
float *x;     /* The unknowns */
float *b;     /* The constants */
float err;    /* The absolute relative error */
int num = 0;  /* number of unknowns */
float *aii;   /* Values at aii for divide*/
float *new_a; /* Enable a for scatter operation */
float *temp;  // Used in serial version

/****** Function declarations */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();    /* Read input from file */
int error_function(float *eval, int num);
void make_a_scatter(int num, int comm_sz, int *num_send);

/* These functions are used in serial version */
int serial_version(int argc, char *argv[]);
float linear_multiply(int pos);
void solve_for_values();
int error_function_serial();
/********************************/

/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

/*
   Conditions for convergence (diagonal dominance):
   1. diagonal element >= sum of all other elements of the row
   2. At least one diagonal element > sum of all other elements of the row
 */
void check_matrix() {
  int bigger = 0; /* Set to 1 if at least one diag element > sum  */
  int i, j;
  float sum = 0;
  float aii = 0;

  for (i = 0; i < num; i++) {
    sum = 0;
    aii = fabs(a[i][i]);

    for (j = 0; j < num; j++)
      if (j != i) sum += fabs(a[i][j]);

    if (aii < sum) {
      printf("The matrix will not converge\n");
      exit(1);
    }

    if (aii > sum) bigger++;
  }

  if (!bigger) {
    printf("The matrix will not converge\n");
    exit(1);
  }
}

/******************************************************/
/* Read input from file */
void get_input(char filename[]) {
  FILE *fp;
  int i, j;

  fp = fopen(filename, "r");
  if (!fp) {
    printf("Cannot open file %s\n", filename);
    exit(1);
  }

  fscanf(fp, "%d ", &num);
  fscanf(fp, "%f ", &err);

  /* Now, time to allocate the matrices and vectors */
  a = (float **)malloc(num * sizeof(float *));

  if (!a) {
    printf("Cannot allocate a!\n");
    exit(1);
  }

  for (i = 0; i < num; i++) {
    a[i] = (float *)malloc(num * sizeof(float));
    if (!a[i]) {
      printf("Cannot allocate a[%d]!\n", i);
      exit(1);
    }
  }

  x = (float *)malloc(num * sizeof(float));
  if (!x) {
    printf("Cannot allocate x!\n");
    exit(1);
  }

  // What is curr?
  /* curr = (float *) malloc(num * sizeof(float));
  if( !curr)
  {
          printf("Cannot allocate curr!\n");
          exit(1);
  } */

  b = (float *)malloc(num * sizeof(float));
  aii = (float *)malloc(num * sizeof(float));

  if (!b) {
    printf("Cannot allocate b!\n");
    exit(1);
  }

  /* Now .. Filling the blanks */

  /* The initial values of Xs */
  for (i = 0; i < num; i++) {
    fscanf(fp, "%f ", &x[i]);
    //printf("%d %f\n",i, x[i]);
  }

  for (i = 0; i < num; i++) {
    for (j = 0; j < num; j++) {
		fscanf(fp, "%f ", &a[i][j]);
	}
    /* reading the b element */
    fscanf(fp, "%f ", &b[i]);
  }

  fclose(fp);
}

/* Used in serial, multiplies each linear equation
 */
float linear_multiply(int pos) {
  float eval = b[pos];
  int i;

  for (i = 0; i < num; i++) {
    if (i != pos) {  // exclude position pos
      eval -= a[pos][i] * x[i];
    }
  }
  return eval / a[pos][pos];
}

/*
 * Multiplies each line of the linear equations
 */
void solve_for_values() {
  temp = (float *)malloc(num * sizeof(float));
  int i;
  for (i = 0; i < num; i++) {
    temp[i] = linear_multiply(i);
  }
}

/* This is the function that calculates the error,
 * Exiting the do-while loop if the calculator error is
 * Below the minimum value
 */
int error_function(float *eval, int num) {
  int i;
  float cur_error;

  for (i = 0; i < num; i++) {
    cur_error = fabsf((eval[i] - x[i]) / eval[i]);
    if (cur_error > err) {
      return 1;
    }
  }
  return 0;
}

/* Error function implemented for serial version*/
int error_function_serial() {
  int i;
  float cur_error;

  for (i = 0; i < num; i++) {
    cur_error = fabsf((temp[i] - x[i]) / temp[i]);
    if (cur_error > err) {
      return 1;
    }
  }
  return 0;
}

/*
 * a[] was allocated an array of pointers.
 * This function makes new_a[] an array of floats
 * To use the Scatter/Gather MPI functions
 */
void make_a_scatter(int num, int comm_sz, int *num_send) {
  int i, j, k = 0;
  new_a = (float *)malloc(num * num * sizeof(float));

  for (i = 0; i < num; i++) {
    for (j = 0; j < num; j++) {
      new_a[k] = a[i][j];
      k++;
      if (i == j) {
        aii[i] = a[i][j];
      }
    }
  }
  free(a);
}

/*
 * My serial version of the code before parallelizing.
 */
int serial_version(int argc, char *argv[]) {
  temp = (float *)malloc(num * sizeof(float));
  int i;
  int nit = 0; /* number of iterations */
  float *save;

  // Find the x values, swap pointers when using new x as old x
  do {
    solve_for_values();
    if (error_function_serial()) {
      nit++;
      memset(x, 0, num * sizeof(float));
      save = x;
      x = temp;
      temp = save;
    } else {
      save = x;
      x = temp;
      temp = save;
      break;
    }
  } while (1);

  /* Writing to the stdout */
  /* Keep that same format */
  for (i = 0; i < num; i++) printf("%f\n", x[i]);

  printf("total number of iterations: %d\n", nit);

  free(temp);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}

int main(int argc, char *argv[]) {
  int i;
  int nit = 0; /* number of iterations */
  int j, k;
  int comm_sz;
  int my_rank;
  
  if (argc != 2) {
    printf("Usage: gsref filename\n");
    exit(1);
  }

  // Read the input file and fill the global data structure above
  get_input(argv[1]);
  check_matrix();

  // Start MPI 
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // If size==1, just use the serial version of my program
  if (comm_sz == 1) {
    int exit = serial_version();
    return exit;
  }
  
  int per_proc = num / comm_sz;
  int leftover = num % comm_sz;
  int num_send[comm_sz];
  int num_send2[comm_sz];
  int findx[comm_sz][2];
  int total = 0;

  /* Used for Scatterv and Gatherv */
  int recvbuf[comm_sz];
  int recvbuf2[comm_sz];
  int displacement[comm_sz];
  int displacement2[comm_sz];
  
  for (i = 0; i < comm_sz; i++) {
	num_send[i] = (i < leftover) ? per_proc + 1 : per_proc;
    num_send2[i] = num_send[i]*num;
	//findx keeps track of the Xi values that each process
	//is responsible for computing
    if (num_send[i] != 0) {
      findx[i][0] = total;
      findx[i][1] = total + num_send[i] - 1;
    }
    recvbuf[i] = num_send[i];
    recvbuf2[i] = recvbuf[i]*num;
    displacement[i] = total;
    displacement2[i] = displacement[i]*num;
    total += num_send[i];
  }

  // Allocate temporary arrays for each process
  float *currX = (float *)malloc(num * sizeof(float));
  float *tempX = (float *)malloc(num_send[my_rank] * sizeof(float));
  float *tempA = (float *)malloc(num_send2[my_rank] * sizeof(float));
  float *tempB = (float *)malloc(num * sizeof(float));
  float *tempAii = (float *)malloc(num * sizeof(float));

  // Change float**a to float*new_a so it is eligible for scattering
  make_a_scatter(num, comm_sz, num_send);

  // Copy to local arrays in order to broadcast elements
  for (i = 0; i < num; i++) {
    tempB[i] = b[i];
    tempAii[i] = aii[i];
  }

  // Scatter and broadcast the appropriate values to use 
  MPI_Bcast(tempB, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(tempAii, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(new_a, num_send2, displacement2, MPI_FLOAT, tempA, 
	recvbuf2[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(x, num_send, displacement, MPI_FLOAT, tempX, 
	recvbuf[my_rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Copy over X values
  for (i = 0; i < num; i++) {
    currX[i] = x[i];
  }

  do {
	  
    // Use new_x to evaluate
    for (i = 0; i < num; i++) {
      x[i] = currX[i];
    }

	int getX = findx[my_rank][0];

	// Evaluate for each Xi
    for (i = 0; i < num_send[my_rank]; i++) {
      //int getb = findx[my_rank][i];
      tempX[i] = tempB[getX];

	  // Subtract all AiXi's from b
      for (j = 0; j < num; j++) {
        tempX[i] -= tempA[i * num + j] * x[j];
      }

	  // tempX[i] already subtracted the Aii value in previous for-loop
	  // So restore it by adding the Aii value
      //int getaii = findx[my_rank][i];
      tempX[i] += tempAii[getX] * x[getX];
      
      // Finally, divide by Aii
      tempX[i] = tempX[i] / tempAii[getX];
      
      getX++;
    }

    MPI_Allgatherv(tempX, num_send[my_rank], MPI_FLOAT, currX, recvbuf,
		displacement, MPI_FLOAT, MPI_COMM_WORLD);
	
    nit++;
  } while (error_function(currX, num));

  /* Only one process should print output, otherwise
  *  All processes would print all outputs
  */
  if (my_rank == 0) {
    /* Writing to the stdout */
    /* Keep that same format */
    for (i = 0; i < num; i++) {
      printf("%f\n", x[i]);
    }

    printf("total number of iterations: %d\n", nit - 1);
    // Minus 1 to account for the first nit++ iteration of do-while

    // Free used global variables
    free(b);
    free(aii);
    free(x);
    free(new_a);
  }

  // Free used local variables
  free(currX);
  free(tempAii);
  free(tempX);
  free(tempA);
  free(tempB);

  // Ensures that a process does not exit before others are done
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  
  return 0;
}
