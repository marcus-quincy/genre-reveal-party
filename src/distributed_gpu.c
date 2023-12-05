#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "csv.h"
#include "point.h"
#include "distributed_gpu_k_clustering.h"

int main() {
	Point* points;

	// mpi meta information
	int comm_sz;
	int my_rank;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if (my_rank == 0) {
		points = readcsv();
		if (points == NULL) return 1;
	}
	//Run k-means with 100 iterations and for 5 clusters
	struct timeval start, end;

	// Record the start time
	gettimeofday(&start, NULL);

	k_means_clustering(points, LINE_COUNT - 1, my_rank, comm_sz);
	// Record the end time
	gettimeofday(&end, NULL);
	// Calculate the elapsed time in seconds
	double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

	// Print the elapsed time
	printf("Elapsed time: %f seconds\n", elapsed_time);

	if (my_rank == 0) {
		writecsv(points);
	}

	MPI_Finalize();

	free(points);

	return 0;
}
