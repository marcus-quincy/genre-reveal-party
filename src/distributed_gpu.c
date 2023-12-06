#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "csv.h"
#include "point.h"
#include "distributed_gpu_k_clustering.h"
#include "validation.h"
#include "constants.h"

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

#ifdef OUTPUT_TIME
	// sometimes it takes a second or 2 for the second rank to start, so have a barrier for timing purposes
	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();
#endif

	dist_gpu_k_means_clustering(points, LINE_COUNT - 1, my_rank, comm_sz);

#ifdef OUTPUT_TIME
	double end = MPI_Wtime();
	if (my_rank == 0) {
		printf("Elapsed time: %lf seconds\n", end - start);
	}
#endif

#ifdef RUN_VALIDATION
	if (my_rank == 0) {
		validate(points, LINE_COUNT - 1);
	}
#endif

	if (my_rank == 0) {
		writecsv(points);
	}

	MPI_Finalize();

	free(points);

	return 0;
}
