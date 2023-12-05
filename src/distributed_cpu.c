#include <stdio.h>
#include <stdlib.h>

#include "point.h"
#include "csv.h"
#include "distributed_cpu_k_clustering.h"
#include "constants.h"
#include "validation.h"

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

	dist_cpu_k_means_clustering(points, LINE_COUNT - 1, my_rank, comm_sz);

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
