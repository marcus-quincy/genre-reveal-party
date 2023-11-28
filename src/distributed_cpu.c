#include <stdio.h>
#include <stdlib.h>

#include "point.h"
#include "csv.h"
#include "distributed_cpu_k_clustering.h"

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

	// Run k-means with 100 iterations and for 5 clusters
	k_means_clustering(points, LINE_COUNT - 1, 100, 5, my_rank, comm_sz);

	if (my_rank == 0) {
		writecsv(points);
	}

	MPI_Finalize();

	free(points);

	return 0;
}
