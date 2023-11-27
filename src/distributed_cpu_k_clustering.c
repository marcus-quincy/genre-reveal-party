#include "distributed_cpu_k_clustering.h"

#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void k_means_clustering(Point* points, int points_size, int epochs, int k) {
	// mpi meta information
	int comm_sz;
	int my_rank;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/// Initialize centroids
	Point centroids[k];
	if (my_rank == 0)
	{
		//srand(time(0));
		srand(42);
		for (int i = 0; i < k; ++i) {
			centroids[i] = points[rand() % points_size];
		}
	}

	/// distribute sections of the array
	int sub_points_size = points_size / comm_sz;
	if (points_size % comm_sz != 0) sub_points_size++;

	int number_full_processes = points_size % comm_sz;
	if (number_full_processes == 0)
		number_full_processes = comm_sz;

	int* send_counts = malloc(comm_sz * sizeof(int));
	int* displs = malloc(comm_sz * sizeof(int));
	for (int i = 0; i < comm_sz; i++) {
		if (i == 0) {
			displs[i] = 0;
			send_counts[i] = sub_points_size;
		}
		else {
			displs[i] = displs[i - 1] + send_counts[i - 1];
			int count = i < number_full_processes ? sub_points_size : sub_points_size - 1;
			send_counts[i] = count;
		}
	}

	Point* sub_points = malloc(send_counts[my_rank] * sizeof(Point));

	MPI_Datatype mpi_point_type = create_point_datatype();

	MPI_Scatterv(points, send_counts, displs, mpi_point_type,
		     sub_points, send_counts[my_rank], mpi_point_type, 0, MPI_COMM_WORLD);

	MPI_Finalize();
}

MPI_Datatype create_point_datatype() {
	MPI_Datatype point_type;
	int count = 5;
	int blocklengths[] = { 1, 1, 1, 1, 1 };
	MPI_Aint displacements[] = {
		offsetof(Point, x),
		offsetof(Point, y),
		offsetof(Point, z),
		offsetof(Point, cluster),
		offsetof(Point, min_dist)
	};

	MPI_Datatype types[] = {
		MPI_DOUBLE,
		MPI_DOUBLE,
		MPI_DOUBLE,
		MPI_INT,
		MPI_DOUBLE
	};

	MPI_Type_create_struct(count, blocklengths, displacements, types, &point_type);
	MPI_Type_commit(&point_type);

	return point_type;
}
