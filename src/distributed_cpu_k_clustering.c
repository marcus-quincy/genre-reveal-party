#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "distributed_cpu_k_clustering.h"
#include "mpi_util.h"
#include "constants.h"

void k_means_clustering(Point* points,
			int points_size,
			int my_rank,
			int comm_sz) {

	MPI_Datatype mpi_point_type = create_point_datatype();

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

	MPI_Scatterv(points, send_counts, displs, mpi_point_type,
		     sub_points, send_counts[my_rank], mpi_point_type,
		     0, MPI_COMM_WORLD);

	/// Initialize centroids
	Point centroids[K_CLUSTERS];
	if (my_rank == 0)
	{
		//srand(time(0));
		srand(42);
		for (int i = 0; i < K_CLUSTERS; ++i) {
			centroids[i] = points[rand() % points_size];
		}
	}

	/// do the k clustering
	for (int i = 0; i < N_EPOCHS; ++i) {
		// make sure the centroids are the same (take from rank 0)
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(centroids, K_CLUSTERS, mpi_point_type, 0, MPI_COMM_WORLD);

		// For each centroid, compute distance from centroid to each point
		// and update point's cluster if necessary
		for (int cluster_id = 0; cluster_id < K_CLUSTERS; ++cluster_id){
			Point c = centroids[cluster_id];

			for (int j = 0; j < sub_points_size; ++j) {
				Point* p = &sub_points[j];
				double dist = point_distance(c, *p);
				if (dist < p->min_dist) {
					p->min_dist = dist;
					p->cluster = cluster_id;
				}
			}
		}

		// compute the local sums
		// Create vectors to keep track of data needed to compute means
		int sub_n_points[K_CLUSTERS];
		double sub_sum_x[K_CLUSTERS];
		double sub_sum_y[K_CLUSTERS];
		double sub_sum_z[K_CLUSTERS];
		for (int j = 0; j < K_CLUSTERS; ++j) {
			sub_n_points[j] = 0;
			sub_sum_x[j] = 0.0;
			sub_sum_y[j] = 0.0;
			sub_sum_z[j] = 0.0;
		}

		// Iterate over points to append data to centroids
		for (int j = 0; j < sub_points_size; j++) {
			Point* p = &sub_points[j];
			sub_n_points[p->cluster] += 1;
			sub_sum_x[p->cluster] += p->x;
			sub_sum_y[p->cluster] += p->y;
			sub_sum_z[p->cluster] += p->z;

			// reset distance
			p->min_dist = DBL_MAX;
		}

		// reduce the sums and then let rank 0 update the centroids
		int n_points[K_CLUSTERS];
		double sum_x[K_CLUSTERS];
		double sum_y[K_CLUSTERS];
		double sum_z[K_CLUSTERS];
		// TODO: is this necessary?
		if (my_rank == 0) {
			for (int j = 0; j < K_CLUSTERS; ++j) {
				n_points[j] = 0;
				sum_x[j] = 0.0;
				sum_y[j] = 0.0;
				sum_z[j] = 0.0;
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(sub_n_points, n_points, K_CLUSTERS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(sub_sum_x, sum_x, K_CLUSTERS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(sub_sum_y, sum_y, K_CLUSTERS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(sub_sum_z, sum_z, K_CLUSTERS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		if (my_rank == 0) {
			for (int cluster_id = 0; cluster_id < K_CLUSTERS; cluster_id++) {
				Point* c = &centroids[cluster_id];
				if (n_points[cluster_id] != 0) {
					c->x = sum_x[cluster_id] / n_points[cluster_id];
					c->y = sum_y[cluster_id] / n_points[cluster_id];
					c->z = sum_z[cluster_id] / n_points[cluster_id];
				}
			}
		}
	}

	/// send all the results back to rank 0 and update
	MPI_Gatherv(sub_points, send_counts[my_rank], mpi_point_type,
		    points, send_counts, displs, mpi_point_type,
		    0, MPI_COMM_WORLD);

	free(displs);
	free(send_counts);
	free(sub_points);
}
