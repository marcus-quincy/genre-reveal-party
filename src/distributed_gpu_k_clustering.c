#include <float.h>
#include <stdlib.h>
#include <stdio.h>

#include "distributed_gpu_k_clustering.h"
#include "constants.h"
#include "mpi_util.h"

extern void cuda_distances_kernel(Point* points_d, int points_size, Point* centroids_h, Point* centroids_d);
extern void cuda_setup(Point* points_h, Point** points_d, int points_size, Point** centroids_d, int** n_points_d, double** sum_x_d, double** sum_y_d, double** sum_z_d);
extern void cuda_cleanup(Point* points_h, Point* points_d, int points_size, Point* centroids_d, int* n_points_d, double* sum_x_d, double* sum_y_d, double* sum_z_d);
extern void cuda_sum_kernel(Point* points_d,
                            int points_size,
                            int* n_points_h,
                            double* sum_x_h,
                            double* sum_y_h,
                            double* sum_z_h,
                            int* n_points_d,
                            double* sum_x_d,
                            double* sum_y_d,
                            double* sum_z_d);

// perform the c clustering
void k_means_clustering(Point* points_h, int points_size, int my_rank, int comm_sz) {
    Point* points_d;
    Point* centroids_d;
    int* n_points_d;
    double* sum_x_d;
    double* sum_y_d;
    double* sum_z_d;

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

	Point* sub_points_h = malloc(send_counts[my_rank] * sizeof(Point));

	MPI_Scatterv(points_h, send_counts, displs, mpi_point_type,
		     sub_points_h, send_counts[my_rank], mpi_point_type,
		     0, MPI_COMM_WORLD);

	/// Initialize centroids
    Point centroids_h[K_CLUSTERS];
	if (my_rank == 0)
	{
		//srand(time(0));
		srand(42);
		for (int i = 0; i < K_CLUSTERS; ++i) {
			centroids_h[i] = points_h[rand() % points_size];
		}
	}

    /// only transfer the subset of points to GPU
    cuda_setup(sub_points_h, &points_d, send_counts[my_rank], &centroids_d, &n_points_d, &sum_x_d, &sum_y_d, &sum_z_d);
    //cuda_setup(points_h, &points_d, points_size, &centroids_d, &n_points_d, &sum_x_d, &sum_y_d, &sum_z_d);

    for (int i = 0; i < N_EPOCHS; ++i) {
        /// ensure each rank's centroids are in sync
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(centroids_h, K_CLUSTERS, mpi_point_type, 0, MPI_COMM_WORLD);

        // For each centroid, compute distance from centroid to each point
        // and update point's cluster if necessary
        cuda_distances_kernel(points_d, send_counts[my_rank], centroids_h, centroids_d);

        // Create vectors to keep track of data needed to compute means
        int sub_n_points_h[K_CLUSTERS];
        double sub_sum_x_h[K_CLUSTERS];
        double sub_sum_y_h[K_CLUSTERS];
        double sub_sum_z_h[K_CLUSTERS];

        /// get the GPU local sums
        cuda_sum_kernel(points_d, send_counts[my_rank], sub_n_points_h, sub_sum_x_h, sub_sum_y_h, sub_sum_z_h, n_points_d, sum_x_d, sum_y_d, sum_z_d);

        int n_points_h[K_CLUSTERS];
        double sum_x_h[K_CLUSTERS];
        double sum_y_h[K_CLUSTERS];
        double sum_z_h[K_CLUSTERS];
		if (my_rank == 0) {
			for (int j = 0; j < K_CLUSTERS; ++j) {
				n_points_h[j] = 0;
				sum_x_h[j] = 0.0;
				sum_y_h[j] = 0.0;
				sum_z_h[j] = 0.0;
			}
		}

        /// add the sums from all the GPUs
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(sub_n_points_h, n_points_h, K_CLUSTERS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(sub_sum_x_h, sum_x_h, K_CLUSTERS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(sub_sum_y_h, sum_y_h, K_CLUSTERS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(sub_sum_z_h, sum_z_h, K_CLUSTERS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		if (my_rank == 0) {
			for (int cluster_id = 0; cluster_id < K_CLUSTERS; cluster_id++) {
				Point* c = &centroids_h[cluster_id];
				if (n_points_h[cluster_id] != 0) {
					c->x = sum_x_h[cluster_id] / n_points_h[cluster_id];
					c->y = sum_y_h[cluster_id] / n_points_h[cluster_id];
					c->z = sum_z_h[cluster_id] / n_points_h[cluster_id];
				}
			}
		}
    }

    cuda_cleanup(sub_points_h, points_d, send_counts[my_rank], centroids_d, n_points_d, sum_x_d, sum_y_d, sum_z_d);

    /// send all the results back to rank 0 and update
    MPI_Gatherv(sub_points_h, send_counts[my_rank], mpi_point_type,
		points_h, send_counts, displs, mpi_point_type,
		0, MPI_COMM_WORLD);

    free(displs);
    free(send_counts);
    free(sub_points_h);
}
