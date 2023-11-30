#include <float.h>
#include <stdlib.h>

#include "shared_gpu_k_clustering.h"

extern void cuda_Kernel_1(Point* points_h, int points_size, Point c, int cluster_id);
extern void cuda_Kernel_2(Point* points_h, int points_size, int* n_points, double* sum_x, double* sum_y, double* sum_z, int k);

// perform the c clustering
void k_means_clustering(Point* points, int points_size, int epochs, int k) {
	Point centroids[k];
	srand(42);
	for (int i = 0; i < k; ++i) {
		centroids[i] = points[rand() % points_size];
	}

	for (int i = 0; i < epochs; ++i) {
		// For each centroid, compute distance from centroid to each point
		// and update point's cluster if necessary
		for (int cluster_id = 0; cluster_id < k; ++cluster_id){
			Point c = centroids[cluster_id];
			cuda_Kernel_1(points, points_size, c, cluster_id);
		}

		// Create vectors to keep track of data needed to compute means
		int n_points[k];
		double sum_x[k];
		double sum_y[k];
		double sum_z[k];
		for (int j = 0; j < k; ++j) {
			n_points[j] = 0;
			sum_x[j] = 0.0;
			sum_y[j] = 0.0;
			sum_z[j] = 0.0;
		}


    //Kernel2
	cuda_Kernel_2(points, points_size, n_points, sum_x, sum_y, sum_z, k);

		for (int cluster_id = 0; cluster_id < k; cluster_id++) {
			Point* c = &centroids[cluster_id];
			if (n_points[cluster_id] != 0) {
				c->x = sum_x[cluster_id] / n_points[cluster_id];
				c->y = sum_y[cluster_id] / n_points[cluster_id];
				c->z = sum_z[cluster_id] / n_points[cluster_id];
			}
		}
	}
}
