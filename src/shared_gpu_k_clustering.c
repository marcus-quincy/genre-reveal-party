#include <float.h>
#include <stdlib.h>
#include <stdio.h>

#include "shared_gpu_k_clustering.h"
#include "constants.h"

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
void share_gpu_k_means_clustering(Point* points_h, int points_size) {
    Point* points_d;
    Point* centroids_d;
    int* n_points_d;
    double* sum_x_d;
    double* sum_y_d;
    double* sum_z_d;
    cuda_setup(points_h, &points_d, points_size, &centroids_d, &n_points_d, &sum_x_d, &sum_y_d, &sum_z_d);
    Point centroids_h[K_CLUSTERS];

    srand(42);
    for (int i = 0; i < K_CLUSTERS; ++i) {
        centroids_h[i] = points_h[rand() % points_size];
    }

    for (int i = 0; i < N_EPOCHS; ++i) {
        // For each centroid, compute distance from centroid to each point
        // and update point's cluster if necessary
        cuda_distances_kernel(points_d, points_size, centroids_h, centroids_d);

        // Create vectors to keep track of data needed to compute means
        int n_points_h[K_CLUSTERS];
        double sum_x_h[K_CLUSTERS];
        double sum_y_h[K_CLUSTERS];
        double sum_z_h[K_CLUSTERS];

        cuda_sum_kernel(points_d, points_size, n_points_h, sum_x_h, sum_y_h, sum_z_h, n_points_d, sum_x_d, sum_y_d, sum_z_d);

        for (int cluster_id = 0; cluster_id < K_CLUSTERS; cluster_id++) {
            Point* c = &centroids_h[cluster_id];
            if (n_points_h[cluster_id] != 0) {
                c->x = sum_x_h[cluster_id] / n_points_h[cluster_id];
                c->y = sum_y_h[cluster_id] / n_points_h[cluster_id];
                c->z = sum_z_h[cluster_id] / n_points_h[cluster_id];
            }
        }
    }

    cuda_cleanup(points_h, points_d, points_size, centroids_d, n_points_d, sum_x_d, sum_y_d, sum_z_d);
}
