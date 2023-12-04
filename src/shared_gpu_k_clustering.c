#include <float.h>
#include <stdlib.h>
#include <stdio.h>

#include "shared_gpu_k_clustering.h"

extern void cuda_distances_kernel(Point* points_d, int points_size, Point* centroids_h, Point* centroids_d, int k);
extern void cuda_setup(Point* points_h, Point** points_d, int points_size, Point** centroids_d, int** n_points_d, double** sum_x_d, double** sum_y_d, double** sum_z_d, int k);
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
                            double* sum_z_d,
                            int k);

// perform the c clustering
void k_means_clustering(Point* points_h, int points_size, int epochs, int k) {
    Point* points_d;
    Point* centroids_d;
    int* n_points_d;
    double* sum_x_d;
    double* sum_y_d;
    double* sum_z_d;
    cuda_setup(points_h, &points_d, points_size, &centroids_d, &n_points_d, &sum_x_d, &sum_y_d, &sum_z_d, k);
    Point* centroids_h = malloc(sizeof(Point) * k);

    srand(42);
    for (int i = 0; i < k; ++i) {
        centroids_h[i] = points_h[rand() % points_size];
    }

    for (int i = 0; i < epochs; ++i) {
        // For each centroid, compute distance from centroid to each point
        // and update point's cluster if necessary
        //fprintf(stderr,"epoch %d\n", i);
        for (int ii = 0; ii < k; ii++) {
		//print_point(centroids_h[ii]);
        }

        cuda_distances_kernel(points_d, points_size, centroids_h, centroids_d, k);

        // Create vectors to keep track of data needed to compute means
        int n_points_h[k];
        double sum_x_h[k];
        double sum_y_h[k];
        double sum_z_h[k];

        cuda_sum_kernel(points_d, points_size, n_points_h, sum_x_h, sum_y_h, sum_z_h, n_points_d, sum_x_d, sum_y_d, sum_z_d, k);

        for (int cluster_id = 0; cluster_id < k; cluster_id++) {
            Point* c = &centroids_h[cluster_id];
            //printf("n_points_h %d\n",n_points_h[cluster_id]);
            if (n_points_h[cluster_id] != 0) {
                c->x = sum_x_h[cluster_id] / n_points_h[cluster_id];
                c->y = sum_y_h[cluster_id] / n_points_h[cluster_id];
                c->z = sum_z_h[cluster_id] / n_points_h[cluster_id];
            }
        }
    }

    free(centroids_h);
    cuda_cleanup(points_h, points_d, points_size, centroids_d, n_points_d, sum_x_d, sum_y_d, sum_z_d);
}
