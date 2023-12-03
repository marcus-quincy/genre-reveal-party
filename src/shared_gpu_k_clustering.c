#include <float.h>
#include <stdlib.h>
#include <stdio.h>

#include "shared_gpu_k_clustering.h"

extern void cuda_distances_kernel(Point* points_d, int points_size, Point* centroids_h, Point* centroids_d, int k);
extern void cuda_setup(Point* points_h, Point** points_d, int points_size, Point** centroids_d, int** n_points_d, double** sum_x_d, double** sum_y_d, double** sum_z_d, int k);
extern void cuda_cleanup(Point* points_h, Point* points_d, int points_size, Point* centroids_d, int* n_points_d, double* sum_x_d, double* sum_y_d, double* sum_z_d);
//extern void cuda_Kernel_2(Point* points_h, int points_size, int* n_points, double* sum_x, double* sum_y, double* sum_z, int k);

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
        cuda_distances_kernel(points_d, points_size, centroids_h, centroids_d, k);

        // Create vectors to keep track of data needed to compute means
        int n_points[k];
        double sum_x[k];
        double sum_y[k];
        double sum_z[k];

        // XXX: for now, just do this in serial. the above loop is much more computation heavy anyways!
        // XXX: this is going to fail, because we aren't copying points back to. This MUST also be
        //  done in GPU so we can keep points over there
        //Kernel2
        //cuda_Kernel_2(points_h, points_size, n_points, sum_x, sum_y, sum_z, k);
        for (int j = 0; j < points_size; j++) {
            Point* p = &points_h[j];
            n_points[p->cluster] += 1;
            sum_x[p->cluster] += p->x;
            sum_y[p->cluster] += p->y;
            sum_z[p->cluster] += p->z;

            // reset distance
            p->min_dist = DBL_MAX;
        }

        for (int cluster_id = 0; cluster_id < k; cluster_id++) {
            Point* c = &centroids_h[cluster_id];
            if (n_points[cluster_id] != 0) {
                c->x = sum_x[cluster_id] / n_points[cluster_id];
                c->y = sum_y[cluster_id] / n_points[cluster_id];
                c->z = sum_z[cluster_id] / n_points[cluster_id];
            }
        }
    }

    free(centroids_h);
    cuda_cleanup(points_h, points_d, points_size, centroids_d, n_points_d, sum_x_d, sum_y_d, sum_z_d);
}
