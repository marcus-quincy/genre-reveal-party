#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "k_clustering.h"
#include "constants.h"

// perform the c clustering
void serial_k_means_clustering(Point* points, int points_size) {
	Point centroids[K_CLUSTERS];
	//srand(time(0));
	srand(42);
	for (int i = 0; i < K_CLUSTERS; ++i) {
		centroids[i] = points[rand() % points_size];
	}

	for (int i = 0; i < N_EPOCHS; ++i) {
		// For each centroid, compute distance from centroid to each point
		// and update point's cluster if necessary
		for (int cluster_id = 0; cluster_id < K_CLUSTERS; ++cluster_id){
			Point c = centroids[cluster_id];

			for (int j = 0; j < points_size; ++j) {
				Point* p = &points[j];
				double dist = point_distance(c, *p);
				if (dist < p->min_dist) {
					p->min_dist = dist;
					p->cluster = cluster_id;
				}
			}
		}

		// Create vectors to keep track of data needed to compute means
		int n_points[K_CLUSTERS];
		double sum_x[K_CLUSTERS];
		double sum_y[K_CLUSTERS];
		double sum_z[K_CLUSTERS];
		for (int j = 0; j < K_CLUSTERS; ++j) {
			n_points[j] = 0;
			sum_x[j] = 0.0;
			sum_y[j] = 0.0;
			sum_z[j] = 0.0;
		}

		// Iterate over points to append data to centroids
		for (int j = 0; j < points_size; j++) {
			Point* p = &points[j];
			n_points[p->cluster] += 1;
			sum_x[p->cluster] += p->x;
			sum_y[p->cluster] += p->y;
			sum_z[p->cluster] += p->z;

			// reset distance
			p->min_dist = DBL_MAX;
		}

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
