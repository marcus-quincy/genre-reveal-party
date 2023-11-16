#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#include "serial.h"
#include "readcsv.h"
#include "utility.h"

int main() {
	Point* points = readcsv();

	// Run k-means with 100 iterations and for 5 clusters
	k_means_clustering(points, LINE_COUNT - 1, 100, 5);

	writecsv(points);

	free(points);

	return 0;
}

void k_means_clustering(Point* points, int points_size, int epochs, int k) {
	Point centroids[k];
	srand(time(0));
	for (int i = 0; i < k; ++i) {
		centroids[i] = points[rand() % points_size];
	}

	for (int i = 0; i < epochs; ++i) {
		// For each centroid, compute distance from centroid to each point
		// and update point's cluster if necessary
		for (int cluster_id = 0; cluster_id < k; ++cluster_id){
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
