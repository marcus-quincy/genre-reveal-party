#include <stdio.h>
#include <stdlib.h>

#include "csv.h"
#include "point.h"
#include "k_clustering.h"
#include "validation.h"

void validate(Point* other_points, int points_size) {
    fprintf(stderr, "validating points against serial\n");
	Point* serial_points = readcsv();

	if(serial_points == NULL) exit(1);

	serial_k_means_clustering(serial_points, points_size);

    compare(serial_points, other_points, points_size);

    free(serial_points);
}

void compare(Point* serial_points, Point* other_points, int points_size) {
    for (int i = 0; i < points_size; i++) {
        Point s = serial_points[i];
        Point o = other_points[i];

        if (s.x != o.x || s.y != o.y || s.z != o.z || s.cluster != o.cluster) {
            fprintf(stderr, "Points at index %d differ:\n\texpected: ", i);
            print_point(s);
            fprintf(stderr, "\tactual: ");
            print_point(o);
            break;
        }
    }
}
