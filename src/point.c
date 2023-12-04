#include <stdio.h>

#include "point.h"

// Computes the (square) euclidean distance between this point and another
double point_distance(Point p0, Point p1) {
    return (p0.x - p1.x) * (p0.x - p1.x) + (p0.y - p1.y) * (p0.y - p1.y) + (p0.z - p1.z) * (p0.z - p1.z) ;
}

void print_point(Point p) {
	fprintf(stderr, "x: %lf y: %lf z: %lf c: %d\n", p.x, p.y, p.z, p.cluster);
}
