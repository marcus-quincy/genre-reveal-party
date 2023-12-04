#pragma once

typedef struct Point {
    double x, y, z;     // coordinates
    int cluster;     // no default cluster
    double min_dist;  // default infinite distance to nearest cluster
} Point;

double point_distance(Point p0, Point p1);

void print_point(Point p);
