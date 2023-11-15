#pragma once

typedef struct Point {
    double x, y, z;     // coordinates
    int cluster;     // no default cluster
    double minDist;  // default infinite distance to nearest cluster
} Point;

double point_distance(Point p0, Point p1);
