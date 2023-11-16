#include "point.h"

// Computes the (square) euclidean distance between this point and another
double point_distance(Point p0, Point p1) {
    return (p0.x - p1.x) * (p0.x - p1.x) + (p0.y - p1.y) * (p0.y - p1.y) + (p0.z - p1.z) * (p0.z - p1.z) ;
}
