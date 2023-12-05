#pragma once

#include "point.h"

void k_means_clustering(Point* points, int points_size, int my_rank, int comm_sz);
