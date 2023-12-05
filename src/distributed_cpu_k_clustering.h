#pragma once

#include <mpi.h>
#include "point.h"

void dist_cpu_k_means_clustering(Point* points,
			int points_size,
			int my_rank,
			int comm_sz);
