#pragma once

#include <mpi.h>
#include "point.h"

void k_means_clustering(Point* points, int points_size, int epochs, int k);
MPI_Datatype create_point_datatype();
