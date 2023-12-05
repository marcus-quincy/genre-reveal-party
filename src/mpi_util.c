#include <stddef.h>

#include "mpi_util.h"
#include "point.h"

MPI_Datatype create_point_datatype() {
	MPI_Datatype point_type;
	int count = 5;
	int blocklengths[] = { 1, 1, 1, 1, 1 };
	MPI_Aint displacements[] = {
		offsetof(Point, x),
		offsetof(Point, y),
		offsetof(Point, z),
		offsetof(Point, cluster),
		offsetof(Point, min_dist)
	};

	MPI_Datatype types[] = {
		MPI_DOUBLE,
		MPI_DOUBLE,
		MPI_DOUBLE,
		MPI_INT,
		MPI_DOUBLE
	};

	MPI_Type_create_struct(count, blocklengths, displacements, types, &point_type);
	MPI_Type_commit(&point_type);

	return point_type;
}
