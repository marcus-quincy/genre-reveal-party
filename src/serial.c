#include <stdio.h>
#include <stdlib.h>

#include "csv.h"
#include "point.h"
#include "k_clustering.h"

int main() {
	Point* points = readcsv();

	if(points == NULL) return 1;

	k_means_clustering(points, LINE_COUNT - 1);

	writecsv(points);

	free(points);

	return 0;
}
