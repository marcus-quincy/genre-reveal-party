#include <stdio.h>
#include <stdlib.h>

#include "csv.h"
#include "point.h"
#include "shared_gpu_k_clustering.h"
#include "constants.h"
#include "validation.h"

int main() {
	Point* points = readcsv();

	if (points == NULL) return 1;

	share_gpu_k_means_clustering(points, LINE_COUNT - 1);
    
#ifdef RUN_VALIDATION
    validate(points, LINE_COUNT - 1);
#endif

    writecsv(points);

	free(points);

	return 0;
}
