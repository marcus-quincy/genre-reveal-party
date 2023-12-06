#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "csv.h"
#include "point.h"
#include "k_clustering.h"
#include "validation.h"
#include "constants.h"

int main() {
	Point* points = readcsv();

	if(points == NULL) return 1;

#ifdef OUTPUT_TIME
    struct timeval start, end;
    gettimeofday(&start, NULL);
#endif

    serial_k_means_clustering(points, LINE_COUNT - 1);

#ifdef OUTPUT_TIME
    gettimeofday(&end, NULL);
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("Elapsed time: %f seconds\n", elapsed_time);
#endif

#ifdef RUN_VALIDATION
    validate(points, LINE_COUNT - 1);
#endif

    writecsv(points);

	free(points);

	return 0;
}
