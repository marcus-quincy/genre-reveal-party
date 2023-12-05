#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "csv.h"
#include "point.h"
#include "shared_gpu_k_clustering.h"
#include "constants.h"
#include "validation.h"

int main() {
	Point* points = readcsv();

	if (points == NULL) return 1;

	//Run k-means with 100 iterations and for 5 clusters
    struct timeval start, end;

    // Record the start time
    gettimeofday(&start, NULL);

	share_gpu_k_means_clustering(points, LINE_COUNT - 1);
    // Record the end time
    gettimeofday(&end, NULL);

    // Calculate the elapsed time in seconds
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    // Print the elapsed time
    printf("Elapsed time: %f seconds\n", elapsed_time);

#ifdef RUN_VALIDATION
    validate(points, LINE_COUNT - 1);
#endif

    writecsv(points);

	free(points);

	return 0;
}
