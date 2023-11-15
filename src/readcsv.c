#include <stdio.h>
#include <stdlib.h>

#include "readcsv.h"

#define MAX_LINE_LENGTH 4096

Point* readcsv() {
	FILE* stream = fopen("input/tracks_features.csv", "r");

	if (stream == NULL) {
		printf("ERROR: could not open file\n");
		return NULL;
	}

	char line[MAX_LINE_LENGTH];

	// get the number of lines in the file
	unsigned long line_count = 0;
	while (fgets(line, MAX_LINE_LENGTH, stream))
		line_count++;

	int firstRowP = 1;
	Point* points = malloc(line_count * sizeof(Point));

	while (fgets(line, MAX_LINE_LENGTH, stream)) {

	}

	return points;
}
