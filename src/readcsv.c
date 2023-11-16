#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "readcsv.h"

Point* readcsv() {
	FILE* stream = fopen("input/tracks_features.csv", "r");

	if (stream == NULL) {
		printf("ERROR: could not open file\n");
		return NULL;
	}

	char line[MAX_LINE_LENGTH];

	int first_row_p = 1;
	Point* points = malloc((LINE_COUNT - 1) * sizeof(Point));
	int current_line = -1;

	while (fgets(line, MAX_LINE_LENGTH, stream)) {
		current_line++;
		char split_line[NUMBER_FIELDS][MAX_FIELD_LENGTH];
		memset(split_line, 0, sizeof(split_line));
		int field_number = 0;
		int field_index = 0;
		int in_double_quote = 0;

		for (int i = 0; i < MAX_FIELD_LENGTH; i++) {
			char c = line[i];
			if (c == ',' && !in_double_quote) {
				field_number++;
				field_index = 0;
				continue;
			}

			if (c == '"' && (i == 0 || line[i - 1] != '\\')) {
				in_double_quote = !in_double_quote;
				continue;
			}

			split_line[field_number][field_index++] = c;

			if (c == 0) {
				break;
			}
		}

		if (first_row_p) {
			/*
			printf("%s\n", split_line[9]);
			printf("%s\n", split_line[10]);
			printf("%s\n", split_line[14]);
			*/

			first_row_p = 0;
			continue;
		}

		double x = atof(split_line[9]); // dancability
		double y = atof(split_line[10]); // energy
		double z = atof(split_line[14]); // speechiness

		Point point = { x, y, z, -1, DBL_MAX };
		points[current_line - 1] = point;
	}

	fclose(stream);

	return points;
}

// return 1 upon success
int writecsv(Point* points) {
	FILE* stream = fopen("output/output.csv", "w");

	if (stream == NULL) {
		printf("ERROR: could not open file\n");
		return 0;
	}

	fprintf(stream, "x,y,z,c\n");
	for (int i = 0; i < LINE_COUNT - 1; i++) {
		Point point = points[i];
		fprintf(stream, "%lf,%lf,%lf,%d\n", point.x, point.y, point.z, point.cluster);
	}

	fclose(stream);

	return 1;
}
