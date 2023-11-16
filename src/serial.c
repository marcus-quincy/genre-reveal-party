#include <stdio.h>
#include <stdlib.h>

#include "readcsv.h"
#include "utility.h"

int main() {
	Point* points = readcsv();

	free(points);

	return 0;
}
