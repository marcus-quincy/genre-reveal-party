#pragma once

#include "point.h"

// constants
#define MAX_LINE_LENGTH 4096  // highest line length we'll handle
#define NUMBER_FIELDS 24      // number of fields in file
#define LINE_COUNT 1204026    // number of lines in file
#define MAX_FIELD_LENGTH 1024 // maximum size of a field in the file


Point* readcsv();
int writecsv(Point* points);
