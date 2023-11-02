//Followed the following tutorial for this program with some modifications
//https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/

#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <float.h>

using namespace std;

struct Point {
    double x, y, z;     // coordinates
    int cluster;     // no default cluster
    double minDist;  // default infinite distance to nearest cluster

    Point() : x(0.0), y(0.0), z(0.0), cluster(-1), minDist(DBL_MAX) {}
    Point(double x, double y, double z) : x(x), y(y), z(z), cluster(-1), minDist(DBL_MAX) {}

    // Computes the (square) euclidean distance between this point and another
    double distance(Point p) {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y) + (p.z - z) * (p.z - z) ;
    }
};

//Reads in the data.csv file into a vector of Points
vector<Point> readcsv() {
    vector<Point> points;
    string line;
    ifstream file("track_features.csv");
    bool firstRow = true;

    while (getline(file, line)) {
        stringstream lineStream(line);
        string bit;
        double x, y, z;

        //skip the first row
        if(!firstRow){

          //skip over input columns that aren't needed
          for(int i = 0; i < 9; i++){
            getline(lineStream, bit, ',');
          }

          //get dancability
          getline(lineStream, bit, ',');
          x = stof(bit);

          //get energy
          getline(lineStream, bit, ',');
          y = stof(bit);

          //skip unneeded coulmns
          for(int i = 0; i < 3; i++){
            getline(lineStream, bit, ',');
          }

          //get speechiness
          getline(lineStream, bit, ',');
          z = stof(bit);

          points.push_back(Point(x, y, z));
        }
        firstRow = false;
    }
    return points;
}

//Perform k-means clustering
void kMeansClustering(vector<Point>* points, int epochs, int k) {
    int n = points->size();

    // Randomly initialise centroids
    // The index of the centroid within the centroids vector
    // represents the cluster label.
    vector<Point> centroids;
    srand(time(0));
    for (int i = 0; i < k; ++i) {
        centroids.push_back(points->at(rand() % n));
    }

    for (int i = 0; i < epochs; ++i) {
        // For each centroid, compute distance from centroid to each point
        // and update point's cluster if necessary
        for (vector<Point>::iterator c = begin(centroids); c != end(centroids);
             ++c) {
            int clusterId = c - begin(centroids);

            for (vector<Point>::iterator it = points->begin();
                 it != points->end(); ++it) {
                Point p = *it;
                double dist = c->distance(p);
                if (dist < p.minDist) {
                    p.minDist = dist;
                    p.cluster = clusterId;
                }
                *it = p;
            }
        }

        // Create vectors to keep track of data needed to compute means
        vector<int> nPoints;
        vector<double> sumX, sumY, sumZ;
        for (int j = 0; j < k; ++j) {
            nPoints.push_back(0);
            sumX.push_back(0.0);
            sumY.push_back(0.0);
            sumZ.push_back(0.0);
        }

        // Iterate over points to append data to centroids
        for (vector<Point>::iterator it = points->begin(); it != points->end();
             ++it) {
            int clusterId = it->cluster;
            nPoints[clusterId] += 1;
            sumX[clusterId] += it->x;
            sumY[clusterId] += it->y;
            sumZ[clusterId] += it->z;

            // reset distance
            it->minDist = DBL_MAX;
        }

        // Compute the new centroids
        for (vector<Point>::iterator c = begin(centroids); c != end(centroids);
             ++c) {
            int clusterId = c - begin(centroids);
            c->x = sumX[clusterId] / nPoints[clusterId];
            c->y = sumY[clusterId] / nPoints[clusterId];
            c->z = sumZ[clusterId] / nPoints[clusterId];
        }
    }

    // Write to csv
    ofstream myfile;
    myfile.open("output.csv");
    myfile << "x,y,z,c" << endl;

    for (vector<Point>::iterator it = points->begin(); it != points->end();
         ++it) {
        myfile << it->x << "," << it->y << "," << it->z << "," << it->cluster << endl;
    }
    myfile.close();
}

int main() {
    vector<Point> points = readcsv();

    // Run k-means with 100 iterations and for 5 clusters
    kMeansClustering(&points, 100, 5);
}