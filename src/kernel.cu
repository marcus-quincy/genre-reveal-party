// -*- mode: c -*-

#include <cstdlib>
#include <stdio.h>
#include <float.h>

#include "point.h"

//Testing for CUDA errors
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//Clustering Calculation
__global__ 
void distances_k(Point* points_d, int points_size, Point* centroids_d, int k) {
    //Get the index for the current point to work with
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    /*
      Potential improvement would be to not interate cluster_id -> k and instead have a separate thread
      per cluster_id. Not sure if this would yield any performance gains though. It also creates race
      conditions that have to be dealt with.
     */

    if(index < points_size) {
        //printf("idx %d\n", index);
        Point* p = &points_d[index];
        for (int cluster_id = 0; cluster_id < k; cluster_id++) {
            Point c = centroids_d[cluster_id];
            double dist = ((c.x - p->x) * (c.x - p->x) + (c.y - p->y) * (c.y - p->y) + (c.z - p->z) * (c.z - p->z));// point_distance(c, *p);
            if (dist < p->min_dist) {
                p->min_dist = dist;
                p->cluster = cluster_id;
            }
        }
    }
}

extern "C" void cuda_setup(Point* points_h, Point** points_d, int points_size, Point** centroids_d, int k) {
    //Allocate device pointers and copy them to the device
    checkCuda(cudaMalloc((void **) points_d, sizeof(Point)*points_size));
    checkCuda(cudaDeviceSynchronize());
    printf("allocating centroids_d\n");
    checkCuda(cudaMalloc((void **) centroids_d, sizeof(Point)*k)); // just allocate the memory now, we will memcpy every kernel call
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(*points_d, points_h, sizeof(Point)*points_size, cudaMemcpyHostToDevice));
    checkCuda(cudaDeviceSynchronize());
}

extern "C" void cuda_cleanup(Point* points_h, Point* points_d, int points_size, Point* centroids_d) {
    //copy device points to host points
    checkCuda(cudaMemcpy(points_h, points_d, sizeof(Point)*points_size, cudaMemcpyDeviceToHost));
    checkCuda(cudaDeviceSynchronize());
    
    //Free device pointers
    checkCuda(cudaFree(points_d));
    checkCuda(cudaFree(centroids_d));
    checkCuda(cudaDeviceSynchronize());
}

// Function that launches the CUDA kernel
extern "C" void cuda_distances_kernel(Point* points_d, int points_size, Point* centroids_h, Point* centroids_d, int k) {
    // XXX: why is it 2d?
    dim3 DimGrid(ceil(points_size/32.0));
    dim3 DimBlock(32);

    // copy the centroids over
    //checkCuda(cudaMalloc((void **) &centroids_d, sizeof(Point)*k)); // just allocate the memory now, we will memcpy every kernel call
    checkCuda(cudaMemcpy(centroids_d, centroids_h, sizeof(Point)*k, cudaMemcpyHostToDevice));
    //checkCuda(cudaMemcpy(points_d, points_h, sizeof(Point)*points_size, cudaMemcpyHostToDevice));
    checkCuda(cudaDeviceSynchronize());

    //Launch kernel
    distances_k<<<DimGrid, DimBlock>>>(points_d, points_size, centroids_d, k);
    //checkCuda(cudaFree(centroids_d));
    checkCuda(cudaDeviceSynchronize());
}

/*
// Function that launches the CUDA kernel
extern "C" void cuda_Kernel_2(Point* points_h, int points_size, int* n_points, double* sum_x, double* sum_y, double* sum_z, int k) {
    Point* points_d;
    int* n_points_d;
    double* sum_x_d;
    double* sum_y_d;
    double* sum_z_d;
    int new_points_size = sqrt(points_size); 

    //Allocate device pointers and copy them to the device
    checkCuda(cudaMalloc((void **) &points_d, sizeof(Point)*points_size));
    checkCuda(cudaMalloc((void **) &n_points_d, sizeof(int)*k));
    checkCuda(cudaMalloc((void **) &sum_x_d, sizeof(double)*k));
    checkCuda(cudaMalloc((void **) &sum_y_d, sizeof(double)*k));
    checkCuda(cudaMalloc((void **) &sum_z_d, sizeof(double)*k));
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(points_d, points_h, sizeof(Point)*points_size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(n_points_d, n_points, sizeof(int)*k, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(sum_x_d, sum_x, sizeof(double)*k, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(sum_y_d, sum_y, sizeof(double)*k, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(sum_z_d, sum_z, sizeof(double)*k, cudaMemcpyHostToDevice));
    checkCuda(cudaDeviceSynchronize());
    
    dim3 DimGrid(ceil(new_points_size/32.0), ceil(new_points_size/32.0), 1);
    dim3 DimBlock(32, 32, 1);

    //Launch the kernel
    my_Kernel_2<<<DimGrid, DimBlock>>>(DBL_MAX, points_d, points_size, n_points_d, sum_x_d, sum_y_d, sum_z_d);
    checkCuda(cudaDeviceSynchronize());
    
    //copy points from device to host
    checkCuda(cudaMemcpy(points_h, points_d, sizeof(Point)*points_size, cudaMemcpyDeviceToHost));  
    checkCuda(cudaDeviceSynchronize());
    
    //Free Allocated pointers
    checkCuda(cudaFree(points_d));
    checkCuda(cudaFree(n_points_d));
    checkCuda(cudaFree(sum_x_d));
    checkCuda(cudaFree(sum_y_d));
    checkCuda(cudaFree(sum_z_d));
    checkCuda(cudaDeviceSynchronize());
}

*/
