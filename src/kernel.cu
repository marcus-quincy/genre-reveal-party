#include <cstdlib>
#include <stdio.h>
#include <float.h>

#include "point.h"

//Testing for CUDA errors
inline cudaError_t checkCuda(cudaError_t result) {
if (result != cudaSuccess) {
fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
}
return result;
}

//Clustering Calculation
__global__ 
void my_Kernel_1(Point* points_d, int points_size, Point* c_d, int cluster_id) {
    
    //Get the index for the current point to work with
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    int index = Col + (Row * blockDim.x *gridDim.x); 
    
    if(index < points_size) {
        Point p = points_d[index];
        double dist = ((c_d->x - p.x) * (c_d->x - p.x) + (c_d->y - p.y) * (c_d->y - p.y) + (c_d->z - p.z) * (c_d->z - p.z));// point_distance(c, *p);
        if (dist < p.min_dist) {
            p.min_dist = dist;
            p.cluster = cluster_id;
        }
        
        points_d[index] = p;
    }
}

__global__
void my_Kernel_2(double max, Point* points_d, int points_size, int* n_points_d, double* sum_x_d, double* sum_y_d, double* sum_z_d) {
    //Get the index of the point to work with
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    int index = Col + (Row * blockDim.x * gridDim.x); 
    
    if(index < points_size) {
        Point p = points_d[index];
        n_points_d[p.cluster] += 1;
        sum_x_d[p.cluster] += p.x;
        sum_y_d[p.cluster] += p.y;
        sum_z_d[p.cluster] += p.z;

        // reset distance
        p.min_dist = max;
        points_d[index] = p;
    }
}

// Function that launches the CUDA kernel
extern "C" void cuda_Kernel_1(Point* points_h, int points_size, Point c, int cluster_id) {
    Point* points_d;
    Point* c_d;
    int new_points_size = sqrt(points_size); 


    //Allocate device pointers and copy them to the device
    checkCuda(cudaMalloc((void **) &points_d, sizeof(Point)*points_size));
    checkCuda(cudaMalloc((void **) &c_d, sizeof(Point)));
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(points_d, points_h, sizeof(Point)*points_size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(c_d, &c, sizeof(Point), cudaMemcpyHostToDevice));  
    checkCuda(cudaDeviceSynchronize());
    
    dim3 DimGrid(ceil(new_points_size/32.0), ceil(new_points_size/32.0), 1);
    dim3 DimBlock(32, 32, 1);

    //Launch kernel
    my_Kernel_1<<<DimGrid, DimBlock>>>(points_d, points_size, c_d, cluster_id);
    checkCuda(cudaDeviceSynchronize());

    //copy device points to host points
    checkCuda(cudaMemcpy(points_h, points_d, sizeof(Point)*points_size, cudaMemcpyDeviceToHost));
    checkCuda(cudaDeviceSynchronize());
    
    //Free device pointers
    checkCuda(cudaFree(points_d));
    checkCuda(cudaFree(c_d));
    checkCuda(cudaDeviceSynchronize());
}

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