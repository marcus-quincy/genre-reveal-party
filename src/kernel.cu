// -*- mode: c -*-

#include <stdio.h>
#include <float.h>

#include "point.h"
#include "cuda_util.h"
#include "constants.h"

// block sizes
#define DISTANCE_BLOCK_SIZE 32
#define SUM_BLOCK_SIZE 256 // can't make this too big, because of limited __shared__ memory

//Clustering Calculation
__global__
void distances_k(Point* points_d, int points_size, Point* centroids_d) {
    //Get the index for the current point to work with
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index < points_size) {
        Point* p = &points_d[index];
        for (int cluster_id = 0; cluster_id < K_CLUSTERS; cluster_id++) {
            Point c = centroids_d[cluster_id];
            // can't call non-kernel function from kernel function, so we do this here.
            double dist = ((c.x - p->x) * (c.x - p->x) + (c.y - p->y) * (c.y - p->y) + (c.z - p->z) * (c.z - p->z));// point_distance(c, *p);
            if (dist < p->min_dist) {
                p->min_dist = dist;
                p->cluster = cluster_id;
            }
        }
    }
}

// Do sum and reduce of the points x, y, and z values in one block
__global__
void sum_reduce_kernel(Point* points_d, int points_size, int* n_points_d, double* sum_x_d, double* sum_y_d, double* sum_z_d) {
    /// initialize variables
    int idx = threadIdx.x;

    __shared__ int n_points[SUM_BLOCK_SIZE][K_CLUSTERS];
    __shared__ double sum_x[SUM_BLOCK_SIZE][K_CLUSTERS];
    __shared__ double sum_y[SUM_BLOCK_SIZE][K_CLUSTERS];
    __shared__ double sum_z[SUM_BLOCK_SIZE][K_CLUSTERS];

    for (int cluster_id = 0; cluster_id < K_CLUSTERS; cluster_id++) {
        n_points[idx][cluster_id] = 0;
        sum_x[idx][cluster_id] = 0;
        sum_y[idx][cluster_id] = 0;
        sum_z[idx][cluster_id] = 0;
    }

    // compute the local sum
    for (int i = idx; i < points_size; i += blockDim.x) {
        Point* p = &points_d[i];
        n_points[idx][p->cluster] += 1;
        sum_x[idx][p->cluster] += p->x;
        sum_y[idx][p->cluster] += p->y;
        sum_z[idx][p->cluster] += p->z;

        p->min_dist = DBL_MAX;
    }

    __syncthreads();

    // do the reduction
    for (int size = blockDim.x / 2; size > 0; size /= 2) {
        if (idx < size) {
            for (int cluster_id = 0; cluster_id < K_CLUSTERS; cluster_id++) {
                n_points[idx][cluster_id] += n_points[idx + size][cluster_id];
                sum_x[idx][cluster_id] += sum_x[idx + size][cluster_id];
                sum_y[idx][cluster_id] += sum_y[idx + size][cluster_id];
                sum_z[idx][cluster_id] += sum_z[idx + size][cluster_id];
            }
        }
        __syncthreads();
    }

    // put into output buffer
    if (idx == 0) {
        for (int cluster_id = 0; cluster_id < K_CLUSTERS; ++cluster_id) {
            n_points_d[cluster_id] = n_points[0][cluster_id];
            sum_x_d[cluster_id] = sum_x[0][cluster_id];
            sum_y_d[cluster_id] = sum_y[0][cluster_id];
            sum_z_d[cluster_id] = sum_z[0][cluster_id];
        }
    }
}

extern "C" void cuda_setup(Point* points_h, Point** points_d, int points_size, Point** centroids_d, int** n_points_d, double** sum_x_d, double** sum_y_d, double** sum_z_d) {
    //Allocate device pointers and copy them to the device
    checkCuda(cudaMalloc((void **) points_d, sizeof(Point)*points_size));
    checkCuda(cudaMalloc((void **) centroids_d, sizeof(Point)*K_CLUSTERS)); // just allocate the memory now, we will memcpy every kernel call
    checkCuda(cudaMalloc((void **) n_points_d, sizeof(int)*K_CLUSTERS));
    checkCuda(cudaMalloc((void **) sum_x_d, sizeof(double)*K_CLUSTERS));
    checkCuda(cudaMalloc((void **) sum_y_d, sizeof(double)*K_CLUSTERS));
    checkCuda(cudaMalloc((void **) sum_z_d, sizeof(double)*K_CLUSTERS));
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(*points_d, points_h, sizeof(Point)*points_size, cudaMemcpyHostToDevice));
    checkCuda(cudaDeviceSynchronize());
}

extern "C" void cuda_cleanup(Point* points_h, Point* points_d, int points_size, Point* centroids_d, int* n_points_d, double* sum_x_d, double* sum_y_d, double* sum_z_d) {
    //copy device points to host points
    checkCuda(cudaMemcpy(points_h, points_d, sizeof(Point)*points_size, cudaMemcpyDeviceToHost));
    checkCuda(cudaDeviceSynchronize());

    //Free device pointers
    checkCuda(cudaFree(points_d));
    checkCuda(cudaFree(centroids_d));
    checkCuda(cudaFree(n_points_d));
    checkCuda(cudaFree(sum_x_d));
    checkCuda(cudaFree(sum_y_d));
    checkCuda(cudaFree(sum_z_d));
    checkCuda(cudaDeviceSynchronize());
}

// Function that launches the CUDA kernel
extern "C" void cuda_distances_kernel(Point* points_d, int points_size, Point* centroids_h, Point* centroids_d) {
    dim3 DimGrid(ceil(points_size/((float)DISTANCE_BLOCK_SIZE)));
    dim3 DimBlock(DISTANCE_BLOCK_SIZE);

    // copy the centroids over
    checkCuda(cudaMemcpy(centroids_d, centroids_h, sizeof(Point)*K_CLUSTERS, cudaMemcpyHostToDevice));
    checkCuda(cudaDeviceSynchronize());

    //Launch kernel
    distances_k<<<DimGrid, DimBlock>>>(points_d, points_size, centroids_d);
    checkCuda(cudaDeviceSynchronize());
}

extern "C" void cuda_sum_kernel(Point* points_d,
                                int points_size,
                                int* n_points_h,
                                double* sum_x_h,
                                double* sum_y_h,
                                double* sum_z_h,
                                int* n_points_d,
                                double* sum_x_d,
                                double* sum_y_d,
                                double* sum_z_d) {
    // do the computation
    sum_reduce_kernel<<<1, SUM_BLOCK_SIZE>>>(points_d, points_size, n_points_d, sum_x_d, sum_y_d, sum_z_d);
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(n_points_h, n_points_d, sizeof(int)*K_CLUSTERS, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(sum_x_h, sum_x_d, sizeof(double)*K_CLUSTERS, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(sum_y_h, sum_y_d, sizeof(double)*K_CLUSTERS, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(sum_z_h, sum_z_d, sizeof(double)*K_CLUSTERS, cudaMemcpyDeviceToHost));
    checkCuda(cudaDeviceSynchronize());
}
