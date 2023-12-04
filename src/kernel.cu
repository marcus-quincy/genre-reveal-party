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

// Do sum and reduce of the points x, y, and z values. This is done in 1 block, but could
// probably be sped up if a multi block reduction algorithm were implemented
__global__
void sum_reduce_kernel(Point* points_d, int points_size, int* n_points_d, double* sum_x_d, double* sum_y_d, double* sum_z_d, int k) {
    /// initialize variables
    int idx = threadIdx.x;

    /* FIXME
       __shared__ int n_points[blockDim.x][k];
       __shared__ double sum_x[blockDim.x][k];
       __shared__ double sum_y[blockDim.x][k];
       __shared__ double sum_z[blockDim.x][k];
    */

    __shared__ int n_points[256][5];
    __shared__ double sum_x[256][5];
    __shared__ double sum_y[256][5];
    __shared__ double sum_z[256][5];

    for (int cluster_id = 0; cluster_id < k; cluster_id++) {
        n_points[idx][cluster_id] = 0;
        sum_x[idx][cluster_id] = 0;
        sum_y[idx][cluster_id] = 0;
        sum_z[idx][cluster_id] = 0;
    }

    // XXX it hates something in this for loop
    //printf("dim %d  ",blockDim.x);
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
            for (int cluster_id = 0; cluster_id < k; cluster_id++) {
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
        for (int cluster_id = 0; cluster_id < k; ++cluster_id) {
            //printf("updating n  points to %d\n",n_points_d[cluster_id]);
            n_points_d[cluster_id] = n_points[0][cluster_id];
            sum_x_d[cluster_id] = sum_x[0][cluster_id];
            sum_y_d[cluster_id] = sum_y[0][cluster_id];
            sum_z_d[cluster_id] = sum_z[0][cluster_id];
        }
    }
}

extern "C" void cuda_setup(Point* points_h, Point** points_d, int points_size, Point** centroids_d, int** n_points_d, double** sum_x_d, double** sum_y_d, double** sum_z_d, int k) {
    //Allocate device pointers and copy them to the device
    checkCuda(cudaMalloc((void **) points_d, sizeof(Point)*points_size));
    checkCuda(cudaMalloc((void **) centroids_d, sizeof(Point)*k)); // just allocate the memory now, we will memcpy every kernel call
    checkCuda(cudaMalloc((void **) n_points_d, sizeof(int)*k));
    checkCuda(cudaMalloc((void **) sum_x_d, sizeof(double)*k));
    checkCuda(cudaMalloc((void **) sum_y_d, sizeof(double)*k));
    checkCuda(cudaMalloc((void **) sum_z_d, sizeof(double)*k));
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
extern "C" void cuda_distances_kernel(Point* points_d, int points_size, Point* centroids_h, Point* centroids_d, int k) {
    dim3 DimGrid(ceil(points_size/32.0));
    dim3 DimBlock(32);

    // copy the centroids over
    checkCuda(cudaMemcpy(centroids_d, centroids_h, sizeof(Point)*k, cudaMemcpyHostToDevice));
    checkCuda(cudaDeviceSynchronize());

    //Launch kernel
    distances_k<<<DimGrid, DimBlock>>>(points_d, points_size, centroids_d, k);
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
                                double* sum_z_d,
                                int k) {
    int block_size = 256; // only using 1 block so make this bigger

    // do the computation
    checkCuda(cudaDeviceSynchronize());
    // we know we get here
    //printf("b4");
//(points_d, points_size, n_points_d, sum_x_d, sum_y_d, sum_z_d, k)
    sum_reduce_kernel<<<1, block_size>>>(points_d, points_size, n_points_d, sum_x_d, sum_y_d, sum_z_d, k);
    //printf("after"); // and here
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(n_points_h, n_points_d, sizeof(int)*k, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(sum_x_h, sum_x_d, sizeof(double)*k, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(sum_y_h, sum_y_d, sizeof(double)*k, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(sum_z_h, sum_z_d, sizeof(double)*k, cudaMemcpyDeviceToHost));
    checkCuda(cudaDeviceSynchronize());
}
