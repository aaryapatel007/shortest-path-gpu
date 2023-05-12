#include "bellman.cuh"

__global__ void initializeInputArray(int *array, int val, int N){
    // initialize thread index
    int t_index = threadIdx.x + blockDim.x * blockIdx.x;

    // check if thread index is within bounds
    if (t_index < N) {
        array[t_index] = val;
        if(t_index == 0) {
            array[t_index] = 0;
        }
    }
}

__global__ void bellmanFordGPUKernel(int *d_adj_list_offsets, int *d_adj_list, int *d_weights, int *d_distances, int *d_distances_i, int N) {
    // initialize thread index
    unsigned int t_index = threadIdx.x + (blockDim.x * blockIdx.x);

    // check if thread index is within bounds
    if (t_index < N) {
        // iterate through all the edges
        for (int j = d_adj_list_offsets[t_index]; j < d_adj_list_offsets[t_index + 1]; j++) {
            int w = d_weights[j];
            int du = d_distances[t_index];
            int updated_dist = du + w;

            // update the distance if the distance from source to i + the weight of the edge between i and j is less than the distance from source to j
            if (du != INT_MAX){
                atomicMin(&d_distances_i[d_adj_list[j]], updated_dist);
            }
        }
    }
}

__global__ void updateDistanceArray(int *d_distances, int *d_distances_i, int N) {
    // initialize thread index
    unsigned int t_index = threadIdx.x + blockDim.x * blockIdx.x;

    // check if thread index is within bounds
    if (t_index < N) {
        // update the distances array if distances_i is less than distances
        if(d_distances_i[t_index] < d_distances[t_index]){
            d_distances[t_index] = d_distances_i[t_index];
        }

        d_distances_i[t_index] = d_distances[t_index];
    }
}

__global__ void bellmanFordGPUGridStrideKernel(int *d_adj_list_offsets, int *d_adj_list, int *d_weights, int *d_distances, int *d_distances_i, int N) {
    // initialize thread index
    unsigned int t_index = threadIdx.x + (blockDim.x * blockIdx.x);
    unsigned int stride = blockDim.x * gridDim.x;

    // check if thread index is within bounds
    for(int index = t_index; index < N; index += stride) {
        // iterate through all the edges
        for (int j = d_adj_list_offsets[index]; j < d_adj_list_offsets[index + 1]; j++) {
            int w = d_weights[j];
            int du = d_distances[index];
            int updated_dist = du + w;

            // update the distance if the distance from source to i + the weight of the edge between i and j is less than the distance from source to j
            if (du != INT_MAX){
                atomicMin(&d_distances_i[d_adj_list[j]], updated_dist);
            }
        }
    }
}

__global__ void updateDistanceArrayGridStride(int *d_distances, int *d_distances_i, int N) {
    // initialize thread index
    unsigned int t_index = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // check if thread index is within bounds
    for(int index = t_index; index < N; index += stride) {
        // update the distances array if distances_i is less than distances
        if(d_distances_i[index] < d_distances[index]){
            d_distances[index] = d_distances_i[index];
        }

        d_distances_i[index] = d_distances[index];
    }
}
