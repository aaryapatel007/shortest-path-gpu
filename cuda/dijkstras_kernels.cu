#include "dijkstras_kernels.cuh"

__global__ void initializeInputArray(bool *d_mask, int *dist_i, int *dist, int val, int N){
    // initialize thread index
    int t_index = threadIdx.x + blockDim.x * blockIdx.x;

    // check if thread index is within bounds
    if (t_index < N) {
        dist[t_index] = val;
        dist_i[t_index] = val;
        d_mask[t_index] = false;
        if(t_index == 0) {
            dist[t_index] = 0;
            dist_i[t_index] = 0;
            d_mask[t_index] = true;
        }
    }
}

__global__ void updateDistanceArray(bool *d_mask, int *d_distances, int *d_distances_i, int N) {
    // initialize thread index
    unsigned int t_index = threadIdx.x + blockDim.x * blockIdx.x;

    // check if thread index is within bounds
    if (t_index < N) {
        // update the distances array if distances_i is less than distances
        if(d_distances_i[t_index] < d_distances[t_index]){
            d_distances[t_index] = d_distances_i[t_index];
            d_mask[t_index] = true;
        }

        d_distances_i[t_index] = d_distances[t_index];
    }
}

__global__ void updateDistanceArrayGridStride(bool *d_mask, int *d_distances, int *d_distances_i, int N) {
    // initialize thread index
    unsigned int t_index = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // check if thread index is within bounds
    for(int index = t_index; index < N; index += stride) {
        // update the distances array if distances_i is less than distances
        if(d_distances_i[index] < d_distances[index]){
            d_distances[index] = d_distances_i[index];
            d_mask[t_index] = true;
        }

        d_distances_i[index] = d_distances[index];
    }
}

__global__ void dijkstrasGPUKernel(int *d_adj_list_offsets, int *d_adj_list, int *d_weights, bool *d_mask, int *d_distances, int *d_distances_i, int N) {
    // initialize thread index
    unsigned int t_index = threadIdx.x + (blockDim.x * blockIdx.x);

    // check if thread index is within bounds
    if (t_index < N) {
        if(d_mask[t_index] == true){
            d_mask[t_index] = false;

            // iterate through all the edges
            for (int j = d_adj_list_offsets[t_index]; j < d_adj_list_offsets[t_index + 1]; j++) {
                int w = d_weights[j];
                int du = d_distances[t_index];
                int updated_dist = du + w;

                // update the distances_i array if updated_dist is less than distances_i
                atomicMin(&d_distances_i[d_adj_list[j]], updated_dist);
            }
        }
    }
}

__global__ void dijkstrasGPUGridStrideKernel(int *d_adj_list_offsets, int *d_adj_list, int *d_weights, bool *d_mask, int *d_distances, int *d_distances_i, int N) {
    // initialize thread index
    unsigned int t_index = threadIdx.x + (blockDim.x * blockIdx.x);
    unsigned int stride = blockDim.x * gridDim.x;

    // check if thread index is within bounds
    for(int index = t_index; index < N; index += stride) {

        if(d_mask[t_index] == true){
            d_mask[t_index] = false;

            // iterate through all the edges
            for (int j = d_adj_list_offsets[index]; j < d_adj_list_offsets[index + 1]; j++) {
                int w = d_weights[j];
                int du = d_distances[index];
                int updated_dist = du + w;

                // update the distance if the distance from source to i + the weight of the edge between i and j is less than the distance from source to j
                atomicMin(&d_distances_i[d_adj_list[j]], updated_dist);
            }
        }
    }
}
