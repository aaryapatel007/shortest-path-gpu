#include "dijkstras.cuh"
#include "../main.h"

// check if all vertices have been visited
bool checkallvisited(bool *mask, int N){
    for(int i = 0;i < N;++i){
        if(mask[i] == true){
            return false;
        }
    }
    return true;
}

int runDijkstrasOnGPU(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, int block_size, string output_file, float &msecs_gpu) {

    int N = numVertices + 1;

    int *d_adj_list_offsets;
    int *d_adj_list;
    int *d_weights;
    int *d_distances; 
    int *d_distances_i;
    bool *d_mask;

    bool *h_mask;

    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);

    //allocate memory
    cudaMalloc((void**) &d_adj_list_offsets, N *sizeof(int));
    cudaMalloc((void**) &d_adj_list, adj_list.size() *sizeof(int));
    cudaMalloc((void**) &d_weights, weights.size() *sizeof(int));
    cudaMalloc((void**) &d_distances, numVertices *sizeof(int));
    cudaMalloc((void**) &d_distances_i, numVertices *sizeof(int));
    cudaMalloc((void**) &d_mask, numVertices *sizeof(bool));

    // allocate memory for mask on host
    h_mask = (bool *)malloc(numVertices * sizeof(bool));

    //copy to device memory
    cudaMemcpy(d_adj_list_offsets, adj_offset_list.data(), N *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj_list, adj_list.data(), adj_list.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), weights.size() *sizeof(int), cudaMemcpyHostToDevice);

    // compute time taken to run Dijkstra's algorithm on GPU
    fprintf(stderr, "Computing shortest path using Dijkstra's algorithm on GPU ");

    dim3 dimBlock(block_size);
	dim3 dimGrid(ceil(numVertices / float(block_size)));

    // initialize the distance array
    initializeInputArray<<<dimGrid, dimBlock>>>(d_mask, d_distances_i, d_distances, INT_MAX, numVertices);

    cudaEventRecord(start_gpu, 0);

    // copy the mask array from device to host
    cudaMemcpy(h_mask, d_mask, numVertices * sizeof(bool), cudaMemcpyDeviceToHost);

    // Dijkstra's algorithm
    while(checkallvisited(h_mask, numVertices) == false){
        dijkstrasGPUKernel<<<dimGrid, dimBlock>>>(d_adj_list_offsets, d_adj_list, d_weights, d_mask, d_distances, d_distances_i, numVertices);
        updateDistanceArray<<<dimGrid, dimBlock>>>(d_mask, d_distances, d_distances_i, numVertices);

        // copy the mask array from device to host
        cudaMemcpy(h_mask, d_mask, numVertices * sizeof(bool), cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(end_gpu, 0);
    cudaEventSynchronize(end_gpu);
    cudaEventElapsedTime(&msecs_gpu, start_gpu, end_gpu);

    fprintf(stderr, "in %f milliseconds.\n", msecs_gpu);

    int *distances = new int[numVertices];

    // copy the distances array from device to host
    cudaMemcpy(distances, d_distances, numVertices * sizeof(int), cudaMemcpyDeviceToHost);

    // Write output to file
    ofstream output_d;
    output_d.open(output_file);

    // write the distances to the output file
    for (int i = 0; i < numVertices; i++)
        output_d << "Distance from 0" << " to " << i << ": " << distances[i] << endl;

    output_d.close();

    // free memory
    free(distances);
    cudaFree(d_adj_list_offsets);
    cudaFree(d_adj_list);
    cudaFree(d_weights);
    cudaFree(d_distances);
    cudaFree(d_distances_i);

    return 0;
}

int runDijkstrasOnGPUGridStride(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, int block_size, string output_file, float &msecs_gpu_gs) {

    int N = numVertices + 1;

    // int *d_in_V;
    int *d_adj_list_offsets;
    int *d_adj_list;
    int *d_weights;
    int *d_distances; 
    int *d_distances_i;
    bool *d_mask;

    bool *h_mask;

    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);

    //allocate memory
    cudaMalloc((void**) &d_adj_list_offsets, N *sizeof(int));
    cudaMalloc((void**) &d_adj_list, adj_list.size() *sizeof(int));
    cudaMalloc((void**) &d_weights, weights.size() *sizeof(int));
    cudaMalloc((void**) &d_distances, numVertices *sizeof(int));
    cudaMalloc((void**) &d_distances_i, numVertices *sizeof(int));
    cudaMalloc((void**) &d_mask, numVertices *sizeof(bool));

    // allocate memory for mask on host
    h_mask = (bool *)malloc(numVertices * sizeof(bool));

    //copy to device memory
    cudaMemcpy(d_adj_list_offsets, adj_offset_list.data(), N *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj_list, adj_list.data(), adj_list.size() *sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), weights.size() *sizeof(int), cudaMemcpyHostToDevice);

    // compute time taken to run Dijkstra's algorithm on GPU
    fprintf(stderr, "Computing shortest path using Dijkstra's algorithm on GPU with grid stride loop ");

    dim3 dimBlock(block_size);
	dim3 dimGrid(ceil(numVertices / float(block_size)));

    // initialize the distance array
    initializeInputArray<<<dimGrid, dimBlock>>>(d_mask, d_distances_i, d_distances, INT_MAX, numVertices);

    cudaEventRecord(start_gpu, 0);

    // copy the mask array from device to host
    cudaMemcpy(h_mask, d_mask, numVertices * sizeof(bool), cudaMemcpyDeviceToHost);

    // Dijkstra's algorithm
    while(checkallvisited(h_mask, numVertices) == false){
        dijkstrasGPUGridStrideKernel<<<dimGrid, dimBlock>>>(d_adj_list_offsets, d_adj_list, d_weights, d_mask, d_distances, d_distances_i, numVertices);
        updateDistanceArrayGridStride<<<dimGrid, dimBlock>>>(d_mask, d_distances, d_distances_i, numVertices);

        // copy the mask array from device to host
        cudaMemcpy(h_mask, d_mask, numVertices * sizeof(bool), cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(end_gpu, 0);
    cudaEventSynchronize(end_gpu);
    cudaEventElapsedTime(&msecs_gpu_gs, start_gpu, end_gpu);

    fprintf(stderr, "in %f milliseconds.\n", msecs_gpu_gs);

    int *distances = new int[numVertices];

    // copy the distances array from device to host
    cudaMemcpy(distances, d_distances, numVertices * sizeof(int), cudaMemcpyDeviceToHost);

    // Write output to file
    ofstream output_d;
    output_d.open(output_file);

    // write the distances to the output file
    for (int i = 0; i < numVertices; i++)
        output_d << "Distance from 0" << " to " << i << ": " << distances[i] << endl;

    output_d.close();

    // free memory
    free(distances);
    cudaFree(d_adj_list_offsets);
    cudaFree(d_adj_list);
    cudaFree(d_weights);
    cudaFree(d_distances);
    cudaFree(d_distances_i);

    return 0;
}
