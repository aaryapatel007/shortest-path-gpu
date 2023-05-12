#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <queue>
#include <stdio.h>
#include <iterator>
#include <time.h>
#include <limits.h>

using namespace std;

__global__ void initializeInputArray(int *array, int val, int N);
__global__ void bellmanFordGPUKernel(int *d_adj_list_offsets, int *d_adj_list, int *d_weights, int *d_distances, int *d_distances_i, int N);
__global__ void updateDistanceArray(int *d_distances, int *d_distances_i, int N);
__global__ void bellmanFordGPUGridStrideKernel(int *d_adj_list_offsets, int *d_adj_list, int *d_weights, int *d_distances, int *d_distances_i, int N);
__global__ void updateDistanceArrayGridStride(int *d_distances, int *d_distances_i, int N);