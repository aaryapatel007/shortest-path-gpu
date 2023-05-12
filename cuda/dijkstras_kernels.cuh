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

__global__ void initializeInputArray(bool *d_mask, int *dist_i, int *dist, int val, int N);
__global__ void updateDistanceArray(bool *d_mask, int *d_distances, int *d_distances_i, int N);
__global__ void updateDistanceArrayGridStride(bool *d_mask, int *d_distances, int *d_distances_i, int N);
__global__ void dijkstrasGPUKernel(int *d_adj_list_offsets, int *d_adj_list, int *d_weights, bool *d_mask, int *d_distances, int *d_distances_i, int N);
__global__ void dijkstrasGPUGridStrideKernel(int *d_adj_list_offsets, int *d_adj_list, int *d_weights, bool *d_mask, int *d_distances, int *d_distances_i, int N);