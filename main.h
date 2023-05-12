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

// bellman ford algorithm
void sequentialBellmanFord(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, vector<int> &distances);
int runBellmanFordOnCPU(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, string output_file, float &msecs_cpu);
int runBellmanFordOnGPU(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, int block_size, string output_file, float &msecs_gpu);
int runBellmanFordOnGPUGridStride(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, int block_size, string output_file, float &msecs_gpu_gs);

// dijkstra's algorithm
void sequentialDijkstras(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, vector<int> &distances);
int runDijkstrasOnCPU(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, string output_file, float &msecs_cpu);
int runDijkstrasOnGPU(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, int block_size, string output_file, float &msecs_gpu);
int runDijkstrasOnGPUGridStride(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, int block_size, string output_file, float &msecs_gpu_gs);