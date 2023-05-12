#include "bellman_kernels.cuh"

int runBellmanFordOnCPU(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, string output_file, float &msecs_cpu);
int runBellmanFordOnGPU(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, int block_size, string output_file, float &msecs_gpu);
int runBellmanFordOnGPUGridStride(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, int block_size, string output_file, float &msecs_gpu_gs);