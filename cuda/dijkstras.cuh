#include "dijkstras_kernels.cuh"

// check if all vertices have been visited
bool checkallvisited(bool *mask, int N);
int runDijkstrasOnGPU(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, int block_size, string output_file, float &msecs_gpu);
int runDijkstrasOnGPUGridStride(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, string output_file, float &msecs_gpu_gs);