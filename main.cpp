#include "main.h"

bool isFileValid(const string& filename) {
    ifstream file(filename.c_str());
    return file.good();
}

int main(int argc, char *argv[]) {
    // check for command line arguments
    if (argc < 2) {
        cout << "Usage: ./gpu_project2 <algorithm> <input_size> <block_size> <output_filepath_cpu> <output_filepath_gpu> <output_filepath_gpu_gs>\n" << endl;
        cout << "<algorithm>: one of 'bellman' or 'dijkstras'.\n"
                "<input_size>: the input size of the graph. One of {1K, 10K, 100K, 200K, 300K, 1M, 5M, 10M}.\n"
                "<block_size>: the block size for the GPU kernels. Default is 1024.\n" 
                "<output_filepath_cpu>: the output filepath for the CPU implementation. Default is '../outputs_<algorithm>/output<input_size>.txt'\n"
                "<output_filepath_gpu>: the output filepath for the GPU implementation. Default is '../outputs_<algorithm>/output<input_size>_gpu.txt'\n"
                "<output_filepath_gpu_gs>: the output filepath for the GPU implementation with grid stride. Default is '../outputs_<algorithm>/output<input_size>_gpu_gs.txt'\n" << endl;
        return 0;
    }
    
    string algorithm = argv[1];
    string size = argv[2];
    int block_size = (argc > 3) ? atoi(argv[3]) : 1024;
    string output_filepath_cpu = (argc > 4) ? argv[4] : "../outputs_" + algorithm + "/output" + size + ".txt";
    string output_filepath_gpu = (argc > 5) ? argv[5] : "../outputs_" + algorithm + "/output" + size + "_gpu.txt";
    string output_filepath_gpu_gs = (argc > 6) ? argv[6] : "../outputs_" + algorithm + "/output" + size + "_gpu_gs.txt";

    string file = "../inputs/input" + size;

    // check if input file exists
    if (!isFileValid(file + "/adjacency_list.txt") || !isFileValid(file + "/adjacency_list_offsets.txt") || !isFileValid(file + "/weights.txt")) {
        cout << "Either file doesn't exist or is invalid. Please run 'src/create_input.cpp' to generate input files." << endl;
        return 0;
    }

    // initialize variables
    int numVertices;

    // initialize vectors
    vector<int> adj_list, adj_offset_list, weights;

    // initialize files
    ifstream adj_list_file, adj_offset_list_file, weights_file;
    ofstream output;

    // initialize timers
    struct timespec start_cpu, end_cpu;
	float msecs_cpu = 0.0f, msecs_gpu = 0.0f, msecs_gpu_gs  = 0.0f;

    // open files
    adj_list_file.open(file + "/adjacency_list.txt");
    adj_offset_list_file.open(file + "/adjacency_list_offsets.txt");
    weights_file.open(file + "/weights.txt");

    ifstream input_file(file + "/adjacency_list.txt");
    istream_iterator<int> input_iterator(input_file);
    istream_iterator<int> end_of_stream;

    copy(input_iterator, end_of_stream, back_inserter(adj_list));

    input_file = ifstream(file + "/adjacency_list_offsets.txt");
    input_iterator = istream_iterator<int>(input_file);

    copy(input_iterator, end_of_stream, back_inserter(adj_offset_list));

    input_file = ifstream(file + "/weights.txt");
    input_iterator = istream_iterator<int>(input_file);
    copy(input_iterator, end_of_stream, back_inserter(weights));

    numVertices = adj_offset_list.size() - 1;

    cout << "Number of vertices: " << numVertices << endl;

    cout << "Number of edges: " << adj_list.size() << endl;

    cout << "Number of weights: " << weights.size() << endl;

    if(algorithm == "bellman"){
        runBellmanFordOnCPU(adj_list, adj_offset_list, weights, numVertices, output_filepath_cpu, msecs_cpu);
        runBellmanFordOnGPU(adj_list, adj_offset_list, weights, numVertices, block_size, output_filepath_gpu, msecs_gpu);
        runBellmanFordOnGPUGridStride(adj_list, adj_offset_list, weights, numVertices, block_size, output_filepath_gpu_gs, msecs_gpu_gs);

        // speedup
        cout << "Bellman GPU Speedup: " << msecs_cpu / msecs_gpu << endl;
        cout << "Bellman GPU Speedup with Grid Stride: " << msecs_cpu / msecs_gpu_gs << endl;
    }
    else if(algorithm == "dijkstras"){
        runDijkstrasOnCPU(adj_list, adj_offset_list, weights, numVertices, output_filepath_cpu, msecs_cpu);
        runDijkstrasOnGPU(adj_list, adj_offset_list, weights, numVertices, block_size, output_filepath_gpu, msecs_gpu);
        runDijkstrasOnGPUGridStride(adj_list, adj_offset_list, weights, numVertices, block_size, output_filepath_gpu_gs, msecs_gpu_gs);

        // speedup
        cout << "Dijkstra's GPU Speedup: " << msecs_cpu / msecs_gpu << endl;
        cout << "Dijkstra's GPU Speedup with Grid Stride: " << msecs_cpu / msecs_gpu_gs << endl;
    }
    
    return 0;
}