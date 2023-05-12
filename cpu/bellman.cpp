#include "../main.h"

// CPU function
void sequentialBellmanFord(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, vector<int> &distances) {
    
    // distance from source to source is 0
    distances[0] = 0;

    // run Bellman-Ford algorithm
    for(int n = 0; n < numVertices - 1; n++){
        for(int i = 0; i < numVertices; i++){
            for(int j = adj_offset_list[i]; j < adj_offset_list[i + 1]; j++){
                if(distances[i] != INT_MAX){
                    // update if the distance from source to i + the weight of the edge between i and j is less than the distance from source to j
                    if(distances[i] + weights[j] < distances[adj_list[j]]){
                        // cout << "Distance from " << i << " to " << adj_list[j] << " is " << distances[i] + weights[j] << endl;
                        distances[adj_list[j]] = distances[i] + weights[j];
                    }
                }
            }
        }
    }
}

int runBellmanFordOnCPU(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, string output_file, float &msecs_cpu){
    // initialize timers
    struct timespec start_cpu, end_cpu;

    vector<int> distances(numVertices, INT_MAX);

    // compute time taken to run Dijkstra's algorithm on CPU
	fprintf(stderr, "Computing shortest path using Bellman-Ford algorithm on CPU ");	
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_cpu);

    // run Dijkstra's algorithm
    sequentialBellmanFord(adj_list, adj_offset_list, weights, numVertices, distances);
    
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_cpu);
	msecs_cpu = 1000.0 * (end_cpu.tv_sec - start_cpu.tv_sec) + (end_cpu.tv_nsec - start_cpu.tv_nsec)/1000000.0;
	fprintf(stderr, "in %f milliseconds.\n", msecs_cpu);

    // write output to file
    ofstream output;
    output.open(output_file);

    // print distances
    for(int i = 0;i < numVertices;++i){
        // cout << "Distance from 0" << " to " << i << ": " << distances[i] << endl;
        output << "Distance from 0" << " to " << i << ": " << distances[i] << endl;
    }

    return 0;
}
