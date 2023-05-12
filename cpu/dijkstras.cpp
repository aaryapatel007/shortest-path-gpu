#include "../main.h"

void sequentialDijkstras(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, vector<int> &distances){
    // distance from source to source is 0
    distances[0] = 0;

    // create a priority queue to store vertices that are being processed
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

    // push source vertex to priority queue
    pq.push(make_pair(0, 0));

    while(!pq.empty()){
        // get vertex with minimum distance from source
        int u = pq.top().second;
        pq.pop();

        // iterate over all neighbors of u
        for(int i = adj_offset_list[u];i < adj_offset_list[u + 1];++i){
            int weight = weights[i];
            int v = adj_list[i];

            // if distance to v through u is smaller than current distance to v
            if(distances[v] > distances[u] + weight){
                // update distance to v
                distances[v] = distances[u] + weight;

                // push v to priority queue
                pq.push(make_pair(distances[v], v));
            }
        }
    }
}

int runDijkstrasOnCPU(vector<int> adj_list, vector<int> adj_offset_list, vector<int> weights, int numVertices, string output_file, float &msecs_cpu){
    // initialize timers
    struct timespec start_cpu, end_cpu;

    vector<int> distances(numVertices, INT_MAX);

    // compute time taken to run Dijkstra's algorithm on CPU
	fprintf(stderr, "Computing shortest path using Dijkstra's algorithm on CPU ");	
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_cpu);

    // run Dijkstra's algorithm
    sequentialDijkstras(adj_list, adj_offset_list, weights, numVertices, distances);
    
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
