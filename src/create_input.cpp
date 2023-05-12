#include <iostream>
#include <fstream>
#include <cstdlib>
#include <climits>
#include <iterator>
#include <vector>
#include<string.h>
#include <ctime>
#include<unordered_set>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

#define THRESHOLD 0.3
#define NUM_THREADS 8
#define MAX_WEIGHT 100
#define MAX_VERTICES 1000000

void generate_random_graph(int num_vertices, int max_degree_per_vertex, char *argv[]){

    if(max_degree_per_vertex >= num_vertices){
        fprintf(stderr, "Error: max_degree_per_vertex >= num_vertices\n");
        return;
    }

    // seed random number generator
    srand(time(0));

    ofstream adj_file, adj_offset_file, weight_file;

    string temp(argv[1]);
    
    if(mkdir(temp.c_str(), 0777) == -1)
        cout << "Error creating directory" << endl;
    else
        cout << "Directory created" << endl;

    adj_file.open(temp + "/adjacency_list.txt");
    adj_offset_file.open(temp + "/adjacency_list_offsets.txt");
    weight_file.open(temp + "/weights.txt");

    vector<int> adjacency_list, adjacency_list_offsets(num_vertices + 1), weights;

    adjacency_list_offsets[0] = 0;

    // create adjacency list, weights, and offsets  
    for(int i = 1; i <= num_vertices; i++){
        int degree_per_vertex = (rand() % max_degree_per_vertex) + 1;

        unordered_set<int> visited;

        for(int j = 0; j < degree_per_vertex; j++){
            int random = rand() % num_vertices;

            while(random == i - 1 || visited.find(random) != visited.end()){
                random = rand() % num_vertices;
            }

            visited.emplace(random);
            adjacency_list.push_back(random);
            weights.push_back((rand() % (MAX_WEIGHT)) + 1);
        }

        adjacency_list_offsets[i] = adjacency_list_offsets[i - 1] + degree_per_vertex;
    }      

    ostream_iterator<int> output_iterator(adj_file, " ");
    copy(begin(adjacency_list), end(adjacency_list), output_iterator);
    adj_file << endl;

    output_iterator = ostream_iterator<int>(adj_offset_file, " ");
    copy(begin(adjacency_list_offsets), end(adjacency_list_offsets), output_iterator);
    adj_offset_file << endl;

    output_iterator = ostream_iterator<int>(weight_file, " ");
    copy(begin(weights), end(weights), output_iterator);
    weight_file << endl;

    adj_file.close();
    adj_offset_file.close();
    weight_file.close();
}


int main(int argc, char *argv[])
{
    if(argc < 4) {
		cout << "Usage: " << argv[0] << " <input_filepath> <num_vertices> <max_degree_per_vertex>\n";
		return 0;
	}

    generate_random_graph(atoi(argv[2]), atoi(argv[3]), argv);


    vector<int> adjacency_list, adjacency_list_offsets, weights;

    string temp(argv[1]);

    //read input file
    ifstream input_file(temp + "/adjacency_list.txt");
    istream_iterator<int> input_iterator(input_file);
    istream_iterator<int> end_of_stream;

    copy(input_iterator, end_of_stream, back_inserter(adjacency_list));

    input_file = ifstream(temp + "/adjacency_list_offsets.txt");
    input_iterator = istream_iterator<int>(input_file);

    copy(input_iterator, end_of_stream, back_inserter(adjacency_list_offsets));

    input_file = ifstream(temp + "/weights.txt");
    input_iterator = istream_iterator<int>(input_file);
    copy(input_iterator, end_of_stream, back_inserter(weights));

    return 0;
}