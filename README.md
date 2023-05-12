# Accelerating Shortest Path algorithms on GPUs

This code is implementation of shortest path algorithms described in the following paper:

Harish, P., Narayanan, P.J. (2007). Accelerating Large Graph Algorithms on the GPU Using CUDA. In: Aluru, S., Parashar, M., Badrinath, R., Prasanna, V.K. (eds) High Performance Computing – HiPC 2007. https://doi.org/10.1007/978-3-540-77220-0_21

This was my project for the course *CSE-560: GPU Computing* offered by IIIT-D under the guidance of **Dr. Ojaswa Sharma**.

## Algorithms implemented:

1. Dijkstra's algorithm
2. Bellman-Ford algorithm

## How to create the input graph file:

1. ```cd src``` to go to the src folder.
2. ```g++ -o create_input create_input.cpp``` to compile the code.
3. ```./create_input <input_filepath> <num_vertices> <max_degree_per_vertex>``` to create a input file folder in the ```input``` folder. Example: ```./create_input ../inputs/input2K 2000 30``` will create a input file with 2000 vertices and maximum degree of 30 per vertex in the ```input``` folder with name ```input2K```.

## How to run the shortest path algorithms:

1. ```mkdir build && cd build``` to create ```build``` folder in the project directory.
2. ```cmake..``` to create makefile in the ```build``` folder.
3. ```make``` to compile the code.
4. ```./gpu_project <algorithm> <input_size>``` to run the code. Example: ```./gpu_project dijkstras 1K``` will run dijkstra's algorithm on the input file ```input1K``` in the ```input``` folder. The output will be stored in the ```output``` folder with name ```output1K```. 
5. ```./gpu_project bellman 1K``` will run bellman-ford algorithm on the input file ```input1K``` in the ```input``` folder. The output will be stored in the ```output``` folder with name ```output1K```.

## Results:

Excellent speedups were obtained in comparison to CPU implementation.

Speedups achieved by GPU and GPU with grid strides over different numbers of vertices by Bellman-Ford algorithm:
![](https://github.com/aaryapatel007/shortest-path-gpu/blob/main/images/speedups_bellman.png)

<br>

Speedups achieved by GPU and GPU with grid strides over different numbers of vertices by Dijkstra's algorithm.
![](https://github.com/aaryapatel007/shortest-path-gpu/blob/main/images/speedups_dijkstras.png)
