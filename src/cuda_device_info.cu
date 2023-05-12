#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

void printCudaDevice(){
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("****** Using device %d ***********\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem,
               (int)devProps.major, (int)devProps.minor,
               (int)devProps.clockRate);
        printf("Number of multiprocessors on device : %d\n", devProps.multiProcessorCount);
        printf("Maximum size of each dimension of a grid : %d\n", devProps.maxGridSize);
        printf("Maximum size of each dimension of a block : %d\n", devProps.maxThreadsDim);
        printf("Maximum number of threads per block : %d\n", devProps.maxThreadsPerBlock);
        //printf("Maximum number of resident blocks per multiprocessor : %d\n", devProps.maxBlocksPerMultiProcessor );
        printf("Maximum resident threads per multiprocessor : %d\n", devProps.maxThreadsPerMultiProcessor);
        printf("Shared memory available per block in bytes : %zu \n", devProps.sharedMemPerBlock );
        printf("Shared memory available per multiprocessor in bytes : %zu \n", devProps.sharedMemPerMultiprocessor );
        printf("Warp size in threads : %d \n", devProps.warpSize );
        printf("****** End of device stats ***********\n");
    }
}

int main(int argc, char **argv)
{
    printCudaDevice();
    return 0;
}