#include <iostream>
#include <vector>
#include <queue>
#include <ctime>
#include <cuda.h>
#include <cooperative_groups.h>

using namespace std;

namespace cg = cooperative_groups;

inline cudaError_t checkCudaErr(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
    }
    return err;
}

void cpuBFS(int node, int nodes, int *adjacencyList, int *edgesOffset)
{
    queue<int> output_queue;
    vector<bool> visited(nodes, false);
    queue<int> temp_queue;

    output_queue.push(node);
    temp_queue.push(node);
    visited[node] = true;

    while (!temp_queue.empty())
    {
        int u = temp_queue.front();
        temp_queue.pop();
        for (int j = edgesOffset[u]; j < edgesOffset[u + 1]; j++)
        {
            int v = adjacencyList[j];
            if (!visited[v])
            {
                // printf("%d\n", v);
                output_queue.push(v);
                temp_queue.push(v);
                visited[v] = true;
            }
        }
    }
}

__global__ void cudaBFS(int node, int nodes, int *adjacencyList, int *edgesOffset, int *currentQueue, int *nextQueue, int *visited)
{
    int thid = threadIdx.x;
    extern __shared__ int queues[];

    int *currentQueueSize = &queues[0];
    int *nextQueueSize = currentQueueSize + 1;

    if (thid == 0)
    {
        // printf("ADJ LIST FROM GPU:\n");
        // for (int i = 0; i < edgesOffset[nodes]; i++)
        // {
        //     printf("%d ", adjacencyList[i]);
        // }
        // printf("\n\n");

        // printf("OFFSET FROM GPU:\n");
        // for (int i = 0; i < nodes + 1; i++)
        // {
        //     printf("%d ", edgesOffset[i]);
        // }
        // printf("\n\n");

        currentQueue[0] = node;
        visited[node] = 1;
        *currentQueueSize = 1;
        *nextQueueSize = 0;

        // printf("QUEUE SIZE: %d\n", *currentQueueSize);
        // for (int i = 0; i < *currentQueueSize; i++)
        // {
        //     printf("%d ", currentQueue[i]);
        // }
        // printf("\n");
    }

    __syncthreads();

    while (*currentQueueSize > 0)
    {
        for (int i = thid; i < *currentQueueSize; i += blockDim.x)
        {
            int u = currentQueue[i];
            // printf("VISITED[%d] = %d\n", u, visited[u]);
            for (int j = edgesOffset[u]; j < edgesOffset[u + 1]; j++)
            {
                int v = adjacencyList[j];
                // printf("u = %d, v = %d, VISITED = %d\n", u, v, visited[v]);
                if (visited[v] == 0)
                {
                    visited[v] = 1;
                    int position = atomicAdd(nextQueueSize, 1);
                    nextQueue[position] = v;
                }
            }
        }
        __syncthreads();

        // if (thid == 0)
        // {
        //     printf("NEXT QUEUE SIZE: %d\n", *nextQueueSize);
        //     for (int i = 0; i < *nextQueueSize; i++)
        //     {
        //         printf("%d ", nextQueue[i]);
        //     }
        //     printf("\n\n\n");
        // }
        int *temp = currentQueue;
        currentQueue = nextQueue;
        nextQueue = temp;

        temp = currentQueueSize;
        currentQueueSize = nextQueueSize;
        nextQueueSize = temp;

        if (thid == 0)
        {
            *nextQueueSize = 0;
        }
        __syncthreads();
    }
}

__global__ void updatedCudaBFS(int node, int nodes, int *adjacencyList, int *edgesOffset, int *currentQueue, int *nextQueue, int *currentQueueSize, int *nextQueueSize, int *visited)
{
    int thid = threadIdx.x;
    int blkid = blockIdx.x;

    auto g = cg::this_grid();

    if (blkid == 0 && thid == 0)
    {
        currentQueue[0] = node;
        visited[node] = 1;
        *currentQueueSize = 1;
        *nextQueueSize = 0;
    }

    g.sync();

    while (*currentQueueSize > 0)
    {
        for (int i = blkid; i < *currentQueueSize; i += gridDim.x)
        {
            int u = currentQueue[i];
            for (int j = thid; j < edgesOffset[u + 1]; j += blockDim.x)
            {
                int v = adjacencyList[j];

                if (visited[v] == 0)
                {
                    visited[v] = 1;
                    int position = atomicAdd(nextQueueSize, 1);
                    nextQueue[position] = v;
                }
            }
        }
        g.sync();

        int *temp = currentQueue;
        currentQueue = nextQueue;
        nextQueue = temp;

        temp = currentQueueSize;
        currentQueueSize = nextQueueSize;
        nextQueueSize = temp;

        if (blkid == 0 && thid == 0)
        {
            *nextQueueSize = 0;
        }
        g.sync();
    }
}

int main(void)
{
    vector<int> v_nodes = {10, 100, 500, 1000, 5000, 10000};
    vector<double> v_prob = {0.05, 0.1, 0.3, 0.5};

    for (double p : v_prob)
    {
        printf("PROBABILITY: %f\n", p);
        for (int nodes : v_nodes)
        {

            vector<int> adjacencyList;
            int *h_edgesOffset = new int[nodes + 1];

            for (int i = 0; i < nodes; i++)
            {
                h_edgesOffset[i] = adjacencyList.size();
                for (int j = 0; j < nodes; j++)
                {
                    if ((float)rand() / RAND_MAX < p)
                    {
                        if (i != j)
                        {
                            adjacencyList.push_back(j);
                        }
                    }
                }
            }
            h_edgesOffset[nodes] = adjacencyList.size();

            int *h_adjacencyList = new int[adjacencyList.size()];
            for (int i = 0; i < adjacencyList.size(); i++)
            {
                h_adjacencyList[i] = adjacencyList[i];
            }

            int *h_visited = new int[nodes];
            int *d_adjacencyList, *d_edgesOffset, *d_currentQueue, *d_nextQueue, *d_visited, *d_currentQueueSize, *d_nextQueueSize;

            checkCudaErr(cudaMalloc((void **)&d_adjacencyList, adjacencyList.size() * sizeof(int)), "cudaMalloc d_adjacencyList");
            checkCudaErr(cudaMalloc((void **)&d_edgesOffset, (nodes + 1) * sizeof(int)), "cudaMalloc d_edgesOffset");
            checkCudaErr(cudaMalloc((void **)&d_currentQueue, nodes * sizeof(int)), "cudaMalloc d_currentQueue");
            checkCudaErr(cudaMalloc((void **)&d_nextQueue, nodes * sizeof(int)), "cudaMalloc d_nextQueue");
            checkCudaErr(cudaMalloc((void **)&d_visited, nodes * sizeof(int)), "cudaMalloc d_visited");
            checkCudaErr(cudaMalloc((void **)&d_currentQueueSize, sizeof(int)), "cudaMalloc d_currentQueueSize");
            checkCudaErr(cudaMalloc((void **)&d_nextQueueSize, sizeof(int)), "cudaMalloc d_nextQueueSize");

            checkCudaErr(cudaMemcpy(d_adjacencyList, h_adjacencyList, adjacencyList.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_adjacencyList");
            checkCudaErr(cudaMemcpy(d_edgesOffset, h_edgesOffset, (nodes + 1) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_edgesOffset");

            int NUMBER_OF_BLOCKS = 128;
            int NUMBER_OF_THREADS = 32;
            int SHARED_MEMORY_SIZE = 2 * sizeof(int);

            clock_t start, end;

            double time_taken_cpu = 0;
            double time_taken_gpu = 0;
            double time_taken_gpu_updated = 0;

            double t_cpu, t_gpu, t_gpu_updated;

            int SAMPLES = 10;

            for (int node = 0; node < SAMPLES; node++)
            {
                // CPU
                start = clock();
                cpuBFS(node, nodes, h_adjacencyList, h_edgesOffset);
                end = clock();

                t_cpu = ((double)(end - start)) * 1000 / CLOCKS_PER_SEC;
                time_taken_cpu += t_cpu;

                // GPU
                cudaMemset(d_visited, 0, nodes * sizeof(int));
                cudaOccupancyMaxPotentialBlockSize(&NUMBER_OF_BLOCKS, &NUMBER_OF_THREADS, cudaBFS, 0, 0);

                start = clock();
                cudaBFS<<<NUMBER_OF_BLOCKS, NUMBER_OF_THREADS, SHARED_MEMORY_SIZE>>>(node, nodes, d_adjacencyList, d_edgesOffset, d_currentQueue, d_nextQueue, d_visited);
                checkCudaErr(cudaDeviceSynchronize(), "device synchronize");
                end = clock();

                t_gpu = ((double)(end - start)) * 1000 / CLOCKS_PER_SEC;
                time_taken_gpu += t_gpu;

                cudaMemcpy(h_visited, d_visited, nodes * sizeof(int), cudaMemcpyDeviceToHost);

                // GPU Updated
                checkCudaErr(cudaMemset(d_visited, 0, nodes * sizeof(int)), "cudaMemset d_visited");
                checkCudaErr(cudaMemset(d_currentQueueSize, 0, sizeof(int)), "cudaMemset d_currentQueueSize");
                checkCudaErr(cudaMemset(d_nextQueueSize, 0, sizeof(int)), "cudaMemset d_nextQueueSize");
                cudaOccupancyMaxPotentialBlockSize(&NUMBER_OF_BLOCKS, &NUMBER_OF_THREADS, updatedCudaBFS, 0, 0);

                start = clock();
                updatedCudaBFS<<<NUMBER_OF_BLOCKS, NUMBER_OF_THREADS>>>(node, nodes, d_adjacencyList, d_edgesOffset, d_currentQueue, d_nextQueue, d_currentQueueSize, d_nextQueueSize, d_visited);
                checkCudaErr(cudaDeviceSynchronize(), "device synchronize");
                end = clock();

                t_gpu_updated = ((double)(end - start)) * 1000 / CLOCKS_PER_SEC;
                time_taken_gpu_updated += t_gpu_updated;

                cudaMemcpy(h_visited, d_visited, nodes * sizeof(int), cudaMemcpyDeviceToHost);
            }

            time_taken_cpu /= SAMPLES;
            time_taken_gpu /= SAMPLES;
            time_taken_gpu_updated /= SAMPLES;

            printf("NODES: %d\t\tCPU: %f ms\t\tGPU: %f ms\t\tGPU Updated: %f ms\n", nodes, time_taken_cpu, time_taken_gpu, time_taken_gpu_updated);

            free(h_adjacencyList);
            free(h_edgesOffset);
            free(h_visited);

            checkCudaErr(cudaFree(d_adjacencyList), "cudaFree d_adjacencyList");
            checkCudaErr(cudaFree(d_edgesOffset), "cudaFree d_edgesOffset");
            checkCudaErr(cudaFree(d_currentQueue), "cudaFree d_currentQueue");
            checkCudaErr(cudaFree(d_nextQueue), "cudaFree d_nextQueue");
            checkCudaErr(cudaFree(d_visited), "cudaFree d_visited");
            checkCudaErr(cudaFree(d_currentQueueSize), "cudaFree d_currentQueueSize");
            checkCudaErr(cudaFree(d_nextQueueSize), "cudaFree d_nextQueueSize");
        }
        printf("\n\n");
    }

    return 0;
}

// nvcc -L /usr/lib/x86_64-linux-gnu --std=c++11 -rdc=true -gencode=arch=compute_75,code=sm_75 updated.cu -o updated.out