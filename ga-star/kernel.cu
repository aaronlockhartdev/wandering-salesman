#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>

#define NUM_QUEUES 1024
#define SIZE 10

const int START[2] = {0, 0};

typedef struct Node Node;

struct Node
{
    float f;
    float g;
    int x;
    int y;
    Node * parent;
    Node * next;
    Node * prev;
};

__device__ Node* open[NUM_QUEUES];
__device__ Node * closed;
__device__ Node * best;
__device__ __constant__ int directions[8][2] = {{1, 1}, {1, 0}, {1, -1}, {0, 1}, {0, -1}, {-1, 1}, {-1, 0}, {-1, -1}};
__device__ __constant__ int goal[2] = {SIZE - 1, SIZE - 1};

__global__ void queueInit(Node * start)
{
    // define start in global memory within GPU (gotta figure this out)
    // use multiple blocks
    open[0] = start;
    closed = NULL;
    best = NULL;
}

int main()
{
    Node * start = (Node*) malloc(sizeof(Node));

    start->f = 0;
    start->g = 0;
    start->x = START[0];
    start->y = START[1];
    start->parent = NULL;
    start->next = NULL;
    start->prev = NULL;

    cudaMallocManaged(&start, sizeof(Node));

    queueInit<<<1, 1>>>(start);

    cudaDeviceSynchronize();

    cudaFree(start);

    return 0;
}