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

__constant__ int directions_c[8][2];
__constant__ int goal_c[2];

__global__ void initBoard(float[][] board)
{
    #include<math.h>
    #include<time.h>

    srand(clock() + blockIdx.x + blockIdx.y * blockDim.x);
    board[blockIdx.x][blockIdx.y] = ((float)(rand() % 100 + 1)) / 100;
    printf("%f\n", board[blockIdx.x][blockIdx.y]);
}

void printBoard(float[][] board)
{
    for (int x = 0; x < SIZE; ++x)
    {
        for (int y = 0; y < SIZE; ++y)
        {
            printf("%d ", (int)(board[x][y] * 100));
        }
        printf("\n");
    }
}

int main()
{
    int directions_h[8][2] = {{1, 1}, {1, 0}, {1, -1}, {0, 1}, {0, -1}, {-1, 1}, {-1, 0}, {-1, -1}};
    int goal_h[2] = {SIZE - 1, SIZE - 1};

    cudaMemcpyToSymbol(directions_c, directions_h, sizeof(int) * 16);
    cudaMemcpyToSymbol(goal_c, goal_h, sizeof(int) * 2);

    float board[SIZE][SIZE];
    cudaMallocManaged(&board, sizeof(float) * SIZE * SIZE);

    initBoard<<<SIZE, SIZE, 1>>>(board);
    cudaDeviceSynchronize();
    printBoard(board);

    Node * start;
    Node * closed;
    Node * best;
    Node* open[NUM_QUEUES];
    float hashTable[SIZE][SIZE];

    cudaMallocManaged(&start, sizeof(Node));
    cudaMallocManaged(&closed, sizeof(Node));
    cudaMallocManaged(&best, sizeof(Node));
    cudaMallocManaged(&open, sizeof(Node*) * NUM_QUEUES);
    cudaMallocManaged(&hashTable, sizeof(float) * SIZE * SIZE);

    start->f = 0;
    start->g = 0;
    start->x = START[0];
    start->y = START[1];
    start->parent = NULL;
    start->next = NULL;
    start->prev = NULL;


    cudaFree(start);

    return 0;
}