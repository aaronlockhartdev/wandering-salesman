#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<GL/glew.h> 
#include<GL/freeglut.h>
#include<GL/gl.h>
#include<GL/glu.h>
#include<GL/glext.h>

// define constants
#define SIZE 620
#define ERROR 0.2
const int DIRECTIONS[26][3] = {{1, 1, 1}, {1, 1, 0}, {1, 1, -1}, {1, 0, 1}, {1, 0, 0}, {1, 0, -1}, {1, -1, 1}, {1, -1, 0}, {1, -1, -1}, {0, 1, 1}, {0, 1, 0}, {0, 1, -1}, {0, 0, 1}, {0, 0, -1}, {0, -1, 1}, {0, -1, 0}, {0, -1, -1}, {-1, 1, 1}, {-1, 1, 0}, {-1, 1, -1}, {-1, 0, 1}, {-1, 0, 0}, {-1, 0, -1}, {-1, -1, 1}, {-1, -1, 0}, {-1, -1, -1}};
const int GOAL[3] = {SIZE - 1, SIZE - 1, SIZE - 1};
const int START[3] = {0, 0, 0};

// Node structure
typedef struct Node Node;

struct Node
{
    Node * next;
    Node * prev;
    int x;
    int y;
    int z;
    Node * parent;
    float f;
    float g;
    float h;
};

// Children structure
typedef struct Children Children;

struct Children
{
    Node ** c;
    int n;
};

// 3D float array structure
typedef struct float3D float3D;

struct float3D
{
    int x;
    int y;
    int z;
    float *** a;
};

// define global variables
float3D * board;
float3D * nodeState;
float3D * map;

Node * openHead;
Node * closeHead;
Node * goal;

// 3D float array creation method
float3D * createFloat3D(int x, int y, int z)
{
    float * zArray = calloc(x * y * z, sizeof(float));
    float ** yArray = calloc(x * y, sizeof(float*));
    float *** xArray = malloc(x * sizeof(float*));

    for (int i = 0; i < x * y; ++i)
    {
        yArray[i] = zArray + i * z;
    }
    for (int i = 0; i < x; ++i)
    {
        xArray[i] = yArray + i * y;
    }

    float3D * f = malloc(sizeof(float3D));
    f->a = xArray;
    f->x = x;
    f->y = y;
    f->z = z;

    return f;
}

// Node creation method
Node * createNode(int x, int y, int z, Node * parent)
{
    Node * node = malloc(sizeof(Node));
    node->x = x;
    node->y = y;
    node->z = z;
    node->parent = parent;
    node->next = NULL;
    node->prev = NULL;
    
    return node;
}

// create init functions
void initGlobal()
{
    // init 3D float arrays
    board = createFloat3D(SIZE, SIZE, SIZE);
    nodeState = createFloat3D(SIZE, SIZE, SIZE);
    map = createFloat3D(SIZE, SIZE, SIZE);

    // set random seed
    srand(clock());

    // init values for 3D float arrays
    for (int x = 0; x < board->x; ++x)
    {
        for (int y = 0; y < board->y; y++)
        {
            for (int z = 0; z < board->z; z++)
            {
                board->a[x][y][z] = ((float)(rand() % 100 + 1)) / 100;
                nodeState->a[x][y][z] = 0;
                map->a[x][y][z] = -1;
            }
        }
    }

    // init heads
    openHead = createNode(START[0], START[1], START[2], NULL);
    openHead->g = board->a[openHead->x][openHead->y][openHead->z];
    openHead->f = openHead->g;

    closeHead = NULL;

    goal = NULL;
}

void initGraphics()
{
    return;
}

void init()
{
    initGlobal();
    initGraphics();
}

// A* function declaration

// gets next steps of a Node
Children * getChildren(Node * node)
{
    Children * children = malloc(sizeof(Children));
    children->n = 0;
    int dirs[26] = {0};
    int moves[26][3] = {0};

    for (int i = 0; i < 26; ++i)
    {
        moves[i][0] = node->x + DIRECTIONS[i][0];
        moves[i][1] = node->y + DIRECTIONS[i][1];
        moves[i][2] = node->z + DIRECTIONS[i][2];
        if (moves[i][0] < SIZE && moves[i][1] < SIZE && moves[i][0] >= 0 && moves[i][1] >= 0 && moves[i][2] < SIZE && moves[i][2] >= 0)
        {
            dirs[i] = 1;
            ++children->n;
        }
    }

    children->c = malloc(children->n * sizeof(Node *));
    Node * child = calloc(children->n, sizeof(Node));

    for (int i = 0; i < children->n; ++i)
    {
        children->c[i] = child + i;
    }
    int counter = 0;
    for (int i = 0; i < 26; ++i)
    {
        if (dirs[i])
        {
            children->c[counter] = createNode(moves[i][0], moves[i][1], moves[i][2], node);
            ++counter;
        }
    }

    return children;
}

// pushes a node to a list sorted by f
Node * pushSorted(Node * node, Node * head)
{
    Node * current = head;
    Node * last;
    while (current != NULL)
    {
        if (node->f < current->f)
        {
            node->prev = current;
            node->next = current->next;
            if (current->next != NULL)
            {
                current->next->prev = node;
            }
            current->next = node;
            if (current == head)
            {
                return node;
            } else
            {
                return head;
            }
        }
        last = current;
        current = current->prev;
    }
    
    last->prev = node;
    node->next = last;
    return head;
}

// calculates observable cost
float g(Node * node, float *** board)
{
    return node->parent->g + board[node->x][node->y][node->z];
}
// calculates predicted cost
float h(Node * node)
{
    return fmax(abs(node->x - GOAL[0]), fmax(abs(node->y - GOAL[1]), abs(node->z - GOAL[2]))) * ERROR;
}

// single A* step
int pathStep()
{
    if (openHead == NULL)
    {
        return 0;
    }
    Children * children = getChildren(openHead);
    Node ** c = children->c;

    for (int i = 0; i < children->n; ++i)
    {
        if ((c[i]->x == GOAL[0]) && (c[i]->y == GOAL[1]) && (c[i]->z == GOAL[2]))
        {
            goal = c[i];
            return 1;
        }
        c[i]->g = g(c[i], board->a);
        c[i]->h = h(c[i]);
        c[i]->f = c[i]->g + c[i]->h;
        if (map->a[c[i]->x][c[i]->y][c[i]->z] <= c[i]->f && map->a[c[i]->x][c[i]->y][c[i]->z] != -1)
        {
            continue;
        } else
        {
            map->a[c[i]->x][c[i]->y][c[i]->z] = c[i]->f;
            openHead = pushSorted(c[i], openHead);
        }
    }
    free(children);
    
    // create temp variable to hold current node
    Node * tmp = openHead;

    // pop current node from open list
    openHead = tmp->prev;
    openHead->next = NULL;

    // push current node to closed list
    if (closeHead != NULL)
    {
        tmp->prev = closeHead;
        closeHead->next = tmp;
    }
    closeHead = tmp;

    return -1;
}

// declare graphics functions
void display()
{
    return;
}

int main()
{
    init();
    while(1)
    {
        int status = pathStep();
        if (status == 0)
        {
            printf("nooo");
            break;
        }
        if (status == 1)
        {
            printf("hell yeah");
            break;
        }
    }
    return 0;
}