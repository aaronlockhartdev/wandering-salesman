#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<GL/glu.h>

#define SIZE 620
#define ERROR 0.2

const int DIRECTIONS[26][3] = {{1, 1, 1}, {1, 1, 0}, {1, 1, -1}, {1, 0, 1}, {1, 0, 0}, {1, 0, -1}, {1, -1, 1}, {1, -1, 0}, {1, -1, -1}, {0, 1, 1}, {0, 1, 0}, {0, 1, -1}, {0, 0, 1}, {0, 0, -1}, {0, -1, 1}, {0, -1, 0}, {0, -1, -1}, {-1, 1, 1}, {-1, 1, 0}, {-1, 1, -1}, {-1, 0, 1}, {-1, 0, 0}, {-1, 0, -1}, {-1, -1, 1}, {-1, -1, 0}, {-1, -1, -1}};
const int GOAL[3] = {SIZE - 1, SIZE - 1, SIZE - 1};
const int START[3] = {0, 0, 0};

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

typedef struct Children
{
    int numChild;
    Node ** children;
} Children;

float *** createBoard()
{
    float * rows = calloc(SIZE * SIZE * SIZE, sizeof(float));
    float ** columns = calloc(SIZE * SIZE, sizeof(float*));
    float *** board = malloc(SIZE * sizeof(float*));

    for (int i = 0; i < SIZE * SIZE; ++i)
    {
        columns[i] = rows + i * SIZE;
    }
    for (int i = 0; i < SIZE; i++)
    {
        board[i] = columns + i * SIZE;
    }


    srand(time(NULL));

    for (int x = 0; x < SIZE; ++x)
    {
        for (int y = 0; y < SIZE; ++y)
        {   
            for (int z = 0; z < SIZE; ++z)
            {
                board[x][y][z] = ((float)(rand() % 100 + 1)) / 100;   
            }
        }
    }
    return board;
}

float *** createMap()
{
    float * rows = calloc(SIZE * SIZE * SIZE, sizeof(float));
    float ** columns = calloc(SIZE * SIZE, sizeof(float*));
    float *** map = malloc(SIZE * sizeof(float*));


    for (int i = 0; i < SIZE * SIZE; ++i)
    {
        columns[i] = rows + i * SIZE;
    }
    for (int i = 0; i < SIZE; i++)
    {
        map[i] = columns + i * SIZE;
    }

    for (int x = 0; x < SIZE; ++x)
    {
        for (int y = 0; y < SIZE; ++y)
        {   
            for (int z = 0; z < SIZE; ++z)
            {
                map[x][y][z] = -1;
            }
        }
    }
    return map;
}

float *** createStatus()
{
    float * rows = calloc(SIZE * SIZE * SIZE, sizeof(float));
    float ** columns = calloc(SIZE * SIZE, sizeof(float*));
    float *** status = malloc(SIZE * sizeof(float*));


    for (int i = 0; i < SIZE * SIZE; ++i)
    {
        columns[i] = rows + i * SIZE;
    }
    for (int i = 0; i < SIZE; i++)
    {
        status[i] = columns + i * SIZE;
    }
    return status;
}

void printBoard(float ** board)
{
    for (int x = 0; x < SIZE; ++x)
    {
        for (int y = 0; y < SIZE; ++y)
        {
            int b = (int)(board[x][y] * 100);
            if(b == 100)
            {
                printf("%d ", (int)(b));
            } else if (b > 9)
            {
                printf("%d  ", (int)(b));
            } else
            {
                printf("%d   ", (int)(b));
            }
            
            
            
        }
        printf("\n");
    }
}

Node * push(Node * node, Node * head)
{
    if (head != NULL)
    {
        head->next = node;
    }
    node->prev = head;
    
    return node;
}
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
void pop(Node * node)
{
    if (node->prev != NULL)
    {
        node->prev->next = node->next;
    }

    if (node->next != NULL)
    {
        node->next->prev = node->prev;
    }

    node->next = NULL;
    node->prev = NULL;
}

Node * findLowest(Node * head)
{
    Node * lowNode = head;
    Node * current = head->prev;
    while (current != NULL)
    {  
        if (current->f < lowNode->f)
        {
            lowNode = current;
        }
        current = current->prev;
    }
    return lowNode;
}

Children * getChildren(Node * node)
{
    int numChild = 0;
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
            ++numChild;
        }
    }


    Node ** children = malloc(numChild * sizeof(Node *));
    Node * child = calloc(numChild, sizeof(Node));
    for (int i = 0; i < numChild; ++i)
    {
        children[i] = child + i;
    }
    int counter = 0;
    for (int i = 0; i < 26; ++i)
    {
        if (dirs[i])
        {
            children[counter]->x = moves[i][0];
            children[counter]->y = moves[i][1];
            children[counter]->z = moves[i][2];
            children[counter]->parent = node;
            children[counter]->next = NULL;
            children[counter]->prev = NULL;
            ++counter;
        }
    }
    Children * c = malloc(sizeof(Children));
    c->children = children;
    c->numChild = numChild;
    return c;
}
int printPath(Node * node, float*** board, int verbose)
{
    int sum = 0;
    Node * current = node;
    while (1)
    {
        if (verbose)
        {
            printf("\n(%d, %d, %d) = %d", current->x, current->y, current->z, (int)(board[current->x][current->y][current->z] * 100));
        }
        sum += (int)(board[current->x][current->y][current->z] * 100);
        if (current->x == START[0] && current->y == START[1] && current->z == START[2])
        {
            return sum;
        }
        current = current->parent;

    }
}
float g(Node * node, float *** board)
{
    return node->parent->g + board[node->x][node->y][node->z];
}
float h(Node * node)
{
    return fmax(abs(node->x - GOAL[0]), fmax(abs(node->y - GOAL[1]), abs(node->z - GOAL[2]))) * ERROR;
}
Node * findPath(float *** board)
{
    Node * start = malloc(sizeof(Node));
    start->x = START[0];
    start->y = START[1];
    start->z = START[2];
    start->g = board[start->x][start->y][start->z];
    start->h = 0;
    start->f = start->g + start->h;
    start->prev = NULL;
    start->next = NULL;

    float *** map = createMap();

    Node * closeHead = NULL;
    Node * openHead = start;

    while (openHead != NULL)
    {
        Node * lowest = openHead;
        // printf("%f vs %f\r", openHead->f, findLowest(openHead)->f);
        Children * c = getChildren(lowest);
        Node ** children = c->children;

        for (int i = 0; i < c->numChild; ++i){
            // printf("Open size: %d, Close size: %d, NumChildren: %d\r", listSize(openHead), listSize(closeHead), c->numChild);
            children[i]->g = g(children[i], board);
            children[i]->h = h(children[i]);
            children[i]->f = children[i]->g + children[i]->h;

            if ((children[i]->x == GOAL[0]) && (children[i]->y == GOAL[1]) && (children[i]->z == GOAL[2]))
            {
                return children[i];
            }
            if (map[children[i]->x][children[i]->y][children[i]->z] <= children[i]->f && map[children[i]->x][children[i]->y][children[i]->z] != -1)
            {
                continue;
            } else
            {
                map[children[i]->x][children[i]->y][children[i]->z] = children[i]->f;
                openHead = pushSorted(children[i], openHead);
            }
        }
        free(children);
        if (lowest == openHead)
        {
            openHead = lowest->prev;
            lowest->prev = NULL;
        }
        pop(lowest);
        closeHead = push(lowest, closeHead);
    }

    return NULL;
}
int main()
{
    float *** board = createBoard();
    // printBoard(board);
    clock_t start, end;
    int cpuTime;

    start = clock();
    Node * path = findPath(board);
    end = clock();
    cpuTime = (int)(((double) (end - start) * 1000) / CLOCKS_PER_SEC);
    if (path == NULL)
    {
        printf("\nMission failed, we'll get em next time");
    }
    printf("\nPath found for %d by %d by %d board in %d milliseconds with distance priority of %.2f and cost of %d\n", SIZE, SIZE, SIZE, cpuTime, ERROR, printPath(path, board, 1));
    free(path);
    return 0;
}