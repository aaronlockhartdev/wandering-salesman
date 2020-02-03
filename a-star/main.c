#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define SIZE 1000
#define ERROR 0.0

const int DIRECTIONS[8][2] = {{1, 1}, {1, 0}, {1, -1}, {0, 1}, {0, -1}, {-1, 1}, {-1, 0}, {-1, -1}};
const int GOAL[2] = {SIZE - 1, SIZE - 1};
const int START[2] = {0, 0};

typedef struct Node Node;

struct Node
{
    Node * next;
    Node * prev;
    int x;
    int y;
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

float ** createBoard()
{
    float * rows = calloc(SIZE * SIZE, sizeof(float));
    float ** board = malloc(SIZE * sizeof(float*));

    for (int i = 0; i < SIZE; ++i)
    {
        board[i] = rows + i * SIZE;
    }

    srand(time(NULL));

    for (int x = 0; x < SIZE; ++x)
    {
        for (int y = 0; y < SIZE; ++y)
        {   
            board[x][y] = ((float)(rand() % 100 + 1)) / 100;

        }
    }
    return board;
}

float ** createMap()
{
    float * rows = calloc(SIZE * SIZE, sizeof(float));
    float ** map = malloc(SIZE * sizeof(float*));

    for (int i = 0; i < SIZE; ++i)
    {
        map[i] = rows + i * SIZE;
    }

    for (int x = 0; x < SIZE; ++x)
    {
        for (int y = 0; y < SIZE; ++y)
        {   
            map[x][y] = -1;

        }
    }
    return map;
}

void printBoard(float ** board)
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

Children * getChildren(Node * node, float ** board)
{
    int numChild = 0;
    int dirs[8] = {0};
    int moves[8][2] = {0};

    for (int i = 0; i < 8; ++i)
    {
        moves[i][0] = node->x + DIRECTIONS[i][0];
        moves[i][1] = node->y + DIRECTIONS[i][1];
        if (moves[i][0] < SIZE && moves[i][1] < SIZE && moves[i][0] >= 0 && moves[i][1] >= 0)
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
    for (int i = 0; i < 8; ++i)
    {
        if (dirs[i])
        {
            children[counter]->x = moves[i][0];
            children[counter]->y = moves[i][1];
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
int listSize(Node * head)
{
    int size = 0;
    Node * current = head;
    while (current != NULL)
    {
        ++size;
        current=current->prev;
    }
    return size;
}
int findLower(Node * node, Node * head)
{
    Node * current = head;
    while (current != NULL)
    {
        if (current->f <= node->f && (current->x == node->x && current->y == node->y))
        {
            return 1;
        }
        current = current->prev;
    }
    return 0;
}
void printPath(Node * node)
{
    Node * current = node;
    while (1)
    {
        printf("\n(%d, %d)", current->x, current->y);
        if (current->x == START[0] && current->y == START[1])
        {
            return;
        }
        current = current->parent;

    }
}
float g(Node * node, float ** board)
{
    return node->parent->g + board[node->parent->x][node->parent->y];
}
float h(Node * node, float ** board)
{
    return fmax(abs(node->x - GOAL[0]), abs(node->y - GOAL[1])) * ERROR;
}
Node * findPath(float ** board)
{
    Node * start = malloc(sizeof(Node));
    start->x = START[0];
    start->y = START[1];
    start->f = 0;
    start->g = 0;
    start->h = 0;
    start->prev = NULL;
    start->next = NULL;

    float ** map = createMap();

    Node * closeHead = NULL;
    Node * openHead = start;

    while (openHead != NULL)
    {
        Node * lowest = openHead;
        // printf("%f vs %f\r", openHead->f, findLowest(openHead)->f);
        Children * c = getChildren(lowest, board);
        Node ** children = c->children;

        for (int i = 0; i < c->numChild; ++i){
            // printf("Open size: %d, Close size: %d, NumChildren: %d\r", listSize(openHead), listSize(closeHead), c->numChild);
            children[i]->g = g(children[i], board);
            children[i]->h = h(children[i], board);
            children[i]->f = children[i]->g + children[i]->h;

            if ((children[i]->x == GOAL[0]) && (children[i]->y == GOAL[1]))
            {
                return children[i];
            }
            if (map[children[i]->x][children[i]->y] <= children[i]->f && map[children[i]->x][children[i]->y] != -1)
            {
                continue;
            } else
            {
                map[children[i]->x][children[i]->y] = children[i]->f;
            }
            
            openHead = pushSorted(children[i], openHead);
        }
        if (lowest == openHead)
        {
            openHead = lowest->prev;
        }
        pop(lowest);
        closeHead = push(lowest, closeHead);
    }

    return NULL;
}
int main()
{
    float ** board = createBoard();
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
    printf("\nPath found for %d by %d board in %d milliseconds with distance priority of %.2f\n", SIZE, SIZE, cpuTime, ERROR);
    printPath(path);
    free(path);
    return 0;
}