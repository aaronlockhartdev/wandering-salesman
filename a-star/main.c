#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define SIZE 10
#define ERROR 0

const int DIRECTIONS[8][2] = {{1, 1}, {1, 0}, {1, -1}, {0, 1}, {0, -1}, {-1, 1}, {-1, 0}, {-1, -1}};
const int GOAL[2] = {SIZE - 1, SIZE - 1};

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
        node->prev = head;
    }
    return node;
}
void pop(Node * node)
{
    if (node->next != NULL)
    {
        node->next->prev = node->prev;
    }
    if (node->prev != NULL)
    {
        node->prev->next = node->next;
    }
}

Node * findLowest(Node * head)
{
    Node * lowNode = head;
    Node * current = head->prev;
    while (1)
    {  
        if (current == NULL)
        {
            break;
        }
        if (current->f < lowNode->f)
        {
            lowNode = current;
        }
        current = current->prev;
    }
    return lowNode;
}

Node ** getChildren(Node * node, float ** board)
{
    int numChild = 0;
    int dirs[8] = {0};
    int moves[8][2];

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

    Node ** children = malloc(numChild * sizeof(Node*));
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
    return children;
}
int findLower(Node * node, Node * head)
{
    Node * current = head;
    while (1)
    {
        if (current == NULL)
        {
            return 0;
        }
        if (current->f < node->f && (current->x == node->x && current->y == node->y))
        {
            return 1;
        }
        current = current->prev;
    }
}
void printPath(Node * node)
{
    Node * current = node;
    while (1)
    {
        if (current == NULL)
        {
            return;
        }
        printf("(%d, %d)", current->x, current->y);
        current = current->parent;
    }
}
int listSize(Node * head)
{
    int size = 0;
    Node * current = head;
    while (1)
    {
        if (current == NULL)
        {
            return size;
        }
        ++size;
        current=current->prev;
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
    start->x = 0;
    start->y = 0;
    start->f = 0;
    start->g = 0;
    start->h = 0;
    start->prev = NULL;
    start->next = NULL;

    Node * closeHead = NULL;
    Node * openHead = start;

    while (openHead != NULL)
    {
        // printf("Open size: %d, Close size: %d\r", listSize(openHead), listSize(closeHead));
        Node * lowest = findLowest(openHead);
        Node ** children = getChildren(lowest, board);
        for (int i = 0; i < sizeof(*children)/sizeof(Node*); ++i){
            children[i]->g = g(children[i], board);
            children[i]->h = h(children[i], board);
            children[i]->f = children[i]->g + children[i]->h;

            if ((children[i]->x == GOAL[0]) && (children[i]->y == GOAL[1]))
            {
                return children[i];
            }
            if (findLower(children[i], openHead))
            {
                continue;
            }
            if (findLower(children[i], closeHead))
            {
                continue;
            } else
            {
                openHead = push(children[i], openHead);
            }
        }
        pop(lowest);
        closeHead = push(lowest, closeHead);
    }

    return NULL;
}
int main()
{
    float ** board = createBoard();
    printBoard(board);
    Node * path = findPath(board);
    printf("Path found: ");
    printPath(path);
    return 0;
}