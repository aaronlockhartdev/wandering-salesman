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
#define M_PI 3.141592653589793
#define SIZE 20
#define ERROR 0.2
const int DIRECTIONS[26][3] = {{1, 1, 1}, {1, 1, 0}, {1, 1, -1}, {1, 0, 1}, {1, 0, 0}, {1, 0, -1}, {1, -1, 1}, {1, -1, 0}, {1, -1, -1}, {0, 1, 1}, {0, 1, 0}, {0, 1, -1}, {0, 0, 1}, {0, 0, -1}, {0, -1, 1}, {0, -1, 0}, {0, -1, -1}, {-1, 1, 1}, {-1, 1, 0}, {-1, 1, -1}, {-1, 0, 1}, {-1, 0, 0}, {-1, 0, -1}, {-1, -1, 1}, {-1, -1, 0}, {-1, -1, -1}};
const int GOAL[3] = {19, 19, 19};
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

typedef struct float4D float4D;

struct float4D
{
    int x;
    int y;
    int z;
    int k;
    float **** a;
};

// define global variables

// A* variables
float3D * board;
float4D * nodeState;
float3D * map;

Node * openHead;
Node * closeHead;
Node * goal;
Node * lowest;

int fullScreen;
int oldX = 1200;
int oldY = 900;

clock_t clck;

// graphics variables
double rotY = 0.0;
double rotZ = 0.0;
double scale = 15.0;

double camX;
double camY;
double camZ;

float distance = 50;

int keys[6] = {0};

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

float4D * createFloat4D(int x, int y, int z, int k)
{
    float * kArray = calloc(x * y * z * k, sizeof(float));
    float ** zArray = calloc(x * y * z, sizeof(float*));
    float *** yArray = calloc(x * y, sizeof(float*));
    float **** xArray = malloc(x * sizeof(float*));

    for (int i = 0; i < x * y * z; ++i)
    {
        zArray[i] = kArray + i * k;
    }
    for (int i = 0; i < x * y; ++i)
    {
        yArray[i] = zArray + i * z;
    }
    for (int i = 0; i < x; ++i)
    {
        xArray[i] = yArray + i * y;
    }
    
    float4D * f = malloc(sizeof(float4D));
    f->a = xArray;
    f->x = x;
    f->y = y;
    f->z = z;
    f->k = k;

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
    nodeState = createFloat4D(SIZE, SIZE, SIZE, 5);
    map = createFloat3D(SIZE, SIZE, SIZE);

    // set random seed
    srand(time(NULL));

    // init values for 3D float arrays

    for (int x = 0; x < board->x; ++x)
    {
        for (int y = 0; y < board->y; y++)
        {
            for (int z = 0; z < board->z; z++)
            {
                board->a[x][y][z] = ((float)(rand() % 100 + 1)) / 100;
                map->a[x][y][z] = -1;
                nodeState->a[x][y][z][0] = board->a[x][y][z];
                nodeState->a[x][y][z][1] = board->a[x][y][z];
                nodeState->a[x][y][z][2] = board->a[x][y][z];
                nodeState->a[x][y][z][3] = 0.5;
                nodeState->a[x][y][z][4] = 1.0 / SIZE;
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
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClearDepth(1.0f);
    glEnable(GL_MULTISAMPLE_ARB);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glShadeModel(GL_SMOOTH);
    glLineWidth(1.3);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
}

void init()
{
    initGlobal();
    initGraphics();
}

// A* function declaration

// prints linked list of nodes
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
// pops a node from a list
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
// pushes a node to a list without sorting
Node * push(Node * node, Node * head)
{
    if (head != NULL)
    {
        head->next = node;
    }
    node->prev = head;
    
    return node;
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
void pathStep()
{
    if (openHead == NULL || goal != NULL)
    {
        return;
    }
    Children * children = getChildren(openHead);
    Node ** c = children->c;
    Node * tmp = openHead;
    lowest = openHead;
    float * state;
    for (int i = 0; i < children->n; ++i)
    {
        if ((c[i]->x == GOAL[0]) && (c[i]->y == GOAL[1]) && (c[i]->z == GOAL[2]))
        {
            lowest = c[i];
            goal = c[i];
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
            state = nodeState->a[c[i]->x][c[i]->y][c[i]->z];
            if (c[i]->x == START[0] && c[i]->y == START[1] && c[i]->z == START[2])
            {
                state[0] = 1.0;
                state[1] = 1.0;
                state[2] = 1.0;
                state[3] = 1.0;
                state[4] = 3.0 / SIZE;
            } else
            {
                state[0] = 1.0;
                state[1] = 0.0;
                state[2] = 0.7;
                state[3] = 0.5;
                state[4] = 3.0 / SIZE;
            }
            openHead = pushSorted(c[i], openHead);
        }
    }
    free(children);
    if (tmp == openHead)
    {
        openHead = tmp->prev;
        tmp->prev = NULL;
    }
    pop(tmp);
    closeHead = push(tmp, closeHead);
    state = nodeState->a[tmp->x][tmp->y][tmp->z];
    if (tmp->x == START[0] && tmp->y == START[1] && tmp->z == START[2])
    {
        state[0] = 1.0;
        state[1] = 1.0;
        state[2] = 1.0;
        state[3] = 1.0;
        state[4] = 3.0 / SIZE;
    } else
    {
        state[0] = 0.0;
        state[1] = 0.8;
        state[2] = 1.0;
        state[3] = 1.0;
        state[4] = 1.0 / SIZE;
    }
    

    if (goal != NULL)
    {
        state = nodeState->a[goal->x][goal->y][goal->z];
        state[0] = 1.0;
        state[1] = 1.0;
        state[2] = 1.0;
        state[3] = 1.0;
        state[4] = 3.0 / SIZE;
    }

    return;
}

// declare graphics functions
void drawCube(float r, float g, float b, float a, float x, float y, float z, float s)
{
    glPushMatrix();

    glTranslatef(x, y, z);

    glBegin(GL_QUADS);                // Begin drawing the color cube with 6 quads
        // Top face (y = 1.0f)
        // Define vertices in counter-clockwise (CCW) order with normal pointing out
        glColor4f(r, g, b, a);     // Green
        glVertex3f( s, s, -s);
        glVertex3f(-s, s, -s);
        glVertex3f(-s, s,  s);
        glVertex3f( s, s,  s);
    
        // Bottom face (y = -1.0f)
        glColor4f(r, g, b, a);    // Orange
        glVertex3f( s, -s, s);
        glVertex3f(-s, -s, s);
        glVertex3f(-s, -s, s);
        glVertex3f( s, -s, s);
    
        // Front face  (z = 1.0f)
        glColor4f(r, g, b, a);     // Red
        glVertex3f( s,  s, s);
        glVertex3f(-s,  s, s);
        glVertex3f(-s, -s, s);
        glVertex3f( s, -s, s);
    
        // Back face (z = -1.0f)
        glColor4f(r, g, b, a);     // Yellow
        glVertex3f( s, s, -s);
        glVertex3f(-s, s, -s);
        glVertex3f(-s, s, -s);
        glVertex3f( s, s, -s);
    
        // Left face (x = -1.0f)
        glColor4f(r, g, b, a);     // Blue
        glVertex3f(-s, s,  s);
        glVertex3f(-s, s, -s);
        glVertex3f(-s, s, -s);
        glVertex3f(-s, s,  s);
    
        // Right face (x = 1.0f)
        glColor4f(r, g, b, a);    // Magenta
        glVertex3f( s,  s, -s);
        glVertex3f( s,  s,  s);
        glVertex3f( s, -s,  s);
        glVertex3f( s, -s, -s);
   glEnd();

   glPopMatrix();
}
void drawLine(float r, float g, float b, float a, float size, Node * one, Node * two)
{
    glPushMatrix();
    float x1, y1, z1, x2, y2, z2;


    x1 = scale * (one->x - (0.5 * nodeState->x))/(0.5 * nodeState->x);
    y1 = scale * (one->y - (0.5 * nodeState->y))/(0.5 * nodeState->y);
    z1 = scale * (one->z - (0.5 * nodeState->z))/(0.5 * nodeState->z);

    x2 = scale * (two->x - (0.5 * nodeState->x))/(0.5 * nodeState->x);
    y2 = scale * (two->y - (0.5 * nodeState->y))/(0.5 * nodeState->y);
    z2 = scale * (two->z - (0.5 * nodeState->z))/(0.5 * nodeState->z);

    glBegin(GL_LINES);
        glColor4f(r, g, b, a);
        glVertex3f(x1, y1, z1);
        glVertex3f(x2, y2, z2);
    glEnd();
    
    glPopMatrix();
}
void drawLines()
{
    Node * current = openHead;
    while (current != NULL)
    {
        Node * current2 = current;
        while (current2->parent != NULL)
        {
            drawLine(1.0f, 1.0f, 1.0f, 0.3f, 8.0f, current2, current2->parent);
            current2 = current2->parent;
        }
        current = current->prev;
    }
    current = lowest;
    while (current != NULL && current->parent != NULL)
    {
        drawLine(0.0f, 0.8f, 1.0f, 0.7f, 20.0f, current, current->parent);
        current = current->parent;
    }
}
void display() 
{
    if ((double)(clock() - clck)/CLOCKS_PER_SEC >= (double) 1.7 / SIZE)
    {
        pathStep();
        clck = clock();
    }

    if (keys[0]) rotY += 0.001 * SIZE;
    if (keys[1]) rotY -= 0.001 * SIZE;
    if (keys[2] && rotZ < 80 * M_PI/180) rotZ += 0.001 * SIZE;
    if (keys[3] && rotZ > -80 * M_PI/180) rotZ -= 0.001 * SIZE;
    if (keys[4] && distance < 100) distance += 0.1 * SIZE;
    if (keys[5] && distance > 10) distance -= 0.1 * SIZE;

    if (rotY >= 2 * M_PI) rotY = 0;


    camX = cos(rotZ) * cos(rotY) * distance;
    camY = sin(rotZ) * distance;
    camZ = cos(rotZ) * sin(rotY) * distance;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW); 
    glLoadIdentity();
    
    gluLookAt(camX, camY, camZ, 0,0,0, 0,1,0);

    for (int x = 0; x < nodeState->x; ++x)
    {
        for (int y = 0; y < nodeState->y; ++y)
        {
            for (int z = 0; z < nodeState->z; ++z)
            {
                float * state = nodeState->a[x][y][z];
                drawCube(state[0], state[1], state[2], state[3], scale * (x - (0.5 * nodeState->x))/(0.5 * nodeState->x), scale * (y - (0.5 * nodeState->y))/(0.5 * nodeState->y), scale * (z - (0.5 * nodeState->z))/(0.5 * nodeState->z), state[4]);
            }
        }
    }

    drawLines();
    
    glutSwapBuffers();
}
void reshape(GLsizei width, GLsizei height) {  // GLsizei for non-negative integer
    // Compute aspect ratio of the new window
    if (height == 0) height = 1;                // To prevent divide by 0
    if (!fullScreen)
    {
        oldX = (int) width;
        oldY = (int) height;
    }

    GLfloat aspect = (GLfloat)width / (GLfloat)height;
    
    // Set the viewport to cover the new window
    glViewport(0, 0, width, height);
    
    // Set the aspect ratio of the clipping volume to match the viewport
    glMatrixMode(GL_PROJECTION);  // To operate on the Projection matrix
    glLoadIdentity();             // Reset
    // Enable perspective projection with fovy, aspect, zNear and zFar
    gluPerspective(45.0f, aspect, 0.1f, 100.0f);
}
void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 'a':
        keys[0] = 1;
        break;
    case 'd':
        keys[1] = 1;
        break;
    case 'w':
        keys[2] = 1;
        break;
    case 's':
        keys[3] = 1;
        break;
    case '-':
        keys[4] = 1;
        break;
    case '=':
        keys[5] = 1;
        break;
    case '3':
        if (!fullScreen)
        {
            fullScreen = 1;
            glutFullScreen();
        } else
        {
            fullScreen = 0;
            glutReshapeWindow(oldX, oldY);
            
        }
        break;
    default:
        break;
    }
}
void keyboardUp(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 'a':
        keys[0] = 0;
        break;
    case 'd':
        keys[1] = 0;
        break;
    case 'w':
        keys[2] = 0;
        break;
    case 's':
        keys[3] = 0;
        break;
    case '-':
        keys[4] = 0;
        break;
    case '=':
        keys[5] = 0;
        break;
    default:
        break;
    }

}
int main(int argc, char** argv)
{
    // init window
    glutInit(&argc, argv);            // Initialize GLUT
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_MULTISAMPLE | GLUT_DEPTH); // Enable double buffered mode
    glutInitWindowPosition(50, 50); // Position the window's initial top-left corner
    glutInitWindowSize(oldX, oldY);
    glutCreateWindow("A* 3D");          // Create window with the given title
    // glutFullScreen();
    glutDisplayFunc(display);       // Register callback handler for window re-paint event
    glutReshapeFunc(reshape);       // Register callback handler for window re-size event
    glutKeyboardFunc(keyboard);
    glutKeyboardUpFunc(keyboardUp);
    glutIdleFunc(display);
    init();                       // Our own OpenGL initialization
    glutMainLoop();    

    return 0;
}