#include <stdio.h>
#include <stdlib.h>

int ** create2DArray(int m, int n) {
    int * points = calloc(m * n, sizeof(int));
    int ** subs = malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i) {
        subs[i] = points + i * m;
    }
    return subs;
}

int ** genBoard(int size) {
    int ** board = create2DArray(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            board[i][j] = rand() % 100;
        }
    }
    return board;
}

void prettyPrint(int ** board, int size){
    for (int i = 0; i < size; ++i) {
        printf("\n");
        for (int j = 0; j < size; ++j) {
            printf("%d ", board[i][j]);
        }
    }
}

int main(){
    int size = 10;

    int ** board = genBoard(size);
    prettyPrint(board, size);
    return 0;
}