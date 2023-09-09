#include <stdio.h>   
#include <stdlib.h> 
#include <math.h>

int hadamard_transform(int m, int n, double *mat);
int abrideged_hadamard_transform(int m, int n, int d, double *mat);


int hadamard_transform(int m, int n, double *mat){
    // 2^d = m
    int d = 0;
    int step = 1;
    double x, y;
    int row1, row2;
    int ind1, ind2;

    while (step < m) {
        d++;
        step *= 2;
    }

    if (step != m) {
        // m is not a power of 2
        return 0;
    }

    for (int i = d-1; i > -1; i--){
        step /= 2;
        for (int j = 0; j < m; j += 2 * step){
            for (int k = j; k < j + step; ++k){
                row1 = k * n;
                row2 = (k + step) * n;
                for (int h = 0; h < n; h++){
                    // row number is k and k + step
                    // column number is h
                    ind1 = row1 + h;
                    ind2 = row2 + h;
                    x = mat[ind1];
                    y = mat[ind2];
                    mat[ind1] = x + y;
                    mat[ind2] = x - y;
                }
            }
        }
    }
    return 1;
}


int abridged_hadamard_transform(int m, int n, int d, double *mat){

    int deg = 0;
    int step = 1;
    double x, y;
    int row1, row2;
    int ind1, ind2; 

    while (step < m) {
        deg++;
        step *= 2;
    }

    if (step != m || deg < d) {
        // m is not a power of 2
        // or the recursion depth is greater than log_2 m
        return 0;
    }

    for (int i = d-1; i > -1; i--){
        step /= 2;
        for (int j = 0; j < m; j += 2 * step){
            for (int k = j; k < j + step; ++k){
                row1 = k * n;
                row2 = (k + step) * n;
                for (int h = 0; h < n; h++){
                    // row number is k and k + step
                    // column number is h
                    ind1 = row1 + h;
                    ind2 = row2 + h;
                    x = mat[ind1];
                    y = mat[ind2];
                    mat[ind1] = x + y;
                    mat[ind2] = x - y;
                }
            }
        }
    }
    return 1;
}
