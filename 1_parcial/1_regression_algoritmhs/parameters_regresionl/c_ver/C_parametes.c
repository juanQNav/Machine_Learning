#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROW 51
#define COLUMN 2
#define MAX_BUFFER_SIZE 256

void fill_column_ones(double matrix[ROW][COLUMN])
{
    for (int i = 0; i < ROW; i++)
        matrix[i][0] = 1;
}

void print_matrix_bidi(int row, int col, double mat[row][col])
{
    for (int i = 0; i < row; i++)
    {
        printf("[");
        for (int j = 0; j < col; j++)
        {
            if (j == col - 1)
                printf("%.2lf", mat[i][j]);
            else
                printf("%.2lf, ", mat[i][j]);
        }
        printf("]\n");
    }
}

void print_matrix_uni(int row, double mat[row])
{
    printf("[");
    for (int i = 0; i < row; i++)
        if (i == row - 1)
            printf("%.6lf", mat[i]);
        else
            printf("%.6lf,", mat[i]);
    printf("]\n");
}

void matrix_transpose(double matrix[ROW][COLUMN], double matrixT[COLUMN][ROW])
{
    for (int i = 0; i < COLUMN; i++)
    {
        for (int j = 0; j < ROW; j++)
        {
            matrixT[i][j] = matrix[j][i];
        }
    }
}

void matrix_multiplication(int row1, int col1, double mat1[row1][col1], int row2, int col2, double mat2[row2][col2], double matRes[row1][col2])
{
    for (int i = 0; i < row1; i++)
    {
        for (int j = 0; j < col2; j++)
        {
            matRes[i][j] = 0; // Initialize to zero before accumulating values
            for (int k = 0; k < col1; k++)
            {
                matRes[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
}
void matrix_vector_multiplication(int rows, int cols, double matrix[rows][cols], double vector[cols], double result[rows])
{
    for (int i = 0; i < rows; i++)
    {
        result[i] = 0;
        for (int j = 0; j < cols; j++)
        {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}
void matrix_inverse(double mat[COLUMN][COLUMN], double matRes[COLUMN][COLUMN])
{
    double determinant = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    if (determinant != 0)
    {
        matRes[0][0] = mat[1][1];
        matRes[0][1] = -mat[0][1];
        matRes[1][0] = -mat[1][0];
        matRes[1][1] = mat[0][0];

        double inverseFactor = 1.0 / determinant;
        matRes[0][0] *= inverseFactor;
        matRes[0][1] *= inverseFactor;
        matRes[1][0] *= inverseFactor;
        matRes[1][1] *= inverseFactor;
    }
}

int main()
{
    // open the file
    FILE *f_csv = fopen("./dataset/accidents.csv", "r");
    if (f_csv == NULL)
    {
        printf("Error: could not open file\n");
        exit(-1);
    }

    // get the values
    double x[ROW][COLUMN] = {0};
    double y[ROW] = {0};

    fill_column_ones(x);

    int count = 0, index_x = 0, index_y = 0;

    char buffer[MAX_BUFFER_SIZE];

    // read the first line "x,y"
    fgets(buffer, sizeof(buffer), f_csv);

    // read a complete line
    while (fgets(buffer, sizeof(buffer), f_csv) != NULL)
    {
        char *token = strtok(buffer, ",");
        while (token != NULL)
        {
            double number = strtod(token, NULL);
            if (count % 2 == 0)
            {
                x[index_x][1] = number;
                index_x++;
            }
            else
            {
                y[index_y] = number;
                index_y++;
            }
            count++;
            token = strtok(NULL, ",");
        }
    }
    printf("----------------X\n");
    print_matrix_bidi(ROW, COLUMN, x);
    printf("----------------Y\n");
    print_matrix_uni(ROW, y);

    double xT[COLUMN][ROW] = {{0}};

    matrix_transpose(x, xT);

    printf("----------------Matrix transpose\n");
    print_matrix_bidi(COLUMN, ROW, xT);

    double xT_mult_X[COLUMN][COLUMN] = {{0}};

    matrix_multiplication(COLUMN, ROW, xT, ROW, COLUMN, x, xT_mult_X);
    printf("----------------Res Mult xT * x\n");
    print_matrix_bidi(COLUMN, COLUMN, xT_mult_X);

    double xT_mult_X_inverse[COLUMN][COLUMN] = {{0}};

    matrix_inverse(xT_mult_X, xT_mult_X_inverse);
    printf("----------------Mat Inverse (xT * x) ^-1\n");
    print_matrix_bidi(COLUMN, COLUMN, xT_mult_X_inverse);

    double xT_mult_X_inver_mult_xT[COLUMN][ROW] = {{0}};

    printf("----------------Res Mult (xT * x)^-1 * xT\n");
    matrix_multiplication(COLUMN, COLUMN, xT_mult_X_inverse, COLUMN, ROW, xT, xT_mult_X_inver_mult_xT);
    print_matrix_bidi(COLUMN, ROW, xT_mult_X_inver_mult_xT);

    double B[COLUMN] = {0};

    printf("----------------Res Mult (xT * x)^-1 * xT * Y\n");
    matrix_vector_multiplication(COLUMN, ROW, xT_mult_X_inver_mult_xT, y, B);
    print_matrix_uni(COLUMN, B);

    fclose(f_csv);
    return 0;
}
