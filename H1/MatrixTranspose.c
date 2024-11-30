#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#define ITERATIONS 10      // Number of runs per thread count

FILE *fptr;
float averageMtSerial = 0;
float averageMtImplicit = 0;
float averageMtOMP = 0;

float averageSerialSymm = 0;
float averageImplicitSymm = 0;
float averageOMPSymm = 0;

float fRand(float fMin, float fMax)
{
    float f = (float)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void printMatrix(int SIZE, float **M)
{
    printf("\n");

    for(int i=0; i<SIZE;i++)
    {
        for(int j=0;j<SIZE;j++)
        {
            printf("[   %f   ]",M[i][j]);
        }
        printf("\n");
    }
}

float** matTranspose(int SIZE, float **M)
{
    float** TM = (float**) malloc (SIZE*sizeof(float*));
    double wt1,wt2;
    for(int i=0;i<SIZE;i++)
    {
        TM[i] = (float*) malloc (SIZE*sizeof(float));
    }
    //init transpose
    wt1 = omp_get_wtime();
    for(int i=0;i<SIZE;i++)
    {
        for(int j=0;j<SIZE;j++)
        {
            TM[i][j] = M[j][i];
        }
    }
    wt2 = omp_get_wtime();
    averageMtSerial += (wt2-wt1);
    return TM;
}

bool checkSym(int SIZE, float **M)
{
    double wt1,wt2;

    //init transpose
    bool symmetric=false;
    float checkNum = 0.0;
    wt1 = omp_get_wtime();

    for(int i=0;i<SIZE;i++)
    {
      for(int j=0;j<SIZE;j++)
        {
            checkNum += M[i][j] - M[j][i];
        }
    }
    wt2 = omp_get_wtime();
    if(checkNum == 0)
    {
        symmetric=true;
    }
    printf( "Serial time for checkSym = %.8g sec\n", (wt2-wt1));
    fprintf(fptr,"Serial time checkSym = %.8g sec\n", (wt2-wt1));
    averageSerialSymm += (wt2-wt1);

    return symmetric;
}

int MIN(int a, int b)
{
    if(a < b)
    {
        return a;
    }
    if(b < a)
    {
        return b;
    }
    return a;
}

bool checkSymImp(int SIZE, float **M)
{
    double wt1,wt2;
    bool symmetric=false;
    float checkNum = 0.0;
    //TILE SIZE Smaller as possible
    int TILE_SIZE = SIZE/16;
    
    wt1 = omp_get_wtime();
    for (int ii = 0; ii < SIZE; ii += TILE_SIZE)
    {
        for (int jj = 0; jj < SIZE; jj += TILE_SIZE)
        {
            for (int i = ii; i < MIN(SIZE, ii + TILE_SIZE); i++)
            {
                for (int j = jj; j < MIN(SIZE, jj + TILE_SIZE); j++)
                {
                    checkNum += M[i][j] -M[j][i];
                }
            }
        }
    }
    wt2 = omp_get_wtime();

    if(checkNum ==0)
    {
        symmetric=true;
    }
    printf( "Implicit time for checkSym = %.8g sec\n", (wt2-wt1));
    fprintf(fptr,"Implicit time checkSym = %.8g sec\n", (wt2-wt1));
    averageImplicitSymm += (wt2-wt1);

    return symmetric;
}

float**  matTransposeImp(int SIZE, float **M)
{
    float** TM = (float**) malloc (SIZE*sizeof(float*));
    double wt1,wt2;
    for(int i=0;i<SIZE;i++)
    {
        TM[i] = (float*) malloc (SIZE*sizeof(float));
    }

    int TILE_SIZE = SIZE/16;
    //init transpose
    wt1 = omp_get_wtime();
    for (int ii = 0; ii < SIZE; ii += TILE_SIZE)
    {
        for (int jj = 0; jj < SIZE; jj += TILE_SIZE)
        {
            for (int i = ii; i < MIN(SIZE, ii + TILE_SIZE); i++)
            {
                for (int j = jj; j < MIN(SIZE, jj + TILE_SIZE); j++)
                {
                    TM[i][j] = M[j][i];
                }
            }
        }
    }
    wt2 = omp_get_wtime();

    averageMtImplicit += (wt2-wt1);
    return TM;
}

float** matTransposeOMP(int SIZE, float **M)
{
    float** TM = (float**) malloc (SIZE*sizeof(float*));
    double wt1,wt2;
    for(int i=0;i<SIZE;i++)
    {
        TM[i] = (float*) malloc (SIZE*sizeof(float));
    }

    //init transpose
    wt1 = omp_get_wtime();
#pragma omp parallel for schedule(guided,1) collapse(2)
    for(int i=0;i<SIZE;i++)
    {
        for(int j=0;j<SIZE;j++)
        {
            TM[i][j] = M[j][i];
        }
    }
    wt2 = omp_get_wtime();

    averageMtOMP += (wt2-wt1);
    return TM;
}

bool checkSymOMP(int SIZE, float **M)
{
    double wt1,wt2;

   bool symmetric=false;
   float checkNum = 0.0;
    
   wt1 = omp_get_wtime();
#pragma omp parallel for collapse(2) reduction(+:checkNum) shared(M)
    for(int i=0;i<SIZE;i++)
    {
      for(int j=0;j<SIZE;j++)
      {
        checkNum += M[i][j] - M[j][i];
      }
    }
    wt2 = omp_get_wtime();
    
    if(checkNum == 0)
    {
        symmetric = true;
    }
    averageOMPSymm += (wt2-wt1);
    
    return symmetric;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "Please insert an integer number as parameter: %s <integer_number>\n", argv[0]);
        return -1;
    }
    
    int N = atoi(argv[1]);
    float** M = (float**) malloc (N*sizeof(float*));
        float** TM1 = (float**) malloc (N*sizeof(float*));
    float** TM2 = (float**) malloc (N*sizeof(float*));
    float** TM3 = (float**) malloc (N*sizeof(float*));

   
    for(int i=0;i<N;i++)
    {
        M[i] = (float*) malloc (N*sizeof(float));
    }
    
    //1. Initialize a random n × n matrix M of floating-point numbers.
    srand(time(NULL));
    for(int i=0; i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            M[i][j] = fRand(0.0,1000.0);
        }
    }
    
    fptr = fopen("Results.csv","a");
    if(fptr==NULL)
    {
        perror("Error while opening the file.\n");
        return 1;
    }
    
    fprintf(fptr,"Matrix: %d X %d\n",N,N);

    //Test to get average serial time:
    if(!checkSym(N,M))
    {

        //Not symmetric so i can do transpose
        for(int i=0;i<ITERATIONS;i++)
        {
            TM1=matTranspose(N,M);
        }
        double avgSerial = averageMtSerial/ITERATIONS;
        fprintf(fptr,"Average Serial matrix transpose time = %.8g sec\n",avgSerial);
        printf("Average Serial matrix transpose time = %.8g sec\n",avgSerial);

        printf("------------------------------------IMPLICIT EXECUTION--------------------------------\n");

        if(!checkSymImp(N,M))
        {
            for(int i=0;i<ITERATIONS;i++)
            {
                TM2=matTransposeImp(N,M);
            }
            double avgImplicit = averageMtImplicit/ITERATIONS;
            fprintf(fptr,"Average Implicit matrix transpose time = %.8g sec\n",avgImplicit);
            printf("Average Implicit matrix transpose time = %.8g sec\n",avgImplicit);
        }
        else
        {
            printf("Matrix is symmetrix so it's already transposed!\n");
        }
        
        printf("------------------------------------EXPLICIT EXECUTION--------------------------------\n");
        printf("                            matTransposeOMP data                                                        checkSymOMP data\n");

        //Test performance with num_threads as parameter

        fprintf(fptr, "Num_Threads,Avg_Parallel_Time,Avg_Speedup,Avg_Efficiency\n");
        for(int num_threads = 1; num_threads <=64; num_threads *= 2)
        {
            omp_set_num_threads(num_threads);
            for(int i=0; i<ITERATIONS; i++)
            {
                if(!checkSymOMP(N,M))
                {
                    TM3=matTransposeOMP(N,M);
                }
                else
                {
                    printf("Matrix is symmetrix so it's already transposed!\n");
                }
            }
            //matTranspose
            double avgParallelTime = averageMtOMP/ITERATIONS;
            double avgSpeedUp = avgSerial / avgParallelTime;
            double avgEfficiency = avgSpeedUp / num_threads;
            
            //checkSym
            double avgParallelTimeS = averageOMPSymm/ITERATIONS;
            double avgSpeedUpS = averageSerialSymm / avgParallelTimeS;
            double avgEfficiencyS = avgSpeedUpS / num_threads;
            
            printf("|Num_Threads  |Avg_Parallel_Time |Avg_Speedup      |Avg_Efficiency                  |Avg_Parallel_Time |Avg_Speedup      |Avg_Efficiency\n");
            fprintf(fptr,"%d\t%.8g\t%.8g\t%.0f%%\n", num_threads, avgParallelTime, avgSpeedUp, (avgEfficiency * 100));
            printf("|    %d        |   %.8g    |   %.8g      |  %.0f %%                           |   %.8g    |   %.8g      |%.0f %%\n", num_threads, avgParallelTime, avgSpeedUp, (avgEfficiency * 100),avgParallelTimeS, avgSpeedUpS, (avgEfficiencyS * 100));
            
            averageMtOMP = 0.0;
            averageOMPSymm = 0.0;
        }
        
        for(int i=0; i<N; i++)
        {
            for(int j=0; j<N; j++)
            {
                if(TM1[i][j] != TM2[i][j] || TM1[i][j] != TM3[i][j] || TM2[i][j] != TM3[i][j])
                {
                    printf("Error! Transpose matrixes are different! \n");
                }
            }
        }
    }
    else
    {
        printf("Matrix is symmetrix so it's already transposed!\n");
    }
    
    fclose(fptr);

    for(int i=0;i<N;i++)
    {
        free(M[i]);
        free(TM1[i]);
        free(TM2[i]);
        free(TM3[i]);
    }
    free(M);
    free(TM1);
    free(TM2);
    free(TM3);
    
    return 0;
}
