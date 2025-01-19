#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdbool.h>
#define ITERATIONS 20      // Number of runs per thread count

FILE *fptr;
float totMtSerial = 0;
float totMtMPI = 0;
float totSerialSymm = 0;
float totMPISymm = 0;
float totMtOMP = 0;
float avgSerial = 0;

float fRand(float fMin, float fMax)
{
    float f = (float)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void printMatrixContiguous(int SIZE, float (*M)[SIZE])
{
    printf("\n");
    for(int i=0;i<SIZE;i++)
    {
        for(int j=0;j<SIZE;j++)
        {
            printf("[   %f   ]",M[i][j]);
        }
        printf("\n");
    }
}

void printMatrixNotContiguous(int SIZE, float **M)
{
    printf("\n");
    for(int i=0;i<SIZE;i++)
    {
        for(int j=0;j<SIZE;j++)
        {
            printf("[   %f   ]",M[i][j]);
        }
        printf("\n");
    }
}

float** matTranspose(int SIZE, float (*M)[SIZE])
{
    float** TM = (float**) malloc (SIZE*sizeof(float*));
    for(int i=0;i<SIZE;i++)
    {
        TM[i] = (float*) malloc (SIZE*sizeof(float));
    }
    double wt1,wt2;
    //init transpose
    wt1 = MPI_Wtime();
    for(int i=0;i<SIZE;i++)
    {
        for(int j=0;j<SIZE;j++)
        {
            TM[i][j] = M[j][i];
        }
    }
    wt2 = MPI_Wtime();
    
   // printf( "Serial time for matTranspose = %.8g sec\n", (wt2-wt1));
    totMtSerial += (wt2-wt1);
    return TM;
}

bool checkSym(int SIZE, float (*M)[SIZE])
{
    double wt1,wt2;

    //init transpose
    bool symmetric=false;
    float checkNum = 0.0;
    
    wt1 = MPI_Wtime();
    for(int i=0;i<SIZE;i++)
    {
      for(int j=0;j<SIZE;j++)
        {
            checkNum += M[i][j] - M[j][i];
        }
    }
    wt2 = MPI_Wtime();
    
    if(checkNum == 0)
    {
        symmetric=true;
    }
   // printf( "Serial time for checkSym = %.8g sec\n", (wt2-wt1));
  // fprintf(fptr,"Serial time checkSym = %.8g sec\n", (wt2-wt1));
    totSerialSymm += (wt2-wt1);

    return symmetric;
}

float**  matTransposeMPI(int SIZE, float (*M)[SIZE],int portion, int rank, int nProcs)
{
    float** TM1 = (float**) malloc (SIZE*sizeof(float*));
    for(int i=0;i<SIZE;i++)
    {
        TM1[i] = (float *) malloc (SIZE*sizeof(float*));
    }
    float (*TM)[SIZE] = malloc(sizeof(*TM) * SIZE);

    double wt1,wt2;
    float *localRows = (float*) malloc(SIZE*portion*sizeof(float));

    MPI_Datatype rows;
    MPI_Type_contiguous(portion*SIZE, MPI_FLOAT, &rows);
    MPI_Type_commit(&rows);
    wt1 = MPI_Wtime();
    MPI_Scatter(M, 1, rows, localRows, 1, rows, 0, MPI_COMM_WORLD);
    
    int start = portion * rank;
    int end = start + portion;
        
    //Perform the transposition
    int k=0;
    for(int i=start;i<end;i++)
    {
        for(int j=0;j<SIZE;j++)
        {
            localRows[k] = M[j][i];
            k+=1;
        }
    }
    //Gather the result
    MPI_Gather(localRows, 1, rows, TM, 1, rows, 0, MPI_COMM_WORLD);
    wt2 = MPI_Wtime();

    if(rank==0)
    {
        //printMatrixContiguous(SIZE, TM);
        totMtMPI += (wt2-wt1);

        for(int i=0;i<SIZE;i++)
        {
            for(int j=0; j<SIZE;j++)
            {
                TM1[i][j] = TM[i][j];
            }
        }
    }
        
    return TM1;
}

bool checkSymMPI(int SIZE, float (*M)[SIZE],int portion, int rank)
{
    double wt1,wt2;
  
    bool symmetric=false;
    float localCheck = 0.0;
    float globalCheck = 0.0;
    
    wt1 = MPI_Wtime();
    int start = portion * rank;
    int end = start + portion;
        
    for(int i=start;i<end;i++)
    {
        for(int j=0;j<SIZE;j++)
        {
            localCheck += M[i][j] - M[j][i];
        }
    }

    MPI_Reduce(&localCheck, &globalCheck, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    wt2 = MPI_Wtime();

    if(rank == 0)
    {
        if(globalCheck==0)
        {
            //printf("Matrix is symmetric!\n");
            symmetric = true;
        }
        else
        {
            //printf("Matrix is not symmetric!\n");
        }
        totMPISymm += (wt2-wt1);
    }
    MPI_Bcast(&symmetric,1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    return symmetric;
}

float** matTransposeOMP(int SIZE, float (*M)[SIZE], int size)
{
    float** TM = (float**) malloc (SIZE*sizeof(float*));
    double wt1,wt2;
    for(int i=0;i<SIZE;i++)
    {
        TM[i] = (float*) malloc (SIZE*sizeof(float));
    }
  
    //init transpose
    wt1 = MPI_Wtime();
#pragma omp parallel for schedule(guided,1) collapse(2) num_threads(size)
    for(int i=0;i<SIZE;i++)
    {
        for(int j=0;j<SIZE;j++)
        {
            TM[i][j] = M[j][i];
        }
    }
    wt2 = MPI_Wtime();

    totMtOMP += (wt2-wt1);
    return TM;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "Please insert an integer number as parameter: %s <integer_number>\n", argv[0]);
        return -1;
    }
    
    int N = atoi(argv[1]);
    
     MPI_Init(&argc, &argv);
    int rank, nProcs, portion,r, wSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    
    if ((r = N % nProcs))
    {
        if(rank==0)
        {
            printf("The number %d in input must be equally divided by the choosen number of processors! (%d processors here).\n", N, nProcs);
        }
    }
    else
    {
        //matrix defined to be contiguous in memory
        float (*M)[N] = malloc(sizeof(*M) * N);
        float (*M1)[wSize] = malloc(sizeof(*M1) * wSize);
        float** TM1 = (float**) malloc (N*sizeof(float*));
        float** TM2 = (float**) malloc (N*sizeof(float*));
        float** TM3 = (float**) malloc (N*sizeof(float*));
        
        wSize = N*nProcs;

        if(rank==0)
        {
            //1. Initialize a random n × n matrix M of floating-point numbers.
            srand(time(NULL));
            for(int i=0;i<N;i++)
            {
                for(int j=0;j<N;j++)
                {
                    M[i][j] = fRand(0.0,10000.0);
                }
            }
           
            //printMatrixContiguous(N,M);
            printf("Matrix: %d X %d with %d processors\n",N,N, nProcs);
            fptr = fopen("Results.csv","a");
            if(fptr==NULL)
            {
                perror("Error while opening the file.\n");
                return 1;
            }
            
            fprintf(fptr,"\n\n\nMatrix: %d X %d\n",N,N);

            //Serial Transpose
            if(!checkSym(N,M))
            {
                printf("Matrix is not symmetric!\n");
                
                for(int i=0;i<ITERATIONS;i++)
                {
                    TM1 = matTranspose(N,M);
                }
                avgSerial = totMtSerial / ITERATIONS;
                double avgSerialSymm = totSerialSymm / ITERATIONS;
                printf("AVERAGE SERIAL TIME TRANSPOSE: %.8g sec\n",avgSerial);
                printf("AVERAGE SERIAL TIME SYMMETRY CHECK: %.8g sec\n",avgSerialSymm);
                fprintf(fptr,"AVERAGE SERIAL TIME MT: %.8g sec\n",avgSerial);
                fprintf(fptr,"AVERAGE SERIAL TIME SYMMETRY CHECK: %.8g sec\n",avgSerialSymm);

            }
            else
            {
                printf("Matrix is already transposed!\n");
            }
            
            //printMatrixNotContiguous(N,TM1);
            
            
            printf("\t\t\t\t\t\t\tMPI EXECUTION\n");
        }
        //Broadcast the matrix to all the processes
        MPI_Bcast(M,N*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
      
        //Start to divide the rows and columns among the processes
        portion = N/nProcs;
        
        //Tell the other process the result of the symm from rank 0
        if(!checkSymMPI(N,M,portion,rank))
        {
            for(int i=0;i<ITERATIONS;i++)
            {
                TM2 = matTransposeMPI(N,M,portion,rank, nProcs);
            }
            
            if(rank==0)
            {
                
                //OMP Transpose for comparison purpouse
                printf("\t\t\t\t\t\t\tOMP EXECUTION\n");

                for(int i=0;i<ITERATIONS;i++)
                {
                    TM3 = matTransposeOMP(N,M,nProcs);
                }
                float averageSymm = totMPISymm / ITERATIONS;
                
                float averageMPI = totMtMPI / ITERATIONS;
                float averageOMP = totMtOMP / ITERATIONS;
                
                float averageSpeedUpMPI = avgSerial / averageMPI;
                float averageSpeedUpOMP = avgSerial / averageOMP;
                
                float averageEfficiencyMPI = averageSpeedUpMPI/nProcs;
                float averageEfficiencyOMP = averageSpeedUpOMP/nProcs;
                
                printf("AVERAGE SYMMETRY CHECK PARALLEL TIME: %.8g sec\n",averageSymm);
                fprintf(fptr,"AVERAGE SYMMETRY CHECK PARALLEL TIME: %.8g sec\n",averageSymm);

                printf("AVERAGE MPI TRANPOSE PARALLEL TIME: %.8g sec\n",averageMPI);
                printf("AVERAGE OMP TRANSPOSE PARALLEL TIME: %.8g sec\n",averageOMP);
                printf("AVERAGE SPEEDUP MPI: %f\n",averageSpeedUpMPI);
                printf("AVERAGE SPEEDUP OMP: %f\n",averageSpeedUpOMP);
                printf("AVERAGE EFFIENCY MPI: %f\n",(averageEfficiencyMPI * 100));
                printf("AVERAGE EFFIENCY OMP: %f\n",(averageEfficiencyOMP * 100));
         
                fprintf(fptr,"Num_Procs/Threads,Avg_Parallel_Time_MPI,Avg_Parallel_Time_OMP,Avg_Speedup_MPI,Avg_Speedup_OMP,Avg_Efficiency_MPI,Avg_Efficiency_OMP\n");
                fprintf(fptr,"%d\t%.8g\t%.8g\t%f\t%f\t%.0f%%\t%.0f%%\n\n", nProcs,averageMPI,averageOMP,averageSpeedUpMPI,averageSpeedUpOMP,(averageEfficiencyMPI*100),(averageEfficiencyOMP*100));

                for(int i=0;i<N;i++)
                {
                    for(int j=0;j<N;j++)
                    {
                        if(TM1[i][j] != TM2[i][j] || TM1[i][j] != TM3[i][j] || TM2[i][j] != TM3[i][j])
                        {
                            printf("Error! Transpose matrixes are different! \n");
                        }
                    }
                }
            }
            
            //Clean up and allocating for weak scaling
            float** TM4 = (float**) malloc (wSize*sizeof(float*));
            
            free(M);
            for(int i=0;i<N;i++)
            {
                free(TM1[i]);
                free(TM2[i]);
                free(TM3[i]);
            }
            free(TM2);
            free(TM1);
            free(TM3);
        
            if(wSize <= 8192)
            {
                if(rank==0)
                {
                    for(int i=0;i<wSize;i++)
                    {
                        for(int j=0;j<wSize;j++)
                        {
                            M1[i][j] = fRand(0.0,10000.0);
                        }
                    }
                    printf("Size for the weak scaling:%d\n",wSize);
                }
                
                MPI_Bcast(M1,wSize*wSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
                totMtMPI = 0;
                
                for(int i=0;i<ITERATIONS;i++)
                {
                    portion = wSize/nProcs;
                    TM4 = matTransposeMPI(wSize,M1,portion,rank, nProcs);
                }
                
                if(rank==0)
                {
                    float averageMPI = totMtMPI / ITERATIONS;
                    float averageWScaling = avgSerial / averageMPI;
                    printf("AVERAGE WEAK SCALING: %f\n", averageWScaling);
                    fprintf(fptr,"WEAK SCALING FACTOR WITH NEW N=%d IS %f\n",wSize, averageWScaling);
                    free(M1);
                    for(int i=0;i<N;i++)
                    {
                        free(TM4[i]);
                    }
                    fclose(fptr);
                }
            }
            else if(rank==0)
            {
                fclose(fptr);
            }
            free(TM4);
        }
    }
 
    MPI_Finalize();
    return 0;
}
