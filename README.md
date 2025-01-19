# ParallelComputing
This repository contains the projects related to the "Introduction to Parallel Computing" course in University of Trento.

INSTRCTIONS TO REPRODUCE THE RESULTS OF H1:
-It's sufficient to connect to the University of Trento's cluster (HPC), clone the repository,open the file "script_matTranspose_C.pbs" and change the following line under the comment "# Select the working directory" with /home/your username/ParallelComputing/H1  
-Type the following command inside the directory H1: qsub script_matTranspose_C.pbs
-Check when it's finished with qstat -u username
-Type "cat matTransposeJob.o" for all the results, or "cat Results.csv" (checkSymOMP excluded).


INSTRUCTIONS TO REPRODUCE THE RESULTS OF H2:
-For replicating the results in your local machine, it is sufficient to clone the repository, having installed mpich version 3.2.1 of further, having installed gcc-9.1.0 or
further, and replacing in the code if having an mpich version different from mpich-3.2.1--gcc-9.1.0 the #include<omp.h> library with the path to the library in your gcc folder.
-For replicating the results in the University of Trento cluster, first clone the repoistory, then open the file "mpiC.pbs" and change the following line 
comment "# Select the working directory" with /home/your username/ParallelComputing/H2
-If you want to launch the program inside an interactive first connect to it with qsub -I -q short_cpuQ -l select=1:ncpus=64:mpiprocs=64:mem=1Gb, then type module load mpich-3.2.1.--gcc-9.1.0 and compile the file "MatrixTranspositionMPI.c" with mpicc -fopenmp -o program MatrixTransposeMPI.c, then run it with mpirun -np n_processors ./program SIZE.
-If you want to run using the pbs file type the following command inside the directory H2: qsub mpiC.pbs, then check when it's finished with qstat -u username and 
finally type cat matTransposeJob.o" for all the results, or "cat Results.csv".

