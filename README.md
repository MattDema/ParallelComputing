# ParallelComputing
This repository contains the projects related to the "Introduction to Parallel Computing" of University of Trento.

INSTRCTIONS TO REPRODUCE THE RESULTS:
-It's sufficient to connect to the University of Trento's cluster (HPC), clone the repository, type the following command: qsub script_matTranspose_C.pbs
-Check when it's finished with qstat -u username
-Type "cat matTransposeJob.o" for all the results, or "cat Results.csv" (checkSymOMP excluded).
