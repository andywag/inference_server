# Graphcore Bert Inference Example

This is an example of BERT PopArt Inference on the IPUs. The design uses loadgen from MLPerf to drive the design and report the performance. 

## Prerequisites

This project depend on loadgen used for MLPerf. Please follow the direction below for installation

- https://github.com/mlcommons/inference/blob/master/loadgen/README_BUILD.md

The project also depends on an external model and external processing files. 

- run make : For downoading

## Commands

The following commands will run the system for a few configurations. 

- 2 IPU : python run_loadgen.py [--accuracy] [--server] 
- 4 IPU : mpirun -np 2 --tag-output --bind-to numa --map-by slot python run_loadgen.py --mpi [--accuracy] [--server] 
- 16 IPU : mpirun -np 9 --tag-output --bind-to numa --map-by slot python run_loadgen.py --mpi --mpi_load [--accuracy] [--server] 

The accuracy option will run a short test showing the accuracy. 
The server option will run the test in server mode. 


## License

Apache License 2.0
