# mean-shift-in-parallel
Mean shift algorithm implemented in sequential and parallel methods for study purposes.

### Project tree

```bash 
├── datas
│   ├── 1000            # folder with 1000 points and 3 centroids
│   │   ├── random_pts_1k.csv # points dataset
│   │   ├── random_cts_1k.csv # centroids dataset
|   ├── ...
├── CUDA           
│   ├── DockerFile           # Docker manaer file
│   ├── docker-compose.yaml  # Docker compose file
│   ├── MS_CUDA                   # handled by @fabian57fabian
│   │   ├── CMakeLists.txt        # C make file
│   │   ├── ...                   # libs and headers
│   │   ├── main.cu               # main CUDA source file
│   │   ├── build.sh              # builds the MS_CUDA project
│   │   ├── execute.sh            # executes the shared memory file
│   │   ├── run.sh                # builds + executes
├── Sequential            # handled by @AngeloDamante
│   ├── CMakeLists.txt        # C make file
│   ├── main.cpp              # main sequential source file
```


## Sequential version
under matainance


## Parallel OpenMP version
under matainance


## Parallel CUDA version

Proposed version was made in naive and shared memory version.

In order to run its execution:

```
cd CUDA/MS_CUDA
./run.sh
```

Inside CUDA/MS_CUDA one can also find a ./build.sh file that creates a bild folder and builds the projects.