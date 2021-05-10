# mean-shift-in-parallel
Mean shift [algorithm](https://en.wikipedia.org/wiki/Mean_shift) implemented in sequential and parallel methods for study purposes.

<p align=center>
    <img src="https://latex.codecogs.com/gif.latex?m(x)&space;=&space;\frac{\sum_{x_i&space;\in&space;N(x)}&space;K(x_i&space;-&space;x)&space;x_i}{\sum_{x_i&space;\in&space;N(x)}&space;K(x_i&space;-&space;x)}" title="m(x) = \frac{\sum_{x_i \in N(x)} K(x_i - x) x_i}{\sum_{x_i \in N(x)} K(x_i - x)}" />
</p>

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
│   ├── include                   # folder with header files
│   │   ├── meanShift.hpp
│   │   ├── utils.hpp
│   │   ├── geometricFunction.hpp 
├── OpenMp               # handled by @AngeloDamante
│   ├── CMakeLists.txt        # C make file
│   ├── main.cpp              # main parallel source file
│   ├── include                   # folder with header files
│   │   ├── meanShiftOmp.hpp
│   │   ├── utils.hpp
│   │   ├── geometricFunction.hpp 
```


## Sequential version
Implementation of mean-shift algorithm in sequential version.

```
cd Sequential 
mkdir build
cd build
cmake .. && make
./ms_sequential
```


## Parallel OpenMP version
Implementation of parallel version with openMp fork-join paradigm. For this version, two versions were implemented, <i>static</i> and <i>dynamic</i>.
```
cd OpenMp
mkdir build
cd build
cmake .. && make
./ms_static     # static version
./ms_dynamic    # dynamic version
```


## Parallel CUDA version

Proposed version was made in naive and shared memory version.

In order to run its execution:

```
cd CUDA/MS_CUDA
./run.sh
```

Inside CUDA/MS_CUDA one can also find a ./build.sh file that creates a bild folder and builds the projects.

## Authors
+ <a href="https://github.com/AngeloDamante"> Angelo D'Amante </a>
+ <a href="https://github.com/fabian57fabian"> Fabian Greavu </a>