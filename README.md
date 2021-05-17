# mean-shift-in-parallel
Mean shift [algorithm](https://en.wikipedia.org/wiki/Mean_shift) implemented in sequential and parallel methods for study purposes.

The parallel optimized part will be the weight mean re-estimation as following:
<p align=center>
   <img src="https://latex.codecogs.com/gif.latex?\bg_white&space;m(x)=&space;\frac{\sum_{x_i&space;\in&space;N(x)}&space;K(x_i&space;-&space;x)&space;x_i}{\sum_{x_i&space;\in&space;N(x)}&space;K(x_i&space;-&space;x)}" title="m(x)= \frac{\sum_{x_i \in N(x)} K(x_i - x) x_i}{\sum_{x_i \in N(x)} K(x_i - x)}" />
</p>
Where K is a gaussian kernel:
<p align=center>
   <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/5f6ba53303ba0848b88b92e129b6f2279f397762" title="Gaussian kernel function" style="background:white;" />
</p>
And centroid reduction parti will be sequential.



### Project tree

```bash 
├── datas
│   ├── 1000            # folder with 1000 points and 3 centroids
│   │   ├── points.csv # points dataset
│   │   ├── centroids.csv # centroids dataset
│   ├── ...  
├── CUDA           
│   ├── DockerFile           # Docker manaer file
│   ├── docker-compose.yaml  # Docker compose file
│   ├── MS_CUDA                   # handled by @fabian57fabian
│   │   ├── CMakeLists.txt        # C make file
│   │   ├── ...                   # libs and headers
│   │   ├── main.cu               # main CUDA source file
│   │   ├── utils.h               # utility file
│   │   ├── build.sh              # builds the MS_CUDA project
│   │   ├── execute_mean_shift.sh            # executes the mean shift algo (naive and shared versions)
│   │   ├── run_mean_shift.sh                # builds + executes
├── Sequential            # handled by @AngeloDamante
│   ├── CMakeLists.txt        # C make file
│   ├── main.cpp              # main sequential source file
│   ├── include                   # folder with header files
│   │   ├── meanShift.hpp
│   │   ├── utils.hpp
│   │   ├── geometricFunction.hpp 
├── OpenMp               # handled by @AngeloDamante
│   ├── CMakeLists.txt        # C make file
│   ├── static.cpp             # parallel main of static version
│   ├── dynamic.cpp             # parallel main of dynamic version
│   ├── include                   # folder with header files
│   │   ├── meanShiftOmp.hpp
│   │   ├── utils.hpp
│   │   ├── geometricFunction.hpp 
```


## Sequential version
Implementation of mean-shift algorithm in sequential version.

```
cd Sequential 
./run.sh
```


## Parallel OpenMP version
Implementation of parallel version with openMp fork-join paradigm. For this version, two versions were implemented, <i>static</i> and <i>dynamic</i>.

To run static version,
```
cd OpenMp
./run_static_version.sh
```

and for dynamic version
```
cd OpenMp
./run_dynamic_version.sh
```

## Parallel CUDA version

Proposed version was made in naive and shared memory version.

In order to run its execution:

```
cd CUDA/MS_CUDA
./run_mean_shift.sh     # Use argument 'info' to view only naive and shared timings
```

It will run the naive and the shared memory versions.

Inside CUDA/MS_CUDA one can also find a ./build.sh file that creates a bild folder and builds the projects.

IF you already runned or build it, you can call the execute bash script.

In order to run all tests, execute:
```
cd CUDA/MS_CUDA
./create_all_timimgs.sh    # will print naive and shared memory timings for each data
```

## Authors
+ <a href="https://github.com/AngeloDamante"> Angelo D'Amante </a>
+ <a href="https://github.com/fabian57fabian"> Fabian Greavu </a>
