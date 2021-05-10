#include <iostream>
#include <omp.h>

int main(int argc, char const *argv[])
{
    int nthreads, tid;

#pragma omp parallel private(nthreads, tid)
    {
        tid = omp_get_thread_num();
        printf("Hello from thread %d \n", tid);

        if (tid == 0)
        {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d \n", nthreads);
        }
    }

    return 0;
}
