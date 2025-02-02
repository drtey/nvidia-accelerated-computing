#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"
#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

__global__ void bodyForce(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float Fx = 0.0f; 
        float Fy = 0.0f; 
        float Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3; 
            Fy += dy * invDist3; 
            Fz += dz * invDist3;
        }

        p[i].vx += dt*Fx; 
        p[i].vy += dt*Fy; 
        p[i].vz += dt*Fz;
    }
}

__global__ void integratePositions(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i].x += p[i].vx*dt;
        p[i].y += p[i].vy*dt;
        p[i].z += p[i].vz*dt;
    }
}

int main(const int argc, const char** argv) {
    int nBodies = 2<<11;
    if (argc > 1) nBodies = 2<<atoi(argv[1]);

    const char * initialized_values;
    const char * solution_values;
    
    if (nBodies == 2<<11) {
        initialized_values = "09-nbody/files/initialized_4096";
        solution_values = "09-nbody/files/solution_4096";
    } else {
        initialized_values = "09-nbody/files/initialized_65536";
        solution_values = "09-nbody/files/solution_65536";
    }
    
    if (argc > 2) initialized_values = argv[2];
    if (argc > 3) solution_values = argv[3];

    const float dt = 0.01f;
    const int nIters = 10;
    int bytes = nBodies * sizeof(Body);
    
    // Allocate unified memory
    Body *p;
    cudaMallocManaged(&p, bytes);

    read_values_from_file(initialized_values, (float*)p, bytes);

    // Use 256 threads per block for good occupancy
    int threadsPerBlock = 256;
    int blocksPerGrid = (nBodies + threadsPerBlock - 1) / threadsPerBlock;

    double totalTime = 0.0;

    // Prefetch data to GPU
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(p, bytes, device, NULL);

    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();

        bodyForce<<<blocksPerGrid, threadsPerBlock>>>(p, dt, nBodies);
        cudaDeviceSynchronize();

        integratePositions<<<blocksPerGrid, threadsPerBlock>>>(p, dt, nBodies);
        cudaDeviceSynchronize();

        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }

    double avgTime = totalTime / (double)(nIters);
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

    // Prefetch back to CPU for file writing
    cudaMemPrefetchAsync(p, bytes, cudaCpuDeviceId, NULL);
    write_values_to_file(solution_values, (float*)p, bytes);

    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

    cudaFree(p);
    return 0;
}