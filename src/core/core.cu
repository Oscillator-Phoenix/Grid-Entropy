#include <cstdio>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "core.h"
#include "cuda_error_check.h"

#define MAX_INT_OF_INPUT 16  //  假设元素为[0,15]的整型

#define RADIUS 2                              // 2 = 5/2
#define DIAMETER ((RADIUS*2)+1)               // 5=2*2+1
#define MAX_WINDOW_AREA (DIAMETER*DIAMETER)   // 25=5*5 Max Window Area

#define TPB_X  16
#define TPB_Y  16


// divup calculates n / m and would round it up if the remainder is non-zero.
int divup(int n, int m) {
    return n % m == 0 ? n/m : n/m + 1;
}


__device__ float d_get_entropy(int *bucket, float area) {
    float ent = 0.0;
    for (int k = 0; k < MAX_INT_OF_INPUT; k++) {
        if (bucket[k] > 0) {
            float p = float(bucket[k]) / area;
            ent -= p*log2f(p);

        }
    }
    return ent;
}


__global__ void kernel_base(const int width, const int height, const float* __restrict__ input, float *output) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ( i >= height || j >= width) {
        return;
    }
    
    int bucket[MAX_INT_OF_INPUT];   // zero initalization
    for (int k = 0; k < MAX_INT_OF_INPUT; k++) bucket[k] = 0;
    int area = 0;

    for (int ii = -RADIUS; ii <= RADIUS; ii++) {
        for (int jj = -RADIUS; jj <= RADIUS; jj++) {
            int x = i+ii; 
            int y = j+jj;
            if (x >= 0 && x < height && y >= 0 && y < width) {  // judge boundary
                bucket[ int(input[x*width+y]) ]++;
                area++;
            }
        }
    }
    
    output[i*width+j] = d_get_entropy(bucket, float(area)) ;
}


void cudaCallback_base(int width, int height, float *sample, float **result) {
    int size = width * height;
    float *input_d, *output_d;

    // Allocate device memory and copy data from host to device
    CHECK(cudaMalloc((void **)&input_d, sizeof(float)*size));
    CHECK(cudaMalloc((void **)&output_d, sizeof(float)*size));
    CHECK(cudaMemcpy(input_d, sample, sizeof(float)*size, cudaMemcpyHostToDevice));

    // Invoke the device function
    dim3 block( TPB_X, TPB_Y, 1 );
    dim3 grid( divup(height, TPB_X), divup(width, TPB_Y), 1);
    // printf("width=%d, height=%d\n", width, height);
    // printf("block(%d, %d), grid(%d, %d)\n", block.x, block.y, grid.x, grid.y);
    kernel_base<<< grid, block >>>(width, height, input_d, output_d);
    cudaDeviceSynchronize();

    // Copy back the results and de-allocate the device memory
    *result = (float *)malloc(sizeof(float)*size);
    CHECK(cudaMemcpy(*result, output_d, sizeof(float)*size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(input_d));
    CHECK(cudaFree(output_d));

    // Note that you don't have to free sample and *result by yourself
}


__global__ void kernel_opt(const int width, const int height, const float* __restrict__ input, float *output) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ( i >= height || j >= width) return;

    __shared__ int smem[ RADIUS+ TPB_X +RADIUS ][ RADIUS+ TPB_Y +RADIUS ];
    int x, y;
    {
        x = i; y = j;
        smem[ RADIUS+threadIdx.x ][ RADIUS+threadIdx.y ] = int(input[x*width + y]);
        if (threadIdx.x < RADIUS) {
            x = i-RADIUS;
            y = j;
            if (x >= 0 && x < height && y >= 0 && y < width) {
                smem[threadIdx.x             ][RADIUS+threadIdx.y] = int(input[x*width + y]);
            }

            x = i+TPB_X;
            y = j;
            if (x >= 0 && x < height && y >= 0 && y < width) {
                smem[RADIUS+TPB_X+threadIdx.x][RADIUS+threadIdx.y] = int(input[x*width + y]);
            }
        }

        if (threadIdx.y < RADIUS) {
            x = i;
            y = j-RADIUS;
            if (x >= 0 && x < height && y >= 0 && y < width) {
                smem[RADIUS+threadIdx.x][threadIdx.y             ] = int(input[x*width + y]);
            }

            x = i;
            y = j+TPB_Y;
            if (x >= 0 && x < height && y >= 0 && y < width) {
                smem[RADIUS+threadIdx.x][RADIUS+TPB_Y+threadIdx.y] = int(input[x*width + y]);
            }
        }

        if (threadIdx.x < RADIUS && threadIdx.y < RADIUS) {
            x = i-RADIUS;
            y = j-RADIUS;
            if (x >= 0 && x < height && y >= 0 && y < width) {
                smem[threadIdx.x             ][threadIdx.y             ] = int(input[x*width + y]);
            }

            x = i-RADIUS;
            y = j+TPB_Y;
            if (x >= 0 && x < height && y >= 0 && y < width) {
                smem[threadIdx.x             ][RADIUS+TPB_Y+threadIdx.y] = int(input[x*width + y]);
            }

            x = i+TPB_X;
            y = j-RADIUS;
            if (x >= 0 && x < height && y >= 0 && y < width) {
                smem[RADIUS+TPB_X+threadIdx.x][threadIdx.y             ] = int(input[x*width + y]);
            }

            x = i+TPB_X;
            y = j+TPB_Y;
            if (x >= 0 && x < height && y >= 0 && y < width) {
                smem[RADIUS+TPB_X+threadIdx.x][RADIUS+TPB_Y+threadIdx.y] = int(input[x*width + y]);
            }
        }
        __syncthreads();
    }

    int bucket[MAX_INT_OF_INPUT];   // zero initalization
    for (int k = 0; k < MAX_INT_OF_INPUT; k++) bucket[k] = 0;

    int sid_x = RADIUS+threadIdx.x;
    int sid_y = RADIUS+threadIdx.y;
    for (int ii = -RADIUS; ii <= RADIUS; ii++) {
        for (int jj = -RADIUS; jj <= RADIUS; jj++) {
            x = i+ii; 
            y = j+jj;
            if (x >= 0 && x < height && y >= 0 && y < width) {
                bucket[ smem[ sid_x+ii ][ sid_y+jj ] ]++;
            }
        }
    }
    
    int area = ( ( j+1 > RADIUS ? RADIUS:j ) + 1 + ( width-j > RADIUS ? RADIUS:(width-1-j)) ) *
               ( ( i+1 > RADIUS ? RADIUS:i ) + 1 + ( height-i > RADIUS ? RADIUS:(height-1-i)) );

    output[i*width+j] = d_get_entropy(bucket, float(area)) ;
}


void cudaCallback_opt(int width, int height, float *sample, float **result) {
    int size = width * height;
    float *input_d, *output_d;

    // Allocate device memory and copy data from host to device
    CHECK(cudaMalloc((void **)&input_d, sizeof(float)*size));
    CHECK(cudaMalloc((void **)&output_d, sizeof(float)*size));
    CHECK(cudaMemcpy(input_d, sample, sizeof(float)*size, cudaMemcpyHostToDevice));

    // Invoke the device function
    dim3 block( TPB_X, TPB_Y, 1 );
    dim3 grid( divup(height, TPB_X), divup(width, TPB_Y), 1);
    kernel_opt<<< grid, block >>>(width, height, input_d, output_d);
    cudaDeviceSynchronize();
    // CHECK_ERROR_MSG("kernel_opt");

    // Copy back the results and de-allocate the device memory
    *result = (float *)malloc(sizeof(float)*size);
    CHECK(cudaMemcpy(*result, output_d, sizeof(float)*size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(input_d));
    CHECK(cudaFree(output_d));
    
    // Note that you don't have to free sample and *result by yourself
}





