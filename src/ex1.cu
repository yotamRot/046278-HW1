#include "ex1.h"

#define HISTOGRAM_SIZE 256
#define NUM_OF_THREADS 256
#define WRAP_SIZE 32

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x % HISTOGRAM_SIZE;
    int increment;
    for (int stride = 1; stride < arr_size; stride *= 2) {
    if (tid >= stride) {
        increment = arr[tid - stride];
    }
    __syncthreads();
    if (tid >= stride) {
        arr[tid] += increment;
    }
    __syncthreads();
    }
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__ 
void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) {

    int ti = threadIdx.x;
    int tg = ti / TILE_WIDTH;
    int workForThread = (TILE_WIDTH * TILE_WIDTH) / NUM_OF_THREADS; // in bytes
    char imageVal;

    __shared__ int sharedHist[HISTOGRAM_SIZE]; // maybe change to 16 bit ? will be confilcits on same bank 

    // zero historgram
    sharedHist[ti] = 0;

    int tileStartIndex;
    int insideTileIndex;
    int curIndex;

    for (int i = 0 ; i < TILE_COUNT * TILE_COUNT; i++)
    {
        tileStartIndex = i % 8 * TILE_WIDTH + (i / 8) * (TILE_WIDTH *TILE_WIDTH) * TILE_COUNT;
        for (int j = 0; j < workForThread; j++)
        {
            insideTileIndex = tg * TILE_WIDTH * TILE_COUNT + ti % 64 + 4 * TILE_WIDTH * TILE_COUNT * j;
            curIndex = tileStartIndex + insideTileIndex;
            imageVal = all_in[curIndex];
            atomicAdd(&sharedHist[imageVal], 1);
        }
    }
    // calc CDF
    prefix_sum(sharedHist, HISTOGRAM_SIZE);
    // calc Map
    maps[ti] = sharedHist[ti] * (1 / (TILE_WIDTH * TILE_WIDTH)) * 256;

    interpolate_device(maps, all_in, all_out);
    return; 
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    uchar *taskMaps;
    uchar *imgIn;
    uchar *imgOut;
    // TODO define task serial memory buffers
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    //TODO: allocate GPU memory for a single input image, a single output image, and maps
    cudaMalloc((void**)&context->taskMaps, N_IMAGES * TILE_COUNT * TILE_COUNT);
    cudaMalloc((void**)&context->imgIn, N_IMAGES * IMG_WIDTH * IMG_WIDTH);
    cudaMalloc((void**)&context->imgOut, N_IMAGES * IMG_WIDTH * IMG_WIDTH);

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: in a for loop:
    //   1. copy the relevant image from images_in to the GPU memory you allocated
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
    for (int i = 0; i < N_IMAGES; i++) {
        cudaMemcpy((void*)context[i].imgIn, &images_in[i * IMG_WIDTH * IMG_HEIGHT],IMG_WIDTH * IMG_HEIGHT * sizeof(char), cudaMemcpyHostToDevice);
        process_image_kernel<<<1, NUM_OF_THREADS>>>(context[i].imgIn, context[i].imgOut, context[i].taskMaps);
        cudaMemcpy(context[i].imgIn, &images_out[i * IMG_WIDTH * IMG_HEIGHT], IMG_WIDTH * IMG_HEIGHT * sizeof(char), cudaMemcpyDeviceToHost);
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init
    cudaFree(context->imgOut);
    cudaFree(context->imgIn);
    cudaFree(context->taskMaps);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for all the input images, output images, and maps

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init

    free(context);
}
