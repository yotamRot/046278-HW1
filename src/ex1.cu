#include "ex1.h"

#define HISTOGRAM_SIZE 256

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
    // TODO

    int ti = threadIdx.x;
    int tG = ti / 32;
    char imageVal;

    __shared__ int sharedHist[HISTOGRAM_SIZE * TILE_COUNT];

    //to do zero hist
    int startIndex = tG * 2 * TILE_WIDTH + sizeof(int) * ti % 32;
    int histogramIndex = tG % (TILE_COUNT * TILE_COUNT) * 2 + (ti % 32 > 16) * 1;

    // block of 32 threads each reponsibole for 2 histograms
    int *myHist = &sharedHist[histogramIndex];


    for (int j = 0 ; j < 3; j ++)
    {
        startIndex += j;
        imageVal = all_in[startIndex];
        atomicAdd(&myHist[imageVal], 1);
    }

    int histogramsPerIteration = (1024 / HISTOGRAM_SIZE);
    int numOfIterations = (TILE_COUNT * TILE_COUNT) / histogramsPerIteration;
    int prefixHistogramIndex = (ti / HISTOGRAM_SIZE) * HISTOGRAM_SIZE;
    int mapIndex = prefixHistogramIndex + ti % HISTOGRAM_SIZE;
    // each iteration we create 4 histograms so we have 16 iterations
    for (int j = 0 ; j < numOfIterations; j++)
    {
        prefixHistogramIndex += j * HISTOGRAM_SIZE * 4;
        prefix_sum(&sharedHist[prefixHistogramIndex], HISTOGRAM_SIZE);
        maps[mapIndex] = sharedHist[mapIndex] * (1 / (TILE_WIDTH * TILE_WIDTH)) * 256;
    }

    interpolate_device(all_in, all_out, maps);
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
        process_image_kernel<<<1, 1024>>>(context[i].imgIn, context[i].imgOut, context[i].taskMaps);
        cudaMemcpy(context[i].imgIn, &images_out[i * IMG_WIDTH * IMG_HEIGHT], IMG_WIDTH * IMG_HEIGHT * sizeof(char), cudaMemcpyDeviceToHost);
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init

    free(context);
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
