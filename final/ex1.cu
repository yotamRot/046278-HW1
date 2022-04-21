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
    int bi = blockIdx.x;
    int workForThread = (TILE_WIDTH * TILE_WIDTH) / NUM_OF_THREADS; // in bytes
    uchar imageVal;

    __shared__ int sharedHist[HISTOGRAM_SIZE]; // maybe change to 16 bit ? will be confilcits on same bank 

    int imageStartIndex = bi * IMG_HEIGHT * IMG_WIDTH;
    int mapStartIndex = bi * TILE_COUNT * TILE_COUNT * HISTOGRAM_SIZE;
    int tileStartIndex;
    int insideTileIndex;
    int curIndex;
    for (int i = 0 ; i < TILE_COUNT * TILE_COUNT; i++)
    {
        // calc tile index in image buffer (shared between al threads in block)
        tileStartIndex = imageStartIndex + i % 8 * TILE_WIDTH + (i / 8) * (TILE_WIDTH *TILE_WIDTH) * TILE_COUNT;
        // zero shared buffer histogram values
        sharedHist[ti] = 0;
        __syncthreads();
        for (int j = 0; j < workForThread; j++)
        {
            // calc index in tile buffer for each thread
            insideTileIndex = tg * TILE_WIDTH * TILE_COUNT + ti % 64 + 4 * TILE_WIDTH * TILE_COUNT * j;
            // sum tile index and index inside tile to find relevant byte for thread in cur iteration
            curIndex = tileStartIndex + insideTileIndex;
            // update histogram
            imageVal = all_in[curIndex];
            atomicAdd(sharedHist + imageVal, 1);
        }
        __syncthreads();
        
        // calc CDF using prefix sumpwdon histogram buffer
        prefix_sum(sharedHist, HISTOGRAM_SIZE);
        __syncthreads();
        // calc map value for each index
        maps[mapStartIndex + HISTOGRAM_SIZE * i + ti] = (float(sharedHist[ti]) * 255)  / (TILE_WIDTH * TILE_WIDTH);
        __syncthreads();
    
    }

    // interpolate image using given maps buffer
    interpolate_device(maps + mapStartIndex, all_in + imageStartIndex, all_out + imageStartIndex);
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
    CUDA_CHECK(cudaMalloc((void**)&context->taskMaps, TILE_COUNT * TILE_COUNT * HISTOGRAM_SIZE));
    CUDA_CHECK(cudaMalloc((void**)&context->imgIn, IMG_WIDTH * IMG_HEIGHT));
    CUDA_CHECK(cudaMalloc((void**)&context->imgOut, IMG_WIDTH * IMG_HEIGHT));

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
    int imageIndex;

    for (int i = 0; i < N_IMAGES; i++) {
        imageIndex = i * IMG_WIDTH * IMG_HEIGHT;
        CUDA_CHECK(cudaMemcpy((void*)context->imgIn, (void*)&images_in[imageIndex], IMG_WIDTH * IMG_HEIGHT * sizeof(char), cudaMemcpyHostToDevice));
        process_image_kernel<<<1, NUM_OF_THREADS>>>(context->imgIn, context->imgOut, context->taskMaps);
        CUDA_CHECK(cudaMemcpy((void*)&images_out[imageIndex], (void*)context->imgOut, IMG_WIDTH * IMG_HEIGHT * sizeof(char), cudaMemcpyDeviceToHost));
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init
    CUDA_CHECK(cudaFree(context->imgOut));
    CUDA_CHECK(cudaFree(context->imgIn));
    CUDA_CHECK(cudaFree(context->taskMaps));
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
    uchar *taskMaps;
    uchar *imgIn;
    uchar *imgOut;
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for all the input images, output images, and maps
    CUDA_CHECK(cudaMalloc((void**)&context->taskMaps, N_IMAGES * TILE_COUNT * TILE_COUNT * HISTOGRAM_SIZE));
    CUDA_CHECK(cudaMalloc((void**)&context->imgIn, N_IMAGES * IMG_HEIGHT * IMG_WIDTH));
    CUDA_CHECK(cudaMalloc((void**)&context->imgOut, N_IMAGES * IMG_WIDTH * IMG_HEIGHT));


    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out

    CUDA_CHECK(cudaMemcpy((void*)context->imgIn, (void*)images_in, N_IMAGES * IMG_HEIGHT * IMG_WIDTH, cudaMemcpyHostToDevice));
    process_image_kernel<<<N_IMAGES , NUM_OF_THREADS>>>(context->imgIn, context->imgOut, context->taskMaps);
    CUDA_CHECK(cudaMemcpy((void*)images_out, (void*)context->imgOut, N_IMAGES *  IMG_WIDTH * IMG_HEIGHT * sizeof(char), cudaMemcpyDeviceToHost));
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    // //TODO: free resources allocated in gpu_bulk_init
    CUDA_CHECK(cudaFree(context->imgOut));
    CUDA_CHECK(cudaFree(context->imgIn));
    CUDA_CHECK(cudaFree(context->taskMaps));
    free(context);
}
