#include "gpu_filter.h"
#include <ctime>
#include <stdio.h>

__global__ void filter( float *in, float *out, int cols,int rows) {
	 int x = blockIdx.x * blockDim.x + threadIdx.x; 
 	 int y = blockIdx.y * blockDim.y + threadIdx.y;

 	 if(x >= cols || y >= rows)
 	 	return;
 	
 	 float res = (in[y*rows + x] - in[y*rows + x + 2]);
 	 //out[y*rows +x] = in[y*rows + x];
 	 out[y*rows +x] = res;
}

void upload_to_gpu(float** cpu_src, float** gpu_dst,float cols,float rows){
	clock_t begin = clock();
	cudaMalloc((void**)gpu_dst, cols * rows * sizeof(float));
	cudaMemcpy( *gpu_dst, *cpu_src, cols * rows * sizeof(float),cudaMemcpyHostToDevice);
	clock_t end = clock();
	printf("Upload time: %d\n",end-begin);
}

void downloadfrom_gpu(float** gpu_src, float** cpu_dst,float cols,float rows){
	//cpu_dst = (float**) malloc(cols * rows * sizeof(float));
	clock_t begin = clock();
	cudaMemcpy(&cpu_dst,gpu_src,cols * rows * sizeof(float),cudaMemcpyDeviceToHost);
	clock_t end = clock();
	printf("Download time: %d\n",end-begin);
}


void gpu_filter(float * input,float * output,int cols,int rows){
	clock_t total_b = clock();
	float *in_gpu,*out_gpu;
	clock_t mal_b = clock();
	cudaMalloc((void**)&out_gpu, cols * rows * sizeof(float));
	clock_t mal_e = clock();
	printf("Mal time: %d\n",mal_e - mal_b);
	upload_to_gpu(&input,&in_gpu,cols,rows);
	dim3 blocks(cols/16,rows/16);
	dim3 threads(16,16);
	clock_t begin = clock();
	filter<<<blocks,threads>>>(in_gpu,out_gpu,cols,rows);
	clock_t end = clock();

	printf("Add time: %d\n",end-begin);


	 begin = clock();
	cudaMemcpy(output,out_gpu,rows*cols*sizeof(float),cudaMemcpyDeviceToHost);
	 end = clock();
	printf("Download time: %d\n",end-begin);
	clock_t total_e = clock();
	printf("Total time: %d\n",total_e - total_b);
	// downloadfrom_gpu(&out_gpu,&output,cols,rows);
	cudaFree( in_gpu );
	cudaFree( out_gpu );
}
