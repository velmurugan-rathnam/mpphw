
/*
 Skeleton CUDA code for Homework 1, CS 264, Harvard, Fall 2009.  

 Here you'll implement the BLAS SAXPY operation (Scalar Alpha X Plus Y):

          Y <= alpha * X + Y 

 To compile:
	     
        nvcc saxpy.cu -o saxpy -I$CUDASDK_HOME/common/inc  \
          -L$CUDASDK_HOME/lib -lcutil
 Usage: 
  
        saxpy -inx="<fname_X>" -iny="<fname_Y>" -alpha=<alpha> \
          {-outy="<outfname_Y>"} {-device=<dev>} ,
	  
 where <fname_X> and <fname_Y> are input files corresponding to vectors
 x and y, respectively, and <alpha> is the alpha parameter for saxpy().  
 You can optionally specify an output file <outfname_Y> to which the
 result vector will be written, and the device number <dev>.

 You can also invoke it as

        saxpy -outy="<outfname_Y>" -n=<n>,

 where <outfname_Y> specifies the output filename, and <n>, vector length, 
 to generate a random vector in [0,1]^n and write it to file.

 The data files here are simply space-delimited text files.

 Kevin Dale <dale@eecs.harvard.edu>
 09.15.09 
*/

#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include "cutil.h"

#include "ioutils.h"


/* CPU saxpy */
__host__ void saxpy_cpu(float *x, float *y, float alpha, int n){
  // Start timers
  unsigned int timer_compute = 0;
  CUT_SAFE_CALL(cutCreateTimer(&timer_compute));
  CUT_SAFE_CALL(cutStartTimer(timer_compute));  

  // Compute
  for(int i=0; i<n; i++)
    y[i]=alpha*x[i]+y[i];

  // Report timing results
  CUT_SAFE_CALL(cutStopTimer(timer_compute));  
  printf("  CPU Processing time   : %f (ms)\n",
         cutGetTimerValue(timer_compute));
  CUT_SAFE_CALL(cutDeleteTimer(timer_compute));
}

/* GPU saxpy kernel */
__global__ void saxpy_gpu_kernel(float *x, float *y, float alpha, int n){
  // ===================================================================
  // Code Segment 5: Determine the output index of each thread,
  // compute the corresponding value and write to the correct location in 
  // vectorr y, making sure to check that the output location is valid.
  // ===================================================================
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n)
    y[i]=alpha*x[i]+y[i];
  // End of Code Segment 5 ============================================
}

/* GPU wrapper: x and y are n-length arrays stored in host memory  */
__host__ void saxpy_gpu(float *x, float *y, float alpha, int n){
  // ===================================================================
  // Code Segment 1: Allocate memory on the device, and copy memory from 
  // host to device memory.
  // ===================================================================
  unsigned int timer_memory = 0;
  CUT_SAFE_CALL(cutCreateTimer(&timer_memory));
  CUT_SAFE_CALL(cutStartTimer(timer_memory));
  
  // Allocate device memory
  float *device_x, *device_y;
  cudaMalloc((void**)&device_x,n*sizeof(float));
  cudaMalloc((void**)&device_y,n*sizeof(float));
  CUT_CHECK_ERROR("device memory allocation");

  // Copy host memory to device  
  cudaMemcpy(device_x,x,n*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(device_y,y,n*sizeof(float),cudaMemcpyHostToDevice);
  CUT_CHECK_ERROR("host to device copy");

  CUT_SAFE_CALL(cutStopTimer(timer_memory));

  // End of Code Segment 1
  // ===================================================================

  // ===================================================================
  // Code Segment 2: Initialize thread block and kernel grid
  // dimensions and invoke the CUDA kernel.
  // ===================================================================
  dim3 blockDim,gridDim; 

  // set the block size to the nearest power of two for vectors smaller 
  // than 256, and 256 otherwise
  blockDim.x=min(256,(unsigned int)pow(2.f,ceil(log(n)/log(2.f))));
  blockDim.y=1; blockDim.z=1;
  gridDim.x=(unsigned int)ceil((float)n/(float)blockDim.x);
  gridDim.y=1; gridDim.z=1;

  printf("  # of threads in a block: %d x %d (%d)\n",
         blockDim.x, blockDim.y, blockDim.x * blockDim.y);
  printf("  # of blocks in a grid  : %d x %d (%d)\n",
         gridDim.x, gridDim.y, gridDim.x * gridDim.y);  

  // Start the timer_compute to calculate how much time we spent on it.
  unsigned int timer_compute = 0;
  CUT_SAFE_CALL(cutCreateTimer(&timer_compute));
  CUT_SAFE_CALL(cutStartTimer(timer_compute));
  
  // Invoke the kernel here (saxpy_gpu_kernel)
  saxpy_gpu_kernel<<<gridDim,blockDim>>>(device_x,device_y,alpha,n);

  // Make sure that all threads have completed
  cudaThreadSynchronize();  

  // Stop the timer
  CUT_SAFE_CALL(cutStopTimer(timer_compute));  

  // End of Code Segment 2
  // ===================================================================

  // Check if kernel execution generated an error
  CUT_CHECK_ERROR("Kernel execution failed");
  
  // ===================================================================
  // Code Segment 3: Copy the results back from the host
  // ===================================================================
  CUT_SAFE_CALL(cutStartTimer(timer_memory));
  cudaMemcpy(y,device_y,n*sizeof(float),cudaMemcpyDeviceToHost);
  CUT_CHECK_ERROR("device to host copy");

  CUT_SAFE_CALL(cutStopTimer(timer_memory));
  
  // End of Code Segment 3
  // ===================================================================

  // ================================================
  // Show timing information
  // ================================================
  printf("  GPU memory access time: %f (ms)\n",
         cutGetTimerValue(timer_memory));
  printf("  GPU computation time  : %f (ms)\n",
         cutGetTimerValue(timer_compute));
  printf("  GPU processing time   : %f (ms)\n",
         cutGetTimerValue(timer_compute) + cutGetTimerValue(timer_memory));

  // Cleanup timers
  CUT_SAFE_CALL(cutDeleteTimer(timer_memory));
  CUT_SAFE_CALL(cutDeleteTimer(timer_compute));  

  // ===================================================================
  // Code Segment 4: Free the device memory
  // ===================================================================
  CUDA_SAFE_CALL(cudaFree(device_x));  
  CUDA_SAFE_CALL(cudaFree(device_y));  

  // End of Code Segment 4
  // ===================================================================

}

// ---------------------------------------------------------------------
// NOTHING TO COMPLETE BELOW 
// ---------------------------------------------------------------------


/* Main driver */
int main(int argc, char **argv){

  // Read command-line args
  char *usage="\n\n(1) To run the saxpy() test, \n"
    "  saxpy -inx=\"<fname_X>\" -iny=\"<fname_Y>\" -alpha=<alpha> \\\n"
    "    {-outy=\"<outfname_Y>\"} {-device=<dev>}\n"
    "\n(2) To generate a random test vector,\n"
    "  saxpy -outy=\"<outfname_Y>\" -n=<n>\n";

  float alpha;
  char *xfname, *yfname, *outfname;
  unsigned int ok=1, genvec=0;
  int n;
  
  if(cutGetCmdLineArgumentstr(argc,(const char**)argv,"inx",&xfname)){
    ok&=cutGetCmdLineArgumentstr(argc,(const char**)argv,"iny",&yfname);    
    ok&=cutGetCmdLineArgumentf(argc,(const char**)argv,"alpha",&alpha);
    cutGetCmdLineArgumentstr(argc,(const char**)argv,"outy",&outfname);
  }else{
    genvec=1;
    ok&=cutGetCmdLineArgumenti(argc,(const char**)argv,"n",&n);    
    ok&=cutGetCmdLineArgumentstr(argc,(const char**)argv,"outy",&outfname);
  }
  if(!ok){
    printf("USAGE: %s\n",usage);
    return -1;      
  }

  // Just generate a random vector and write to file if requested
  if(genvec){
    srand(time(NULL));
    float *randvec=(float*)malloc(sizeof(float)*n);
    for(int i=0; i<n; i++)
      randvec[i]=(float)rand()/(float)RAND_MAX; // in [0,1]
    if(!WriteFile(outfname,randvec,n))
      printf("Warning: failed to write to file %s\n",outfname);
    free(randvec);
    return 0;
  }
 
  // Read input data
  float *x=NULL,*y=NULL;
  int nx,ny;
  if(!ReadFile(xfname,&x,&nx)){
    printf("Error: failed to read file %s.  Exiting.\n",xfname);
    return -2;
  }
  if(!ReadFile(yfname,&y,&ny)){
    printf("Error: failed to read file %s.  Exiting.\n",yfname);
    return -3;
  }
  if(nx!=ny){
    printf("Error: input vectors are different lengths (%d and %d). Exiting\n",
           nx,ny);
    return -4;
  }
  if(nx==0){
    printf("Error: inputs empty. Exiting\n");
    return -5;
  }
  n=nx; 

  // Initialize the device
  CUT_DEVICE_INIT(argc,argv);  

  // saxpy() is in place, so just point CPU result to y array and allocate
  // space for gpu result and copy y 
  float *cpu_y=y;
  float *gpu_y=(float*)malloc(sizeof(float)*n); 
  memcpy(gpu_y,y,sizeof(float)*n);

  // Invoke reference 
  saxpy_cpu(x,cpu_y,alpha,n);

  // Invoke GPU implementation 
  saxpy_gpu(x,gpu_y,alpha,n);

  // Compare and report results
  float epsilon=1e-6f;
  unsigned int test=1;
  for(int i=0; i<n; i++)
    test&=abs(gpu_y[i]-cpu_y[i])<=epsilon;
  printf("Test %s\n", test ? "PASSED" : "FAILED");

  // Write output file if requested
  if(outfname){
    if(!WriteFile(outfname,gpu_y,n))
      printf("Warning: failed to write to file %s\n",outfname);
  }

  // Cleanup and return
  free(x);
  free(y);
  free(gpu_y);

  return 0;
};

