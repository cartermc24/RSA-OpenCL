//
//  CasRSA_CL.c
//  CasRSA_CL
//
//  Created by Carter McCardwell on 12/7/14.
//  Licensed under the MIT license, attribution is required.
//

#define __NO_STD_VECTOR
#define MAX_SOURCE_SIZE (0x100000)
#ifndef NUMWORDS //May be manually defined at compile time
#define NUMWORDS 1000
#endif

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include <CL/cl.h>

const unsigned long WordMask = 0x00000000FFFFFFFFUL;

char* stradd(const char* a, const char* b){
    size_t len = strlen(a) + strlen(b);
    char *ret = (char*)malloc(len * sizeof(char) + 1);
    *ret = '\0';
    return strcat(strcat(ret, a) ,b);
} //This function adds two string pointers together

bool is_prime(int target)
{
  for (int i = 2; i <= sqrt(target); i++)
  {
    if ((target % i) == 0)
    {
      return false;
    }
  }
  return true;
}

int main(int argc, const char * argv[])
{
    printf("\nCasRSA_CL OpenCL 1.2 implementation of RSA\nCarter McCardwell, Northeastern University NUCAR - http://coe.neu.edu/~cmccardw - mccardwell.net");

    clock_t c_start, c_stop;
    c_start = clock(); //Create a clock to benchmark the time taken for execution

    FILE *rsa_file;
    FILE *outfile;
    FILE *cl_code;

    rsa_file = fopen(argv[1], "r");
    if (onefile == NULL) { printf("\nerror_rsa_file\n"); return(1); }
    outfile = fopen(argv[2], "w"); //The outfile, the result will be written here
    if (outfile == NULL) { printf("\nerror (permission error: run with sudo or in directory the user owns)\n"); return(1); }

    char n[NUMWORDS], totient[NUMWORDS], result[NUMWORDS];
    char p[NUMWORDS], q[NUMWORDS], message[NUMWORDS];
    int e;

    fscanf(rsa_file, "%s %s %i %s", p, q, &e, message);
    if (e < 1 || e > 3120 || !is_prime(e) || !(3120 % e == 0))
    {
        printf("\nRSA Fundamental Error: e must be 1<e<3120, a prime number, and not a devisor of 3120");
        return(1);
    }

    int l_p = strlen(p);
    int l_q = strlen(q);
    int l_msg = strlen(message);

    printf("\nSize of p: %i, q: %i, message: %i", l_p, l_q, l_msg);

    for (int i = 0; i < l_p; i++) { p[i] -= '0'; }
    for (int i = 0; i < l_q; i++) { q[i] -= '0'; }
    for (int i = 0; i < l_msg; i++) { message[i] -= '0'; }

    char *cl_headers = "#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable\n#define NUMWORDS ";
    char num[10];
    sprintf(num, "%d", NUMWORDS);
    char *append_str = stradd(cl_headers, num);

    cl_code = fopen("kernel_rsa.cl", "r");
    if (cl_code == NULL) { printf("\nerror: clfile\n"); return(1); }
    char *source_str = (char *)malloc(MAX_SOURCE_SIZE);
    fread(source_str, 1, MAX_SOURCE_SIZE, cl_code);
    fclose(cl_code);

    append_str = stradd(append_str, source_str);
    size_t length = strlen(append_str);

    //Set OpenCL Context
    cl_int err;
    cl_platform_id platform;
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) { printf("platformid"); }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) { printf("getdeivceid %i", err); }

    cl_uint numberOfCores;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(numberOfCores), &numberOfCores, NULL);
    printf("\nRunning with %i compute units", numberOfCores); //Utilize the maximum number of compute units

    cl_uint maxThreads;
    clGetDeviceInfo(device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maxThreads), &maxThreads, NULL);
    printf("\nRunning with %i threads per compute units", maxThreads); //Utilize the maximum number of threads/cu

    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) { printf("context"); }

    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) { printf("queue"); }

    cl_program program = clCreateProgramWithSource(context, 1, &append_str, &length, &err); //Compile program with expanded key included in the source
    if (err != CL_SUCCESS) { printf("createprogram"); }

    err = clBuildProgram(program, 1, &device, "-I ./ -cl-std=CL1.2", NULL, NULL);

    if (err == CL_BUILD_PROGRAM_FAILURE) {
        printf("\nBuild Error = %i", err);
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
    }

    cl_mem cl_p, cl_q, cl_msg, cl_result;
    cl_int status = CL_SUCCESS;
    
    cl_p = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(char)*NUMWORDS, &p, &status);
    if (status != CL_SUCCESS || cl_num1 == NULL) { printf("\nCreate num1: %i", status); }

    cl_q = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(char)*NUMWORDS, &q, &status);
    if (status != CL_SUCCESS || cl_num2 == NULL) { printf("\nCreate num2: %i", status); }

    cl_msg = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(char)*NUMWORDS, &message, &status);
    if (status != CL_SUCCESS || cl_num2 == NULL) { printf("\nCreate message: %i", status); }

    cl_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(char)*NUMWORDS, &result, &status);
    if (status != CL_SUCCESS || cl_result == NULL) { printf("\nCreate acc: %i", status); }

    cl_kernel rsa_kernel = clCreateKernel(program, "rsa_enc", &status);
    if (status != CL_SUCCESS) { printf("\nclCreateKernel: %i", status); }

    status = clSetKernelArg(rsa_kernel, 0, sizeof(cl_mem), &cl_num1);
    status = clSetKernelArg(rsa_kernel, 1, sizeof(cl_mem), &cl_num2);
    status = clSetKernelArg(rsa_kernel, 2, sizeof(cl_mem), &cl_msg);
    status = clSetKernelArg(rsa_kernel, 3, sizeof(cl_mem), &cl_result);
    status = clSetKernelArg(rsa_kernel, 4, sizeof(int), &l_p);
    status = clSetKernelArg(rsa_kernel, 5, sizeof(int), &l_q);
    status = clSetKernelArg(rsa_kernel, 6, sizeof(int), &l_msg);
    if (status != CL_SUCCESS) { printf("\nclSetKernelArg: %i", status); }

    size_t local_ws = 1, global_ws = 1;
    printf("\nLocal: %d - Global: %d", local_ws, global_ws);

    status = clEnqueueNDRangeKernel(queue, rsa_kernel, 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
    if (status != CL_SUCCESS) { printf("\nclEnqueueNDRangeKernel: %i", status); }

    clFinish(queue);

    status = clEnqueueReadBuffer(queue, cl_result, CL_TRUE, 0, sizeof(char)*NUMWORDS, &result, 0, NULL, NULL);
    if (status != CL_SUCCESS) { printf("\nclEnqueueReadBuffer: %i", status); }

    c_stop = clock();
    float diff = (((float)c_stop - (float)c_start) / CLOCKS_PER_SEC ) * 1000;

    printf("\nResult:\n%s\n", result);

    printf("\nDone - Time taken: %f ms\n", diff);
    clReleaseMemObject(cl_num1);
    clReleaseMemObject(cl_num2);
    clReleaseMemObject(cl_result);
    fclose(onefile);
    fclose(twofile);
    fclose(outfile);
    free(source_str);
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    return 0;
}