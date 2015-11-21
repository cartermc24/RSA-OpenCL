//
//  CasRSA_CL.c
//  CasRSA_CL
//
//  Created by Carter McCardwell on 11/18/15.
//  Copyright (c) 2015 Carter McCardwell. All rights reserved.
//

#define __NO_STD_VECTOR
#define MAX_SOURCE_SIZE (0x100000)
#ifndef MAXDIGITS
#define	MAXDIGITS	500		/* maximum length bignum */
#endif

#define PLUS		1		/* positive sign bit */
#define MINUS		-1		/* negative sign bit */

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#include <OpenCL/OpenCL.h>

typedef struct {
    char digits[MAXDIGITS];         /* represent the number */
    int signbit;			/* 1 if positive, -1 if negative */
    int lastdigit;			/* index of high-order digit */
} bignum;

const unsigned long WordMask = 0x00000000FFFFFFFFUL;

char* stradd(const char* a, const char* b){
    size_t len = strlen(a) + strlen(b);
    char *ret = (char*)malloc(len * sizeof(char) + 1);
    *ret = '\0';
    return strcat(strcat(ret, a) ,b);
} //This function adds two string pointers together

void print_bignum(bignum *n)
{
    int i;
    
    if (n->signbit == MINUS) printf("- ");
    for (i=n->lastdigit; i>=0; i--)
        printf("%c",'0'+ n->digits[i]);
    
    printf("\n");
}

void int_to_bignum(int s, bignum *n)
{
    int i;				/* counter */
    int t;				/* int to work with */
    
    if (s >= 0) n->signbit = PLUS;
    else n->signbit = MINUS;
    
    for (i=0; i<MAXDIGITS; i++) n->digits[i] = (char) 0;
    
    n->lastdigit = -1;
    
    t = abs(s);
    
    while (t > 0) {
        n->lastdigit ++;
        n->digits[ n->lastdigit ] = (t % 10);
        t = t / 10;
    }
    
    if (s == 0) n->lastdigit = 0;
}

void initialize_bignum(bignum *n)
{
    int_to_bignum(0,n);
}

int scan_for_start_str(char* num_str)
{
    int places = MAXDIGITS;
    for (int i = 0; i < MAXDIGITS; i++)
    {
        if (num_str[i] != 0) { places = i; }
    }
    return places;
}

void initialize_bignum_with_str(char* num_str, bignum *big_num)
{
    int_to_bignum(0, big_num);
    for (int i = 0; i < MAXDIGITS; i++) { big_num->digits[i] = num_str[i]; }
    big_num->lastdigit = scan_for_start_str(num_str);
}

int main(int argc, const char * argv[])
{
    printf("CasRSA_CL OpenCL 1.2 implementation of RSA\nCarter McCardwell, Northeastern University NUCAR\nhttp://coe.neu.edu/~cmccardw - mccardwell.net\n--------------------------------------\n");

    if (argc != 3)
    {
        printf("Usage: ./CasRSA_CL conf_file outfile\nWhere the conf_file is a textfile that contains: [p] [q] [e] [message]\n");
        return 1;
    }
    
    clock_t c_start, c_stop;
    c_start = clock(); //Create a clock to benchmark the time taken for execution

    char p_s[MAXDIGITS], q_s[MAXDIGITS], M_s[MAXDIGITS];
    char ps[MAXDIGITS], qs[MAXDIGITS], Ms[MAXDIGITS];
    bignum p, q, M, result;
    int e;

    FILE *conf_file;
    FILE *outfile;
    FILE *cl_code;

    conf_file = fopen(argv[1], "r"); //First number
    if (conf_file == NULL) { printf("error_conf_file"); return(1); }
    outfile = fopen(argv[2], "w"); //The outfile, the result will be written here
    if (outfile == NULL) { printf("error (permission error: run with sudo or in directory the user owns)"); return (1); }

    for (int i = 0; i < MAXDIGITS; i++)
    {
        p_s[i] = 0; q_s[i] = 0; M_s[i] = 0;
        ps[i] = 0; qs[i] = 0; Ms[i] = 0;
    }
    
    int_to_bignum(0, &result);

    fscanf(conf_file, "%s %s %i %s", &ps, &qs, &e, &Ms);
    
	for (int i = 0; i < strlen(Ms); i++) { M_s[i] = Ms[strlen(Ms)-1-i] - '0'; }
	for (int i = 0; i < strlen(ps); i++) { p_s[i] = ps[strlen(ps)-1-i] - '0'; }
	for (int i = 0; i < strlen(qs); i++) { q_s[i] = qs[strlen(qs)-1-i] - '0'; }

	initialize_bignum_with_str(M_s, &M);
	initialize_bignum_with_str(p_s, &p);
	initialize_bignum_with_str(q_s, &q);
    
    printf("INPUT:\n\tM:"); print_bignum(&M);
    printf("\n\tp:"); print_bignum(&p);
    printf("\n\tq:"); print_bignum(&q);
    
    char *cl_headers = "#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable\n#define PLUS 1\n#define MINUS -1\n#define MAXDIGITS ";
    char num[10];
    sprintf(num, "%d", MAXDIGITS);
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
    printf("\nThis GPU supports %i compute units", numberOfCores); //Utilize the maximum number of compute units

    cl_uint maxThreads;
    clGetDeviceInfo(device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maxThreads), &maxThreads, NULL);
    //printf("\nRunning with %i threads per compute units", maxThreads); //Utilize the maximum number of threads/cu

    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) { printf("context"); }

    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) { printf("queue"); }

    cl_program program = clCreateProgramWithSource(context, 1, &append_str, &length, &err); //Compile program with expanded key included in the source
    if (err != CL_SUCCESS) { printf("createprogram"); }

    printf("\nBuilding CL Kernel...");
    err = clBuildProgram(program, 1, &device, "-I ./ -cl-std=CL1.2", NULL, NULL); //The fourth parameter is specific to OpenCL 2.0

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
    printf("\t[Done]");

    cl_mem cl_p, cl_q, cl_M, cl_result;
    cl_int status = CL_SUCCESS;
    
    cl_p = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(bignum), &p, &status);
    if (status != CL_SUCCESS || cl_p == NULL) { printf("\nCreate p: %i", status); }

    cl_q = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(bignum), &q, &status);
    if (status != CL_SUCCESS || cl_q == NULL) { printf("\nCreate q: %i", status); }

    cl_M = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(bignum), &M, &status);
    if (status != CL_SUCCESS || cl_M == NULL) { printf("\nCreate M: %i", status); }

    cl_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(bignum), &result, &status);
    if (status != CL_SUCCESS || cl_result == NULL) { printf("\nCreate result: %i", status); }

    cl_kernel rsa_kernel = clCreateKernel(program, "rsa_cypher", &status);
    if (status != CL_SUCCESS) { printf("\nclCreateKernel: %i", status); }

    status = clSetKernelArg(rsa_kernel, 0, sizeof(cl_mem), &cl_p);
    status = clSetKernelArg(rsa_kernel, 1, sizeof(cl_mem), &cl_q);
    status = clSetKernelArg(rsa_kernel, 2, sizeof(cl_mem), &cl_M);
    status = clSetKernelArg(rsa_kernel, 3, sizeof(cl_mem), &cl_result);
    status = clSetKernelArg(rsa_kernel, 4, sizeof(int), &e);
    if (status != CL_SUCCESS) { printf("\nclSetKernelArg: %i", status); }

    size_t local_ws = 1, global_ws = 1;
    printf("\nRun Parameters: Local: %zu - Global: %zu", local_ws, global_ws);

    status = clEnqueueNDRangeKernel(queue, rsa_kernel, 1, NULL, &global_ws, &local_ws, 0, NULL, NULL);
    if (status != CL_SUCCESS) { printf("\nclEnqueueNDRangeKernel: %i", status); }

    clFinish(queue);

    status = clEnqueueReadBuffer(queue, cl_result, CL_TRUE, 0, sizeof(bignum), &result, 0, NULL, NULL);
    if (status != CL_SUCCESS) { printf("\nclEnqueueReadBuffer: %i", status); }

    printf("\nEncrypted Result: ");
    print_bignum(&result);

    c_stop = clock();
    float diff = (((float)c_stop - (float)c_start) / CLOCKS_PER_SEC ) * 1000;

    printf("\nWriting result to outfile...");
    for (int i = result.lastdigit+1; i --> 0;)
    {
        fprintf(outfile, "%c", result.digits[i]+'0');
    }
    printf("\t[Done]");
    
    printf("\nDone - Time taken: %f ms\n", diff);
    clReleaseMemObject(cl_p);
    clReleaseMemObject(cl_q);
    clReleaseMemObject(cl_M);
    clReleaseMemObject(cl_result);
    fclose(conf_file);
    fclose(outfile);
    free(source_str);
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    return 0;
}

