
/*
 * FLAME GPU v 1.5.X for CUDA 9
 * Copyright University of Sheffield.
 * Original Author: Dr Paul Richmond (user contributions tracked on https://github.com/FLAMEGPU/FLAMEGPU)
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence
 * on www.flamegpu.com website.
 *
 */


  //Disable internal thrust warnings about conversions
  #ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning (disable : 4267)
  #pragma warning (disable : 4244)
  #endif
  #ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-parameter"
  #endif

  // includes
  #include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cub/cub.cuh>

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"


#ifdef _MSC_VER
#pragma warning(pop)
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort=true)
{
	gpuAssert( cudaPeekAtLastError(), file, line );
#ifdef _DEBUG
	gpuAssert( cudaDeviceSynchronize(), file, line );
#endif
   
}

/* SM padding and offset variables */
int SM_START;
int PADDING;

unsigned int g_iterationNumber;

/* Agent Memory */

/* crystal Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_crystal_list* d_crystals;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_crystal_list* d_crystals_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_crystal_list* d_crystals_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_crystal_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_crystal_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_crystal_values;  /**< Agent sort identifiers value */

/* crystal state variables */
xmachine_memory_crystal_list* h_crystals_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_crystal_list* d_crystals_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_crystal_default_count;   /**< Agent population size counter */ 


/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
unsigned int h_crystals_default_variable_rank_data_iteration;
unsigned int h_crystals_default_variable_l_data_iteration;
unsigned int h_crystals_default_variable_bin_data_iteration;


/* Message Memory */

/* internal_coord Message variables */
xmachine_message_internal_coord_list* h_internal_coords;         /**< Pointer to message list on host*/
xmachine_message_internal_coord_list* d_internal_coords;         /**< Pointer to message list on device*/
xmachine_message_internal_coord_list* d_internal_coords_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_internal_coord_count;         /**< message list counter*/
int h_message_internal_coord_output_type;   /**< message output type (single or optional)*/

  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_crystal;
size_t temp_scan_storage_bytes_crystal;


/*Global condition counts*/

/* RNG rand48 */
RNG_rand48* h_rand48;    /**< Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**< Pointer to RNG_rand48 seed list on device*/

/* Cuda Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEvent_t instrument_iteration_start, instrument_iteration_stop;
	float instrument_iteration_milliseconds = 0.0f;
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEvent_t instrument_start, instrument_stop;
	float instrument_milliseconds = 0.0f;
#endif

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**< Indicates if the position (in message list) of last message*/
int scan_last_included;      /**< Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */

/** crystal_create_ranks
 * Agent function prototype for create_ranks function of crystal agent
 */
void crystal_create_ranks(cudaStream_t &stream);

/** crystal_nucleate
 * Agent function prototype for nucleate function of crystal agent
 */
void crystal_nucleate(cudaStream_t &stream);

/** crystal_growth
 * Agent function prototype for growth function of crystal agent
 */
void crystal_growth(cudaStream_t &stream);

  
void setPaddingAndOffset()
{
    PROFILE_SCOPED_RANGE("setPaddingAndOffset");
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int x64_sys = 0;

	// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 && deviceProp.minor == 9999){
		printf("Error: There is no device supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
    
    //check if double is used and supported
#ifdef _DOUBLE_SUPPORT_REQUIRED_
	printf("Simulation requires full precision double values\n");
	if ((deviceProp.major < 2)&&(deviceProp.minor < 3)){
		printf("Error: Hardware does not support full precision double values!\n");
		exit(EXIT_FAILURE);
	}
    
#endif

	//check 32 or 64bit
	x64_sys = (sizeof(void*)==8);
	if (x64_sys)
	{
		printf("64Bit System Detected\n");
	}
	else
	{
		printf("32Bit System Detected\n");
	}

	SM_START = 0;
	PADDING = 0;
  
	//copy padding and offset to GPU
	gpuErrchk(cudaMemcpyToSymbol( d_SM_START, &SM_START, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol( d_PADDING, &PADDING, sizeof(int)));     
}

int is_sqr_pow2(int x){
	int r = (int)pow(4, ceil(log(x)/log(4)));
	return (r == x);
}

int lowest_sqr_pow2(int x){
	int l;
	
	//escape early if x is square power of 2
	if (is_sqr_pow2(x))
		return x;
	
	//lower bound		
	l = (int)pow(4, floor(log(x)/log(4)));
	
	return l;
}

/* Unary function required for cudaOccupancyMaxPotentialBlockSizeVariableSMem to avoid warnings */
int no_sm(int b){
	return 0;
}

/* Unary function to return shared memory size for reorder message kernels */
int reorder_messages_sm_size(int blockSize)
{
	return sizeof(unsigned int)*(blockSize+1);
}


/** getIterationNumber
 *  Get the iteration number (host)
 *  @return a 1 indexed value for the iteration number, which is incremented at the start of each simulation step.
 *      I.e. it is 0 on up until the first call to singleIteration()
 */
extern unsigned int getIterationNumber(){
    return g_iterationNumber;
}

void initialise(char * inputfile){
    PROFILE_SCOPED_RANGE("initialise");

	//set the padding and offset values depending on architecture and OS
	setPaddingAndOffset();
  
    // Initialise some global variables
    g_iterationNumber = 0;

    // Initialise variables for tracking which iterations' data is accessible on the host.
    h_crystals_default_variable_rank_data_iteration = 0;
    h_crystals_default_variable_l_data_iteration = 0;
    h_crystals_default_variable_bin_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_crystal_SoA_size = sizeof(xmachine_memory_crystal_list);
	h_crystals_default = (xmachine_memory_crystal_list*)malloc(xmachine_crystal_SoA_size);

	/* Message memory allocation (CPU) */
	int message_internal_coord_SoA_size = sizeof(xmachine_message_internal_coord_list);
	h_internal_coords = (xmachine_message_internal_coord_list*)malloc(message_internal_coord_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs

  /* Graph memory allocation (CPU) */
  

    PROFILE_POP_RANGE(); //"allocate host"
	

	//read initial states
	readInitialStates(inputfile, h_crystals_default, &h_xmachine_memory_crystal_default_count);

  // Read graphs from disk
  

  PROFILE_PUSH_RANGE("allocate device");
	
	/* crystal Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_crystals, xmachine_crystal_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_crystals_swap, xmachine_crystal_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_crystals_new, xmachine_crystal_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_crystal_keys, xmachine_memory_crystal_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_crystal_values, xmachine_memory_crystal_MAX* sizeof(uint)));
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_crystals_default, xmachine_crystal_SoA_size));
	gpuErrchk( cudaMemcpy( d_crystals_default, h_crystals_default, xmachine_crystal_SoA_size, cudaMemcpyHostToDevice));
    
	/* internal_coord Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_internal_coords, message_internal_coord_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_internal_coords_swap, message_internal_coord_SoA_size));
	gpuErrchk( cudaMemcpy( d_internal_coords, h_internal_coords, message_internal_coord_SoA_size, cudaMemcpyHostToDevice));
		


  /* Allocate device memory for graphs */
  

    PROFILE_POP_RANGE(); // "allocate device"

    /* Calculate and allocate CUB temporary memory for exclusive scans */
    
    d_temp_scan_storage_crystal = nullptr;
    temp_scan_storage_bytes_crystal = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_crystal, 
        temp_scan_storage_bytes_crystal, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_crystal_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_crystal, temp_scan_storage_bytes_crystal));
    

	/*Set global condition counts*/

	/* RNG rand48 */
    PROFILE_PUSH_RANGE("Initialse RNG_rand48");
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	gpuErrchk( cudaMalloc( (void**) &d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
	int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
	h_rand48->A.x = A & 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) & 0xFFFFFFLL;
	h_rand48->C.x = C & 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) & 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) << 16) | 0x330E;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x & 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) & 0xFFFFFFLL;
	}
	//copy to device
	gpuErrchk( cudaMemcpy( d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));

    PROFILE_POP_RANGE();

	/* Call all init functions */
	/* Prepare cuda event timers for instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventCreate(&instrument_iteration_start);
	cudaEventCreate(&instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventCreate(&instrument_start);
	cudaEventCreate(&instrument_stop);
#endif

	
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
    initConstants();
    PROFILE_PUSH_RANGE("initConstants");
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: initConstants = %f (ms)\n", instrument_milliseconds);
#endif
	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_crystal_default_count: %u\n",get_agent_crystal_default_count());
	
#endif
} 


void sort_crystals_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_crystal_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_crystal_default_count); 
	gridSize = (h_xmachine_memory_crystal_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_crystal_keys, d_xmachine_memory_crystal_values, d_crystals_default);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_crystal_keys),  thrust::device_pointer_cast(d_xmachine_memory_crystal_keys) + h_xmachine_memory_crystal_default_count,  thrust::device_pointer_cast(d_xmachine_memory_crystal_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_crystal_agents, no_sm, h_xmachine_memory_crystal_default_count); 
	gridSize = (h_xmachine_memory_crystal_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_crystal_agents<<<gridSize, blockSize>>>(d_xmachine_memory_crystal_values, d_crystals_default, d_crystals_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_crystal_list* d_crystals_temp = d_crystals_default;
	d_crystals_default = d_crystals_swap;
	d_crystals_swap = d_crystals_temp;	
}


void cleanup(){
    PROFILE_SCOPED_RANGE("cleanup");

    /* Call all exit functions */
	
#if defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif

    hist_func();
    PROFILE_PUSH_RANGE("hist_func");
	PROFILE_POP_RANGE();

#if defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: hist_func = %f (ms)\n", instrument_milliseconds);
#endif
	

	/* Agent data free*/
	
	/* crystal Agent variables */
	gpuErrchk(cudaFree(d_crystals));
	gpuErrchk(cudaFree(d_crystals_swap));
	gpuErrchk(cudaFree(d_crystals_new));
	
	free( h_crystals_default);
	gpuErrchk(cudaFree(d_crystals_default));
	

	/* Message data free */
	
	/* internal_coord Message variables */
	free( h_internal_coords);
	gpuErrchk(cudaFree(d_internal_coords));
	gpuErrchk(cudaFree(d_internal_coords_swap));
	

    /* Free temporary CUB memory if required. */
    
    if(d_temp_scan_storage_crystal != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_crystal));
      d_temp_scan_storage_crystal = nullptr;
      temp_scan_storage_bytes_crystal = 0;
    }
    

  /* Graph data free */
  
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));

  /* CUDA Event Timers for Instrumentation */
#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventDestroy(instrument_iteration_start);
	cudaEventDestroy(instrument_iteration_stop);
#endif
#if (defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS) || (defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS) || (defined(INSTRUMENT_STEP_FUNCTIONS) && INSTRUMENT_STEP_FUNCTIONS) || (defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS)
	cudaEventDestroy(instrument_start);
	cudaEventDestroy(instrument_stop);
#endif
}

void singleIteration(){
PROFILE_SCOPED_RANGE("singleIteration");

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_start);
#endif

    // Increment the iteration number.
    g_iterationNumber++;

  /* set all non partitioned, spatial partitioned and On-Graph Partitioned message counts to 0*/
	h_message_internal_coord_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_internal_coord_count, &h_message_internal_coord_count, sizeof(int)));
	

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("crystal_create_ranks");
	crystal_create_ranks(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: crystal_create_ranks = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("crystal_nucleate");
	crystal_nucleate(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: crystal_nucleate = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("crystal_growth");
	crystal_growth(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: crystal_growth = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_crystal_default_count: %u\n",get_agent_crystal_default_count());
	
#endif

#if defined(INSTRUMENT_ITERATIONS) && INSTRUMENT_ITERATIONS
	cudaEventRecord(instrument_iteration_stop);
	cudaEventSynchronize(instrument_iteration_stop);
	cudaEventElapsedTime(&instrument_iteration_milliseconds, instrument_iteration_start, instrument_iteration_stop);
	printf("Instrumentation: Iteration Time = %f (ms)\n", instrument_iteration_milliseconds);
#endif
}

/* Environment functions */

//host constant declaration
float h_env_DELTA_T;
float h_env_B0;
float h_env_G0;
float h_env_b;
float h_env_y;
int h_env_t;
float h_env_m0;
float h_env_delta_m0;
int h_env_nuc_no;
int h_env_exit_no;
int h_env_agg_no;


//constant setter
void set_DELTA_T(float* h_DELTA_T){
    gpuErrchk(cudaMemcpyToSymbol(DELTA_T, h_DELTA_T, sizeof(float)));
    memcpy(&h_env_DELTA_T, h_DELTA_T,sizeof(float));
}

//constant getter
const float* get_DELTA_T(){
    return &h_env_DELTA_T;
}



//constant setter
void set_B0(float* h_B0){
    gpuErrchk(cudaMemcpyToSymbol(B0, h_B0, sizeof(float)));
    memcpy(&h_env_B0, h_B0,sizeof(float));
}

//constant getter
const float* get_B0(){
    return &h_env_B0;
}



//constant setter
void set_G0(float* h_G0){
    gpuErrchk(cudaMemcpyToSymbol(G0, h_G0, sizeof(float)));
    memcpy(&h_env_G0, h_G0,sizeof(float));
}

//constant getter
const float* get_G0(){
    return &h_env_G0;
}



//constant setter
void set_b(float* h_b){
    gpuErrchk(cudaMemcpyToSymbol(b, h_b, sizeof(float)));
    memcpy(&h_env_b, h_b,sizeof(float));
}

//constant getter
const float* get_b(){
    return &h_env_b;
}



//constant setter
void set_y(float* h_y){
    gpuErrchk(cudaMemcpyToSymbol(y, h_y, sizeof(float)));
    memcpy(&h_env_y, h_y,sizeof(float));
}

//constant getter
const float* get_y(){
    return &h_env_y;
}



//constant setter
void set_t(int* h_t){
    gpuErrchk(cudaMemcpyToSymbol(t, h_t, sizeof(int)));
    memcpy(&h_env_t, h_t,sizeof(int));
}

//constant getter
const int* get_t(){
    return &h_env_t;
}



//constant setter
void set_m0(float* h_m0){
    gpuErrchk(cudaMemcpyToSymbol(m0, h_m0, sizeof(float)));
    memcpy(&h_env_m0, h_m0,sizeof(float));
}

//constant getter
const float* get_m0(){
    return &h_env_m0;
}



//constant setter
void set_delta_m0(float* h_delta_m0){
    gpuErrchk(cudaMemcpyToSymbol(delta_m0, h_delta_m0, sizeof(float)));
    memcpy(&h_env_delta_m0, h_delta_m0,sizeof(float));
}

//constant getter
const float* get_delta_m0(){
    return &h_env_delta_m0;
}



//constant setter
void set_nuc_no(int* h_nuc_no){
    gpuErrchk(cudaMemcpyToSymbol(nuc_no, h_nuc_no, sizeof(int)));
    memcpy(&h_env_nuc_no, h_nuc_no,sizeof(int));
}

//constant getter
const int* get_nuc_no(){
    return &h_env_nuc_no;
}



//constant setter
void set_exit_no(int* h_exit_no){
    gpuErrchk(cudaMemcpyToSymbol(exit_no, h_exit_no, sizeof(int)));
    memcpy(&h_env_exit_no, h_exit_no,sizeof(int));
}

//constant getter
const int* get_exit_no(){
    return &h_env_exit_no;
}



//constant setter
void set_agg_no(int* h_agg_no){
    gpuErrchk(cudaMemcpyToSymbol(agg_no, h_agg_no, sizeof(int)));
    memcpy(&h_env_agg_no, h_agg_no,sizeof(int));
}

//constant getter
const int* get_agg_no(){
    return &h_env_agg_no;
}




/* Agent data access functions*/

    
int get_agent_crystal_MAX_count(){
    return xmachine_memory_crystal_MAX;
}


int get_agent_crystal_default_count(){
	//continuous agent
	return h_xmachine_memory_crystal_default_count;
	
}

xmachine_memory_crystal_list* get_device_crystal_default_agents(){
	return d_crystals_default;
}

xmachine_memory_crystal_list* get_host_crystal_default_agents(){
	return h_crystals_default;
}



/* Host based access of agent variables*/

/** float get_crystal_default_variable_rank(unsigned int index)
 * Gets the value of the rank variable of an crystal agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable rank
 */
__host__ float get_crystal_default_variable_rank(unsigned int index){
    unsigned int count = get_agent_crystal_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_crystals_default_variable_rank_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_crystals_default->rank,
                    d_crystals_default->rank,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_crystals_default_variable_rank_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_crystals_default->rank[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access rank for the %u th member of crystal_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_crystal_default_variable_l(unsigned int index)
 * Gets the value of the l variable of an crystal agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable l
 */
__host__ float get_crystal_default_variable_l(unsigned int index){
    unsigned int count = get_agent_crystal_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_crystals_default_variable_l_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_crystals_default->l,
                    d_crystals_default->l,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_crystals_default_variable_l_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_crystals_default->l[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access l for the %u th member of crystal_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_crystal_default_variable_bin(unsigned int index)
 * Gets the value of the bin variable of an crystal agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable bin
 */
__host__ int get_crystal_default_variable_bin(unsigned int index){
    unsigned int count = get_agent_crystal_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_crystals_default_variable_bin_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_crystals_default->bin,
                    d_crystals_default->bin,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_crystals_default_variable_bin_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_crystals_default->bin[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access bin for the %u th member of crystal_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}



/* Host based agent creation functions */
// These are only available for continuous agents.



/* copy_single_xmachine_memory_crystal_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_crystal_hostToDevice(xmachine_memory_crystal_list * d_dst, xmachine_memory_crystal * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->rank, &h_agent->rank, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->l, &h_agent->l, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->bin, &h_agent->bin, sizeof(int), cudaMemcpyHostToDevice));

}
/*
 * Private function to copy some elements from a host based struct of arrays to a device based struct of arrays for a single agent state.
 * Individual copies of `count` elements are performed for each agent variable or each component of agent array variables, to avoid wasted data transfer.
 * There will be a point at which a single cudaMemcpy will outperform many smaller memcpys, however host based agent creation should typically only populate a fraction of the maximum buffer size, so this should be more efficient.
 * @optimisation - experimentally find the proportion at which transferring the whole SoA would be better and incorporate this. The same will apply to agent variable arrays.
 * 
 * @param d_dst device destination SoA
 * @oaram h_src host source SoA
 * @param count the number of agents to transfer data for
 */
void copy_partial_xmachine_memory_crystal_hostToDevice(xmachine_memory_crystal_list * d_dst, xmachine_memory_crystal_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->rank, h_src->rank, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->l, h_src->l, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->bin, h_src->bin, count * sizeof(int), cudaMemcpyHostToDevice));

    }
}

xmachine_memory_crystal* h_allocate_agent_crystal(){
	xmachine_memory_crystal* agent = (xmachine_memory_crystal*)malloc(sizeof(xmachine_memory_crystal));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_crystal));

	return agent;
}
void h_free_agent_crystal(xmachine_memory_crystal** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_crystal** h_allocate_agent_crystal_array(unsigned int count){
	xmachine_memory_crystal ** agents = (xmachine_memory_crystal**)malloc(count * sizeof(xmachine_memory_crystal*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_crystal();
	}
	return agents;
}
void h_free_agent_crystal_array(xmachine_memory_crystal*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_crystal(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_crystal_AoS_to_SoA(xmachine_memory_crystal_list * dst, xmachine_memory_crystal** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->rank[i] = src[i]->rank;
			 
			dst->l[i] = src[i]->l;
			 
			dst->bin[i] = src[i]->bin;
			
		}
	}
}


void h_add_agent_crystal_default(xmachine_memory_crystal* agent){
	if (h_xmachine_memory_crystal_count + 1 > xmachine_memory_crystal_MAX){
		printf("Error: Buffer size of crystal agents in state default will be exceeded by h_add_agent_crystal_default\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_crystal_hostToDevice(d_crystals_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_crystal_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_crystal_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_crystals_default, d_crystals_new, h_xmachine_memory_crystal_default_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_crystal_default_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_crystal_default_count, &h_xmachine_memory_crystal_default_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_crystals_default_variable_rank_data_iteration = 0;
    h_crystals_default_variable_l_data_iteration = 0;
    h_crystals_default_variable_bin_data_iteration = 0;
    

}
void h_add_agents_crystal_default(xmachine_memory_crystal** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_crystal_count + count > xmachine_memory_crystal_MAX){
			printf("Error: Buffer size of crystal agents in state default will be exceeded by h_add_agents_crystal_default\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_crystal_AoS_to_SoA(h_crystals_default, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_crystal_hostToDevice(d_crystals_new, h_crystals_default, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_crystal_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_crystal_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_crystals_default, d_crystals_new, h_xmachine_memory_crystal_default_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_crystal_default_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_crystal_default_count, &h_xmachine_memory_crystal_default_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_crystals_default_variable_rank_data_iteration = 0;
        h_crystals_default_variable_l_data_iteration = 0;
        h_crystals_default_variable_bin_data_iteration = 0;
        

	}
}


/*  Analytics Functions */

float reduce_crystal_default_rank_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_crystals_default->rank),  thrust::device_pointer_cast(d_crystals_default->rank) + h_xmachine_memory_crystal_default_count);
}

float min_crystal_default_rank_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_crystals_default->rank);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_crystal_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_crystal_default_rank_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_crystals_default->rank);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_crystal_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_crystal_default_l_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_crystals_default->l),  thrust::device_pointer_cast(d_crystals_default->l) + h_xmachine_memory_crystal_default_count);
}

float min_crystal_default_l_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_crystals_default->l);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_crystal_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_crystal_default_l_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_crystals_default->l);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_crystal_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_crystal_default_bin_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_crystals_default->bin),  thrust::device_pointer_cast(d_crystals_default->bin) + h_xmachine_memory_crystal_default_count);
}

int count_crystal_default_bin_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_crystals_default->bin),  thrust::device_pointer_cast(d_crystals_default->bin) + h_xmachine_memory_crystal_default_count, count_value);
}
int min_crystal_default_bin_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_crystals_default->bin);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_crystal_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_crystal_default_bin_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_crystals_default->bin);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_crystal_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}



/* Agent functions */


	
/* Shared memory size calculator for agent function */
int crystal_create_ranks_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** crystal_create_ranks
 * Agent function prototype for create_ranks function of crystal agent
 */
void crystal_create_ranks(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_crystal_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_crystal_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_crystal_list* crystals_default_temp = d_crystals;
	d_crystals = d_crystals_default;
	d_crystals_default = crystals_default_temp;
	//set working count to current state count
	h_xmachine_memory_crystal_count = h_xmachine_memory_crystal_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_crystal_count, &h_xmachine_memory_crystal_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_crystal_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_crystal_default_count, &h_xmachine_memory_crystal_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_internal_coord_count + h_xmachine_memory_crystal_count > xmachine_message_internal_coord_MAX){
		printf("Error: Buffer size of internal_coord message will be exceeded in function create_ranks\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_create_ranks, crystal_create_ranks_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = crystal_create_ranks_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_internal_coord_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_internal_coord_output_type, &h_message_internal_coord_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (create_ranks)
	//Reallocate   : false
	//Input        : 
	//Output       : internal_coord
	//Agent Output : 
	GPUFLAME_create_ranks<<<g, b, sm_size, stream>>>(d_crystals, d_internal_coords, d_rand48);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_internal_coord_count += h_xmachine_memory_crystal_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_internal_coord_count, &h_message_internal_coord_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_crystal_default_count+h_xmachine_memory_crystal_count > xmachine_memory_crystal_MAX){
		printf("Error: Buffer size of create_ranks agents in state default will be exceeded moving working agents to next state in function create_ranks\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  crystals_default_temp = d_crystals;
  d_crystals = d_crystals_default;
  d_crystals_default = crystals_default_temp;
        
	//update new state agent size
	h_xmachine_memory_crystal_default_count += h_xmachine_memory_crystal_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_crystal_default_count, &h_xmachine_memory_crystal_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int crystal_nucleate_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_internal_coord));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** crystal_nucleate
 * Agent function prototype for nucleate function of crystal agent
 */
void crystal_nucleate(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_crystal_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_crystal_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_crystal_list* crystals_default_temp = d_crystals;
	d_crystals = d_crystals_default;
	d_crystals_default = crystals_default_temp;
	//set working count to current state count
	h_xmachine_memory_crystal_count = h_xmachine_memory_crystal_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_crystal_count, &h_xmachine_memory_crystal_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_crystal_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_crystal_default_count, &h_xmachine_memory_crystal_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_nucleate, crystal_nucleate_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = crystal_nucleate_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_crystal_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_crystal_scan_input<<<gridSize, blockSize, 0, stream>>>(d_crystals);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (nucleate)
	//Reallocate   : true
	//Input        : internal_coord
	//Output       : 
	//Agent Output : 
	GPUFLAME_nucleate<<<g, b, sm_size, stream>>>(d_crystals, d_internal_coords);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_crystal, 
        temp_scan_storage_bytes_crystal, 
        d_crystals->_scan_input,
        d_crystals->_position,
        h_xmachine_memory_crystal_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_crystal_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_crystal_Agents<<<gridSize, blockSize, 0, stream>>>(d_crystals_swap, d_crystals, 0, h_xmachine_memory_crystal_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_crystal_list* nucleate_crystals_temp = d_crystals;
	d_crystals = d_crystals_swap;
	d_crystals_swap = nucleate_crystals_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_crystals_swap->_position[h_xmachine_memory_crystal_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_crystals_swap->_scan_input[h_xmachine_memory_crystal_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_crystal_count = scan_last_sum+1;
	else
		h_xmachine_memory_crystal_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_crystal_count, &h_xmachine_memory_crystal_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_crystal_default_count+h_xmachine_memory_crystal_count > xmachine_memory_crystal_MAX){
		printf("Error: Buffer size of nucleate agents in state default will be exceeded moving working agents to next state in function nucleate\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_crystal_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_crystal_Agents<<<gridSize, blockSize, 0, stream>>>(d_crystals_default, d_crystals, h_xmachine_memory_crystal_default_count, h_xmachine_memory_crystal_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_crystal_default_count += h_xmachine_memory_crystal_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_crystal_default_count, &h_xmachine_memory_crystal_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int crystal_growth_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** crystal_growth
 * Agent function prototype for growth function of crystal agent
 */
void crystal_growth(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_crystal_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_crystal_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_crystal_list* crystals_default_temp = d_crystals;
	d_crystals = d_crystals_default;
	d_crystals_default = crystals_default_temp;
	//set working count to current state count
	h_xmachine_memory_crystal_count = h_xmachine_memory_crystal_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_crystal_count, &h_xmachine_memory_crystal_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_crystal_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_crystal_default_count, &h_xmachine_memory_crystal_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_growth, crystal_growth_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = crystal_growth_sm_size(blockSize);
	
	
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_crystal_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_crystal_scan_input<<<gridSize, blockSize, 0, stream>>>(d_crystals);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (growth)
	//Reallocate   : true
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_growth<<<g, b, sm_size, stream>>>(d_crystals);
	gpuErrchkLaunch();
	
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_crystal, 
        temp_scan_storage_bytes_crystal, 
        d_crystals->_scan_input,
        d_crystals->_position,
        h_xmachine_memory_crystal_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_crystal_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_crystal_Agents<<<gridSize, blockSize, 0, stream>>>(d_crystals_swap, d_crystals, 0, h_xmachine_memory_crystal_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_crystal_list* growth_crystals_temp = d_crystals;
	d_crystals = d_crystals_swap;
	d_crystals_swap = growth_crystals_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_crystals_swap->_position[h_xmachine_memory_crystal_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_crystals_swap->_scan_input[h_xmachine_memory_crystal_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_crystal_count = scan_last_sum+1;
	else
		h_xmachine_memory_crystal_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_crystal_count, &h_xmachine_memory_crystal_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_crystal_default_count+h_xmachine_memory_crystal_count > xmachine_memory_crystal_MAX){
		printf("Error: Buffer size of growth agents in state default will be exceeded moving working agents to next state in function growth\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_crystal_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_crystal_Agents<<<gridSize, blockSize, 0, stream>>>(d_crystals_default, d_crystals, h_xmachine_memory_crystal_default_count, h_xmachine_memory_crystal_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_crystal_default_count += h_xmachine_memory_crystal_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_crystal_default_count, &h_xmachine_memory_crystal_default_count, sizeof(int)));	
	
	
}


 
extern void reset_crystal_default_count()
{
    h_xmachine_memory_crystal_default_count = 0;
}
