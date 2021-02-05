
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

/* agent Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_agent_list* d_agents;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_agent_list* d_agents_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_agent_list* d_agents_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_agent_count;   /**< Agent population size counter */ 
int h_xmachine_memory_agent_pop_width;   /**< Agent population width */
uint * d_xmachine_memory_agent_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_agent_values;  /**< Agent sort identifiers value */

/* agent state variables */
xmachine_memory_agent_list* h_agents_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_agent_list* d_agents_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_agent_default_count;   /**< Agent population size counter */ 


/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
unsigned int h_agents_default_variable_location_id_data_iteration;
unsigned int h_agents_default_variable_agent_id_data_iteration;
unsigned int h_agents_default_variable_state_data_iteration;
unsigned int h_agents_default_variable_sugar_level_data_iteration;
unsigned int h_agents_default_variable_metabolism_data_iteration;
unsigned int h_agents_default_variable_env_sugar_level_data_iteration;


/* Message Memory */

/* cell_state Message variables */
xmachine_message_cell_state_list* h_cell_states;         /**< Pointer to message list on host*/
xmachine_message_cell_state_list* d_cell_states;         /**< Pointer to message list on device*/
xmachine_message_cell_state_list* d_cell_states_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Discrete Partitioning Variables*/
int h_message_cell_state_range;     /**< range of the discrete message*/
int h_message_cell_state_width;     /**< with of the message grid*/
/* Texture offset values for host */
int h_tex_xmachine_message_cell_state_location_id_offset;
int h_tex_xmachine_message_cell_state_state_offset;
int h_tex_xmachine_message_cell_state_env_sugar_level_offset;
/* movement_request Message variables */
xmachine_message_movement_request_list* h_movement_requests;         /**< Pointer to message list on host*/
xmachine_message_movement_request_list* d_movement_requests;         /**< Pointer to message list on device*/
xmachine_message_movement_request_list* d_movement_requests_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Discrete Partitioning Variables*/
int h_message_movement_request_range;     /**< range of the discrete message*/
int h_message_movement_request_width;     /**< with of the message grid*/
/* Texture offset values for host */
int h_tex_xmachine_message_movement_request_agent_id_offset;
int h_tex_xmachine_message_movement_request_location_id_offset;
int h_tex_xmachine_message_movement_request_sugar_level_offset;
int h_tex_xmachine_message_movement_request_metabolism_offset;
/* movement_response Message variables */
xmachine_message_movement_response_list* h_movement_responses;         /**< Pointer to message list on host*/
xmachine_message_movement_response_list* d_movement_responses;         /**< Pointer to message list on device*/
xmachine_message_movement_response_list* d_movement_responses_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Discrete Partitioning Variables*/
int h_message_movement_response_range;     /**< range of the discrete message*/
int h_message_movement_response_width;     /**< with of the message grid*/
/* Texture offset values for host */
int h_tex_xmachine_message_movement_response_location_id_offset;
int h_tex_xmachine_message_movement_response_agent_id_offset;
  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_agent;
size_t temp_scan_storage_bytes_agent;


/*Global condition counts*/
int h_metabolise_and_growback_condition_count;


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

/** agent_metabolise_and_growback
 * Agent function prototype for metabolise_and_growback function of agent agent
 */
void agent_metabolise_and_growback(cudaStream_t &stream);

/** agent_output_cell_state
 * Agent function prototype for output_cell_state function of agent agent
 */
void agent_output_cell_state(cudaStream_t &stream);

/** agent_movement_request
 * Agent function prototype for movement_request function of agent agent
 */
void agent_movement_request(cudaStream_t &stream);

/** agent_movement_response
 * Agent function prototype for movement_response function of agent agent
 */
void agent_movement_response(cudaStream_t &stream);

/** agent_movement_transaction
 * Agent function prototype for movement_transaction function of agent agent
 */
void agent_movement_transaction(cudaStream_t &stream);

  
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
    h_agents_default_variable_location_id_data_iteration = 0;
    h_agents_default_variable_agent_id_data_iteration = 0;
    h_agents_default_variable_state_data_iteration = 0;
    h_agents_default_variable_sugar_level_data_iteration = 0;
    h_agents_default_variable_metabolism_data_iteration = 0;
    h_agents_default_variable_env_sugar_level_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_agent_SoA_size = sizeof(xmachine_memory_agent_list);
	h_agents_default = (xmachine_memory_agent_list*)malloc(xmachine_agent_SoA_size);

	/* Message memory allocation (CPU) */
	int message_cell_state_SoA_size = sizeof(xmachine_message_cell_state_list);
	h_cell_states = (xmachine_message_cell_state_list*)malloc(message_cell_state_SoA_size);
	int message_movement_request_SoA_size = sizeof(xmachine_message_movement_request_list);
	h_movement_requests = (xmachine_message_movement_request_list*)malloc(message_movement_request_SoA_size);
	int message_movement_response_SoA_size = sizeof(xmachine_message_movement_response_list);
	h_movement_responses = (xmachine_message_movement_response_list*)malloc(message_movement_response_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs

  /* Graph memory allocation (CPU) */
  

    PROFILE_POP_RANGE(); //"allocate host"
	
	
	/* Set discrete cell_state message variables (range, width)*/
	h_message_cell_state_range = 1; //from xml
	h_message_cell_state_width = (int)floor(sqrt((float)xmachine_message_cell_state_MAX));
	//check the width
	if (!is_sqr_pow2(xmachine_message_cell_state_MAX)){
		printf("ERROR: cell_state message max must be a square power of 2 for a 2D discrete message grid!\n");
		exit(EXIT_FAILURE);
	}
	gpuErrchk(cudaMemcpyToSymbol( d_message_cell_state_range, &h_message_cell_state_range, sizeof(int)));	
	gpuErrchk(cudaMemcpyToSymbol( d_message_cell_state_width, &h_message_cell_state_width, sizeof(int)));
	
	
	/* Set discrete movement_request message variables (range, width)*/
	h_message_movement_request_range = 1; //from xml
	h_message_movement_request_width = (int)floor(sqrt((float)xmachine_message_movement_request_MAX));
	//check the width
	if (!is_sqr_pow2(xmachine_message_movement_request_MAX)){
		printf("ERROR: movement_request message max must be a square power of 2 for a 2D discrete message grid!\n");
		exit(EXIT_FAILURE);
	}
	gpuErrchk(cudaMemcpyToSymbol( d_message_movement_request_range, &h_message_movement_request_range, sizeof(int)));	
	gpuErrchk(cudaMemcpyToSymbol( d_message_movement_request_width, &h_message_movement_request_width, sizeof(int)));
	
	
	/* Set discrete movement_response message variables (range, width)*/
	h_message_movement_response_range = 1; //from xml
	h_message_movement_response_width = (int)floor(sqrt((float)xmachine_message_movement_response_MAX));
	//check the width
	if (!is_sqr_pow2(xmachine_message_movement_response_MAX)){
		printf("ERROR: movement_response message max must be a square power of 2 for a 2D discrete message grid!\n");
		exit(EXIT_FAILURE);
	}
	gpuErrchk(cudaMemcpyToSymbol( d_message_movement_response_range, &h_message_movement_response_range, sizeof(int)));	
	gpuErrchk(cudaMemcpyToSymbol( d_message_movement_response_width, &h_message_movement_response_width, sizeof(int)));
	
	/* Check that population size is a square power of 2*/
	if (!is_sqr_pow2(xmachine_memory_agent_MAX)){
		printf("ERROR: agents agent count must be a square power of 2!\n");
		exit(EXIT_FAILURE);
	}
	h_xmachine_memory_agent_pop_width = (int)sqrt(xmachine_memory_agent_MAX);
	

	//read initial states
	readInitialStates(inputfile, h_agents_default, &h_xmachine_memory_agent_default_count);

  // Read graphs from disk
  

  PROFILE_PUSH_RANGE("allocate device");
	
	/* agent Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_agents, xmachine_agent_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_agents_swap, xmachine_agent_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_agents_new, xmachine_agent_SoA_size));
    
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_agents_default, xmachine_agent_SoA_size));
	gpuErrchk( cudaMemcpy( d_agents_default, h_agents_default, xmachine_agent_SoA_size, cudaMemcpyHostToDevice));
    
	/* cell_state Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_cell_states, message_cell_state_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_cell_states_swap, message_cell_state_SoA_size));
	gpuErrchk( cudaMemcpy( d_cell_states, h_cell_states, message_cell_state_SoA_size, cudaMemcpyHostToDevice));
	
	/* movement_request Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_movement_requests, message_movement_request_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_movement_requests_swap, message_movement_request_SoA_size));
	gpuErrchk( cudaMemcpy( d_movement_requests, h_movement_requests, message_movement_request_SoA_size, cudaMemcpyHostToDevice));
	
	/* movement_response Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_movement_responses, message_movement_response_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_movement_responses_swap, message_movement_response_SoA_size));
	gpuErrchk( cudaMemcpy( d_movement_responses, h_movement_responses, message_movement_response_SoA_size, cudaMemcpyHostToDevice));
		


  /* Allocate device memory for graphs */
  

    PROFILE_POP_RANGE(); // "allocate device"

    /* Calculate and allocate CUB temporary memory for exclusive scans */
    
    d_temp_scan_storage_agent = nullptr;
    temp_scan_storage_bytes_agent = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_agent_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_agent, temp_scan_storage_bytes_agent));
    

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

	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_agent_default_count: %u\n",get_agent_agent_default_count());
	
#endif
} 



void cleanup(){
    PROFILE_SCOPED_RANGE("cleanup");

    /* Call all exit functions */
	

	/* Agent data free*/
	
	/* agent Agent variables */
	gpuErrchk(cudaFree(d_agents));
	gpuErrchk(cudaFree(d_agents_swap));
	gpuErrchk(cudaFree(d_agents_new));
	
	free( h_agents_default);
	gpuErrchk(cudaFree(d_agents_default));
	

	/* Message data free */
	
	/* cell_state Message variables */
	free( h_cell_states);
	gpuErrchk(cudaFree(d_cell_states));
	gpuErrchk(cudaFree(d_cell_states_swap));
	
	/* movement_request Message variables */
	free( h_movement_requests);
	gpuErrchk(cudaFree(d_movement_requests));
	gpuErrchk(cudaFree(d_movement_requests_swap));
	
	/* movement_response Message variables */
	free( h_movement_responses);
	gpuErrchk(cudaFree(d_movement_responses));
	gpuErrchk(cudaFree(d_movement_responses_swap));
	

    /* Free temporary CUB memory if required. */
    
    if(d_temp_scan_storage_agent != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_agent));
      d_temp_scan_storage_agent = nullptr;
      temp_scan_storage_bytes_agent = 0;
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

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_metabolise_and_growback");
	agent_metabolise_and_growback(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_metabolise_and_growback = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_output_cell_state");
	agent_output_cell_state(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_output_cell_state = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_movement_request");
	agent_movement_request(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_movement_request = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 4*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_movement_response");
	agent_movement_response(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_movement_response = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 5*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_movement_transaction");
	agent_movement_transaction(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_movement_transaction = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_agent_default_count: %u\n",get_agent_agent_default_count());
	
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



/* Agent data access functions*/

    
int get_agent_agent_MAX_count(){
    return xmachine_memory_agent_MAX;
}


int get_agent_agent_default_count(){
	//discrete agent 
	return xmachine_memory_agent_MAX;
}

xmachine_memory_agent_list* get_device_agent_default_agents(){
	return d_agents_default;
}

xmachine_memory_agent_list* get_host_agent_default_agents(){
	return h_agents_default;
}

int get_agent_population_width(){
  return h_xmachine_memory_agent_pop_width;
}



/* Host based access of agent variables*/

/** int get_agent_default_variable_location_id(unsigned int index)
 * Gets the value of the location_id variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable location_id
 */
__host__ int get_agent_default_variable_location_id(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_location_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->location_id,
                    d_agents_default->location_id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_location_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->location_id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access location_id for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_default_variable_agent_id(unsigned int index)
 * Gets the value of the agent_id variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable agent_id
 */
__host__ int get_agent_default_variable_agent_id(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_agent_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->agent_id,
                    d_agents_default->agent_id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_agent_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->agent_id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access agent_id for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_default_variable_state(unsigned int index)
 * Gets the value of the state variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable state
 */
__host__ int get_agent_default_variable_state(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_state_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->state,
                    d_agents_default->state,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_state_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->state[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access state for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_default_variable_sugar_level(unsigned int index)
 * Gets the value of the sugar_level variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable sugar_level
 */
__host__ int get_agent_default_variable_sugar_level(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_sugar_level_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->sugar_level,
                    d_agents_default->sugar_level,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_sugar_level_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->sugar_level[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access sugar_level for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_default_variable_metabolism(unsigned int index)
 * Gets the value of the metabolism variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable metabolism
 */
__host__ int get_agent_default_variable_metabolism(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_metabolism_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->metabolism,
                    d_agents_default->metabolism,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_metabolism_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->metabolism[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access metabolism for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_default_variable_env_sugar_level(unsigned int index)
 * Gets the value of the env_sugar_level variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable env_sugar_level
 */
__host__ int get_agent_default_variable_env_sugar_level(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_env_sugar_level_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->env_sugar_level,
                    d_agents_default->env_sugar_level,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_env_sugar_level_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->env_sugar_level[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access env_sugar_level for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}



/* Host based agent creation functions */
// These are only available for continuous agents.



/*  Analytics Functions */

int reduce_agent_default_location_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->location_id),  thrust::device_pointer_cast(d_agents_default->location_id) + h_xmachine_memory_agent_default_count);
}

int count_agent_default_location_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agents_default->location_id),  thrust::device_pointer_cast(d_agents_default->location_id) + h_xmachine_memory_agent_default_count, count_value);
}
int min_agent_default_location_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->location_id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_default_location_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->location_id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_default_agent_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->agent_id),  thrust::device_pointer_cast(d_agents_default->agent_id) + h_xmachine_memory_agent_default_count);
}

int count_agent_default_agent_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agents_default->agent_id),  thrust::device_pointer_cast(d_agents_default->agent_id) + h_xmachine_memory_agent_default_count, count_value);
}
int min_agent_default_agent_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->agent_id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_default_agent_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->agent_id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_default_state_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->state),  thrust::device_pointer_cast(d_agents_default->state) + h_xmachine_memory_agent_default_count);
}

int count_agent_default_state_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agents_default->state),  thrust::device_pointer_cast(d_agents_default->state) + h_xmachine_memory_agent_default_count, count_value);
}
int min_agent_default_state_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->state);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_default_state_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->state);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_default_sugar_level_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->sugar_level),  thrust::device_pointer_cast(d_agents_default->sugar_level) + h_xmachine_memory_agent_default_count);
}

int count_agent_default_sugar_level_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agents_default->sugar_level),  thrust::device_pointer_cast(d_agents_default->sugar_level) + h_xmachine_memory_agent_default_count, count_value);
}
int min_agent_default_sugar_level_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->sugar_level);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_default_sugar_level_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->sugar_level);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_default_metabolism_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->metabolism),  thrust::device_pointer_cast(d_agents_default->metabolism) + h_xmachine_memory_agent_default_count);
}

int count_agent_default_metabolism_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agents_default->metabolism),  thrust::device_pointer_cast(d_agents_default->metabolism) + h_xmachine_memory_agent_default_count, count_value);
}
int min_agent_default_metabolism_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->metabolism);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_default_metabolism_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->metabolism);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_default_env_sugar_level_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->env_sugar_level),  thrust::device_pointer_cast(d_agents_default->env_sugar_level) + h_xmachine_memory_agent_default_count);
}

int count_agent_default_env_sugar_level_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agents_default->env_sugar_level),  thrust::device_pointer_cast(d_agents_default->env_sugar_level) + h_xmachine_memory_agent_default_count, count_value);
}
int min_agent_default_env_sugar_level_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->env_sugar_level);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_default_env_sugar_level_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->env_sugar_level);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}



/* Agent functions */


	
/* Shared memory size calculator for agent function */
int agent_metabolise_and_growback_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_metabolise_and_growback
 * Agent function prototype for metabolise_and_growback function of agent agent
 */
void agent_metabolise_and_growback(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_agent_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS A GLOBAL CONDITION
	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_agent_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agents_default);
	gpuErrchkLaunch();
	
	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, metabolise_and_growback_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	metabolise_and_growback_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default);
	gpuErrchkLaunch();
	
	//GET CONDTIONS TRUE COUNT FROM CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_agents_default->_scan_input,
        d_agents_default->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agents_default->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agents_default->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	int global_conditions_true = 0;
	if (scan_last_included == 1)
		global_conditions_true = scan_last_sum+1;
	else		
		global_conditions_true = scan_last_sum;
	//check if condition is true for all agents or if max condition count is reached
	if ((global_conditions_true != h_xmachine_memory_agent_count)&&(h_metabolise_and_growback_condition_count < 9))
	{
		h_metabolise_and_growback_condition_count ++;
		return;
	}
	if ((h_metabolise_and_growback_condition_count == 9))
	{
		printf("Global agent condition for metabolise_and_growback function reached the maximum number of 9 conditions\n");
	}
	
	//RESET THE CONDITION COUNT
	h_metabolise_and_growback_condition_count = 0;
	
	//MAP CURRENT STATE TO WORKING LIST
	xmachine_memory_agent_list* agents_default_temp = d_agents;
	d_agents = d_agents_default;
	d_agents_default = agents_default_temp;
	//set current state count to 0
	h_xmachine_memory_agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	
	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_metabolise_and_growback, agent_metabolise_and_growback_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = agent_metabolise_and_growback_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (metabolise_and_growback)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_metabolise_and_growback<<<g, b, sm_size, stream>>>(d_agents);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	agents_default_temp = d_agents_default;
	d_agents_default = d_agents;
	d_agents = agents_default_temp;
    //set current state count
	h_xmachine_memory_agent_default_count = h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_output_cell_state_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_output_cell_state
 * Agent function prototype for output_cell_state function of agent agent
 */
void agent_output_cell_state(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_agent_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_agent_list* agents_default_temp = d_agents;
	d_agents = d_agents_default;
	d_agents_default = agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_cell_state, agent_output_cell_state_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = agent_output_cell_state_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	
	
	//MAIN XMACHINE FUNCTION CALL (output_cell_state)
	//Reallocate   : false
	//Input        : 
	//Output       : cell_state
	//Agent Output : 
	GPUFLAME_output_cell_state<<<g, b, sm_size, stream>>>(d_agents, d_cell_states);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	agents_default_temp = d_agents_default;
	d_agents_default = d_agents;
	d_agents = agents_default_temp;
    //set current state count
	h_xmachine_memory_agent_default_count = h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_movement_request_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Discrete agent and message input has discrete partitioning
	int sm_grid_width = (int)ceil(sqrt(blockSize));
	int sm_grid_size = (int)pow((float)sm_grid_width+(h_message_cell_state_range*2), 2);
	sm_size += (sm_grid_size *sizeof(xmachine_message_cell_state)); //update sm size
	sm_size += (sm_grid_size * PADDING);  //offset for avoiding conflicts
	
	return sm_size;
}

/** agent_movement_request
 * Agent function prototype for movement_request function of agent agent
 */
void agent_movement_request(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_agent_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_agent_list* agents_default_temp = d_agents;
	d_agents = d_agents_default;
	d_agents_default = agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_movement_request, agent_movement_request_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = agent_movement_request_sm_size(blockSize);
	
	
	
	//check that the range is not greater than the square of the block size. If so then there will be too many uncoalesded reads
	if (h_message_cell_state_range > (int)blockSize){
		printf("ERROR: Message range is greater than the thread block size. Increase thread block size or reduce the range!\n");
		exit(EXIT_FAILURE);
	}
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_cell_state_location_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_cell_state_location_id_byte_offset, tex_xmachine_message_cell_state_location_id, d_cell_states->location_id, sizeof(int)*xmachine_message_cell_state_MAX));
	h_tex_xmachine_message_cell_state_location_id_offset = (int)tex_xmachine_message_cell_state_location_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_cell_state_location_id_offset, &h_tex_xmachine_message_cell_state_location_id_offset, sizeof(int)));
	size_t tex_xmachine_message_cell_state_state_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_cell_state_state_byte_offset, tex_xmachine_message_cell_state_state, d_cell_states->state, sizeof(int)*xmachine_message_cell_state_MAX));
	h_tex_xmachine_message_cell_state_state_offset = (int)tex_xmachine_message_cell_state_state_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_cell_state_state_offset, &h_tex_xmachine_message_cell_state_state_offset, sizeof(int)));
	size_t tex_xmachine_message_cell_state_env_sugar_level_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_cell_state_env_sugar_level_byte_offset, tex_xmachine_message_cell_state_env_sugar_level, d_cell_states->env_sugar_level, sizeof(int)*xmachine_message_cell_state_MAX));
	h_tex_xmachine_message_cell_state_env_sugar_level_offset = (int)tex_xmachine_message_cell_state_env_sugar_level_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_cell_state_env_sugar_level_offset, &h_tex_xmachine_message_cell_state_env_sugar_level_offset, sizeof(int)));
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	
	
	//MAIN XMACHINE FUNCTION CALL (movement_request)
	//Reallocate   : false
	//Input        : cell_state
	//Output       : movement_request
	//Agent Output : 
	GPUFLAME_movement_request<<<g, b, sm_size, stream>>>(d_agents, d_cell_states, d_movement_requests);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_cell_state_location_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_cell_state_state));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_cell_state_env_sugar_level));
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	agents_default_temp = d_agents_default;
	d_agents_default = d_agents;
	d_agents = agents_default_temp;
    //set current state count
	h_xmachine_memory_agent_default_count = h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_movement_response_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Discrete agent and message input has discrete partitioning
	int sm_grid_width = (int)ceil(sqrt(blockSize));
	int sm_grid_size = (int)pow((float)sm_grid_width+(h_message_movement_request_range*2), 2);
	sm_size += (sm_grid_size *sizeof(xmachine_message_movement_request)); //update sm size
	sm_size += (sm_grid_size * PADDING);  //offset for avoiding conflicts
	
	return sm_size;
}

/** agent_movement_response
 * Agent function prototype for movement_response function of agent agent
 */
void agent_movement_response(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_agent_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_agent_list* agents_default_temp = d_agents;
	d_agents = d_agents_default;
	d_agents_default = agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_movement_response, agent_movement_response_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = agent_movement_response_sm_size(blockSize);
	
	
	
	//check that the range is not greater than the square of the block size. If so then there will be too many uncoalesded reads
	if (h_message_movement_request_range > (int)blockSize){
		printf("ERROR: Message range is greater than the thread block size. Increase thread block size or reduce the range!\n");
		exit(EXIT_FAILURE);
	}
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_movement_request_agent_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_movement_request_agent_id_byte_offset, tex_xmachine_message_movement_request_agent_id, d_movement_requests->agent_id, sizeof(int)*xmachine_message_movement_request_MAX));
	h_tex_xmachine_message_movement_request_agent_id_offset = (int)tex_xmachine_message_movement_request_agent_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_movement_request_agent_id_offset, &h_tex_xmachine_message_movement_request_agent_id_offset, sizeof(int)));
	size_t tex_xmachine_message_movement_request_location_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_movement_request_location_id_byte_offset, tex_xmachine_message_movement_request_location_id, d_movement_requests->location_id, sizeof(int)*xmachine_message_movement_request_MAX));
	h_tex_xmachine_message_movement_request_location_id_offset = (int)tex_xmachine_message_movement_request_location_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_movement_request_location_id_offset, &h_tex_xmachine_message_movement_request_location_id_offset, sizeof(int)));
	size_t tex_xmachine_message_movement_request_sugar_level_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_movement_request_sugar_level_byte_offset, tex_xmachine_message_movement_request_sugar_level, d_movement_requests->sugar_level, sizeof(int)*xmachine_message_movement_request_MAX));
	h_tex_xmachine_message_movement_request_sugar_level_offset = (int)tex_xmachine_message_movement_request_sugar_level_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_movement_request_sugar_level_offset, &h_tex_xmachine_message_movement_request_sugar_level_offset, sizeof(int)));
	size_t tex_xmachine_message_movement_request_metabolism_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_movement_request_metabolism_byte_offset, tex_xmachine_message_movement_request_metabolism, d_movement_requests->metabolism, sizeof(int)*xmachine_message_movement_request_MAX));
	h_tex_xmachine_message_movement_request_metabolism_offset = (int)tex_xmachine_message_movement_request_metabolism_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_movement_request_metabolism_offset, &h_tex_xmachine_message_movement_request_metabolism_offset, sizeof(int)));
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	
	
	//MAIN XMACHINE FUNCTION CALL (movement_response)
	//Reallocate   : false
	//Input        : movement_request
	//Output       : movement_response
	//Agent Output : 
	GPUFLAME_movement_response<<<g, b, sm_size, stream>>>(d_agents, d_movement_requests, d_movement_responses, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_movement_request_agent_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_movement_request_location_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_movement_request_sugar_level));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_movement_request_metabolism));
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	agents_default_temp = d_agents_default;
	d_agents_default = d_agents;
	d_agents = agents_default_temp;
    //set current state count
	h_xmachine_memory_agent_default_count = h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_movement_transaction_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Discrete agent and message input has discrete partitioning
	int sm_grid_width = (int)ceil(sqrt(blockSize));
	int sm_grid_size = (int)pow((float)sm_grid_width+(h_message_movement_response_range*2), 2);
	sm_size += (sm_grid_size *sizeof(xmachine_message_movement_response)); //update sm size
	sm_size += (sm_grid_size * PADDING);  //offset for avoiding conflicts
	
	return sm_size;
}

/** agent_movement_transaction
 * Agent function prototype for movement_transaction function of agent agent
 */
void agent_movement_transaction(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_agent_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_agent_list* agents_default_temp = d_agents;
	d_agents = d_agents_default;
	d_agents_default = agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_movement_transaction, agent_movement_transaction_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = agent_movement_transaction_sm_size(blockSize);
	
	
	
	//check that the range is not greater than the square of the block size. If so then there will be too many uncoalesded reads
	if (h_message_movement_response_range > (int)blockSize){
		printf("ERROR: Message range is greater than the thread block size. Increase thread block size or reduce the range!\n");
		exit(EXIT_FAILURE);
	}
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_movement_response_location_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_movement_response_location_id_byte_offset, tex_xmachine_message_movement_response_location_id, d_movement_responses->location_id, sizeof(int)*xmachine_message_movement_response_MAX));
	h_tex_xmachine_message_movement_response_location_id_offset = (int)tex_xmachine_message_movement_response_location_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_movement_response_location_id_offset, &h_tex_xmachine_message_movement_response_location_id_offset, sizeof(int)));
	size_t tex_xmachine_message_movement_response_agent_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_movement_response_agent_id_byte_offset, tex_xmachine_message_movement_response_agent_id, d_movement_responses->agent_id, sizeof(int)*xmachine_message_movement_response_MAX));
	h_tex_xmachine_message_movement_response_agent_id_offset = (int)tex_xmachine_message_movement_response_agent_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_movement_response_agent_id_offset, &h_tex_xmachine_message_movement_response_agent_id_offset, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (movement_transaction)
	//Reallocate   : false
	//Input        : movement_response
	//Output       : 
	//Agent Output : 
	GPUFLAME_movement_transaction<<<g, b, sm_size, stream>>>(d_agents, d_movement_responses);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_movement_response_location_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_movement_response_agent_id));
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	agents_default_temp = d_agents_default;
	d_agents_default = d_agents;
	d_agents = agents_default_temp;
    //set current state count
	h_xmachine_memory_agent_default_count = h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}


 
extern void reset_agent_default_count()
{
    h_xmachine_memory_agent_default_count = 0;
}
