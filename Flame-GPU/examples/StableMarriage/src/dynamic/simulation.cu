
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

/* Man Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_Man_list* d_Mans;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_Man_list* d_Mans_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_Man_list* d_Mans_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_Man_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_Man_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_Man_values;  /**< Agent sort identifiers value */

/* Man state variables */
xmachine_memory_Man_list* h_Mans_unengaged;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Man_list* d_Mans_unengaged;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Man_unengaged_count;   /**< Agent population size counter */ 

/* Man state variables */
xmachine_memory_Man_list* h_Mans_engaged;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Man_list* d_Mans_engaged;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Man_engaged_count;   /**< Agent population size counter */ 

/* Woman Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_Woman_list* d_Womans;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_Woman_list* d_Womans_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_Woman_list* d_Womans_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_Woman_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_Woman_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_Woman_values;  /**< Agent sort identifiers value */

/* Woman state variables */
xmachine_memory_Woman_list* h_Womans_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Woman_list* d_Womans_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Woman_default_count;   /**< Agent population size counter */ 


/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
unsigned int h_Mans_unengaged_variable_id_data_iteration;
unsigned int h_Mans_unengaged_variable_round_data_iteration;
unsigned int h_Mans_unengaged_variable_engaged_to_data_iteration;
unsigned int h_Mans_unengaged_variable_preferred_woman_data_iteration;
unsigned int h_Mans_engaged_variable_id_data_iteration;
unsigned int h_Mans_engaged_variable_round_data_iteration;
unsigned int h_Mans_engaged_variable_engaged_to_data_iteration;
unsigned int h_Mans_engaged_variable_preferred_woman_data_iteration;
unsigned int h_Womans_default_variable_id_data_iteration;
unsigned int h_Womans_default_variable_current_suitor_data_iteration;
unsigned int h_Womans_default_variable_current_suitor_rank_data_iteration;
unsigned int h_Womans_default_variable_preferred_man_data_iteration;


/* Message Memory */

/* proposal Message variables */
xmachine_message_proposal_list* h_proposals;         /**< Pointer to message list on host*/
xmachine_message_proposal_list* d_proposals;         /**< Pointer to message list on device*/
xmachine_message_proposal_list* d_proposals_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_proposal_count;         /**< message list counter*/
int h_message_proposal_output_type;   /**< message output type (single or optional)*/

/* notification Message variables */
xmachine_message_notification_list* h_notifications;         /**< Pointer to message list on host*/
xmachine_message_notification_list* d_notifications;         /**< Pointer to message list on device*/
xmachine_message_notification_list* d_notifications_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_notification_count;         /**< message list counter*/
int h_message_notification_output_type;   /**< message output type (single or optional)*/

  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_Man;
size_t temp_scan_storage_bytes_Man;

void * d_temp_scan_storage_Woman;
size_t temp_scan_storage_bytes_Woman;


/*Global condition counts*/
int h_check_resolved_condition_count;


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

/** Man_make_proposals
 * Agent function prototype for make_proposals function of Man agent
 */
void Man_make_proposals(cudaStream_t &stream);

/** Man_check_notifications
 * Agent function prototype for check_notifications function of Man agent
 */
void Man_check_notifications(cudaStream_t &stream);

/** Man_check_resolved
 * Agent function prototype for check_resolved function of Man agent
 */
void Man_check_resolved(cudaStream_t &stream);

/** Woman_check_proposals
 * Agent function prototype for check_proposals function of Woman agent
 */
void Woman_check_proposals(cudaStream_t &stream);

/** Woman_notify_suitors
 * Agent function prototype for notify_suitors function of Woman agent
 */
void Woman_notify_suitors(cudaStream_t &stream);

  
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
    h_Mans_unengaged_variable_id_data_iteration = 0;
    h_Mans_unengaged_variable_round_data_iteration = 0;
    h_Mans_unengaged_variable_engaged_to_data_iteration = 0;
    h_Mans_unengaged_variable_preferred_woman_data_iteration = 0;
    h_Mans_engaged_variable_id_data_iteration = 0;
    h_Mans_engaged_variable_round_data_iteration = 0;
    h_Mans_engaged_variable_engaged_to_data_iteration = 0;
    h_Mans_engaged_variable_preferred_woman_data_iteration = 0;
    h_Womans_default_variable_id_data_iteration = 0;
    h_Womans_default_variable_current_suitor_data_iteration = 0;
    h_Womans_default_variable_current_suitor_rank_data_iteration = 0;
    h_Womans_default_variable_preferred_man_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_Man_SoA_size = sizeof(xmachine_memory_Man_list);
	h_Mans_unengaged = (xmachine_memory_Man_list*)malloc(xmachine_Man_SoA_size);
	h_Mans_engaged = (xmachine_memory_Man_list*)malloc(xmachine_Man_SoA_size);
	int xmachine_Woman_SoA_size = sizeof(xmachine_memory_Woman_list);
	h_Womans_default = (xmachine_memory_Woman_list*)malloc(xmachine_Woman_SoA_size);

	/* Message memory allocation (CPU) */
	int message_proposal_SoA_size = sizeof(xmachine_message_proposal_list);
	h_proposals = (xmachine_message_proposal_list*)malloc(message_proposal_SoA_size);
	int message_notification_SoA_size = sizeof(xmachine_message_notification_list);
	h_notifications = (xmachine_message_notification_list*)malloc(message_notification_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs

  /* Graph memory allocation (CPU) */
  

    PROFILE_POP_RANGE(); //"allocate host"
	

	//read initial states
	readInitialStates(inputfile, h_Mans_unengaged, &h_xmachine_memory_Man_unengaged_count, h_Womans_default, &h_xmachine_memory_Woman_default_count);

  // Read graphs from disk
  

  PROFILE_PUSH_RANGE("allocate device");
	
	/* Man Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Mans, xmachine_Man_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Mans_swap, xmachine_Man_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Mans_new, xmachine_Man_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Man_keys, xmachine_memory_Man_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Man_values, xmachine_memory_Man_MAX* sizeof(uint)));
	/* unengaged memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Mans_unengaged, xmachine_Man_SoA_size));
	gpuErrchk( cudaMemcpy( d_Mans_unengaged, h_Mans_unengaged, xmachine_Man_SoA_size, cudaMemcpyHostToDevice));
    
	/* engaged memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Mans_engaged, xmachine_Man_SoA_size));
	gpuErrchk( cudaMemcpy( d_Mans_engaged, h_Mans_engaged, xmachine_Man_SoA_size, cudaMemcpyHostToDevice));
    
	/* Woman Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Womans, xmachine_Woman_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Womans_swap, xmachine_Woman_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Womans_new, xmachine_Woman_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Woman_keys, xmachine_memory_Woman_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Woman_values, xmachine_memory_Woman_MAX* sizeof(uint)));
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Womans_default, xmachine_Woman_SoA_size));
	gpuErrchk( cudaMemcpy( d_Womans_default, h_Womans_default, xmachine_Woman_SoA_size, cudaMemcpyHostToDevice));
    
	/* proposal Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_proposals, message_proposal_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_proposals_swap, message_proposal_SoA_size));
	gpuErrchk( cudaMemcpy( d_proposals, h_proposals, message_proposal_SoA_size, cudaMemcpyHostToDevice));
	
	/* notification Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_notifications, message_notification_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_notifications_swap, message_notification_SoA_size));
	gpuErrchk( cudaMemcpy( d_notifications, h_notifications, message_notification_SoA_size, cudaMemcpyHostToDevice));
		


  /* Allocate device memory for graphs */
  

    PROFILE_POP_RANGE(); // "allocate device"

    /* Calculate and allocate CUB temporary memory for exclusive scans */
    
    d_temp_scan_storage_Man = nullptr;
    temp_scan_storage_bytes_Man = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Man, 
        temp_scan_storage_bytes_Man, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_Man_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_Man, temp_scan_storage_bytes_Man));
    
    d_temp_scan_storage_Woman = nullptr;
    temp_scan_storage_bytes_Woman = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Woman, 
        temp_scan_storage_bytes_Woman, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_Woman_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_Woman, temp_scan_storage_bytes_Woman));
    

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
	
		printf("Init agent_Man_unengaged_count: %u\n",get_agent_Man_unengaged_count());
	
		printf("Init agent_Man_engaged_count: %u\n",get_agent_Man_engaged_count());
	
		printf("Init agent_Woman_default_count: %u\n",get_agent_Woman_default_count());
	
#endif
} 


void sort_Mans_unengaged(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Man_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Man_unengaged_count); 
	gridSize = (h_xmachine_memory_Man_unengaged_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Man_keys, d_xmachine_memory_Man_values, d_Mans_unengaged);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Man_keys),  thrust::device_pointer_cast(d_xmachine_memory_Man_keys) + h_xmachine_memory_Man_unengaged_count,  thrust::device_pointer_cast(d_xmachine_memory_Man_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Man_agents, no_sm, h_xmachine_memory_Man_unengaged_count); 
	gridSize = (h_xmachine_memory_Man_unengaged_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Man_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Man_values, d_Mans_unengaged, d_Mans_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Man_list* d_Mans_temp = d_Mans_unengaged;
	d_Mans_unengaged = d_Mans_swap;
	d_Mans_swap = d_Mans_temp;	
}

void sort_Mans_engaged(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Man_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Man_engaged_count); 
	gridSize = (h_xmachine_memory_Man_engaged_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Man_keys, d_xmachine_memory_Man_values, d_Mans_engaged);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Man_keys),  thrust::device_pointer_cast(d_xmachine_memory_Man_keys) + h_xmachine_memory_Man_engaged_count,  thrust::device_pointer_cast(d_xmachine_memory_Man_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Man_agents, no_sm, h_xmachine_memory_Man_engaged_count); 
	gridSize = (h_xmachine_memory_Man_engaged_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Man_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Man_values, d_Mans_engaged, d_Mans_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Man_list* d_Mans_temp = d_Mans_engaged;
	d_Mans_engaged = d_Mans_swap;
	d_Mans_swap = d_Mans_temp;	
}

void sort_Womans_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Woman_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Woman_default_count); 
	gridSize = (h_xmachine_memory_Woman_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Woman_keys, d_xmachine_memory_Woman_values, d_Womans_default);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Woman_keys),  thrust::device_pointer_cast(d_xmachine_memory_Woman_keys) + h_xmachine_memory_Woman_default_count,  thrust::device_pointer_cast(d_xmachine_memory_Woman_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Woman_agents, no_sm, h_xmachine_memory_Woman_default_count); 
	gridSize = (h_xmachine_memory_Woman_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Woman_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Woman_values, d_Womans_default, d_Womans_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Woman_list* d_Womans_temp = d_Womans_default;
	d_Womans_default = d_Womans_swap;
	d_Womans_swap = d_Womans_temp;	
}


void cleanup(){
    PROFILE_SCOPED_RANGE("cleanup");

    /* Call all exit functions */
	

	/* Agent data free*/
	
	/* Man Agent variables */
	gpuErrchk(cudaFree(d_Mans));
	gpuErrchk(cudaFree(d_Mans_swap));
	gpuErrchk(cudaFree(d_Mans_new));
	
	free( h_Mans_unengaged);
	gpuErrchk(cudaFree(d_Mans_unengaged));
	
	free( h_Mans_engaged);
	gpuErrchk(cudaFree(d_Mans_engaged));
	
	/* Woman Agent variables */
	gpuErrchk(cudaFree(d_Womans));
	gpuErrchk(cudaFree(d_Womans_swap));
	gpuErrchk(cudaFree(d_Womans_new));
	
	free( h_Womans_default);
	gpuErrchk(cudaFree(d_Womans_default));
	

	/* Message data free */
	
	/* proposal Message variables */
	free( h_proposals);
	gpuErrchk(cudaFree(d_proposals));
	gpuErrchk(cudaFree(d_proposals_swap));
	
	/* notification Message variables */
	free( h_notifications);
	gpuErrchk(cudaFree(d_notifications));
	gpuErrchk(cudaFree(d_notifications_swap));
	

    /* Free temporary CUB memory if required. */
    
    if(d_temp_scan_storage_Man != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_Man));
      d_temp_scan_storage_Man = nullptr;
      temp_scan_storage_bytes_Man = 0;
    }
    
    if(d_temp_scan_storage_Woman != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_Woman));
      d_temp_scan_storage_Woman = nullptr;
      temp_scan_storage_bytes_Woman = 0;
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
	h_message_proposal_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_proposal_count, &h_message_proposal_count, sizeof(int)));
	
	h_message_notification_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_notification_count, &h_message_notification_count, sizeof(int)));
	

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Man_make_proposals");
	Man_make_proposals(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Man_make_proposals = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Woman_check_proposals");
	Woman_check_proposals(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Woman_check_proposals = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Woman_notify_suitors");
	Woman_notify_suitors(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Woman_notify_suitors = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 4*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Man_check_notifications");
	Man_check_notifications(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Man_check_notifications = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 5*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Man_check_resolved");
	Man_check_resolved(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Man_check_resolved = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_Man_unengaged_count: %u\n",get_agent_Man_unengaged_count());
	
		printf("agent_Man_engaged_count: %u\n",get_agent_Man_engaged_count());
	
		printf("agent_Woman_default_count: %u\n",get_agent_Woman_default_count());
	
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

    
int get_agent_Man_MAX_count(){
    return xmachine_memory_Man_MAX;
}


int get_agent_Man_unengaged_count(){
	//continuous agent
	return h_xmachine_memory_Man_unengaged_count;
	
}

xmachine_memory_Man_list* get_device_Man_unengaged_agents(){
	return d_Mans_unengaged;
}

xmachine_memory_Man_list* get_host_Man_unengaged_agents(){
	return h_Mans_unengaged;
}

int get_agent_Man_engaged_count(){
	//continuous agent
	return h_xmachine_memory_Man_engaged_count;
	
}

xmachine_memory_Man_list* get_device_Man_engaged_agents(){
	return d_Mans_engaged;
}

xmachine_memory_Man_list* get_host_Man_engaged_agents(){
	return h_Mans_engaged;
}

    
int get_agent_Woman_MAX_count(){
    return xmachine_memory_Woman_MAX;
}


int get_agent_Woman_default_count(){
	//continuous agent
	return h_xmachine_memory_Woman_default_count;
	
}

xmachine_memory_Woman_list* get_device_Woman_default_agents(){
	return d_Womans_default;
}

xmachine_memory_Woman_list* get_host_Woman_default_agents(){
	return h_Womans_default;
}



/* Host based access of agent variables*/

/** int get_Man_unengaged_variable_id(unsigned int index)
 * Gets the value of the id variable of an Man agent in the unengaged state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_Man_unengaged_variable_id(unsigned int index){
    unsigned int count = get_agent_Man_unengaged_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Mans_unengaged_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Mans_unengaged->id,
                    d_Mans_unengaged->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Mans_unengaged_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Mans_unengaged->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of Man_unengaged. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Man_unengaged_variable_round(unsigned int index)
 * Gets the value of the round variable of an Man agent in the unengaged state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable round
 */
__host__ int get_Man_unengaged_variable_round(unsigned int index){
    unsigned int count = get_agent_Man_unengaged_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Mans_unengaged_variable_round_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Mans_unengaged->round,
                    d_Mans_unengaged->round,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Mans_unengaged_variable_round_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Mans_unengaged->round[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access round for the %u th member of Man_unengaged. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Man_unengaged_variable_engaged_to(unsigned int index)
 * Gets the value of the engaged_to variable of an Man agent in the unengaged state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable engaged_to
 */
__host__ int get_Man_unengaged_variable_engaged_to(unsigned int index){
    unsigned int count = get_agent_Man_unengaged_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Mans_unengaged_variable_engaged_to_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Mans_unengaged->engaged_to,
                    d_Mans_unengaged->engaged_to,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Mans_unengaged_variable_engaged_to_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Mans_unengaged->engaged_to[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access engaged_to for the %u th member of Man_unengaged. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Man_unengaged_variable_preferred_woman(unsigned int index, unsigned int element)
 * Gets the element-th value of the preferred_woman variable array of an Man agent in the unengaged state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable preferred_woman
 */
__host__ int get_Man_unengaged_variable_preferred_woman(unsigned int index, unsigned int element){
    unsigned int count = get_agent_Man_unengaged_count();
    unsigned int numElements = 1024;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Mans_unengaged_variable_preferred_woman_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_Mans_unengaged->preferred_woman + (e * xmachine_memory_Man_MAX),
                        d_Mans_unengaged->preferred_woman + (e * xmachine_memory_Man_MAX), 
                        count * sizeof(int), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_Mans_unengaged_variable_preferred_woman_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Mans_unengaged->preferred_woman[index + (element * xmachine_memory_Man_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of preferred_woman for the %u th member of Man_unengaged. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Man_engaged_variable_id(unsigned int index)
 * Gets the value of the id variable of an Man agent in the engaged state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_Man_engaged_variable_id(unsigned int index){
    unsigned int count = get_agent_Man_engaged_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Mans_engaged_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Mans_engaged->id,
                    d_Mans_engaged->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Mans_engaged_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Mans_engaged->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of Man_engaged. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Man_engaged_variable_round(unsigned int index)
 * Gets the value of the round variable of an Man agent in the engaged state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable round
 */
__host__ int get_Man_engaged_variable_round(unsigned int index){
    unsigned int count = get_agent_Man_engaged_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Mans_engaged_variable_round_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Mans_engaged->round,
                    d_Mans_engaged->round,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Mans_engaged_variable_round_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Mans_engaged->round[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access round for the %u th member of Man_engaged. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Man_engaged_variable_engaged_to(unsigned int index)
 * Gets the value of the engaged_to variable of an Man agent in the engaged state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable engaged_to
 */
__host__ int get_Man_engaged_variable_engaged_to(unsigned int index){
    unsigned int count = get_agent_Man_engaged_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Mans_engaged_variable_engaged_to_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Mans_engaged->engaged_to,
                    d_Mans_engaged->engaged_to,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Mans_engaged_variable_engaged_to_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Mans_engaged->engaged_to[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access engaged_to for the %u th member of Man_engaged. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Man_engaged_variable_preferred_woman(unsigned int index, unsigned int element)
 * Gets the element-th value of the preferred_woman variable array of an Man agent in the engaged state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable preferred_woman
 */
__host__ int get_Man_engaged_variable_preferred_woman(unsigned int index, unsigned int element){
    unsigned int count = get_agent_Man_engaged_count();
    unsigned int numElements = 1024;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Mans_engaged_variable_preferred_woman_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_Mans_engaged->preferred_woman + (e * xmachine_memory_Man_MAX),
                        d_Mans_engaged->preferred_woman + (e * xmachine_memory_Man_MAX), 
                        count * sizeof(int), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_Mans_engaged_variable_preferred_woman_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Mans_engaged->preferred_woman[index + (element * xmachine_memory_Man_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of preferred_woman for the %u th member of Man_engaged. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Woman_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an Woman agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_Woman_default_variable_id(unsigned int index){
    unsigned int count = get_agent_Woman_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Womans_default_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Womans_default->id,
                    d_Womans_default->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Womans_default_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Womans_default->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of Woman_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Woman_default_variable_current_suitor(unsigned int index)
 * Gets the value of the current_suitor variable of an Woman agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_suitor
 */
__host__ int get_Woman_default_variable_current_suitor(unsigned int index){
    unsigned int count = get_agent_Woman_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Womans_default_variable_current_suitor_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Womans_default->current_suitor,
                    d_Womans_default->current_suitor,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Womans_default_variable_current_suitor_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Womans_default->current_suitor[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access current_suitor for the %u th member of Woman_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Woman_default_variable_current_suitor_rank(unsigned int index)
 * Gets the value of the current_suitor_rank variable of an Woman agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_suitor_rank
 */
__host__ int get_Woman_default_variable_current_suitor_rank(unsigned int index){
    unsigned int count = get_agent_Woman_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Womans_default_variable_current_suitor_rank_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Womans_default->current_suitor_rank,
                    d_Womans_default->current_suitor_rank,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Womans_default_variable_current_suitor_rank_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Womans_default->current_suitor_rank[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access current_suitor_rank for the %u th member of Woman_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_Woman_default_variable_preferred_man(unsigned int index, unsigned int element)
 * Gets the element-th value of the preferred_man variable array of an Woman agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable preferred_man
 */
__host__ int get_Woman_default_variable_preferred_man(unsigned int index, unsigned int element){
    unsigned int count = get_agent_Woman_default_count();
    unsigned int numElements = 1024;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Womans_default_variable_preferred_man_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_Womans_default->preferred_man + (e * xmachine_memory_Woman_MAX),
                        d_Womans_default->preferred_man + (e * xmachine_memory_Woman_MAX), 
                        count * sizeof(int), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_Womans_default_variable_preferred_man_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Womans_default->preferred_man[index + (element * xmachine_memory_Woman_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of preferred_man for the %u th member of Woman_default. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}



/* Host based agent creation functions */
// These are only available for continuous agents.



/* copy_single_xmachine_memory_Man_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_Man_hostToDevice(xmachine_memory_Man_list * d_dst, xmachine_memory_Man * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->round, &h_agent->round, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->engaged_to, &h_agent->engaged_to, sizeof(int), cudaMemcpyHostToDevice));
 
	for(unsigned int i = 0; i < 1024; i++){
		gpuErrchk(cudaMemcpy(d_dst->preferred_woman + (i * xmachine_memory_Man_MAX), h_agent->preferred_woman + i, sizeof(int), cudaMemcpyHostToDevice));
    }

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
void copy_partial_xmachine_memory_Man_hostToDevice(xmachine_memory_Man_list * d_dst, xmachine_memory_Man_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->round, h_src->round, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->engaged_to, h_src->engaged_to, count * sizeof(int), cudaMemcpyHostToDevice));
 
		for(unsigned int i = 0; i < 1024; i++){
			gpuErrchk(cudaMemcpy(d_dst->preferred_woman + (i * xmachine_memory_Man_MAX), h_src->preferred_woman + (i * xmachine_memory_Man_MAX), count * sizeof(int), cudaMemcpyHostToDevice));
        }


    }
}


/* copy_single_xmachine_memory_Woman_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_Woman_hostToDevice(xmachine_memory_Woman_list * d_dst, xmachine_memory_Woman * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->current_suitor, &h_agent->current_suitor, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->current_suitor_rank, &h_agent->current_suitor_rank, sizeof(int), cudaMemcpyHostToDevice));
 
	for(unsigned int i = 0; i < 1024; i++){
		gpuErrchk(cudaMemcpy(d_dst->preferred_man + (i * xmachine_memory_Woman_MAX), h_agent->preferred_man + i, sizeof(int), cudaMemcpyHostToDevice));
    }

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
void copy_partial_xmachine_memory_Woman_hostToDevice(xmachine_memory_Woman_list * d_dst, xmachine_memory_Woman_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->current_suitor, h_src->current_suitor, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->current_suitor_rank, h_src->current_suitor_rank, count * sizeof(int), cudaMemcpyHostToDevice));
 
		for(unsigned int i = 0; i < 1024; i++){
			gpuErrchk(cudaMemcpy(d_dst->preferred_man + (i * xmachine_memory_Woman_MAX), h_src->preferred_man + (i * xmachine_memory_Woman_MAX), count * sizeof(int), cudaMemcpyHostToDevice));
        }


    }
}

xmachine_memory_Man* h_allocate_agent_Man(){
	xmachine_memory_Man* agent = (xmachine_memory_Man*)malloc(sizeof(xmachine_memory_Man));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_Man));
	// Agent variable arrays must be allocated
    agent->preferred_woman = (int*)malloc(1024 * sizeof(int));
	
    // If there is no default value, memset to 0.
    memset(agent->preferred_woman, 0, sizeof(int)*1024);
	return agent;
}
void h_free_agent_Man(xmachine_memory_Man** agent){

    free((*agent)->preferred_woman);
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_Man** h_allocate_agent_Man_array(unsigned int count){
	xmachine_memory_Man ** agents = (xmachine_memory_Man**)malloc(count * sizeof(xmachine_memory_Man*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_Man();
	}
	return agents;
}
void h_free_agent_Man_array(xmachine_memory_Man*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_Man(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_Man_AoS_to_SoA(xmachine_memory_Man_list * dst, xmachine_memory_Man** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->round[i] = src[i]->round;
			 
			dst->engaged_to[i] = src[i]->engaged_to;
			 
			for(unsigned int j = 0; j < 1024; j++){
				dst->preferred_woman[(j * xmachine_memory_Man_MAX) + i] = src[i]->preferred_woman[j];
			}
			
		}
	}
}


void h_add_agent_Man_unengaged(xmachine_memory_Man* agent){
	if (h_xmachine_memory_Man_count + 1 > xmachine_memory_Man_MAX){
		printf("Error: Buffer size of Man agents in state unengaged will be exceeded by h_add_agent_Man_unengaged\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_Man_hostToDevice(d_Mans_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Man_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_Man_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Mans_unengaged, d_Mans_new, h_xmachine_memory_Man_unengaged_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_Man_unengaged_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Man_unengaged_count, &h_xmachine_memory_Man_unengaged_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Mans_unengaged_variable_id_data_iteration = 0;
    h_Mans_unengaged_variable_round_data_iteration = 0;
    h_Mans_unengaged_variable_engaged_to_data_iteration = 0;
    h_Mans_unengaged_variable_preferred_woman_data_iteration = 0;
    

}
void h_add_agents_Man_unengaged(xmachine_memory_Man** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_Man_count + count > xmachine_memory_Man_MAX){
			printf("Error: Buffer size of Man agents in state unengaged will be exceeded by h_add_agents_Man_unengaged\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_Man_AoS_to_SoA(h_Mans_unengaged, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_Man_hostToDevice(d_Mans_new, h_Mans_unengaged, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Man_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_Man_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Mans_unengaged, d_Mans_new, h_xmachine_memory_Man_unengaged_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_Man_unengaged_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Man_unengaged_count, &h_xmachine_memory_Man_unengaged_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Mans_unengaged_variable_id_data_iteration = 0;
        h_Mans_unengaged_variable_round_data_iteration = 0;
        h_Mans_unengaged_variable_engaged_to_data_iteration = 0;
        h_Mans_unengaged_variable_preferred_woman_data_iteration = 0;
        

	}
}


void h_add_agent_Man_engaged(xmachine_memory_Man* agent){
	if (h_xmachine_memory_Man_count + 1 > xmachine_memory_Man_MAX){
		printf("Error: Buffer size of Man agents in state engaged will be exceeded by h_add_agent_Man_engaged\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_Man_hostToDevice(d_Mans_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Man_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_Man_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Mans_engaged, d_Mans_new, h_xmachine_memory_Man_engaged_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_Man_engaged_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Man_engaged_count, &h_xmachine_memory_Man_engaged_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Mans_engaged_variable_id_data_iteration = 0;
    h_Mans_engaged_variable_round_data_iteration = 0;
    h_Mans_engaged_variable_engaged_to_data_iteration = 0;
    h_Mans_engaged_variable_preferred_woman_data_iteration = 0;
    

}
void h_add_agents_Man_engaged(xmachine_memory_Man** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_Man_count + count > xmachine_memory_Man_MAX){
			printf("Error: Buffer size of Man agents in state engaged will be exceeded by h_add_agents_Man_engaged\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_Man_AoS_to_SoA(h_Mans_engaged, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_Man_hostToDevice(d_Mans_new, h_Mans_engaged, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Man_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_Man_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Mans_engaged, d_Mans_new, h_xmachine_memory_Man_engaged_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_Man_engaged_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Man_engaged_count, &h_xmachine_memory_Man_engaged_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Mans_engaged_variable_id_data_iteration = 0;
        h_Mans_engaged_variable_round_data_iteration = 0;
        h_Mans_engaged_variable_engaged_to_data_iteration = 0;
        h_Mans_engaged_variable_preferred_woman_data_iteration = 0;
        

	}
}

xmachine_memory_Woman* h_allocate_agent_Woman(){
	xmachine_memory_Woman* agent = (xmachine_memory_Woman*)malloc(sizeof(xmachine_memory_Woman));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_Woman));
	// Agent variable arrays must be allocated
    agent->preferred_man = (int*)malloc(1024 * sizeof(int));
	
    // If there is no default value, memset to 0.
    memset(agent->preferred_man, 0, sizeof(int)*1024);
	return agent;
}
void h_free_agent_Woman(xmachine_memory_Woman** agent){

    free((*agent)->preferred_man);
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_Woman** h_allocate_agent_Woman_array(unsigned int count){
	xmachine_memory_Woman ** agents = (xmachine_memory_Woman**)malloc(count * sizeof(xmachine_memory_Woman*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_Woman();
	}
	return agents;
}
void h_free_agent_Woman_array(xmachine_memory_Woman*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_Woman(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_Woman_AoS_to_SoA(xmachine_memory_Woman_list * dst, xmachine_memory_Woman** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->current_suitor[i] = src[i]->current_suitor;
			 
			dst->current_suitor_rank[i] = src[i]->current_suitor_rank;
			 
			for(unsigned int j = 0; j < 1024; j++){
				dst->preferred_man[(j * xmachine_memory_Woman_MAX) + i] = src[i]->preferred_man[j];
			}
			
		}
	}
}


void h_add_agent_Woman_default(xmachine_memory_Woman* agent){
	if (h_xmachine_memory_Woman_count + 1 > xmachine_memory_Woman_MAX){
		printf("Error: Buffer size of Woman agents in state default will be exceeded by h_add_agent_Woman_default\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_Woman_hostToDevice(d_Womans_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Woman_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_Woman_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Womans_default, d_Womans_new, h_xmachine_memory_Woman_default_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_Woman_default_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Woman_default_count, &h_xmachine_memory_Woman_default_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Womans_default_variable_id_data_iteration = 0;
    h_Womans_default_variable_current_suitor_data_iteration = 0;
    h_Womans_default_variable_current_suitor_rank_data_iteration = 0;
    h_Womans_default_variable_preferred_man_data_iteration = 0;
    

}
void h_add_agents_Woman_default(xmachine_memory_Woman** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_Woman_count + count > xmachine_memory_Woman_MAX){
			printf("Error: Buffer size of Woman agents in state default will be exceeded by h_add_agents_Woman_default\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_Woman_AoS_to_SoA(h_Womans_default, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_Woman_hostToDevice(d_Womans_new, h_Womans_default, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Woman_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_Woman_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Womans_default, d_Womans_new, h_xmachine_memory_Woman_default_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_Woman_default_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Woman_default_count, &h_xmachine_memory_Woman_default_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Womans_default_variable_id_data_iteration = 0;
        h_Womans_default_variable_current_suitor_data_iteration = 0;
        h_Womans_default_variable_current_suitor_rank_data_iteration = 0;
        h_Womans_default_variable_preferred_man_data_iteration = 0;
        

	}
}


/*  Analytics Functions */

int reduce_Man_unengaged_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Mans_unengaged->id),  thrust::device_pointer_cast(d_Mans_unengaged->id) + h_xmachine_memory_Man_unengaged_count);
}

int count_Man_unengaged_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Mans_unengaged->id),  thrust::device_pointer_cast(d_Mans_unengaged->id) + h_xmachine_memory_Man_unengaged_count, count_value);
}
int min_Man_unengaged_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Mans_unengaged->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Man_unengaged_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Man_unengaged_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Mans_unengaged->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Man_unengaged_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Man_unengaged_round_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Mans_unengaged->round),  thrust::device_pointer_cast(d_Mans_unengaged->round) + h_xmachine_memory_Man_unengaged_count);
}

int count_Man_unengaged_round_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Mans_unengaged->round),  thrust::device_pointer_cast(d_Mans_unengaged->round) + h_xmachine_memory_Man_unengaged_count, count_value);
}
int min_Man_unengaged_round_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Mans_unengaged->round);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Man_unengaged_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Man_unengaged_round_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Mans_unengaged->round);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Man_unengaged_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Man_unengaged_engaged_to_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Mans_unengaged->engaged_to),  thrust::device_pointer_cast(d_Mans_unengaged->engaged_to) + h_xmachine_memory_Man_unengaged_count);
}

int count_Man_unengaged_engaged_to_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Mans_unengaged->engaged_to),  thrust::device_pointer_cast(d_Mans_unengaged->engaged_to) + h_xmachine_memory_Man_unengaged_count, count_value);
}
int min_Man_unengaged_engaged_to_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Mans_unengaged->engaged_to);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Man_unengaged_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Man_unengaged_engaged_to_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Mans_unengaged->engaged_to);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Man_unengaged_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Man_engaged_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Mans_engaged->id),  thrust::device_pointer_cast(d_Mans_engaged->id) + h_xmachine_memory_Man_engaged_count);
}

int count_Man_engaged_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Mans_engaged->id),  thrust::device_pointer_cast(d_Mans_engaged->id) + h_xmachine_memory_Man_engaged_count, count_value);
}
int min_Man_engaged_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Mans_engaged->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Man_engaged_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Man_engaged_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Mans_engaged->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Man_engaged_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Man_engaged_round_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Mans_engaged->round),  thrust::device_pointer_cast(d_Mans_engaged->round) + h_xmachine_memory_Man_engaged_count);
}

int count_Man_engaged_round_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Mans_engaged->round),  thrust::device_pointer_cast(d_Mans_engaged->round) + h_xmachine_memory_Man_engaged_count, count_value);
}
int min_Man_engaged_round_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Mans_engaged->round);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Man_engaged_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Man_engaged_round_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Mans_engaged->round);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Man_engaged_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Man_engaged_engaged_to_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Mans_engaged->engaged_to),  thrust::device_pointer_cast(d_Mans_engaged->engaged_to) + h_xmachine_memory_Man_engaged_count);
}

int count_Man_engaged_engaged_to_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Mans_engaged->engaged_to),  thrust::device_pointer_cast(d_Mans_engaged->engaged_to) + h_xmachine_memory_Man_engaged_count, count_value);
}
int min_Man_engaged_engaged_to_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Mans_engaged->engaged_to);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Man_engaged_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Man_engaged_engaged_to_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Mans_engaged->engaged_to);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Man_engaged_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Woman_default_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Womans_default->id),  thrust::device_pointer_cast(d_Womans_default->id) + h_xmachine_memory_Woman_default_count);
}

int count_Woman_default_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Womans_default->id),  thrust::device_pointer_cast(d_Womans_default->id) + h_xmachine_memory_Woman_default_count, count_value);
}
int min_Woman_default_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Womans_default->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Woman_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Woman_default_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Womans_default->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Woman_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Woman_default_current_suitor_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Womans_default->current_suitor),  thrust::device_pointer_cast(d_Womans_default->current_suitor) + h_xmachine_memory_Woman_default_count);
}

int count_Woman_default_current_suitor_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Womans_default->current_suitor),  thrust::device_pointer_cast(d_Womans_default->current_suitor) + h_xmachine_memory_Woman_default_count, count_value);
}
int min_Woman_default_current_suitor_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Womans_default->current_suitor);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Woman_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Woman_default_current_suitor_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Womans_default->current_suitor);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Woman_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_Woman_default_current_suitor_rank_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Womans_default->current_suitor_rank),  thrust::device_pointer_cast(d_Womans_default->current_suitor_rank) + h_xmachine_memory_Woman_default_count);
}

int count_Woman_default_current_suitor_rank_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_Womans_default->current_suitor_rank),  thrust::device_pointer_cast(d_Womans_default->current_suitor_rank) + h_xmachine_memory_Woman_default_count, count_value);
}
int min_Woman_default_current_suitor_rank_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Womans_default->current_suitor_rank);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Woman_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_Woman_default_current_suitor_rank_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_Womans_default->current_suitor_rank);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Woman_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}



/* Agent functions */


	
/* Shared memory size calculator for agent function */
int Man_make_proposals_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Man_make_proposals
 * Agent function prototype for make_proposals function of Man agent
 */
void Man_make_proposals(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Man_unengaged_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Man_unengaged_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_Man_count = h_xmachine_memory_Man_unengaged_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_count, &h_xmachine_memory_Man_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Man_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Man_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Mans_unengaged);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_Man_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Mans);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, make_proposals_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	make_proposals_function_filter<<<gridSize, blockSize, 0, stream>>>(d_Mans_unengaged, d_Mans);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_Man_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Man, 
        temp_scan_storage_bytes_Man, 
        d_Mans_unengaged->_scan_input,
        d_Mans_unengaged->_position,
        h_xmachine_memory_Man_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Mans_unengaged->_position[h_xmachine_memory_Man_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Mans_unengaged->_scan_input[h_xmachine_memory_Man_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_Man_unengaged_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Man_unengaged_count = scan_last_sum;
	//Scatter into swap
	scatter_Man_Agents<<<gridSize, blockSize, 0, stream>>>(d_Mans_swap, d_Mans_unengaged, 0, h_xmachine_memory_Man_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_Man_list* Mans_unengaged_temp = d_Mans_unengaged;
	d_Mans_unengaged = d_Mans_swap;
	d_Mans_swap = Mans_unengaged_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_unengaged_count, &h_xmachine_memory_Man_unengaged_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Man, 
        temp_scan_storage_bytes_Man, 
        d_Mans->_scan_input,
        d_Mans->_position,
        h_xmachine_memory_Man_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Mans->_position[h_xmachine_memory_Man_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Mans->_scan_input[h_xmachine_memory_Man_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_Man_Agents<<<gridSize, blockSize, 0, stream>>>(d_Mans_swap, d_Mans, 0, h_xmachine_memory_Man_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_Man_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Man_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_Man_list* Mans_temp = d_Mans;
	d_Mans = d_Mans_swap;
	d_Mans_swap = Mans_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_count, &h_xmachine_memory_Man_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_Man_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_Man_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_proposal_count + h_xmachine_memory_Man_count > xmachine_message_proposal_MAX){
		printf("Error: Buffer size of proposal message will be exceeded in function make_proposals\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_make_proposals, Man_make_proposals_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Man_make_proposals_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_proposal_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_proposal_output_type, &h_message_proposal_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (make_proposals)
	//Reallocate   : false
	//Input        : 
	//Output       : proposal
	//Agent Output : 
	GPUFLAME_make_proposals<<<g, b, sm_size, stream>>>(d_Mans, d_proposals);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_proposal_count += h_xmachine_memory_Man_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_proposal_count, &h_message_proposal_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Man_unengaged_count+h_xmachine_memory_Man_count > xmachine_memory_Man_MAX){
		printf("Error: Buffer size of make_proposals agents in state unengaged will be exceeded moving working agents to next state in function make_proposals\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Man_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Man_Agents<<<gridSize, blockSize, 0, stream>>>(d_Mans_unengaged, d_Mans, h_xmachine_memory_Man_unengaged_count, h_xmachine_memory_Man_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Man_unengaged_count += h_xmachine_memory_Man_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_unengaged_count, &h_xmachine_memory_Man_unengaged_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Man_check_notifications_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_notification));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Man_check_notifications
 * Agent function prototype for check_notifications function of Man agent
 */
void Man_check_notifications(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Man_unengaged_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Man_unengaged_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Man_list* Mans_unengaged_temp = d_Mans;
	d_Mans = d_Mans_unengaged;
	d_Mans_unengaged = Mans_unengaged_temp;
	//set working count to current state count
	h_xmachine_memory_Man_count = h_xmachine_memory_Man_unengaged_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_count, &h_xmachine_memory_Man_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Man_unengaged_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_unengaged_count, &h_xmachine_memory_Man_unengaged_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_check_notifications, Man_check_notifications_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Man_check_notifications_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (check_notifications)
	//Reallocate   : false
	//Input        : notification
	//Output       : 
	//Agent Output : 
	GPUFLAME_check_notifications<<<g, b, sm_size, stream>>>(d_Mans, d_notifications);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Man_unengaged_count+h_xmachine_memory_Man_count > xmachine_memory_Man_MAX){
		printf("Error: Buffer size of check_notifications agents in state unengaged will be exceeded moving working agents to next state in function check_notifications\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Mans_unengaged_temp = d_Mans;
  d_Mans = d_Mans_unengaged;
  d_Mans_unengaged = Mans_unengaged_temp;
        
	//update new state agent size
	h_xmachine_memory_Man_unengaged_count += h_xmachine_memory_Man_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_unengaged_count, &h_xmachine_memory_Man_unengaged_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Man_check_resolved_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Man_check_resolved
 * Agent function prototype for check_resolved function of Man agent
 */
void Man_check_resolved(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Man_unengaged_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Man_unengaged_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS A GLOBAL CONDITION
	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_Man_count = h_xmachine_memory_Man_unengaged_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_count, &h_xmachine_memory_Man_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Man_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Man_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Mans_unengaged);
	gpuErrchkLaunch();
	
	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, check_resolved_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	check_resolved_function_filter<<<gridSize, blockSize, 0, stream>>>(d_Mans_unengaged);
	gpuErrchkLaunch();
	
	//GET CONDTIONS TRUE COUNT FROM CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Man, 
        temp_scan_storage_bytes_Man, 
        d_Mans_unengaged->_scan_input,
        d_Mans_unengaged->_position,
        h_xmachine_memory_Man_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Mans_unengaged->_position[h_xmachine_memory_Man_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Mans_unengaged->_scan_input[h_xmachine_memory_Man_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	int global_conditions_true = 0;
	if (scan_last_included == 1)
		global_conditions_true = scan_last_sum+1;
	else		
		global_conditions_true = scan_last_sum;
	//check if condition is true for all agents or if max condition count is reached
	if ((global_conditions_true != h_xmachine_memory_Man_count)&&(h_check_resolved_condition_count < 5000))
	{
		h_check_resolved_condition_count ++;
		return;
	}
	if ((h_check_resolved_condition_count == 5000))
	{
		printf("Global agent condition for check_resolved function reached the maximum number of 5000 conditions\n");
	}
	
	//RESET THE CONDITION COUNT
	h_check_resolved_condition_count = 0;
	
	//MAP CURRENT STATE TO WORKING LIST
	xmachine_memory_Man_list* Mans_unengaged_temp = d_Mans;
	d_Mans = d_Mans_unengaged;
	d_Mans_unengaged = Mans_unengaged_temp;
	//set current state count to 0
	h_xmachine_memory_Man_unengaged_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_count, &h_xmachine_memory_Man_count, sizeof(int)));	
	
	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_check_resolved, Man_check_resolved_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Man_check_resolved_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (check_resolved)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_check_resolved<<<g, b, sm_size, stream>>>(d_Mans);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Man_engaged_count+h_xmachine_memory_Man_count > xmachine_memory_Man_MAX){
		printf("Error: Buffer size of check_resolved agents in state engaged will be exceeded moving working agents to next state in function check_resolved\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Man_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Man_Agents<<<gridSize, blockSize, 0, stream>>>(d_Mans_engaged, d_Mans, h_xmachine_memory_Man_engaged_count, h_xmachine_memory_Man_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Man_engaged_count += h_xmachine_memory_Man_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Man_engaged_count, &h_xmachine_memory_Man_engaged_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Woman_check_proposals_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_proposal));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Woman_check_proposals
 * Agent function prototype for check_proposals function of Woman agent
 */
void Woman_check_proposals(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Woman_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Woman_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Woman_list* Womans_default_temp = d_Womans;
	d_Womans = d_Womans_default;
	d_Womans_default = Womans_default_temp;
	//set working count to current state count
	h_xmachine_memory_Woman_count = h_xmachine_memory_Woman_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Woman_count, &h_xmachine_memory_Woman_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Woman_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Woman_default_count, &h_xmachine_memory_Woman_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_check_proposals, Woman_check_proposals_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Woman_check_proposals_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (check_proposals)
	//Reallocate   : false
	//Input        : proposal
	//Output       : 
	//Agent Output : 
	GPUFLAME_check_proposals<<<g, b, sm_size, stream>>>(d_Womans, d_proposals);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Woman_default_count+h_xmachine_memory_Woman_count > xmachine_memory_Woman_MAX){
		printf("Error: Buffer size of check_proposals agents in state default will be exceeded moving working agents to next state in function check_proposals\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Womans_default_temp = d_Womans;
  d_Womans = d_Womans_default;
  d_Womans_default = Womans_default_temp;
        
	//update new state agent size
	h_xmachine_memory_Woman_default_count += h_xmachine_memory_Woman_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Woman_default_count, &h_xmachine_memory_Woman_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Woman_notify_suitors_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Woman_notify_suitors
 * Agent function prototype for notify_suitors function of Woman agent
 */
void Woman_notify_suitors(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Woman_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Woman_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_Woman_count = h_xmachine_memory_Woman_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Woman_count, &h_xmachine_memory_Woman_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Woman_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Woman_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Womans_default);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_Woman_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Womans);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, notify_suitors_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	notify_suitors_function_filter<<<gridSize, blockSize, 0, stream>>>(d_Womans_default, d_Womans);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_Woman_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Woman, 
        temp_scan_storage_bytes_Woman, 
        d_Womans_default->_scan_input,
        d_Womans_default->_position,
        h_xmachine_memory_Woman_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Womans_default->_position[h_xmachine_memory_Woman_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Womans_default->_scan_input[h_xmachine_memory_Woman_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_Woman_default_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Woman_default_count = scan_last_sum;
	//Scatter into swap
	scatter_Woman_Agents<<<gridSize, blockSize, 0, stream>>>(d_Womans_swap, d_Womans_default, 0, h_xmachine_memory_Woman_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_Woman_list* Womans_default_temp = d_Womans_default;
	d_Womans_default = d_Womans_swap;
	d_Womans_swap = Womans_default_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Woman_default_count, &h_xmachine_memory_Woman_default_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Woman, 
        temp_scan_storage_bytes_Woman, 
        d_Womans->_scan_input,
        d_Womans->_position,
        h_xmachine_memory_Woman_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Womans->_position[h_xmachine_memory_Woman_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Womans->_scan_input[h_xmachine_memory_Woman_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_Woman_Agents<<<gridSize, blockSize, 0, stream>>>(d_Womans_swap, d_Womans, 0, h_xmachine_memory_Woman_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_Woman_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Woman_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_Woman_list* Womans_temp = d_Womans;
	d_Womans = d_Womans_swap;
	d_Womans_swap = Womans_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Woman_count, &h_xmachine_memory_Woman_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_Woman_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_Woman_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_notification_count + h_xmachine_memory_Woman_count > xmachine_message_notification_MAX){
		printf("Error: Buffer size of notification message will be exceeded in function notify_suitors\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_notify_suitors, Woman_notify_suitors_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Woman_notify_suitors_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_notification_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_notification_output_type, &h_message_notification_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (notify_suitors)
	//Reallocate   : false
	//Input        : 
	//Output       : notification
	//Agent Output : 
	GPUFLAME_notify_suitors<<<g, b, sm_size, stream>>>(d_Womans, d_notifications);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_notification_count += h_xmachine_memory_Woman_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_notification_count, &h_message_notification_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Woman_default_count+h_xmachine_memory_Woman_count > xmachine_memory_Woman_MAX){
		printf("Error: Buffer size of notify_suitors agents in state default will be exceeded moving working agents to next state in function notify_suitors\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Woman_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Woman_Agents<<<gridSize, blockSize, 0, stream>>>(d_Womans_default, d_Womans, h_xmachine_memory_Woman_default_count, h_xmachine_memory_Woman_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Woman_default_count += h_xmachine_memory_Woman_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Woman_default_count, &h_xmachine_memory_Woman_default_count, sizeof(int)));	
	
	
}


 
extern void reset_Man_unengaged_count()
{
    h_xmachine_memory_Man_unengaged_count = 0;
}
 
extern void reset_Man_engaged_count()
{
    h_xmachine_memory_Man_engaged_count = 0;
}
 
extern void reset_Woman_default_count()
{
    h_xmachine_memory_Woman_default_count = 0;
}
