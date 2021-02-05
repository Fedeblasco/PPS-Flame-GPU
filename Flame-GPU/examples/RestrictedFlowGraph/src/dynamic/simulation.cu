
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

/* Agent Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_Agent_list* d_Agents;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_Agent_list* d_Agents_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_Agent_list* d_Agents_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_Agent_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_Agent_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_Agent_values;  /**< Agent sort identifiers value */

/* Agent state variables */
xmachine_memory_Agent_list* h_Agents_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_Agent_list* d_Agents_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_Agent_default_count;   /**< Agent population size counter */ 


/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
unsigned int h_Agents_default_variable_id_data_iteration;
unsigned int h_Agents_default_variable_currentEdge_data_iteration;
unsigned int h_Agents_default_variable_nextEdge_data_iteration;
unsigned int h_Agents_default_variable_nextEdgeRemainingCapacity_data_iteration;
unsigned int h_Agents_default_variable_hasIntent_data_iteration;
unsigned int h_Agents_default_variable_position_data_iteration;
unsigned int h_Agents_default_variable_distanceTravelled_data_iteration;
unsigned int h_Agents_default_variable_blockedIterationCount_data_iteration;
unsigned int h_Agents_default_variable_speed_data_iteration;
unsigned int h_Agents_default_variable_x_data_iteration;
unsigned int h_Agents_default_variable_y_data_iteration;
unsigned int h_Agents_default_variable_z_data_iteration;
unsigned int h_Agents_default_variable_colour_data_iteration;


/* Message Memory */

/* location Message variables */
xmachine_message_location_list* h_locations;         /**< Pointer to message list on host*/
xmachine_message_location_list* d_locations;         /**< Pointer to message list on device*/
xmachine_message_location_list* d_locations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* On-Graph Partitioned message variables  */
unsigned int h_message_location_count;         /**< message list counter*/
int h_message_location_output_type;   /**< message output type (single or optional)*/
/* On-Graph Partitioning Variables */
// Message bounds structure
xmachine_message_location_bounds * d_xmachine_message_location_bounds;
// Temporary data used during the scattering of messages
xmachine_message_location_scatterer * d_xmachine_message_location_scatterer; 
// Values for CUB exclusive scan of spatially partitioned variables
void * d_temp_scan_storage_xmachine_message_location;
size_t temp_scan_bytes_xmachine_message_location;

/* intent Message variables */
xmachine_message_intent_list* h_intents;         /**< Pointer to message list on host*/
xmachine_message_intent_list* d_intents;         /**< Pointer to message list on device*/
xmachine_message_intent_list* d_intents_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* On-Graph Partitioned message variables  */
unsigned int h_message_intent_count;         /**< message list counter*/
int h_message_intent_output_type;   /**< message output type (single or optional)*/
/* On-Graph Partitioning Variables */
// Message bounds structure
xmachine_message_intent_bounds * d_xmachine_message_intent_bounds;
// Temporary data used during the scattering of messages
xmachine_message_intent_scatterer * d_xmachine_message_intent_scatterer; 
// Values for CUB exclusive scan of spatially partitioned variables
void * d_temp_scan_storage_xmachine_message_intent;
size_t temp_scan_bytes_xmachine_message_intent;

  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_Agent;
size_t temp_scan_storage_bytes_Agent;


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

/** Agent_output_location
 * Agent function prototype for output_location function of Agent agent
 */
void Agent_output_location(cudaStream_t &stream);

/** Agent_read_locations
 * Agent function prototype for read_locations function of Agent agent
 */
void Agent_read_locations(cudaStream_t &stream);

/** Agent_resolve_intent
 * Agent function prototype for resolve_intent function of Agent agent
 */
void Agent_resolve_intent(cudaStream_t &stream);

/** Agent_move
 * Agent function prototype for move function of Agent agent
 */
void Agent_move(cudaStream_t &stream);

  
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
    h_Agents_default_variable_id_data_iteration = 0;
    h_Agents_default_variable_currentEdge_data_iteration = 0;
    h_Agents_default_variable_nextEdge_data_iteration = 0;
    h_Agents_default_variable_nextEdgeRemainingCapacity_data_iteration = 0;
    h_Agents_default_variable_hasIntent_data_iteration = 0;
    h_Agents_default_variable_position_data_iteration = 0;
    h_Agents_default_variable_distanceTravelled_data_iteration = 0;
    h_Agents_default_variable_blockedIterationCount_data_iteration = 0;
    h_Agents_default_variable_speed_data_iteration = 0;
    h_Agents_default_variable_x_data_iteration = 0;
    h_Agents_default_variable_y_data_iteration = 0;
    h_Agents_default_variable_z_data_iteration = 0;
    h_Agents_default_variable_colour_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_Agent_SoA_size = sizeof(xmachine_memory_Agent_list);
	h_Agents_default = (xmachine_memory_Agent_list*)malloc(xmachine_Agent_SoA_size);

	/* Message memory allocation (CPU) */
	int message_location_SoA_size = sizeof(xmachine_message_location_list);
	h_locations = (xmachine_message_location_list*)malloc(message_location_SoA_size);
	int message_intent_SoA_size = sizeof(xmachine_message_intent_list);
	h_intents = (xmachine_message_intent_list*)malloc(message_intent_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs

  /* Graph memory allocation (CPU) */
  
    // Allocate host structure used to load data for device copying
    h_staticGraph_memory_network = (staticGraph_memory_network*) malloc(sizeof(staticGraph_memory_network));
    // Ensure allocation was successful.
    if(h_staticGraph_memory_network == nullptr ){
        printf("FATAL ERROR: Could not allocate host memory for static network network \n");
        PROFILE_POP_RANGE();
        exit(EXIT_FAILURE);
    }
  

    PROFILE_POP_RANGE(); //"allocate host"
	

	//read initial states
	readInitialStates(inputfile, h_Agents_default, &h_xmachine_memory_Agent_default_count);

  // Read graphs from disk
  load_staticGraph_network_from_json("network.json", h_staticGraph_memory_network);
  

  PROFILE_PUSH_RANGE("allocate device");
	
	/* Agent Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Agents, xmachine_Agent_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Agents_swap, xmachine_Agent_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_Agents_new, xmachine_Agent_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Agent_keys, xmachine_memory_Agent_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_Agent_values, xmachine_memory_Agent_MAX* sizeof(uint)));
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_Agents_default, xmachine_Agent_SoA_size));
	gpuErrchk( cudaMemcpy( d_Agents_default, h_Agents_default, xmachine_Agent_SoA_size, cudaMemcpyHostToDevice));
    
	/* location Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_locations, message_location_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_locations_swap, message_location_SoA_size));
	gpuErrchk( cudaMemcpy( d_locations, h_locations, message_location_SoA_size, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**)&d_xmachine_message_location_bounds, sizeof(xmachine_message_location_bounds)));
  gpuErrchk(cudaMalloc((void**)&d_xmachine_message_location_scatterer, sizeof(xmachine_message_location_scatterer)));
  /* Calculate and allocate CUB temporary memory for exclusive scans */
    d_temp_scan_storage_xmachine_message_location = nullptr;
    temp_scan_bytes_xmachine_message_location = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_xmachine_message_location, 
        temp_scan_bytes_xmachine_message_location, 
        (unsigned int*) nullptr, 
        (unsigned int*) nullptr, 
        staticGraph_network_edge_bufferSize
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_xmachine_message_location, temp_scan_bytes_xmachine_message_location));
  
	
	/* intent Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_intents, message_intent_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_intents_swap, message_intent_SoA_size));
	gpuErrchk( cudaMemcpy( d_intents, h_intents, message_intent_SoA_size, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc((void**)&d_xmachine_message_intent_bounds, sizeof(xmachine_message_intent_bounds)));
  gpuErrchk(cudaMalloc((void**)&d_xmachine_message_intent_scatterer, sizeof(xmachine_message_intent_scatterer)));
  /* Calculate and allocate CUB temporary memory for exclusive scans */
    d_temp_scan_storage_xmachine_message_intent = nullptr;
    temp_scan_bytes_xmachine_message_intent = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_xmachine_message_intent, 
        temp_scan_bytes_xmachine_message_intent, 
        (unsigned int*) nullptr, 
        (unsigned int*) nullptr, 
        staticGraph_network_edge_bufferSize
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_xmachine_message_intent, temp_scan_bytes_xmachine_message_intent));
  
		


  /* Allocate device memory for graphs */
  
  // Allocate device memory, this is freed by cleanup() in simulation.cu
  gpuErrchk(cudaMalloc((void**)&d_staticGraph_memory_network, sizeof(staticGraph_memory_network)));

  // Copy data to the Device
  gpuErrchk(cudaMemcpy(d_staticGraph_memory_network, h_staticGraph_memory_network, sizeof(staticGraph_memory_network), cudaMemcpyHostToDevice));

  // Copy device pointer(s) to CUDA constant(s)
  gpuErrchk(cudaMemcpyToSymbol(d_staticGraph_memory_network_ptr, &d_staticGraph_memory_network, sizeof(staticGraph_memory_network*)));
  

    PROFILE_POP_RANGE(); // "allocate device"

    /* Calculate and allocate CUB temporary memory for exclusive scans */
    
    d_temp_scan_storage_Agent = nullptr;
    temp_scan_storage_bytes_Agent = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Agent, 
        temp_scan_storage_bytes_Agent, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_Agent_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_Agent, temp_scan_storage_bytes_Agent));
    

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
    initialiseHost();
    PROFILE_PUSH_RANGE("initialiseHost");
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: initialiseHost = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
    generateAgents();
    PROFILE_PUSH_RANGE("generateAgents");
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: generateAgents = %f (ms)\n", instrument_milliseconds);
#endif
	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_Agent_default_count: %u\n",get_agent_Agent_default_count());
	
#endif
} 


void sort_Agents_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Agent_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_Agent_default_count); 
	gridSize = (h_xmachine_memory_Agent_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_Agent_keys, d_xmachine_memory_Agent_values, d_Agents_default);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_Agent_keys),  thrust::device_pointer_cast(d_xmachine_memory_Agent_keys) + h_xmachine_memory_Agent_default_count,  thrust::device_pointer_cast(d_xmachine_memory_Agent_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_Agent_agents, no_sm, h_xmachine_memory_Agent_default_count); 
	gridSize = (h_xmachine_memory_Agent_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_Agent_agents<<<gridSize, blockSize>>>(d_xmachine_memory_Agent_values, d_Agents_default, d_Agents_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_Agent_list* d_Agents_temp = d_Agents_default;
	d_Agents_default = d_Agents_swap;
	d_Agents_swap = d_Agents_temp;	
}


void cleanup(){
    PROFILE_SCOPED_RANGE("cleanup");

    /* Call all exit functions */
	
#if defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif

    exitFunc();
    PROFILE_PUSH_RANGE("exitFunc");
	PROFILE_POP_RANGE();

#if defined(INSTRUMENT_EXIT_FUNCTIONS) && INSTRUMENT_EXIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: exitFunc = %f (ms)\n", instrument_milliseconds);
#endif
	

	/* Agent data free*/
	
	/* Agent Agent variables */
	gpuErrchk(cudaFree(d_Agents));
	gpuErrchk(cudaFree(d_Agents_swap));
	gpuErrchk(cudaFree(d_Agents_new));
	
	free( h_Agents_default);
	gpuErrchk(cudaFree(d_Agents_default));
	

	/* Message data free */
	
	/* location Message variables */
	free( h_locations);
	gpuErrchk(cudaFree(d_locations));
	gpuErrchk(cudaFree(d_locations_swap));
  gpuErrchk(cudaFree(d_xmachine_message_location_bounds));
  gpuErrchk(cudaFree(d_xmachine_message_location_scatterer));
  gpuErrchk(cudaFree(d_temp_scan_storage_xmachine_message_location));
  d_temp_scan_storage_xmachine_message_location = nullptr;
  temp_scan_bytes_xmachine_message_location = 0;
  
	
	/* intent Message variables */
	free( h_intents);
	gpuErrchk(cudaFree(d_intents));
	gpuErrchk(cudaFree(d_intents_swap));
  gpuErrchk(cudaFree(d_xmachine_message_intent_bounds));
  gpuErrchk(cudaFree(d_xmachine_message_intent_scatterer));
  gpuErrchk(cudaFree(d_temp_scan_storage_xmachine_message_intent));
  d_temp_scan_storage_xmachine_message_intent = nullptr;
  temp_scan_bytes_xmachine_message_intent = 0;
  
	

    /* Free temporary CUB memory if required. */
    
    if(d_temp_scan_storage_Agent != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_Agent));
      d_temp_scan_storage_Agent = nullptr;
      temp_scan_storage_bytes_Agent = 0;
    }
    

  /* Graph data free */
  
  gpuErrchk(cudaFree(d_staticGraph_memory_network));
  d_staticGraph_memory_network = nullptr;
  // Free host memory
  free(h_staticGraph_memory_network);
  h_staticGraph_memory_network = nullptr;
  
  
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
	h_message_location_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));
	
	h_message_intent_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_intent_count, &h_message_intent_count, sizeof(int)));
	

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Agent_output_location");
	Agent_output_location(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Agent_output_location = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Agent_read_locations");
	Agent_read_locations(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Agent_read_locations = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Agent_resolve_intent");
	Agent_resolve_intent(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Agent_resolve_intent = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 4*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("Agent_move");
	Agent_move(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: Agent_move = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_Agent_default_count: %u\n",get_agent_Agent_default_count());
	
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
unsigned int h_env_SEED;
unsigned int h_env_INIT_POPULATION;
float h_env_PARAM_MIN_SPEED;
float h_env_PARAM_MAX_SPEED;


//constant setter
void set_SEED(unsigned int* h_SEED){
    gpuErrchk(cudaMemcpyToSymbol(SEED, h_SEED, sizeof(unsigned int)));
    memcpy(&h_env_SEED, h_SEED,sizeof(unsigned int));
}

//constant getter
const unsigned int* get_SEED(){
    return &h_env_SEED;
}



//constant setter
void set_INIT_POPULATION(unsigned int* h_INIT_POPULATION){
    gpuErrchk(cudaMemcpyToSymbol(INIT_POPULATION, h_INIT_POPULATION, sizeof(unsigned int)));
    memcpy(&h_env_INIT_POPULATION, h_INIT_POPULATION,sizeof(unsigned int));
}

//constant getter
const unsigned int* get_INIT_POPULATION(){
    return &h_env_INIT_POPULATION;
}



//constant setter
void set_PARAM_MIN_SPEED(float* h_PARAM_MIN_SPEED){
    gpuErrchk(cudaMemcpyToSymbol(PARAM_MIN_SPEED, h_PARAM_MIN_SPEED, sizeof(float)));
    memcpy(&h_env_PARAM_MIN_SPEED, h_PARAM_MIN_SPEED,sizeof(float));
}

//constant getter
const float* get_PARAM_MIN_SPEED(){
    return &h_env_PARAM_MIN_SPEED;
}



//constant setter
void set_PARAM_MAX_SPEED(float* h_PARAM_MAX_SPEED){
    gpuErrchk(cudaMemcpyToSymbol(PARAM_MAX_SPEED, h_PARAM_MAX_SPEED, sizeof(float)));
    memcpy(&h_env_PARAM_MAX_SPEED, h_PARAM_MAX_SPEED,sizeof(float));
}

//constant getter
const float* get_PARAM_MAX_SPEED(){
    return &h_env_PARAM_MAX_SPEED;
}




/* Agent data access functions*/

    
int get_agent_Agent_MAX_count(){
    return xmachine_memory_Agent_MAX;
}


int get_agent_Agent_default_count(){
	//continuous agent
	return h_xmachine_memory_Agent_default_count;
	
}

xmachine_memory_Agent_list* get_device_Agent_default_agents(){
	return d_Agents_default;
}

xmachine_memory_Agent_list* get_host_Agent_default_agents(){
	return h_Agents_default;
}



/* Host based access of agent variables*/

/** unsigned int get_Agent_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Agent_default_variable_id(unsigned int index){
    unsigned int count = get_agent_Agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Agents_default_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Agents_default->id,
                    d_Agents_default->id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Agents_default_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Agents_default->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of Agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Agent_default_variable_currentEdge(unsigned int index)
 * Gets the value of the currentEdge variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable currentEdge
 */
__host__ unsigned int get_Agent_default_variable_currentEdge(unsigned int index){
    unsigned int count = get_agent_Agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Agents_default_variable_currentEdge_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Agents_default->currentEdge,
                    d_Agents_default->currentEdge,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Agents_default_variable_currentEdge_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Agents_default->currentEdge[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access currentEdge for the %u th member of Agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Agent_default_variable_nextEdge(unsigned int index)
 * Gets the value of the nextEdge variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable nextEdge
 */
__host__ unsigned int get_Agent_default_variable_nextEdge(unsigned int index){
    unsigned int count = get_agent_Agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Agents_default_variable_nextEdge_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Agents_default->nextEdge,
                    d_Agents_default->nextEdge,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Agents_default_variable_nextEdge_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Agents_default->nextEdge[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access nextEdge for the %u th member of Agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Agent_default_variable_nextEdgeRemainingCapacity(unsigned int index)
 * Gets the value of the nextEdgeRemainingCapacity variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable nextEdgeRemainingCapacity
 */
__host__ unsigned int get_Agent_default_variable_nextEdgeRemainingCapacity(unsigned int index){
    unsigned int count = get_agent_Agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Agents_default_variable_nextEdgeRemainingCapacity_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Agents_default->nextEdgeRemainingCapacity,
                    d_Agents_default->nextEdgeRemainingCapacity,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Agents_default_variable_nextEdgeRemainingCapacity_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Agents_default->nextEdgeRemainingCapacity[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access nextEdgeRemainingCapacity for the %u th member of Agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** bool get_Agent_default_variable_hasIntent(unsigned int index)
 * Gets the value of the hasIntent variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hasIntent
 */
__host__ bool get_Agent_default_variable_hasIntent(unsigned int index){
    unsigned int count = get_agent_Agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Agents_default_variable_hasIntent_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Agents_default->hasIntent,
                    d_Agents_default->hasIntent,
                    count * sizeof(bool),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Agents_default_variable_hasIntent_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Agents_default->hasIntent[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access hasIntent for the %u th member of Agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Agent_default_variable_position(unsigned int index)
 * Gets the value of the position variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable position
 */
__host__ float get_Agent_default_variable_position(unsigned int index){
    unsigned int count = get_agent_Agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Agents_default_variable_position_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Agents_default->position,
                    d_Agents_default->position,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Agents_default_variable_position_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Agents_default->position[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access position for the %u th member of Agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Agent_default_variable_distanceTravelled(unsigned int index)
 * Gets the value of the distanceTravelled variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable distanceTravelled
 */
__host__ float get_Agent_default_variable_distanceTravelled(unsigned int index){
    unsigned int count = get_agent_Agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Agents_default_variable_distanceTravelled_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Agents_default->distanceTravelled,
                    d_Agents_default->distanceTravelled,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Agents_default_variable_distanceTravelled_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Agents_default->distanceTravelled[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access distanceTravelled for the %u th member of Agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_Agent_default_variable_blockedIterationCount(unsigned int index)
 * Gets the value of the blockedIterationCount variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable blockedIterationCount
 */
__host__ unsigned int get_Agent_default_variable_blockedIterationCount(unsigned int index){
    unsigned int count = get_agent_Agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Agents_default_variable_blockedIterationCount_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Agents_default->blockedIterationCount,
                    d_Agents_default->blockedIterationCount,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Agents_default_variable_blockedIterationCount_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Agents_default->blockedIterationCount[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access blockedIterationCount for the %u th member of Agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Agent_default_variable_speed(unsigned int index)
 * Gets the value of the speed variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable speed
 */
__host__ float get_Agent_default_variable_speed(unsigned int index){
    unsigned int count = get_agent_Agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Agents_default_variable_speed_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Agents_default->speed,
                    d_Agents_default->speed,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Agents_default_variable_speed_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Agents_default->speed[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access speed for the %u th member of Agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Agent_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_Agent_default_variable_x(unsigned int index){
    unsigned int count = get_agent_Agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Agents_default_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Agents_default->x,
                    d_Agents_default->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Agents_default_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Agents_default->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of Agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Agent_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_Agent_default_variable_y(unsigned int index){
    unsigned int count = get_agent_Agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Agents_default_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Agents_default->y,
                    d_Agents_default->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Agents_default_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Agents_default->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of Agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Agent_default_variable_z(unsigned int index)
 * Gets the value of the z variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_Agent_default_variable_z(unsigned int index){
    unsigned int count = get_agent_Agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Agents_default_variable_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Agents_default->z,
                    d_Agents_default->z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Agents_default_variable_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Agents_default->z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z for the %u th member of Agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_Agent_default_variable_colour(unsigned int index)
 * Gets the value of the colour variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable colour
 */
__host__ float get_Agent_default_variable_colour(unsigned int index){
    unsigned int count = get_agent_Agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_Agents_default_variable_colour_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_Agents_default->colour,
                    d_Agents_default->colour,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_Agents_default_variable_colour_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_Agents_default->colour[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access colour for the %u th member of Agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}



/* Host based agent creation functions */
// These are only available for continuous agents.



/* copy_single_xmachine_memory_Agent_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_Agent_hostToDevice(xmachine_memory_Agent_list * d_dst, xmachine_memory_Agent * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->currentEdge, &h_agent->currentEdge, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->nextEdge, &h_agent->nextEdge, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->nextEdgeRemainingCapacity, &h_agent->nextEdgeRemainingCapacity, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->hasIntent, &h_agent->hasIntent, sizeof(bool), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->position, &h_agent->position, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->distanceTravelled, &h_agent->distanceTravelled, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->blockedIterationCount, &h_agent->blockedIterationCount, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->speed, &h_agent->speed, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, &h_agent->x, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, &h_agent->y, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->z, &h_agent->z, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->colour, &h_agent->colour, sizeof(float), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_Agent_hostToDevice(xmachine_memory_Agent_list * d_dst, xmachine_memory_Agent_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->currentEdge, h_src->currentEdge, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->nextEdge, h_src->nextEdge, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->nextEdgeRemainingCapacity, h_src->nextEdgeRemainingCapacity, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->hasIntent, h_src->hasIntent, count * sizeof(bool), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->position, h_src->position, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->distanceTravelled, h_src->distanceTravelled, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->blockedIterationCount, h_src->blockedIterationCount, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->speed, h_src->speed, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, h_src->x, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, h_src->y, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->z, h_src->z, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->colour, h_src->colour, count * sizeof(float), cudaMemcpyHostToDevice));

    }
}

xmachine_memory_Agent* h_allocate_agent_Agent(){
	xmachine_memory_Agent* agent = (xmachine_memory_Agent*)malloc(sizeof(xmachine_memory_Agent));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_Agent));

    agent->nextEdgeRemainingCapacity = 0;

    agent->colour = 0.0;

	return agent;
}
void h_free_agent_Agent(xmachine_memory_Agent** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_Agent** h_allocate_agent_Agent_array(unsigned int count){
	xmachine_memory_Agent ** agents = (xmachine_memory_Agent**)malloc(count * sizeof(xmachine_memory_Agent*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_Agent();
	}
	return agents;
}
void h_free_agent_Agent_array(xmachine_memory_Agent*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_Agent(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_Agent_AoS_to_SoA(xmachine_memory_Agent_list * dst, xmachine_memory_Agent** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->currentEdge[i] = src[i]->currentEdge;
			 
			dst->nextEdge[i] = src[i]->nextEdge;
			 
			dst->nextEdgeRemainingCapacity[i] = src[i]->nextEdgeRemainingCapacity;
			 
			dst->hasIntent[i] = src[i]->hasIntent;
			 
			dst->position[i] = src[i]->position;
			 
			dst->distanceTravelled[i] = src[i]->distanceTravelled;
			 
			dst->blockedIterationCount[i] = src[i]->blockedIterationCount;
			 
			dst->speed[i] = src[i]->speed;
			 
			dst->x[i] = src[i]->x;
			 
			dst->y[i] = src[i]->y;
			 
			dst->z[i] = src[i]->z;
			 
			dst->colour[i] = src[i]->colour;
			
		}
	}
}


void h_add_agent_Agent_default(xmachine_memory_Agent* agent){
	if (h_xmachine_memory_Agent_count + 1 > xmachine_memory_Agent_MAX){
		printf("Error: Buffer size of Agent agents in state default will be exceeded by h_add_agent_Agent_default\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_Agent_hostToDevice(d_Agents_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Agent_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_Agent_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Agents_default, d_Agents_new, h_xmachine_memory_Agent_default_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_Agent_default_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Agent_default_count, &h_xmachine_memory_Agent_default_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_Agents_default_variable_id_data_iteration = 0;
    h_Agents_default_variable_currentEdge_data_iteration = 0;
    h_Agents_default_variable_nextEdge_data_iteration = 0;
    h_Agents_default_variable_nextEdgeRemainingCapacity_data_iteration = 0;
    h_Agents_default_variable_hasIntent_data_iteration = 0;
    h_Agents_default_variable_position_data_iteration = 0;
    h_Agents_default_variable_distanceTravelled_data_iteration = 0;
    h_Agents_default_variable_blockedIterationCount_data_iteration = 0;
    h_Agents_default_variable_speed_data_iteration = 0;
    h_Agents_default_variable_x_data_iteration = 0;
    h_Agents_default_variable_y_data_iteration = 0;
    h_Agents_default_variable_z_data_iteration = 0;
    h_Agents_default_variable_colour_data_iteration = 0;
    

}
void h_add_agents_Agent_default(xmachine_memory_Agent** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_Agent_count + count > xmachine_memory_Agent_MAX){
			printf("Error: Buffer size of Agent agents in state default will be exceeded by h_add_agents_Agent_default\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_Agent_AoS_to_SoA(h_Agents_default, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_Agent_hostToDevice(d_Agents_new, h_Agents_default, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_Agent_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_Agent_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_Agents_default, d_Agents_new, h_xmachine_memory_Agent_default_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_Agent_default_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_Agent_default_count, &h_xmachine_memory_Agent_default_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_Agents_default_variable_id_data_iteration = 0;
        h_Agents_default_variable_currentEdge_data_iteration = 0;
        h_Agents_default_variable_nextEdge_data_iteration = 0;
        h_Agents_default_variable_nextEdgeRemainingCapacity_data_iteration = 0;
        h_Agents_default_variable_hasIntent_data_iteration = 0;
        h_Agents_default_variable_position_data_iteration = 0;
        h_Agents_default_variable_distanceTravelled_data_iteration = 0;
        h_Agents_default_variable_blockedIterationCount_data_iteration = 0;
        h_Agents_default_variable_speed_data_iteration = 0;
        h_Agents_default_variable_x_data_iteration = 0;
        h_Agents_default_variable_y_data_iteration = 0;
        h_Agents_default_variable_z_data_iteration = 0;
        h_Agents_default_variable_colour_data_iteration = 0;
        

	}
}


/*  Analytics Functions */

unsigned int reduce_Agent_default_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Agents_default->id),  thrust::device_pointer_cast(d_Agents_default->id) + h_xmachine_memory_Agent_default_count);
}

unsigned int count_Agent_default_id_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_Agents_default->id),  thrust::device_pointer_cast(d_Agents_default->id) + h_xmachine_memory_Agent_default_count, count_value);
}
unsigned int min_Agent_default_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Agent_default_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Agent_default_currentEdge_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Agents_default->currentEdge),  thrust::device_pointer_cast(d_Agents_default->currentEdge) + h_xmachine_memory_Agent_default_count);
}

unsigned int count_Agent_default_currentEdge_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_Agents_default->currentEdge),  thrust::device_pointer_cast(d_Agents_default->currentEdge) + h_xmachine_memory_Agent_default_count, count_value);
}
unsigned int min_Agent_default_currentEdge_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->currentEdge);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Agent_default_currentEdge_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->currentEdge);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Agent_default_nextEdge_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Agents_default->nextEdge),  thrust::device_pointer_cast(d_Agents_default->nextEdge) + h_xmachine_memory_Agent_default_count);
}

unsigned int count_Agent_default_nextEdge_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_Agents_default->nextEdge),  thrust::device_pointer_cast(d_Agents_default->nextEdge) + h_xmachine_memory_Agent_default_count, count_value);
}
unsigned int min_Agent_default_nextEdge_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->nextEdge);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Agent_default_nextEdge_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->nextEdge);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Agent_default_nextEdgeRemainingCapacity_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Agents_default->nextEdgeRemainingCapacity),  thrust::device_pointer_cast(d_Agents_default->nextEdgeRemainingCapacity) + h_xmachine_memory_Agent_default_count);
}

unsigned int count_Agent_default_nextEdgeRemainingCapacity_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_Agents_default->nextEdgeRemainingCapacity),  thrust::device_pointer_cast(d_Agents_default->nextEdgeRemainingCapacity) + h_xmachine_memory_Agent_default_count, count_value);
}
unsigned int min_Agent_default_nextEdgeRemainingCapacity_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->nextEdgeRemainingCapacity);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Agent_default_nextEdgeRemainingCapacity_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->nextEdgeRemainingCapacity);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
bool reduce_Agent_default_hasIntent_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Agents_default->hasIntent),  thrust::device_pointer_cast(d_Agents_default->hasIntent) + h_xmachine_memory_Agent_default_count);
}

bool min_Agent_default_hasIntent_variable(){
    //min in default stream
    thrust::device_ptr<bool> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->hasIntent);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
bool max_Agent_default_hasIntent_variable(){
    //max in default stream
    thrust::device_ptr<bool> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->hasIntent);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Agent_default_position_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Agents_default->position),  thrust::device_pointer_cast(d_Agents_default->position) + h_xmachine_memory_Agent_default_count);
}

float min_Agent_default_position_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->position);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Agent_default_position_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->position);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Agent_default_distanceTravelled_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Agents_default->distanceTravelled),  thrust::device_pointer_cast(d_Agents_default->distanceTravelled) + h_xmachine_memory_Agent_default_count);
}

float min_Agent_default_distanceTravelled_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->distanceTravelled);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Agent_default_distanceTravelled_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->distanceTravelled);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_Agent_default_blockedIterationCount_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Agents_default->blockedIterationCount),  thrust::device_pointer_cast(d_Agents_default->blockedIterationCount) + h_xmachine_memory_Agent_default_count);
}

unsigned int count_Agent_default_blockedIterationCount_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_Agents_default->blockedIterationCount),  thrust::device_pointer_cast(d_Agents_default->blockedIterationCount) + h_xmachine_memory_Agent_default_count, count_value);
}
unsigned int min_Agent_default_blockedIterationCount_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->blockedIterationCount);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_Agent_default_blockedIterationCount_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->blockedIterationCount);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Agent_default_speed_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Agents_default->speed),  thrust::device_pointer_cast(d_Agents_default->speed) + h_xmachine_memory_Agent_default_count);
}

float min_Agent_default_speed_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->speed);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Agent_default_speed_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->speed);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Agent_default_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Agents_default->x),  thrust::device_pointer_cast(d_Agents_default->x) + h_xmachine_memory_Agent_default_count);
}

float min_Agent_default_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Agent_default_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Agent_default_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Agents_default->y),  thrust::device_pointer_cast(d_Agents_default->y) + h_xmachine_memory_Agent_default_count);
}

float min_Agent_default_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Agent_default_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Agent_default_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Agents_default->z),  thrust::device_pointer_cast(d_Agents_default->z) + h_xmachine_memory_Agent_default_count);
}

float min_Agent_default_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Agent_default_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_Agent_default_colour_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_Agents_default->colour),  thrust::device_pointer_cast(d_Agents_default->colour) + h_xmachine_memory_Agent_default_count);
}

float min_Agent_default_colour_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->colour);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_Agent_default_colour_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_Agents_default->colour);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_Agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}



/* Agent functions */


	
/* Shared memory size calculator for agent function */
int Agent_output_location_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Agent_output_location
 * Agent function prototype for output_location function of Agent agent
 */
void Agent_output_location(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Agent_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Agent_list* Agents_default_temp = d_Agents;
	d_Agents = d_Agents_default;
	d_Agents_default = Agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_Agent_count = h_xmachine_memory_Agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Agent_count, &h_xmachine_memory_Agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Agent_default_count, &h_xmachine_memory_Agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_location_count + h_xmachine_memory_Agent_count > xmachine_message_location_MAX){
		printf("Error: Buffer size of location message will be exceeded in function output_location\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_location, Agent_output_location_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Agent_output_location_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_location_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_location_output_type, &h_message_location_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (output_location)
	//Reallocate   : false
	//Input        : 
	//Output       : location
	//Agent Output : 
	GPUFLAME_output_location<<<g, b, sm_size, stream>>>(d_Agents, d_locations);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_location_count += h_xmachine_memory_Agent_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));	
	
  // Sort messages based on the edge index, and construct the relevant data structure for graph edge based messaging. Keys are sorted and then message data is scattered. 

  // Reset the message bounds data structure to 0
  gpuErrchk(cudaMemset((void*)d_xmachine_message_location_bounds, 0, sizeof(xmachine_message_location_bounds)));

  // If there are any messages output (to account for 0 optional messages)
  if (h_message_location_count > 0){
  // Build histogram using atomics
  cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, hist_location_messages, no_sm, h_message_location_count);
  gridSize = (h_message_location_count + blockSize - 1) / blockSize;
  hist_location_messages <<<gridSize, blockSize, 0, stream >>>(d_xmachine_message_location_scatterer->edge_local_index, d_xmachine_message_location_scatterer->unsorted_edge_index, d_xmachine_message_location_bounds->count, d_locations, h_message_location_count);
  gpuErrchkLaunch();

  // Exclusive scan on histogram output to find the index for each message for each edge/bucket
  cub::DeviceScan::ExclusiveSum(
      d_temp_scan_storage_xmachine_message_location,
      temp_scan_bytes_xmachine_message_location,
      d_xmachine_message_location_bounds->count,
      d_xmachine_message_location_bounds->start,
      staticGraph_network_edge_bufferSize, 
      stream
  );
  gpuErrchkLaunch();

  // Launch kernel to re-order (scatter) the messages
  cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, reorder_location_messages, no_sm, h_message_location_count);
  gridSize = (h_message_location_count + blockSize - 1) / blockSize;  // Round up according to array size
  reorder_location_messages <<<gridSize, blockSize, 0, stream >>>(d_xmachine_message_location_scatterer->edge_local_index, d_xmachine_message_location_scatterer->unsorted_edge_index, d_xmachine_message_location_bounds->start, d_locations, d_locations_swap, h_message_location_count);
  gpuErrchkLaunch();
  }
  // Pointer swap the double buffers.
  xmachine_message_location_list* d_locations_temp = d_locations;
  d_locations = d_locations_swap;
  d_locations_swap = d_locations_temp;

  
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Agent_default_count+h_xmachine_memory_Agent_count > xmachine_memory_Agent_MAX){
		printf("Error: Buffer size of output_location agents in state default will be exceeded moving working agents to next state in function output_location\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Agents_default_temp = d_Agents;
  d_Agents = d_Agents_default;
  d_Agents_default = Agents_default_temp;
        
	//update new state agent size
	h_xmachine_memory_Agent_default_count += h_xmachine_memory_Agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Agent_default_count, &h_xmachine_memory_Agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Agent_read_locations_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is On-Graph Partitioned
  sm_size += (blockSize * sizeof(xmachine_message_location));
  
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Agent_read_locations
 * Agent function prototype for read_locations function of Agent agent
 */
void Agent_read_locations(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Agent_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Agent_list* Agents_default_temp = d_Agents;
	d_Agents = d_Agents_default;
	d_Agents_default = Agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_Agent_count = h_xmachine_memory_Agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Agent_count, &h_xmachine_memory_Agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Agent_default_count, &h_xmachine_memory_Agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_intent_count + h_xmachine_memory_Agent_count > xmachine_message_intent_MAX){
		printf("Error: Buffer size of intent message will be exceeded in function read_locations\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_read_locations, Agent_read_locations_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Agent_read_locations_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_intent_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_intent_output_type, &h_message_intent_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_intent_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_intent_swaps<<<gridSize, blockSize, 0, stream>>>(d_intents); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (read_locations)
	//Reallocate   : false
	//Input        : location
	//Output       : intent
	//Agent Output : 
	GPUFLAME_read_locations<<<g, b, sm_size, stream>>>(d_Agents, d_locations, d_xmachine_message_location_bounds, d_intents);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//intent Message Type Prefix Sum
	
	//swap output
	xmachine_message_intent_list* d_intents_scanswap_temp = d_intents;
	d_intents = d_intents_swap;
	d_intents_swap = d_intents_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Agent, 
        temp_scan_storage_bytes_Agent, 
        d_intents_swap->_scan_input,
        d_intents_swap->_position,
        h_xmachine_memory_Agent_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_intent_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_intent_messages<<<gridSize, blockSize, 0, stream>>>(d_intents, d_intents_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_intents_swap->_position[h_xmachine_memory_Agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_intents_swap->_scan_input[h_xmachine_memory_Agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_intent_count += scan_last_sum+1;
	}else{
		h_message_intent_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_intent_count, &h_message_intent_count, sizeof(int)));	
	
  // Sort messages based on the edge index, and construct the relevant data structure for graph edge based messaging. Keys are sorted and then message data is scattered. 

  // Reset the message bounds data structure to 0
  gpuErrchk(cudaMemset((void*)d_xmachine_message_intent_bounds, 0, sizeof(xmachine_message_intent_bounds)));

  // If there are any messages output (to account for 0 optional messages)
  if (h_message_intent_count > 0){
  // Build histogram using atomics
  cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, hist_intent_messages, no_sm, h_message_intent_count);
  gridSize = (h_message_intent_count + blockSize - 1) / blockSize;
  hist_intent_messages <<<gridSize, blockSize, 0, stream >>>(d_xmachine_message_intent_scatterer->edge_local_index, d_xmachine_message_intent_scatterer->unsorted_edge_index, d_xmachine_message_intent_bounds->count, d_intents, h_message_intent_count);
  gpuErrchkLaunch();

  // Exclusive scan on histogram output to find the index for each message for each edge/bucket
  cub::DeviceScan::ExclusiveSum(
      d_temp_scan_storage_xmachine_message_intent,
      temp_scan_bytes_xmachine_message_intent,
      d_xmachine_message_intent_bounds->count,
      d_xmachine_message_intent_bounds->start,
      staticGraph_network_edge_bufferSize, 
      stream
  );
  gpuErrchkLaunch();

  // Launch kernel to re-order (scatter) the messages
  cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, reorder_intent_messages, no_sm, h_message_intent_count);
  gridSize = (h_message_intent_count + blockSize - 1) / blockSize;  // Round up according to array size
  reorder_intent_messages <<<gridSize, blockSize, 0, stream >>>(d_xmachine_message_intent_scatterer->edge_local_index, d_xmachine_message_intent_scatterer->unsorted_edge_index, d_xmachine_message_intent_bounds->start, d_intents, d_intents_swap, h_message_intent_count);
  gpuErrchkLaunch();
  }
  // Pointer swap the double buffers.
  xmachine_message_intent_list* d_intents_temp = d_intents;
  d_intents = d_intents_swap;
  d_intents_swap = d_intents_temp;

  
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Agent_default_count+h_xmachine_memory_Agent_count > xmachine_memory_Agent_MAX){
		printf("Error: Buffer size of read_locations agents in state default will be exceeded moving working agents to next state in function read_locations\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Agents_default_temp = d_Agents;
  d_Agents = d_Agents_default;
  d_Agents_default = Agents_default_temp;
        
	//update new state agent size
	h_xmachine_memory_Agent_default_count += h_xmachine_memory_Agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Agent_default_count, &h_xmachine_memory_Agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Agent_resolve_intent_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is On-Graph Partitioned
  sm_size += (blockSize * sizeof(xmachine_message_intent));
  
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** Agent_resolve_intent
 * Agent function prototype for resolve_intent function of Agent agent
 */
void Agent_resolve_intent(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Agent_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_Agent_count = h_xmachine_memory_Agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Agent_count, &h_xmachine_memory_Agent_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Agent_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Agents_default);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_Agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Agents);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, resolve_intent_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	resolve_intent_function_filter<<<gridSize, blockSize, 0, stream>>>(d_Agents_default, d_Agents);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_Agent_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Agent, 
        temp_scan_storage_bytes_Agent, 
        d_Agents_default->_scan_input,
        d_Agents_default->_position,
        h_xmachine_memory_Agent_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Agents_default->_position[h_xmachine_memory_Agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Agents_default->_scan_input[h_xmachine_memory_Agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_Agent_default_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Agent_default_count = scan_last_sum;
	//Scatter into swap
	scatter_Agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_Agents_swap, d_Agents_default, 0, h_xmachine_memory_Agent_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_Agent_list* Agents_default_temp = d_Agents_default;
	d_Agents_default = d_Agents_swap;
	d_Agents_swap = Agents_default_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Agent_default_count, &h_xmachine_memory_Agent_default_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Agent, 
        temp_scan_storage_bytes_Agent, 
        d_Agents->_scan_input,
        d_Agents->_position,
        h_xmachine_memory_Agent_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Agents->_position[h_xmachine_memory_Agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Agents->_scan_input[h_xmachine_memory_Agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_Agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_Agents_swap, d_Agents, 0, h_xmachine_memory_Agent_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_Agent_count = scan_last_sum+1;
	else		
		h_xmachine_memory_Agent_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_Agent_list* Agents_temp = d_Agents;
	d_Agents = d_Agents_swap;
	d_Agents_swap = Agents_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Agent_count, &h_xmachine_memory_Agent_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_Agent_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_Agent_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_resolve_intent, Agent_resolve_intent_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Agent_resolve_intent_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_Agent_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_Agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_Agents);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (resolve_intent)
	//Reallocate   : true
	//Input        : intent
	//Output       : 
	//Agent Output : 
	GPUFLAME_resolve_intent<<<g, b, sm_size, stream>>>(d_Agents, d_intents, d_xmachine_message_intent_bounds, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_Agent, 
        temp_scan_storage_bytes_Agent, 
        d_Agents->_scan_input,
        d_Agents->_position,
        h_xmachine_memory_Agent_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_Agent_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_Agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_Agents_swap, d_Agents, 0, h_xmachine_memory_Agent_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_Agent_list* resolve_intent_Agents_temp = d_Agents;
	d_Agents = d_Agents_swap;
	d_Agents_swap = resolve_intent_Agents_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_Agents_swap->_position[h_xmachine_memory_Agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_Agents_swap->_scan_input[h_xmachine_memory_Agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_Agent_count = scan_last_sum+1;
	else
		h_xmachine_memory_Agent_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Agent_count, &h_xmachine_memory_Agent_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Agent_default_count+h_xmachine_memory_Agent_count > xmachine_memory_Agent_MAX){
		printf("Error: Buffer size of resolve_intent agents in state default will be exceeded moving working agents to next state in function resolve_intent\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_Agent_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_Agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_Agents_default, d_Agents, h_xmachine_memory_Agent_default_count, h_xmachine_memory_Agent_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_Agent_default_count += h_xmachine_memory_Agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Agent_default_count, &h_xmachine_memory_Agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int Agent_move_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** Agent_move
 * Agent function prototype for move function of Agent agent
 */
void Agent_move(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_Agent_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_Agent_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_Agent_list* Agents_default_temp = d_Agents;
	d_Agents = d_Agents_default;
	d_Agents_default = Agents_default_temp;
	//set working count to current state count
	h_xmachine_memory_Agent_count = h_xmachine_memory_Agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Agent_count, &h_xmachine_memory_Agent_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_Agent_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Agent_default_count, &h_xmachine_memory_Agent_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_move, Agent_move_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = Agent_move_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (move)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_move<<<g, b, sm_size, stream>>>(d_Agents);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_Agent_default_count+h_xmachine_memory_Agent_count > xmachine_memory_Agent_MAX){
		printf("Error: Buffer size of move agents in state default will be exceeded moving working agents to next state in function move\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  Agents_default_temp = d_Agents;
  d_Agents = d_Agents_default;
  d_Agents_default = Agents_default_temp;
        
	//update new state agent size
	h_xmachine_memory_Agent_default_count += h_xmachine_memory_Agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_Agent_default_count, &h_xmachine_memory_Agent_default_count, sizeof(int)));	
	
	
}


 
extern void reset_Agent_default_count()
{
    h_xmachine_memory_Agent_default_count = 0;
}
