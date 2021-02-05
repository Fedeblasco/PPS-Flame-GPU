
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

/* keratinocyte Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_keratinocyte_list* d_keratinocytes;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_keratinocyte_list* d_keratinocytes_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_keratinocyte_list* d_keratinocytes_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_keratinocyte_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_keratinocyte_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_keratinocyte_values;  /**< Agent sort identifiers value */

/* keratinocyte state variables */
xmachine_memory_keratinocyte_list* h_keratinocytes_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_keratinocyte_list* d_keratinocytes_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_keratinocyte_default_count;   /**< Agent population size counter */ 

/* keratinocyte state variables */
xmachine_memory_keratinocyte_list* h_keratinocytes_resolve;      /**< Pointer to agent list (population) on host*/
xmachine_memory_keratinocyte_list* d_keratinocytes_resolve;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_keratinocyte_resolve_count;   /**< Agent population size counter */ 


/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
unsigned int h_keratinocytes_default_variable_id_data_iteration;
unsigned int h_keratinocytes_default_variable_type_data_iteration;
unsigned int h_keratinocytes_default_variable_x_data_iteration;
unsigned int h_keratinocytes_default_variable_y_data_iteration;
unsigned int h_keratinocytes_default_variable_z_data_iteration;
unsigned int h_keratinocytes_default_variable_force_x_data_iteration;
unsigned int h_keratinocytes_default_variable_force_y_data_iteration;
unsigned int h_keratinocytes_default_variable_force_z_data_iteration;
unsigned int h_keratinocytes_default_variable_num_xy_bonds_data_iteration;
unsigned int h_keratinocytes_default_variable_num_z_bonds_data_iteration;
unsigned int h_keratinocytes_default_variable_num_stem_bonds_data_iteration;
unsigned int h_keratinocytes_default_variable_cycle_data_iteration;
unsigned int h_keratinocytes_default_variable_diff_noise_factor_data_iteration;
unsigned int h_keratinocytes_default_variable_dead_ticks_data_iteration;
unsigned int h_keratinocytes_default_variable_contact_inhibited_ticks_data_iteration;
unsigned int h_keratinocytes_default_variable_motility_data_iteration;
unsigned int h_keratinocytes_default_variable_dir_data_iteration;
unsigned int h_keratinocytes_default_variable_movement_data_iteration;
unsigned int h_keratinocytes_resolve_variable_id_data_iteration;
unsigned int h_keratinocytes_resolve_variable_type_data_iteration;
unsigned int h_keratinocytes_resolve_variable_x_data_iteration;
unsigned int h_keratinocytes_resolve_variable_y_data_iteration;
unsigned int h_keratinocytes_resolve_variable_z_data_iteration;
unsigned int h_keratinocytes_resolve_variable_force_x_data_iteration;
unsigned int h_keratinocytes_resolve_variable_force_y_data_iteration;
unsigned int h_keratinocytes_resolve_variable_force_z_data_iteration;
unsigned int h_keratinocytes_resolve_variable_num_xy_bonds_data_iteration;
unsigned int h_keratinocytes_resolve_variable_num_z_bonds_data_iteration;
unsigned int h_keratinocytes_resolve_variable_num_stem_bonds_data_iteration;
unsigned int h_keratinocytes_resolve_variable_cycle_data_iteration;
unsigned int h_keratinocytes_resolve_variable_diff_noise_factor_data_iteration;
unsigned int h_keratinocytes_resolve_variable_dead_ticks_data_iteration;
unsigned int h_keratinocytes_resolve_variable_contact_inhibited_ticks_data_iteration;
unsigned int h_keratinocytes_resolve_variable_motility_data_iteration;
unsigned int h_keratinocytes_resolve_variable_dir_data_iteration;
unsigned int h_keratinocytes_resolve_variable_movement_data_iteration;


/* Message Memory */

/* location Message variables */
xmachine_message_location_list* h_locations;         /**< Pointer to message list on host*/
xmachine_message_location_list* d_locations;         /**< Pointer to message list on device*/
xmachine_message_location_list* d_locations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_location_count;         /**< message list counter*/
int h_message_location_output_type;   /**< message output type (single or optional)*/
/* Spatial Partitioning Variables*/
#ifdef FAST_ATOMIC_SORTING
	uint * d_xmachine_message_location_local_bin_index;	  /**< index offset within the assigned bin */
	uint * d_xmachine_message_location_unsorted_index;		/**< unsorted index (hash) value for message */
    // Values for CUB exclusive scan of spatially partitioned variables
    void * d_temp_scan_storage_xmachine_message_location;
    size_t temp_scan_bytes_xmachine_message_location;
#else
	uint * d_xmachine_message_location_keys;	  /**< message sort identifier keys*/
	uint * d_xmachine_message_location_values;  /**< message sort identifier values */
#endif
xmachine_message_location_PBM * d_location_partition_matrix;  /**< Pointer to PCB matrix */
glm::vec3 h_message_location_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
glm::vec3 h_message_location_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
glm::ivec3 h_message_location_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
float h_message_location_radius;                 /**< partition radius (used to determin the size of the partitions) */
/* Texture offset values for host */
int h_tex_xmachine_message_location_id_offset;
int h_tex_xmachine_message_location_type_offset;
int h_tex_xmachine_message_location_x_offset;
int h_tex_xmachine_message_location_y_offset;
int h_tex_xmachine_message_location_z_offset;
int h_tex_xmachine_message_location_dir_offset;
int h_tex_xmachine_message_location_motility_offset;
int h_tex_xmachine_message_location_range_offset;
int h_tex_xmachine_message_location_iteration_offset;
int h_tex_xmachine_message_location_pbm_start_offset;
int h_tex_xmachine_message_location_pbm_end_or_count_offset;

/* force Message variables */
xmachine_message_force_list* h_forces;         /**< Pointer to message list on host*/
xmachine_message_force_list* d_forces;         /**< Pointer to message list on device*/
xmachine_message_force_list* d_forces_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_force_count;         /**< message list counter*/
int h_message_force_output_type;   /**< message output type (single or optional)*/
/* Spatial Partitioning Variables*/
#ifdef FAST_ATOMIC_SORTING
	uint * d_xmachine_message_force_local_bin_index;	  /**< index offset within the assigned bin */
	uint * d_xmachine_message_force_unsorted_index;		/**< unsorted index (hash) value for message */
    // Values for CUB exclusive scan of spatially partitioned variables
    void * d_temp_scan_storage_xmachine_message_force;
    size_t temp_scan_bytes_xmachine_message_force;
#else
	uint * d_xmachine_message_force_keys;	  /**< message sort identifier keys*/
	uint * d_xmachine_message_force_values;  /**< message sort identifier values */
#endif
xmachine_message_force_PBM * d_force_partition_matrix;  /**< Pointer to PCB matrix */
glm::vec3 h_message_force_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
glm::vec3 h_message_force_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
glm::ivec3 h_message_force_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
float h_message_force_radius;                 /**< partition radius (used to determin the size of the partitions) */
/* Texture offset values for host */
int h_tex_xmachine_message_force_type_offset;
int h_tex_xmachine_message_force_x_offset;
int h_tex_xmachine_message_force_y_offset;
int h_tex_xmachine_message_force_z_offset;
int h_tex_xmachine_message_force_id_offset;
int h_tex_xmachine_message_force_pbm_start_offset;
int h_tex_xmachine_message_force_pbm_end_or_count_offset;

  
/* CUDA Streams for function layers */
cudaStream_t stream1;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_keratinocyte;
size_t temp_scan_storage_bytes_keratinocyte;


/*Global condition counts*/
int h_output_location_condition_count;


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

/** keratinocyte_output_location
 * Agent function prototype for output_location function of keratinocyte agent
 */
void keratinocyte_output_location(cudaStream_t &stream);

/** keratinocyte_cycle
 * Agent function prototype for cycle function of keratinocyte agent
 */
void keratinocyte_cycle(cudaStream_t &stream);

/** keratinocyte_differentiate
 * Agent function prototype for differentiate function of keratinocyte agent
 */
void keratinocyte_differentiate(cudaStream_t &stream);

/** keratinocyte_death_signal
 * Agent function prototype for death_signal function of keratinocyte agent
 */
void keratinocyte_death_signal(cudaStream_t &stream);

/** keratinocyte_migrate
 * Agent function prototype for migrate function of keratinocyte agent
 */
void keratinocyte_migrate(cudaStream_t &stream);

/** keratinocyte_force_resolution_output
 * Agent function prototype for force_resolution_output function of keratinocyte agent
 */
void keratinocyte_force_resolution_output(cudaStream_t &stream);

/** keratinocyte_resolve_forces
 * Agent function prototype for resolve_forces function of keratinocyte agent
 */
void keratinocyte_resolve_forces(cudaStream_t &stream);

  
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
    h_keratinocytes_default_variable_id_data_iteration = 0;
    h_keratinocytes_default_variable_type_data_iteration = 0;
    h_keratinocytes_default_variable_x_data_iteration = 0;
    h_keratinocytes_default_variable_y_data_iteration = 0;
    h_keratinocytes_default_variable_z_data_iteration = 0;
    h_keratinocytes_default_variable_force_x_data_iteration = 0;
    h_keratinocytes_default_variable_force_y_data_iteration = 0;
    h_keratinocytes_default_variable_force_z_data_iteration = 0;
    h_keratinocytes_default_variable_num_xy_bonds_data_iteration = 0;
    h_keratinocytes_default_variable_num_z_bonds_data_iteration = 0;
    h_keratinocytes_default_variable_num_stem_bonds_data_iteration = 0;
    h_keratinocytes_default_variable_cycle_data_iteration = 0;
    h_keratinocytes_default_variable_diff_noise_factor_data_iteration = 0;
    h_keratinocytes_default_variable_dead_ticks_data_iteration = 0;
    h_keratinocytes_default_variable_contact_inhibited_ticks_data_iteration = 0;
    h_keratinocytes_default_variable_motility_data_iteration = 0;
    h_keratinocytes_default_variable_dir_data_iteration = 0;
    h_keratinocytes_default_variable_movement_data_iteration = 0;
    h_keratinocytes_resolve_variable_id_data_iteration = 0;
    h_keratinocytes_resolve_variable_type_data_iteration = 0;
    h_keratinocytes_resolve_variable_x_data_iteration = 0;
    h_keratinocytes_resolve_variable_y_data_iteration = 0;
    h_keratinocytes_resolve_variable_z_data_iteration = 0;
    h_keratinocytes_resolve_variable_force_x_data_iteration = 0;
    h_keratinocytes_resolve_variable_force_y_data_iteration = 0;
    h_keratinocytes_resolve_variable_force_z_data_iteration = 0;
    h_keratinocytes_resolve_variable_num_xy_bonds_data_iteration = 0;
    h_keratinocytes_resolve_variable_num_z_bonds_data_iteration = 0;
    h_keratinocytes_resolve_variable_num_stem_bonds_data_iteration = 0;
    h_keratinocytes_resolve_variable_cycle_data_iteration = 0;
    h_keratinocytes_resolve_variable_diff_noise_factor_data_iteration = 0;
    h_keratinocytes_resolve_variable_dead_ticks_data_iteration = 0;
    h_keratinocytes_resolve_variable_contact_inhibited_ticks_data_iteration = 0;
    h_keratinocytes_resolve_variable_motility_data_iteration = 0;
    h_keratinocytes_resolve_variable_dir_data_iteration = 0;
    h_keratinocytes_resolve_variable_movement_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_keratinocyte_SoA_size = sizeof(xmachine_memory_keratinocyte_list);
	h_keratinocytes_default = (xmachine_memory_keratinocyte_list*)malloc(xmachine_keratinocyte_SoA_size);
	h_keratinocytes_resolve = (xmachine_memory_keratinocyte_list*)malloc(xmachine_keratinocyte_SoA_size);

	/* Message memory allocation (CPU) */
	int message_location_SoA_size = sizeof(xmachine_message_location_list);
	h_locations = (xmachine_message_location_list*)malloc(message_location_SoA_size);
	int message_force_SoA_size = sizeof(xmachine_message_force_list);
	h_forces = (xmachine_message_force_list*)malloc(message_force_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs

  /* Graph memory allocation (CPU) */
  

    PROFILE_POP_RANGE(); //"allocate host"
	
			
	/* Set spatial partitioning location message variables (min_bounds, max_bounds)*/
	h_message_location_radius = (float)125;
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_radius, &h_message_location_radius, sizeof(float)));	
	    h_message_location_min_bounds = glm::vec3((float)0.0, (float)0.0, (float)0.0);
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_min_bounds, &h_message_location_min_bounds, sizeof(glm::vec3)));	
	h_message_location_max_bounds = glm::vec3((float)500, (float)500, (float)500);
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_max_bounds, &h_message_location_max_bounds, sizeof(glm::vec3)));	
	h_message_location_partitionDim.x = (int)ceil((h_message_location_max_bounds.x - h_message_location_min_bounds.x)/h_message_location_radius);
	h_message_location_partitionDim.y = (int)ceil((h_message_location_max_bounds.y - h_message_location_min_bounds.y)/h_message_location_radius);
	h_message_location_partitionDim.z = (int)ceil((h_message_location_max_bounds.z - h_message_location_min_bounds.z)/h_message_location_radius);
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_partitionDim, &h_message_location_partitionDim, sizeof(glm::ivec3)));	
	
			
	/* Set spatial partitioning force message variables (min_bounds, max_bounds)*/
	h_message_force_radius = (float)15.625;
	gpuErrchk(cudaMemcpyToSymbol( d_message_force_radius, &h_message_force_radius, sizeof(float)));	
	    h_message_force_min_bounds = glm::vec3((float)0.0, (float)0.0, (float)0.0);
	gpuErrchk(cudaMemcpyToSymbol( d_message_force_min_bounds, &h_message_force_min_bounds, sizeof(glm::vec3)));	
	h_message_force_max_bounds = glm::vec3((float)500, (float)500, (float)500);
	gpuErrchk(cudaMemcpyToSymbol( d_message_force_max_bounds, &h_message_force_max_bounds, sizeof(glm::vec3)));	
	h_message_force_partitionDim.x = (int)ceil((h_message_force_max_bounds.x - h_message_force_min_bounds.x)/h_message_force_radius);
	h_message_force_partitionDim.y = (int)ceil((h_message_force_max_bounds.y - h_message_force_min_bounds.y)/h_message_force_radius);
	h_message_force_partitionDim.z = (int)ceil((h_message_force_max_bounds.z - h_message_force_min_bounds.z)/h_message_force_radius);
	gpuErrchk(cudaMemcpyToSymbol( d_message_force_partitionDim, &h_message_force_partitionDim, sizeof(glm::ivec3)));	
	

	//read initial states
	readInitialStates(inputfile, h_keratinocytes_resolve, &h_xmachine_memory_keratinocyte_resolve_count);

  // Read graphs from disk
  

  PROFILE_PUSH_RANGE("allocate device");
	
	/* keratinocyte Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_keratinocytes, xmachine_keratinocyte_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_keratinocytes_swap, xmachine_keratinocyte_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_keratinocytes_new, xmachine_keratinocyte_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_keratinocyte_keys, xmachine_memory_keratinocyte_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_keratinocyte_values, xmachine_memory_keratinocyte_MAX* sizeof(uint)));
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_keratinocytes_default, xmachine_keratinocyte_SoA_size));
	gpuErrchk( cudaMemcpy( d_keratinocytes_default, h_keratinocytes_default, xmachine_keratinocyte_SoA_size, cudaMemcpyHostToDevice));
    
	/* resolve memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_keratinocytes_resolve, xmachine_keratinocyte_SoA_size));
	gpuErrchk( cudaMemcpy( d_keratinocytes_resolve, h_keratinocytes_resolve, xmachine_keratinocyte_SoA_size, cudaMemcpyHostToDevice));
    
	/* location Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_locations, message_location_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_locations_swap, message_location_SoA_size));
	gpuErrchk( cudaMemcpy( d_locations, h_locations, message_location_SoA_size, cudaMemcpyHostToDevice));
	gpuErrchk( cudaMalloc( (void**) &d_location_partition_matrix, sizeof(xmachine_message_location_PBM)));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_location_local_bin_index, xmachine_message_location_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_location_unsorted_index, xmachine_message_location_MAX* sizeof(uint)));
    /* Calculate and allocate CUB temporary memory for exclusive scans */
    d_temp_scan_storage_xmachine_message_location = nullptr;
    temp_scan_bytes_xmachine_message_location = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_xmachine_message_location, 
        temp_scan_bytes_xmachine_message_location, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_message_location_grid_size
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_xmachine_message_location, temp_scan_bytes_xmachine_message_location));
#else
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_location_keys, xmachine_message_location_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_location_values, xmachine_message_location_MAX* sizeof(uint)));
#endif
	
	/* force Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_forces, message_force_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_forces_swap, message_force_SoA_size));
	gpuErrchk( cudaMemcpy( d_forces, h_forces, message_force_SoA_size, cudaMemcpyHostToDevice));
	gpuErrchk( cudaMalloc( (void**) &d_force_partition_matrix, sizeof(xmachine_message_force_PBM)));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_force_local_bin_index, xmachine_message_force_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_force_unsorted_index, xmachine_message_force_MAX* sizeof(uint)));
    /* Calculate and allocate CUB temporary memory for exclusive scans */
    d_temp_scan_storage_xmachine_message_force = nullptr;
    temp_scan_bytes_xmachine_message_force = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_xmachine_message_force, 
        temp_scan_bytes_xmachine_message_force, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_message_force_grid_size
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_xmachine_message_force, temp_scan_bytes_xmachine_message_force));
#else
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_force_keys, xmachine_message_force_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_force_values, xmachine_message_force_MAX* sizeof(uint)));
#endif
		


  /* Allocate device memory for graphs */
  

    PROFILE_POP_RANGE(); // "allocate device"

    /* Calculate and allocate CUB temporary memory for exclusive scans */
    
    d_temp_scan_storage_keratinocyte = nullptr;
    temp_scan_storage_bytes_keratinocyte = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_keratinocyte, 
        temp_scan_storage_bytes_keratinocyte, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_keratinocyte_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_keratinocyte, temp_scan_storage_bytes_keratinocyte));
    

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
    setConstants();
    PROFILE_PUSH_RANGE("setConstants");
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: setConstants = %f (ms)\n", instrument_milliseconds);
#endif
	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_keratinocyte_default_count: %u\n",get_agent_keratinocyte_default_count());
	
		printf("Init agent_keratinocyte_resolve_count: %u\n",get_agent_keratinocyte_resolve_count());
	
#endif
} 


void sort_keratinocytes_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_keratinocyte_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_keratinocyte_default_count); 
	gridSize = (h_xmachine_memory_keratinocyte_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_keratinocyte_keys, d_xmachine_memory_keratinocyte_values, d_keratinocytes_default);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_keratinocyte_keys),  thrust::device_pointer_cast(d_xmachine_memory_keratinocyte_keys) + h_xmachine_memory_keratinocyte_default_count,  thrust::device_pointer_cast(d_xmachine_memory_keratinocyte_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_keratinocyte_agents, no_sm, h_xmachine_memory_keratinocyte_default_count); 
	gridSize = (h_xmachine_memory_keratinocyte_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_keratinocyte_agents<<<gridSize, blockSize>>>(d_xmachine_memory_keratinocyte_values, d_keratinocytes_default, d_keratinocytes_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_keratinocyte_list* d_keratinocytes_temp = d_keratinocytes_default;
	d_keratinocytes_default = d_keratinocytes_swap;
	d_keratinocytes_swap = d_keratinocytes_temp;	
}

void sort_keratinocytes_resolve(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_keratinocyte_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_keratinocyte_resolve_count); 
	gridSize = (h_xmachine_memory_keratinocyte_resolve_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_keratinocyte_keys, d_xmachine_memory_keratinocyte_values, d_keratinocytes_resolve);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_keratinocyte_keys),  thrust::device_pointer_cast(d_xmachine_memory_keratinocyte_keys) + h_xmachine_memory_keratinocyte_resolve_count,  thrust::device_pointer_cast(d_xmachine_memory_keratinocyte_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_keratinocyte_agents, no_sm, h_xmachine_memory_keratinocyte_resolve_count); 
	gridSize = (h_xmachine_memory_keratinocyte_resolve_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_keratinocyte_agents<<<gridSize, blockSize>>>(d_xmachine_memory_keratinocyte_values, d_keratinocytes_resolve, d_keratinocytes_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_keratinocyte_list* d_keratinocytes_temp = d_keratinocytes_resolve;
	d_keratinocytes_resolve = d_keratinocytes_swap;
	d_keratinocytes_swap = d_keratinocytes_temp;	
}


void cleanup(){
    PROFILE_SCOPED_RANGE("cleanup");

    /* Call all exit functions */
	

	/* Agent data free*/
	
	/* keratinocyte Agent variables */
	gpuErrchk(cudaFree(d_keratinocytes));
	gpuErrchk(cudaFree(d_keratinocytes_swap));
	gpuErrchk(cudaFree(d_keratinocytes_new));
	
	free( h_keratinocytes_default);
	gpuErrchk(cudaFree(d_keratinocytes_default));
	
	free( h_keratinocytes_resolve);
	gpuErrchk(cudaFree(d_keratinocytes_resolve));
	

	/* Message data free */
	
	/* location Message variables */
	free( h_locations);
	gpuErrchk(cudaFree(d_locations));
	gpuErrchk(cudaFree(d_locations_swap));
	gpuErrchk(cudaFree(d_location_partition_matrix));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaFree(d_xmachine_message_location_local_bin_index));
	gpuErrchk(cudaFree(d_xmachine_message_location_unsorted_index));
  gpuErrchk(cudaFree(d_temp_scan_storage_xmachine_message_location));
  d_temp_scan_storage_xmachine_message_location = nullptr;
  temp_scan_bytes_xmachine_message_location = 0;
#else
	gpuErrchk(cudaFree(d_xmachine_message_location_keys));
	gpuErrchk(cudaFree(d_xmachine_message_location_values));
#endif
	
	/* force Message variables */
	free( h_forces);
	gpuErrchk(cudaFree(d_forces));
	gpuErrchk(cudaFree(d_forces_swap));
	gpuErrchk(cudaFree(d_force_partition_matrix));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaFree(d_xmachine_message_force_local_bin_index));
	gpuErrchk(cudaFree(d_xmachine_message_force_unsorted_index));
  gpuErrchk(cudaFree(d_temp_scan_storage_xmachine_message_force));
  d_temp_scan_storage_xmachine_message_force = nullptr;
  temp_scan_bytes_xmachine_message_force = 0;
#else
	gpuErrchk(cudaFree(d_xmachine_message_force_keys));
	gpuErrchk(cudaFree(d_xmachine_message_force_values));
#endif
	

    /* Free temporary CUB memory if required. */
    
    if(d_temp_scan_storage_keratinocyte != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_keratinocyte));
      d_temp_scan_storage_keratinocyte = nullptr;
      temp_scan_storage_bytes_keratinocyte = 0;
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
	h_message_location_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));
	
	h_message_force_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_force_count, &h_message_force_count, sizeof(int)));
	

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("keratinocyte_output_location");
	keratinocyte_output_location(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: keratinocyte_output_location = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("keratinocyte_cycle");
	keratinocyte_cycle(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: keratinocyte_cycle = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("keratinocyte_differentiate");
	keratinocyte_differentiate(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: keratinocyte_differentiate = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 4*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("keratinocyte_death_signal");
	keratinocyte_death_signal(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: keratinocyte_death_signal = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 5*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("keratinocyte_migrate");
	keratinocyte_migrate(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: keratinocyte_migrate = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 6*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("keratinocyte_force_resolution_output");
	keratinocyte_force_resolution_output(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: keratinocyte_force_resolution_output = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 7*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("keratinocyte_resolve_forces");
	keratinocyte_resolve_forces(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: keratinocyte_resolve_forces = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_keratinocyte_default_count: %u\n",get_agent_keratinocyte_default_count());
	
		printf("agent_keratinocyte_resolve_count: %u\n",get_agent_keratinocyte_resolve_count());
	
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
float h_env_calcium_level;
int h_env_CYCLE_LENGTH[5];
float h_env_SUBSTRATE_FORCE[5];
float h_env_DOWNWARD_FORCE[5];
float h_env_FORCE_MATRIX[25];
float h_env_FORCE_REP;
float h_env_FORCE_DAMPENER;
int h_env_BASEMENT_MAX_Z;


//constant setter
void set_calcium_level(float* h_calcium_level){
    gpuErrchk(cudaMemcpyToSymbol(calcium_level, h_calcium_level, sizeof(float)));
    memcpy(&h_env_calcium_level, h_calcium_level,sizeof(float));
}

//constant getter
const float* get_calcium_level(){
    return &h_env_calcium_level;
}



//constant setter
void set_CYCLE_LENGTH(int* h_CYCLE_LENGTH){
    gpuErrchk(cudaMemcpyToSymbol(CYCLE_LENGTH, h_CYCLE_LENGTH, sizeof(int)*5));
    memcpy(&h_env_CYCLE_LENGTH, h_CYCLE_LENGTH,sizeof(int)*5);
}

//constant getter
const int* get_CYCLE_LENGTH(){
    return h_env_CYCLE_LENGTH;
}



//constant setter
void set_SUBSTRATE_FORCE(float* h_SUBSTRATE_FORCE){
    gpuErrchk(cudaMemcpyToSymbol(SUBSTRATE_FORCE, h_SUBSTRATE_FORCE, sizeof(float)*5));
    memcpy(&h_env_SUBSTRATE_FORCE, h_SUBSTRATE_FORCE,sizeof(float)*5);
}

//constant getter
const float* get_SUBSTRATE_FORCE(){
    return h_env_SUBSTRATE_FORCE;
}



//constant setter
void set_DOWNWARD_FORCE(float* h_DOWNWARD_FORCE){
    gpuErrchk(cudaMemcpyToSymbol(DOWNWARD_FORCE, h_DOWNWARD_FORCE, sizeof(float)*5));
    memcpy(&h_env_DOWNWARD_FORCE, h_DOWNWARD_FORCE,sizeof(float)*5);
}

//constant getter
const float* get_DOWNWARD_FORCE(){
    return h_env_DOWNWARD_FORCE;
}



//constant setter
void set_FORCE_MATRIX(float* h_FORCE_MATRIX){
    gpuErrchk(cudaMemcpyToSymbol(FORCE_MATRIX, h_FORCE_MATRIX, sizeof(float)*25));
    memcpy(&h_env_FORCE_MATRIX, h_FORCE_MATRIX,sizeof(float)*25);
}

//constant getter
const float* get_FORCE_MATRIX(){
    return h_env_FORCE_MATRIX;
}



//constant setter
void set_FORCE_REP(float* h_FORCE_REP){
    gpuErrchk(cudaMemcpyToSymbol(FORCE_REP, h_FORCE_REP, sizeof(float)));
    memcpy(&h_env_FORCE_REP, h_FORCE_REP,sizeof(float));
}

//constant getter
const float* get_FORCE_REP(){
    return &h_env_FORCE_REP;
}



//constant setter
void set_FORCE_DAMPENER(float* h_FORCE_DAMPENER){
    gpuErrchk(cudaMemcpyToSymbol(FORCE_DAMPENER, h_FORCE_DAMPENER, sizeof(float)));
    memcpy(&h_env_FORCE_DAMPENER, h_FORCE_DAMPENER,sizeof(float));
}

//constant getter
const float* get_FORCE_DAMPENER(){
    return &h_env_FORCE_DAMPENER;
}



//constant setter
void set_BASEMENT_MAX_Z(int* h_BASEMENT_MAX_Z){
    gpuErrchk(cudaMemcpyToSymbol(BASEMENT_MAX_Z, h_BASEMENT_MAX_Z, sizeof(int)));
    memcpy(&h_env_BASEMENT_MAX_Z, h_BASEMENT_MAX_Z,sizeof(int));
}

//constant getter
const int* get_BASEMENT_MAX_Z(){
    return &h_env_BASEMENT_MAX_Z;
}




/* Agent data access functions*/

    
int get_agent_keratinocyte_MAX_count(){
    return xmachine_memory_keratinocyte_MAX;
}


int get_agent_keratinocyte_default_count(){
	//continuous agent
	return h_xmachine_memory_keratinocyte_default_count;
	
}

xmachine_memory_keratinocyte_list* get_device_keratinocyte_default_agents(){
	return d_keratinocytes_default;
}

xmachine_memory_keratinocyte_list* get_host_keratinocyte_default_agents(){
	return h_keratinocytes_default;
}

int get_agent_keratinocyte_resolve_count(){
	//continuous agent
	return h_xmachine_memory_keratinocyte_resolve_count;
	
}

xmachine_memory_keratinocyte_list* get_device_keratinocyte_resolve_agents(){
	return d_keratinocytes_resolve;
}

xmachine_memory_keratinocyte_list* get_host_keratinocyte_resolve_agents(){
	return h_keratinocytes_resolve;
}



/* Host based access of agent variables*/

/** int get_keratinocyte_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_keratinocyte_default_variable_id(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->id,
                    d_keratinocytes_default->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_keratinocyte_default_variable_type(unsigned int index)
 * Gets the value of the type variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable type
 */
__host__ int get_keratinocyte_default_variable_type(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_type_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->type,
                    d_keratinocytes_default->type,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_type_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->type[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access type for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_keratinocyte_default_variable_x(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->x,
                    d_keratinocytes_default->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_keratinocyte_default_variable_y(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->y,
                    d_keratinocytes_default->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_default_variable_z(unsigned int index)
 * Gets the value of the z variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_keratinocyte_default_variable_z(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->z,
                    d_keratinocytes_default->z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_default_variable_force_x(unsigned int index)
 * Gets the value of the force_x variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable force_x
 */
__host__ float get_keratinocyte_default_variable_force_x(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_force_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->force_x,
                    d_keratinocytes_default->force_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_force_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->force_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access force_x for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_default_variable_force_y(unsigned int index)
 * Gets the value of the force_y variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable force_y
 */
__host__ float get_keratinocyte_default_variable_force_y(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_force_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->force_y,
                    d_keratinocytes_default->force_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_force_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->force_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access force_y for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_default_variable_force_z(unsigned int index)
 * Gets the value of the force_z variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable force_z
 */
__host__ float get_keratinocyte_default_variable_force_z(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_force_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->force_z,
                    d_keratinocytes_default->force_z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_force_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->force_z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access force_z for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_keratinocyte_default_variable_num_xy_bonds(unsigned int index)
 * Gets the value of the num_xy_bonds variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable num_xy_bonds
 */
__host__ int get_keratinocyte_default_variable_num_xy_bonds(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_num_xy_bonds_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->num_xy_bonds,
                    d_keratinocytes_default->num_xy_bonds,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_num_xy_bonds_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->num_xy_bonds[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access num_xy_bonds for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_keratinocyte_default_variable_num_z_bonds(unsigned int index)
 * Gets the value of the num_z_bonds variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable num_z_bonds
 */
__host__ int get_keratinocyte_default_variable_num_z_bonds(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_num_z_bonds_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->num_z_bonds,
                    d_keratinocytes_default->num_z_bonds,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_num_z_bonds_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->num_z_bonds[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access num_z_bonds for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_keratinocyte_default_variable_num_stem_bonds(unsigned int index)
 * Gets the value of the num_stem_bonds variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable num_stem_bonds
 */
__host__ int get_keratinocyte_default_variable_num_stem_bonds(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_num_stem_bonds_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->num_stem_bonds,
                    d_keratinocytes_default->num_stem_bonds,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_num_stem_bonds_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->num_stem_bonds[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access num_stem_bonds for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_keratinocyte_default_variable_cycle(unsigned int index)
 * Gets the value of the cycle variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable cycle
 */
__host__ int get_keratinocyte_default_variable_cycle(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_cycle_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->cycle,
                    d_keratinocytes_default->cycle,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_cycle_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->cycle[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access cycle for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_default_variable_diff_noise_factor(unsigned int index)
 * Gets the value of the diff_noise_factor variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable diff_noise_factor
 */
__host__ float get_keratinocyte_default_variable_diff_noise_factor(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_diff_noise_factor_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->diff_noise_factor,
                    d_keratinocytes_default->diff_noise_factor,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_diff_noise_factor_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->diff_noise_factor[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access diff_noise_factor for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_keratinocyte_default_variable_dead_ticks(unsigned int index)
 * Gets the value of the dead_ticks variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable dead_ticks
 */
__host__ int get_keratinocyte_default_variable_dead_ticks(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_dead_ticks_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->dead_ticks,
                    d_keratinocytes_default->dead_ticks,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_dead_ticks_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->dead_ticks[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access dead_ticks for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_keratinocyte_default_variable_contact_inhibited_ticks(unsigned int index)
 * Gets the value of the contact_inhibited_ticks variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable contact_inhibited_ticks
 */
__host__ int get_keratinocyte_default_variable_contact_inhibited_ticks(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_contact_inhibited_ticks_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->contact_inhibited_ticks,
                    d_keratinocytes_default->contact_inhibited_ticks,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_contact_inhibited_ticks_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->contact_inhibited_ticks[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access contact_inhibited_ticks for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_default_variable_motility(unsigned int index)
 * Gets the value of the motility variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable motility
 */
__host__ float get_keratinocyte_default_variable_motility(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_motility_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->motility,
                    d_keratinocytes_default->motility,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_motility_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->motility[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access motility for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_default_variable_dir(unsigned int index)
 * Gets the value of the dir variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable dir
 */
__host__ float get_keratinocyte_default_variable_dir(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_dir_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->dir,
                    d_keratinocytes_default->dir,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_dir_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->dir[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access dir for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_default_variable_movement(unsigned int index)
 * Gets the value of the movement variable of an keratinocyte agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable movement
 */
__host__ float get_keratinocyte_default_variable_movement(unsigned int index){
    unsigned int count = get_agent_keratinocyte_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_default_variable_movement_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_default->movement,
                    d_keratinocytes_default->movement,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_default_variable_movement_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_default->movement[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access movement for the %u th member of keratinocyte_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_keratinocyte_resolve_variable_id(unsigned int index)
 * Gets the value of the id variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_keratinocyte_resolve_variable_id(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->id,
                    d_keratinocytes_resolve->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_keratinocyte_resolve_variable_type(unsigned int index)
 * Gets the value of the type variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable type
 */
__host__ int get_keratinocyte_resolve_variable_type(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_type_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->type,
                    d_keratinocytes_resolve->type,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_type_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->type[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access type for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_resolve_variable_x(unsigned int index)
 * Gets the value of the x variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_keratinocyte_resolve_variable_x(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->x,
                    d_keratinocytes_resolve->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_resolve_variable_y(unsigned int index)
 * Gets the value of the y variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_keratinocyte_resolve_variable_y(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->y,
                    d_keratinocytes_resolve->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_resolve_variable_z(unsigned int index)
 * Gets the value of the z variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_keratinocyte_resolve_variable_z(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->z,
                    d_keratinocytes_resolve->z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access z for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_resolve_variable_force_x(unsigned int index)
 * Gets the value of the force_x variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable force_x
 */
__host__ float get_keratinocyte_resolve_variable_force_x(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_force_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->force_x,
                    d_keratinocytes_resolve->force_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_force_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->force_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access force_x for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_resolve_variable_force_y(unsigned int index)
 * Gets the value of the force_y variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable force_y
 */
__host__ float get_keratinocyte_resolve_variable_force_y(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_force_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->force_y,
                    d_keratinocytes_resolve->force_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_force_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->force_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access force_y for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_resolve_variable_force_z(unsigned int index)
 * Gets the value of the force_z variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable force_z
 */
__host__ float get_keratinocyte_resolve_variable_force_z(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_force_z_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->force_z,
                    d_keratinocytes_resolve->force_z,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_force_z_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->force_z[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access force_z for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_keratinocyte_resolve_variable_num_xy_bonds(unsigned int index)
 * Gets the value of the num_xy_bonds variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable num_xy_bonds
 */
__host__ int get_keratinocyte_resolve_variable_num_xy_bonds(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_num_xy_bonds_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->num_xy_bonds,
                    d_keratinocytes_resolve->num_xy_bonds,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_num_xy_bonds_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->num_xy_bonds[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access num_xy_bonds for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_keratinocyte_resolve_variable_num_z_bonds(unsigned int index)
 * Gets the value of the num_z_bonds variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable num_z_bonds
 */
__host__ int get_keratinocyte_resolve_variable_num_z_bonds(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_num_z_bonds_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->num_z_bonds,
                    d_keratinocytes_resolve->num_z_bonds,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_num_z_bonds_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->num_z_bonds[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access num_z_bonds for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_keratinocyte_resolve_variable_num_stem_bonds(unsigned int index)
 * Gets the value of the num_stem_bonds variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable num_stem_bonds
 */
__host__ int get_keratinocyte_resolve_variable_num_stem_bonds(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_num_stem_bonds_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->num_stem_bonds,
                    d_keratinocytes_resolve->num_stem_bonds,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_num_stem_bonds_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->num_stem_bonds[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access num_stem_bonds for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_keratinocyte_resolve_variable_cycle(unsigned int index)
 * Gets the value of the cycle variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable cycle
 */
__host__ int get_keratinocyte_resolve_variable_cycle(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_cycle_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->cycle,
                    d_keratinocytes_resolve->cycle,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_cycle_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->cycle[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access cycle for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_resolve_variable_diff_noise_factor(unsigned int index)
 * Gets the value of the diff_noise_factor variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable diff_noise_factor
 */
__host__ float get_keratinocyte_resolve_variable_diff_noise_factor(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_diff_noise_factor_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->diff_noise_factor,
                    d_keratinocytes_resolve->diff_noise_factor,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_diff_noise_factor_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->diff_noise_factor[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access diff_noise_factor for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_keratinocyte_resolve_variable_dead_ticks(unsigned int index)
 * Gets the value of the dead_ticks variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable dead_ticks
 */
__host__ int get_keratinocyte_resolve_variable_dead_ticks(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_dead_ticks_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->dead_ticks,
                    d_keratinocytes_resolve->dead_ticks,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_dead_ticks_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->dead_ticks[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access dead_ticks for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_keratinocyte_resolve_variable_contact_inhibited_ticks(unsigned int index)
 * Gets the value of the contact_inhibited_ticks variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable contact_inhibited_ticks
 */
__host__ int get_keratinocyte_resolve_variable_contact_inhibited_ticks(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_contact_inhibited_ticks_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->contact_inhibited_ticks,
                    d_keratinocytes_resolve->contact_inhibited_ticks,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_contact_inhibited_ticks_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->contact_inhibited_ticks[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access contact_inhibited_ticks for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_resolve_variable_motility(unsigned int index)
 * Gets the value of the motility variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable motility
 */
__host__ float get_keratinocyte_resolve_variable_motility(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_motility_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->motility,
                    d_keratinocytes_resolve->motility,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_motility_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->motility[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access motility for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_resolve_variable_dir(unsigned int index)
 * Gets the value of the dir variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable dir
 */
__host__ float get_keratinocyte_resolve_variable_dir(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_dir_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->dir,
                    d_keratinocytes_resolve->dir,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_dir_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->dir[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access dir for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_keratinocyte_resolve_variable_movement(unsigned int index)
 * Gets the value of the movement variable of an keratinocyte agent in the resolve state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable movement
 */
__host__ float get_keratinocyte_resolve_variable_movement(unsigned int index){
    unsigned int count = get_agent_keratinocyte_resolve_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_keratinocytes_resolve_variable_movement_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_keratinocytes_resolve->movement,
                    d_keratinocytes_resolve->movement,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_keratinocytes_resolve_variable_movement_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_keratinocytes_resolve->movement[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access movement for the %u th member of keratinocyte_resolve. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}



/* Host based agent creation functions */
// These are only available for continuous agents.



/* copy_single_xmachine_memory_keratinocyte_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_keratinocyte_hostToDevice(xmachine_memory_keratinocyte_list * d_dst, xmachine_memory_keratinocyte * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->type, &h_agent->type, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, &h_agent->x, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, &h_agent->y, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->z, &h_agent->z, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->force_x, &h_agent->force_x, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->force_y, &h_agent->force_y, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->force_z, &h_agent->force_z, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->num_xy_bonds, &h_agent->num_xy_bonds, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->num_z_bonds, &h_agent->num_z_bonds, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->num_stem_bonds, &h_agent->num_stem_bonds, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->cycle, &h_agent->cycle, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->diff_noise_factor, &h_agent->diff_noise_factor, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->dead_ticks, &h_agent->dead_ticks, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->contact_inhibited_ticks, &h_agent->contact_inhibited_ticks, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->motility, &h_agent->motility, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->dir, &h_agent->dir, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->movement, &h_agent->movement, sizeof(float), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_keratinocyte_hostToDevice(xmachine_memory_keratinocyte_list * d_dst, xmachine_memory_keratinocyte_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->type, h_src->type, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, h_src->x, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, h_src->y, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->z, h_src->z, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->force_x, h_src->force_x, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->force_y, h_src->force_y, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->force_z, h_src->force_z, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->num_xy_bonds, h_src->num_xy_bonds, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->num_z_bonds, h_src->num_z_bonds, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->num_stem_bonds, h_src->num_stem_bonds, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->cycle, h_src->cycle, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->diff_noise_factor, h_src->diff_noise_factor, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->dead_ticks, h_src->dead_ticks, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->contact_inhibited_ticks, h_src->contact_inhibited_ticks, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->motility, h_src->motility, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->dir, h_src->dir, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->movement, h_src->movement, count * sizeof(float), cudaMemcpyHostToDevice));

    }
}

xmachine_memory_keratinocyte* h_allocate_agent_keratinocyte(){
	xmachine_memory_keratinocyte* agent = (xmachine_memory_keratinocyte*)malloc(sizeof(xmachine_memory_keratinocyte));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_keratinocyte));

	return agent;
}
void h_free_agent_keratinocyte(xmachine_memory_keratinocyte** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_keratinocyte** h_allocate_agent_keratinocyte_array(unsigned int count){
	xmachine_memory_keratinocyte ** agents = (xmachine_memory_keratinocyte**)malloc(count * sizeof(xmachine_memory_keratinocyte*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_keratinocyte();
	}
	return agents;
}
void h_free_agent_keratinocyte_array(xmachine_memory_keratinocyte*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_keratinocyte(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_keratinocyte_AoS_to_SoA(xmachine_memory_keratinocyte_list * dst, xmachine_memory_keratinocyte** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->type[i] = src[i]->type;
			 
			dst->x[i] = src[i]->x;
			 
			dst->y[i] = src[i]->y;
			 
			dst->z[i] = src[i]->z;
			 
			dst->force_x[i] = src[i]->force_x;
			 
			dst->force_y[i] = src[i]->force_y;
			 
			dst->force_z[i] = src[i]->force_z;
			 
			dst->num_xy_bonds[i] = src[i]->num_xy_bonds;
			 
			dst->num_z_bonds[i] = src[i]->num_z_bonds;
			 
			dst->num_stem_bonds[i] = src[i]->num_stem_bonds;
			 
			dst->cycle[i] = src[i]->cycle;
			 
			dst->diff_noise_factor[i] = src[i]->diff_noise_factor;
			 
			dst->dead_ticks[i] = src[i]->dead_ticks;
			 
			dst->contact_inhibited_ticks[i] = src[i]->contact_inhibited_ticks;
			 
			dst->motility[i] = src[i]->motility;
			 
			dst->dir[i] = src[i]->dir;
			 
			dst->movement[i] = src[i]->movement;
			
		}
	}
}


void h_add_agent_keratinocyte_default(xmachine_memory_keratinocyte* agent){
	if (h_xmachine_memory_keratinocyte_count + 1 > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of keratinocyte agents in state default will be exceeded by h_add_agent_keratinocyte_default\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_keratinocyte_hostToDevice(d_keratinocytes_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_keratinocyte_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_keratinocyte_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_keratinocytes_default, d_keratinocytes_new, h_xmachine_memory_keratinocyte_default_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_keratinocyte_default_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_keratinocytes_default_variable_id_data_iteration = 0;
    h_keratinocytes_default_variable_type_data_iteration = 0;
    h_keratinocytes_default_variable_x_data_iteration = 0;
    h_keratinocytes_default_variable_y_data_iteration = 0;
    h_keratinocytes_default_variable_z_data_iteration = 0;
    h_keratinocytes_default_variable_force_x_data_iteration = 0;
    h_keratinocytes_default_variable_force_y_data_iteration = 0;
    h_keratinocytes_default_variable_force_z_data_iteration = 0;
    h_keratinocytes_default_variable_num_xy_bonds_data_iteration = 0;
    h_keratinocytes_default_variable_num_z_bonds_data_iteration = 0;
    h_keratinocytes_default_variable_num_stem_bonds_data_iteration = 0;
    h_keratinocytes_default_variable_cycle_data_iteration = 0;
    h_keratinocytes_default_variable_diff_noise_factor_data_iteration = 0;
    h_keratinocytes_default_variable_dead_ticks_data_iteration = 0;
    h_keratinocytes_default_variable_contact_inhibited_ticks_data_iteration = 0;
    h_keratinocytes_default_variable_motility_data_iteration = 0;
    h_keratinocytes_default_variable_dir_data_iteration = 0;
    h_keratinocytes_default_variable_movement_data_iteration = 0;
    

}
void h_add_agents_keratinocyte_default(xmachine_memory_keratinocyte** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_keratinocyte_count + count > xmachine_memory_keratinocyte_MAX){
			printf("Error: Buffer size of keratinocyte agents in state default will be exceeded by h_add_agents_keratinocyte_default\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_keratinocyte_AoS_to_SoA(h_keratinocytes_default, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_keratinocyte_hostToDevice(d_keratinocytes_new, h_keratinocytes_default, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_keratinocyte_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_keratinocyte_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_keratinocytes_default, d_keratinocytes_new, h_xmachine_memory_keratinocyte_default_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_keratinocyte_default_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_keratinocytes_default_variable_id_data_iteration = 0;
        h_keratinocytes_default_variable_type_data_iteration = 0;
        h_keratinocytes_default_variable_x_data_iteration = 0;
        h_keratinocytes_default_variable_y_data_iteration = 0;
        h_keratinocytes_default_variable_z_data_iteration = 0;
        h_keratinocytes_default_variable_force_x_data_iteration = 0;
        h_keratinocytes_default_variable_force_y_data_iteration = 0;
        h_keratinocytes_default_variable_force_z_data_iteration = 0;
        h_keratinocytes_default_variable_num_xy_bonds_data_iteration = 0;
        h_keratinocytes_default_variable_num_z_bonds_data_iteration = 0;
        h_keratinocytes_default_variable_num_stem_bonds_data_iteration = 0;
        h_keratinocytes_default_variable_cycle_data_iteration = 0;
        h_keratinocytes_default_variable_diff_noise_factor_data_iteration = 0;
        h_keratinocytes_default_variable_dead_ticks_data_iteration = 0;
        h_keratinocytes_default_variable_contact_inhibited_ticks_data_iteration = 0;
        h_keratinocytes_default_variable_motility_data_iteration = 0;
        h_keratinocytes_default_variable_dir_data_iteration = 0;
        h_keratinocytes_default_variable_movement_data_iteration = 0;
        

	}
}


void h_add_agent_keratinocyte_resolve(xmachine_memory_keratinocyte* agent){
	if (h_xmachine_memory_keratinocyte_count + 1 > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of keratinocyte agents in state resolve will be exceeded by h_add_agent_keratinocyte_resolve\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_keratinocyte_hostToDevice(d_keratinocytes_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_keratinocyte_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_keratinocyte_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_keratinocytes_resolve, d_keratinocytes_new, h_xmachine_memory_keratinocyte_resolve_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_keratinocyte_resolve_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_keratinocyte_resolve_count, &h_xmachine_memory_keratinocyte_resolve_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_keratinocytes_resolve_variable_id_data_iteration = 0;
    h_keratinocytes_resolve_variable_type_data_iteration = 0;
    h_keratinocytes_resolve_variable_x_data_iteration = 0;
    h_keratinocytes_resolve_variable_y_data_iteration = 0;
    h_keratinocytes_resolve_variable_z_data_iteration = 0;
    h_keratinocytes_resolve_variable_force_x_data_iteration = 0;
    h_keratinocytes_resolve_variable_force_y_data_iteration = 0;
    h_keratinocytes_resolve_variable_force_z_data_iteration = 0;
    h_keratinocytes_resolve_variable_num_xy_bonds_data_iteration = 0;
    h_keratinocytes_resolve_variable_num_z_bonds_data_iteration = 0;
    h_keratinocytes_resolve_variable_num_stem_bonds_data_iteration = 0;
    h_keratinocytes_resolve_variable_cycle_data_iteration = 0;
    h_keratinocytes_resolve_variable_diff_noise_factor_data_iteration = 0;
    h_keratinocytes_resolve_variable_dead_ticks_data_iteration = 0;
    h_keratinocytes_resolve_variable_contact_inhibited_ticks_data_iteration = 0;
    h_keratinocytes_resolve_variable_motility_data_iteration = 0;
    h_keratinocytes_resolve_variable_dir_data_iteration = 0;
    h_keratinocytes_resolve_variable_movement_data_iteration = 0;
    

}
void h_add_agents_keratinocyte_resolve(xmachine_memory_keratinocyte** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_keratinocyte_count + count > xmachine_memory_keratinocyte_MAX){
			printf("Error: Buffer size of keratinocyte agents in state resolve will be exceeded by h_add_agents_keratinocyte_resolve\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_keratinocyte_AoS_to_SoA(h_keratinocytes_resolve, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_keratinocyte_hostToDevice(d_keratinocytes_new, h_keratinocytes_resolve, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_keratinocyte_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_keratinocyte_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_keratinocytes_resolve, d_keratinocytes_new, h_xmachine_memory_keratinocyte_resolve_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_keratinocyte_resolve_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_keratinocyte_resolve_count, &h_xmachine_memory_keratinocyte_resolve_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_keratinocytes_resolve_variable_id_data_iteration = 0;
        h_keratinocytes_resolve_variable_type_data_iteration = 0;
        h_keratinocytes_resolve_variable_x_data_iteration = 0;
        h_keratinocytes_resolve_variable_y_data_iteration = 0;
        h_keratinocytes_resolve_variable_z_data_iteration = 0;
        h_keratinocytes_resolve_variable_force_x_data_iteration = 0;
        h_keratinocytes_resolve_variable_force_y_data_iteration = 0;
        h_keratinocytes_resolve_variable_force_z_data_iteration = 0;
        h_keratinocytes_resolve_variable_num_xy_bonds_data_iteration = 0;
        h_keratinocytes_resolve_variable_num_z_bonds_data_iteration = 0;
        h_keratinocytes_resolve_variable_num_stem_bonds_data_iteration = 0;
        h_keratinocytes_resolve_variable_cycle_data_iteration = 0;
        h_keratinocytes_resolve_variable_diff_noise_factor_data_iteration = 0;
        h_keratinocytes_resolve_variable_dead_ticks_data_iteration = 0;
        h_keratinocytes_resolve_variable_contact_inhibited_ticks_data_iteration = 0;
        h_keratinocytes_resolve_variable_motility_data_iteration = 0;
        h_keratinocytes_resolve_variable_dir_data_iteration = 0;
        h_keratinocytes_resolve_variable_movement_data_iteration = 0;
        

	}
}


/*  Analytics Functions */

int reduce_keratinocyte_default_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->id),  thrust::device_pointer_cast(d_keratinocytes_default->id) + h_xmachine_memory_keratinocyte_default_count);
}

int count_keratinocyte_default_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_keratinocytes_default->id),  thrust::device_pointer_cast(d_keratinocytes_default->id) + h_xmachine_memory_keratinocyte_default_count, count_value);
}
int min_keratinocyte_default_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_keratinocyte_default_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_keratinocyte_default_type_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->type),  thrust::device_pointer_cast(d_keratinocytes_default->type) + h_xmachine_memory_keratinocyte_default_count);
}

int count_keratinocyte_default_type_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_keratinocytes_default->type),  thrust::device_pointer_cast(d_keratinocytes_default->type) + h_xmachine_memory_keratinocyte_default_count, count_value);
}
int min_keratinocyte_default_type_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->type);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_keratinocyte_default_type_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->type);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_default_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->x),  thrust::device_pointer_cast(d_keratinocytes_default->x) + h_xmachine_memory_keratinocyte_default_count);
}

float min_keratinocyte_default_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_default_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_default_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->y),  thrust::device_pointer_cast(d_keratinocytes_default->y) + h_xmachine_memory_keratinocyte_default_count);
}

float min_keratinocyte_default_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_default_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_default_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->z),  thrust::device_pointer_cast(d_keratinocytes_default->z) + h_xmachine_memory_keratinocyte_default_count);
}

float min_keratinocyte_default_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_default_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_default_force_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->force_x),  thrust::device_pointer_cast(d_keratinocytes_default->force_x) + h_xmachine_memory_keratinocyte_default_count);
}

float min_keratinocyte_default_force_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->force_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_default_force_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->force_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_default_force_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->force_y),  thrust::device_pointer_cast(d_keratinocytes_default->force_y) + h_xmachine_memory_keratinocyte_default_count);
}

float min_keratinocyte_default_force_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->force_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_default_force_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->force_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_default_force_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->force_z),  thrust::device_pointer_cast(d_keratinocytes_default->force_z) + h_xmachine_memory_keratinocyte_default_count);
}

float min_keratinocyte_default_force_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->force_z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_default_force_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->force_z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_keratinocyte_default_num_xy_bonds_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->num_xy_bonds),  thrust::device_pointer_cast(d_keratinocytes_default->num_xy_bonds) + h_xmachine_memory_keratinocyte_default_count);
}

int count_keratinocyte_default_num_xy_bonds_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_keratinocytes_default->num_xy_bonds),  thrust::device_pointer_cast(d_keratinocytes_default->num_xy_bonds) + h_xmachine_memory_keratinocyte_default_count, count_value);
}
int min_keratinocyte_default_num_xy_bonds_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->num_xy_bonds);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_keratinocyte_default_num_xy_bonds_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->num_xy_bonds);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_keratinocyte_default_num_z_bonds_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->num_z_bonds),  thrust::device_pointer_cast(d_keratinocytes_default->num_z_bonds) + h_xmachine_memory_keratinocyte_default_count);
}

int count_keratinocyte_default_num_z_bonds_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_keratinocytes_default->num_z_bonds),  thrust::device_pointer_cast(d_keratinocytes_default->num_z_bonds) + h_xmachine_memory_keratinocyte_default_count, count_value);
}
int min_keratinocyte_default_num_z_bonds_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->num_z_bonds);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_keratinocyte_default_num_z_bonds_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->num_z_bonds);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_keratinocyte_default_num_stem_bonds_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->num_stem_bonds),  thrust::device_pointer_cast(d_keratinocytes_default->num_stem_bonds) + h_xmachine_memory_keratinocyte_default_count);
}

int count_keratinocyte_default_num_stem_bonds_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_keratinocytes_default->num_stem_bonds),  thrust::device_pointer_cast(d_keratinocytes_default->num_stem_bonds) + h_xmachine_memory_keratinocyte_default_count, count_value);
}
int min_keratinocyte_default_num_stem_bonds_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->num_stem_bonds);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_keratinocyte_default_num_stem_bonds_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->num_stem_bonds);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_keratinocyte_default_cycle_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->cycle),  thrust::device_pointer_cast(d_keratinocytes_default->cycle) + h_xmachine_memory_keratinocyte_default_count);
}

int count_keratinocyte_default_cycle_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_keratinocytes_default->cycle),  thrust::device_pointer_cast(d_keratinocytes_default->cycle) + h_xmachine_memory_keratinocyte_default_count, count_value);
}
int min_keratinocyte_default_cycle_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->cycle);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_keratinocyte_default_cycle_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->cycle);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_default_diff_noise_factor_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->diff_noise_factor),  thrust::device_pointer_cast(d_keratinocytes_default->diff_noise_factor) + h_xmachine_memory_keratinocyte_default_count);
}

float min_keratinocyte_default_diff_noise_factor_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->diff_noise_factor);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_default_diff_noise_factor_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->diff_noise_factor);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_keratinocyte_default_dead_ticks_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->dead_ticks),  thrust::device_pointer_cast(d_keratinocytes_default->dead_ticks) + h_xmachine_memory_keratinocyte_default_count);
}

int count_keratinocyte_default_dead_ticks_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_keratinocytes_default->dead_ticks),  thrust::device_pointer_cast(d_keratinocytes_default->dead_ticks) + h_xmachine_memory_keratinocyte_default_count, count_value);
}
int min_keratinocyte_default_dead_ticks_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->dead_ticks);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_keratinocyte_default_dead_ticks_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->dead_ticks);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_keratinocyte_default_contact_inhibited_ticks_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->contact_inhibited_ticks),  thrust::device_pointer_cast(d_keratinocytes_default->contact_inhibited_ticks) + h_xmachine_memory_keratinocyte_default_count);
}

int count_keratinocyte_default_contact_inhibited_ticks_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_keratinocytes_default->contact_inhibited_ticks),  thrust::device_pointer_cast(d_keratinocytes_default->contact_inhibited_ticks) + h_xmachine_memory_keratinocyte_default_count, count_value);
}
int min_keratinocyte_default_contact_inhibited_ticks_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->contact_inhibited_ticks);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_keratinocyte_default_contact_inhibited_ticks_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->contact_inhibited_ticks);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_default_motility_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->motility),  thrust::device_pointer_cast(d_keratinocytes_default->motility) + h_xmachine_memory_keratinocyte_default_count);
}

float min_keratinocyte_default_motility_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->motility);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_default_motility_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->motility);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_default_dir_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->dir),  thrust::device_pointer_cast(d_keratinocytes_default->dir) + h_xmachine_memory_keratinocyte_default_count);
}

float min_keratinocyte_default_dir_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->dir);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_default_dir_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->dir);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_default_movement_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_default->movement),  thrust::device_pointer_cast(d_keratinocytes_default->movement) + h_xmachine_memory_keratinocyte_default_count);
}

float min_keratinocyte_default_movement_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->movement);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_default_movement_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_default->movement);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_keratinocyte_resolve_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->id),  thrust::device_pointer_cast(d_keratinocytes_resolve->id) + h_xmachine_memory_keratinocyte_resolve_count);
}

int count_keratinocyte_resolve_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_keratinocytes_resolve->id),  thrust::device_pointer_cast(d_keratinocytes_resolve->id) + h_xmachine_memory_keratinocyte_resolve_count, count_value);
}
int min_keratinocyte_resolve_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_keratinocyte_resolve_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_keratinocyte_resolve_type_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->type),  thrust::device_pointer_cast(d_keratinocytes_resolve->type) + h_xmachine_memory_keratinocyte_resolve_count);
}

int count_keratinocyte_resolve_type_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_keratinocytes_resolve->type),  thrust::device_pointer_cast(d_keratinocytes_resolve->type) + h_xmachine_memory_keratinocyte_resolve_count, count_value);
}
int min_keratinocyte_resolve_type_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->type);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_keratinocyte_resolve_type_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->type);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_resolve_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->x),  thrust::device_pointer_cast(d_keratinocytes_resolve->x) + h_xmachine_memory_keratinocyte_resolve_count);
}

float min_keratinocyte_resolve_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_resolve_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_resolve_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->y),  thrust::device_pointer_cast(d_keratinocytes_resolve->y) + h_xmachine_memory_keratinocyte_resolve_count);
}

float min_keratinocyte_resolve_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_resolve_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_resolve_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->z),  thrust::device_pointer_cast(d_keratinocytes_resolve->z) + h_xmachine_memory_keratinocyte_resolve_count);
}

float min_keratinocyte_resolve_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_resolve_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_resolve_force_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->force_x),  thrust::device_pointer_cast(d_keratinocytes_resolve->force_x) + h_xmachine_memory_keratinocyte_resolve_count);
}

float min_keratinocyte_resolve_force_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->force_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_resolve_force_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->force_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_resolve_force_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->force_y),  thrust::device_pointer_cast(d_keratinocytes_resolve->force_y) + h_xmachine_memory_keratinocyte_resolve_count);
}

float min_keratinocyte_resolve_force_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->force_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_resolve_force_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->force_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_resolve_force_z_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->force_z),  thrust::device_pointer_cast(d_keratinocytes_resolve->force_z) + h_xmachine_memory_keratinocyte_resolve_count);
}

float min_keratinocyte_resolve_force_z_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->force_z);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_resolve_force_z_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->force_z);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_keratinocyte_resolve_num_xy_bonds_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->num_xy_bonds),  thrust::device_pointer_cast(d_keratinocytes_resolve->num_xy_bonds) + h_xmachine_memory_keratinocyte_resolve_count);
}

int count_keratinocyte_resolve_num_xy_bonds_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_keratinocytes_resolve->num_xy_bonds),  thrust::device_pointer_cast(d_keratinocytes_resolve->num_xy_bonds) + h_xmachine_memory_keratinocyte_resolve_count, count_value);
}
int min_keratinocyte_resolve_num_xy_bonds_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->num_xy_bonds);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_keratinocyte_resolve_num_xy_bonds_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->num_xy_bonds);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_keratinocyte_resolve_num_z_bonds_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->num_z_bonds),  thrust::device_pointer_cast(d_keratinocytes_resolve->num_z_bonds) + h_xmachine_memory_keratinocyte_resolve_count);
}

int count_keratinocyte_resolve_num_z_bonds_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_keratinocytes_resolve->num_z_bonds),  thrust::device_pointer_cast(d_keratinocytes_resolve->num_z_bonds) + h_xmachine_memory_keratinocyte_resolve_count, count_value);
}
int min_keratinocyte_resolve_num_z_bonds_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->num_z_bonds);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_keratinocyte_resolve_num_z_bonds_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->num_z_bonds);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_keratinocyte_resolve_num_stem_bonds_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->num_stem_bonds),  thrust::device_pointer_cast(d_keratinocytes_resolve->num_stem_bonds) + h_xmachine_memory_keratinocyte_resolve_count);
}

int count_keratinocyte_resolve_num_stem_bonds_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_keratinocytes_resolve->num_stem_bonds),  thrust::device_pointer_cast(d_keratinocytes_resolve->num_stem_bonds) + h_xmachine_memory_keratinocyte_resolve_count, count_value);
}
int min_keratinocyte_resolve_num_stem_bonds_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->num_stem_bonds);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_keratinocyte_resolve_num_stem_bonds_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->num_stem_bonds);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_keratinocyte_resolve_cycle_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->cycle),  thrust::device_pointer_cast(d_keratinocytes_resolve->cycle) + h_xmachine_memory_keratinocyte_resolve_count);
}

int count_keratinocyte_resolve_cycle_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_keratinocytes_resolve->cycle),  thrust::device_pointer_cast(d_keratinocytes_resolve->cycle) + h_xmachine_memory_keratinocyte_resolve_count, count_value);
}
int min_keratinocyte_resolve_cycle_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->cycle);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_keratinocyte_resolve_cycle_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->cycle);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_resolve_diff_noise_factor_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->diff_noise_factor),  thrust::device_pointer_cast(d_keratinocytes_resolve->diff_noise_factor) + h_xmachine_memory_keratinocyte_resolve_count);
}

float min_keratinocyte_resolve_diff_noise_factor_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->diff_noise_factor);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_resolve_diff_noise_factor_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->diff_noise_factor);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_keratinocyte_resolve_dead_ticks_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->dead_ticks),  thrust::device_pointer_cast(d_keratinocytes_resolve->dead_ticks) + h_xmachine_memory_keratinocyte_resolve_count);
}

int count_keratinocyte_resolve_dead_ticks_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_keratinocytes_resolve->dead_ticks),  thrust::device_pointer_cast(d_keratinocytes_resolve->dead_ticks) + h_xmachine_memory_keratinocyte_resolve_count, count_value);
}
int min_keratinocyte_resolve_dead_ticks_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->dead_ticks);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_keratinocyte_resolve_dead_ticks_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->dead_ticks);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_keratinocyte_resolve_contact_inhibited_ticks_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->contact_inhibited_ticks),  thrust::device_pointer_cast(d_keratinocytes_resolve->contact_inhibited_ticks) + h_xmachine_memory_keratinocyte_resolve_count);
}

int count_keratinocyte_resolve_contact_inhibited_ticks_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_keratinocytes_resolve->contact_inhibited_ticks),  thrust::device_pointer_cast(d_keratinocytes_resolve->contact_inhibited_ticks) + h_xmachine_memory_keratinocyte_resolve_count, count_value);
}
int min_keratinocyte_resolve_contact_inhibited_ticks_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->contact_inhibited_ticks);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_keratinocyte_resolve_contact_inhibited_ticks_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->contact_inhibited_ticks);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_resolve_motility_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->motility),  thrust::device_pointer_cast(d_keratinocytes_resolve->motility) + h_xmachine_memory_keratinocyte_resolve_count);
}

float min_keratinocyte_resolve_motility_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->motility);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_resolve_motility_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->motility);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_resolve_dir_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->dir),  thrust::device_pointer_cast(d_keratinocytes_resolve->dir) + h_xmachine_memory_keratinocyte_resolve_count);
}

float min_keratinocyte_resolve_dir_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->dir);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_resolve_dir_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->dir);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_keratinocyte_resolve_movement_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_keratinocytes_resolve->movement),  thrust::device_pointer_cast(d_keratinocytes_resolve->movement) + h_xmachine_memory_keratinocyte_resolve_count);
}

float min_keratinocyte_resolve_movement_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->movement);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_keratinocyte_resolve_movement_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_keratinocytes_resolve->movement);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_keratinocyte_resolve_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}



/* Agent functions */


	
/* Shared memory size calculator for agent function */
int keratinocyte_output_location_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** keratinocyte_output_location
 * Agent function prototype for output_location function of keratinocyte agent
 */
void keratinocyte_output_location(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_keratinocyte_resolve_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_keratinocyte_resolve_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS A GLOBAL CONDITION
	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_keratinocyte_count = h_xmachine_memory_keratinocyte_resolve_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_count, &h_xmachine_memory_keratinocyte_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_keratinocyte_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_keratinocyte_scan_input<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_resolve);
	gpuErrchkLaunch();
	
	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, output_location_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	output_location_function_filter<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_resolve);
	gpuErrchkLaunch();
	
	//GET CONDTIONS TRUE COUNT FROM CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_keratinocyte, 
        temp_scan_storage_bytes_keratinocyte, 
        d_keratinocytes_resolve->_scan_input,
        d_keratinocytes_resolve->_position,
        h_xmachine_memory_keratinocyte_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_keratinocytes_resolve->_position[h_xmachine_memory_keratinocyte_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_keratinocytes_resolve->_scan_input[h_xmachine_memory_keratinocyte_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	int global_conditions_true = 0;
	if (scan_last_included == 1)
		global_conditions_true = scan_last_sum+1;
	else		
		global_conditions_true = scan_last_sum;
	//check if condition is true for all agents or if max condition count is reached
	if ((global_conditions_true != h_xmachine_memory_keratinocyte_count)&&(h_output_location_condition_count < 200))
	{
		h_output_location_condition_count ++;
		return;
	}
	if ((h_output_location_condition_count == 200))
	{
		printf("Global agent condition for output_location function reached the maximum number of 200 conditions\n");
	}
	
	//RESET THE CONDITION COUNT
	h_output_location_condition_count = 0;
	
	//MAP CURRENT STATE TO WORKING LIST
	xmachine_memory_keratinocyte_list* keratinocytes_resolve_temp = d_keratinocytes;
	d_keratinocytes = d_keratinocytes_resolve;
	d_keratinocytes_resolve = keratinocytes_resolve_temp;
	//set current state count to 0
	h_xmachine_memory_keratinocyte_resolve_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_count, &h_xmachine_memory_keratinocyte_count, sizeof(int)));	
	
	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_location_count + h_xmachine_memory_keratinocyte_count > xmachine_message_location_MAX){
		printf("Error: Buffer size of location message will be exceeded in function output_location\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_location, keratinocyte_output_location_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = keratinocyte_output_location_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_location_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_location_output_type, &h_message_location_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (output_location)
	//Reallocate   : false
	//Input        : 
	//Output       : location
	//Agent Output : 
	GPUFLAME_output_location<<<g, b, sm_size, stream>>>(d_keratinocytes, d_locations);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_location_count += h_xmachine_memory_keratinocyte_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_location_count, &h_message_location_count, sizeof(int)));	
	
	//reset partition matrix
	gpuErrchk( cudaMemset( (void*) d_location_partition_matrix, 0, sizeof(xmachine_message_location_PBM)));
    //PR Bug fix: Second fix. This should prevent future problems when multiple agents write the same message as now the message structure is completely rebuilt after an output.
    if (h_message_location_count > 0){
#ifdef FAST_ATOMIC_SORTING
      //USE ATOMICS TO BUILD PARTITION BOUNDARY
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hist_location_messages, no_sm, h_message_location_count); 
	  gridSize = (h_message_location_count + blockSize - 1) / blockSize;
	  hist_location_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_location_local_bin_index, d_xmachine_message_location_unsorted_index, d_location_partition_matrix->end_or_count, d_locations, h_message_location_count);
	  gpuErrchkLaunch();
	
      // Scan
      cub::DeviceScan::ExclusiveSum(
          d_temp_scan_storage_xmachine_message_location, 
          temp_scan_bytes_xmachine_message_location, 
          d_location_partition_matrix->end_or_count,
          d_location_partition_matrix->start,
          xmachine_message_location_grid_size, 
          stream
      );
	
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_location_messages, no_sm, h_message_location_count); 
	  gridSize = (h_message_location_count + blockSize - 1) / blockSize; 	// Round up according to array size 
	  reorder_location_messages <<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_location_local_bin_index, d_xmachine_message_location_unsorted_index, d_location_partition_matrix->start, d_locations, d_locations_swap, h_message_location_count);
	  gpuErrchkLaunch();
#else
	  //HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
	  //Get message hash values for sorting
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hash_location_messages, no_sm, h_message_location_count); 
	  gridSize = (h_message_location_count + blockSize - 1) / blockSize;
	  hash_location_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_location_keys, d_xmachine_message_location_values, d_locations);
	  gpuErrchkLaunch();
	  //Sort
	  thrust::sort_by_key(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_xmachine_message_location_keys),  thrust::device_pointer_cast(d_xmachine_message_location_keys) + h_message_location_count,  thrust::device_pointer_cast(d_xmachine_message_location_values));
	  gpuErrchkLaunch();
	  //reorder and build pcb
	  gpuErrchk(cudaMemset(d_location_partition_matrix->start, 0xffffffff, xmachine_message_location_grid_size* sizeof(int)));
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_location_messages, reorder_messages_sm_size, h_message_location_count); 
	  gridSize = (h_message_location_count + blockSize - 1) / blockSize;
	  int reorder_sm_size = reorder_messages_sm_size(blockSize);
	  reorder_location_messages<<<gridSize, blockSize, reorder_sm_size, stream>>>(d_xmachine_message_location_keys, d_xmachine_message_location_values, d_location_partition_matrix, d_locations, d_locations_swap);
	  gpuErrchkLaunch();
#endif
  }
	//swap ordered list
	xmachine_message_location_list* d_locations_temp = d_locations;
	d_locations = d_locations_swap;
	d_locations_swap = d_locations_temp;
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_keratinocyte_default_count+h_xmachine_memory_keratinocyte_count > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of output_location agents in state default will be exceeded moving working agents to next state in function output_location\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_keratinocyte_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_keratinocyte_Agents<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_default, d_keratinocytes, h_xmachine_memory_keratinocyte_default_count, h_xmachine_memory_keratinocyte_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_keratinocyte_default_count += h_xmachine_memory_keratinocyte_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int keratinocyte_cycle_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** keratinocyte_cycle
 * Agent function prototype for cycle function of keratinocyte agent
 */
void keratinocyte_cycle(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_keratinocyte_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_keratinocyte_default_count;

	
	//FOR keratinocyte AGENT OUTPUT, RESET THE AGENT NEW LIST SCAN INPUT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_keratinocyte_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_keratinocyte_scan_input<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_new);
	gpuErrchkLaunch();
	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_keratinocyte_list* keratinocytes_default_temp = d_keratinocytes;
	d_keratinocytes = d_keratinocytes_default;
	d_keratinocytes_default = keratinocytes_default_temp;
	//set working count to current state count
	h_xmachine_memory_keratinocyte_count = h_xmachine_memory_keratinocyte_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_count, &h_xmachine_memory_keratinocyte_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_keratinocyte_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_cycle, keratinocyte_cycle_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = keratinocyte_cycle_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (cycle)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : keratinocyte
	GPUFLAME_cycle<<<g, b, sm_size, stream>>>(d_keratinocytes, d_keratinocytes_new, d_rand48);
	gpuErrchkLaunch();
	
	
    //COPY ANY AGENT COUNT BEFORE keratinocyte AGENTS ARE KILLED (needed for scatter)
	int keratinocytes_pre_death_count = h_xmachine_memory_keratinocyte_count;
	
	//FOR keratinocyte AGENT OUTPUT SCATTER AGENTS 

    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_keratinocyte, 
        temp_scan_storage_bytes_keratinocyte, 
        d_keratinocytes_new->_scan_input, 
        d_keratinocytes_new->_position, 
        keratinocytes_pre_death_count,
        stream
    );

	//reset agent count
	int keratinocyte_after_birth_count;
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_keratinocytes_new->_position[keratinocytes_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_keratinocytes_new->_scan_input[keratinocytes_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		keratinocyte_after_birth_count = h_xmachine_memory_keratinocyte_default_count + scan_last_sum+1;
	else
		keratinocyte_after_birth_count = h_xmachine_memory_keratinocyte_default_count + scan_last_sum;
	//check buffer is not exceeded
	if (keratinocyte_after_birth_count > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of keratinocyte agents in state default will be exceeded writing new agents in function cycle\n");
		exit(EXIT_FAILURE);
	}
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_keratinocyte_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_keratinocyte_Agents<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_default, d_keratinocytes_new, h_xmachine_memory_keratinocyte_default_count, keratinocytes_pre_death_count);
	gpuErrchkLaunch();
	//Copy count to device
	h_xmachine_memory_keratinocyte_default_count = keratinocyte_after_birth_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_keratinocyte_default_count+h_xmachine_memory_keratinocyte_count > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of cycle agents in state default will be exceeded moving working agents to next state in function cycle\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_keratinocyte_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_keratinocyte_Agents<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_default, d_keratinocytes, h_xmachine_memory_keratinocyte_default_count, h_xmachine_memory_keratinocyte_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_keratinocyte_default_count += h_xmachine_memory_keratinocyte_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int keratinocyte_differentiate_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** keratinocyte_differentiate
 * Agent function prototype for differentiate function of keratinocyte agent
 */
void keratinocyte_differentiate(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_keratinocyte_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_keratinocyte_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_keratinocyte_list* keratinocytes_default_temp = d_keratinocytes;
	d_keratinocytes = d_keratinocytes_default;
	d_keratinocytes_default = keratinocytes_default_temp;
	//set working count to current state count
	h_xmachine_memory_keratinocyte_count = h_xmachine_memory_keratinocyte_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_count, &h_xmachine_memory_keratinocyte_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_keratinocyte_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_differentiate, keratinocyte_differentiate_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = keratinocyte_differentiate_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_location_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_id_byte_offset, tex_xmachine_message_location_id, d_locations->id, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_id_offset = (int)tex_xmachine_message_location_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_id_offset, &h_tex_xmachine_message_location_id_offset, sizeof(int)));
	size_t tex_xmachine_message_location_type_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_type_byte_offset, tex_xmachine_message_location_type, d_locations->type, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_type_offset = (int)tex_xmachine_message_location_type_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_type_offset, &h_tex_xmachine_message_location_type_offset, sizeof(int)));
	size_t tex_xmachine_message_location_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_x_byte_offset, tex_xmachine_message_location_x, d_locations->x, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_x_offset = (int)tex_xmachine_message_location_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_x_offset, &h_tex_xmachine_message_location_x_offset, sizeof(int)));
	size_t tex_xmachine_message_location_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_y_byte_offset, tex_xmachine_message_location_y, d_locations->y, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_y_offset = (int)tex_xmachine_message_location_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_y_offset, &h_tex_xmachine_message_location_y_offset, sizeof(int)));
	size_t tex_xmachine_message_location_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_z_byte_offset, tex_xmachine_message_location_z, d_locations->z, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_z_offset = (int)tex_xmachine_message_location_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_z_offset, &h_tex_xmachine_message_location_z_offset, sizeof(int)));
	size_t tex_xmachine_message_location_dir_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_dir_byte_offset, tex_xmachine_message_location_dir, d_locations->dir, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_dir_offset = (int)tex_xmachine_message_location_dir_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_dir_offset, &h_tex_xmachine_message_location_dir_offset, sizeof(int)));
	size_t tex_xmachine_message_location_motility_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_motility_byte_offset, tex_xmachine_message_location_motility, d_locations->motility, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_motility_offset = (int)tex_xmachine_message_location_motility_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_motility_offset, &h_tex_xmachine_message_location_motility_offset, sizeof(int)));
	size_t tex_xmachine_message_location_range_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_range_byte_offset, tex_xmachine_message_location_range, d_locations->range, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_range_offset = (int)tex_xmachine_message_location_range_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_range_offset, &h_tex_xmachine_message_location_range_offset, sizeof(int)));
	size_t tex_xmachine_message_location_iteration_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_iteration_byte_offset, tex_xmachine_message_location_iteration, d_locations->iteration, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_iteration_offset = (int)tex_xmachine_message_location_iteration_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_iteration_offset, &h_tex_xmachine_message_location_iteration_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_location_pbm_start_byte_offset;
	size_t tex_xmachine_message_location_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_pbm_start_byte_offset, tex_xmachine_message_location_pbm_start, d_location_partition_matrix->start, sizeof(int)*xmachine_message_location_grid_size));
	h_tex_xmachine_message_location_pbm_start_offset = (int)tex_xmachine_message_location_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_pbm_start_offset, &h_tex_xmachine_message_location_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_pbm_end_or_count_byte_offset, tex_xmachine_message_location_pbm_end_or_count, d_location_partition_matrix->end_or_count, sizeof(int)*xmachine_message_location_grid_size));
  h_tex_xmachine_message_location_pbm_end_or_count_offset = (int)tex_xmachine_message_location_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_pbm_end_or_count_offset, &h_tex_xmachine_message_location_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (differentiate)
	//Reallocate   : false
	//Input        : location
	//Output       : 
	//Agent Output : 
	GPUFLAME_differentiate<<<g, b, sm_size, stream>>>(d_keratinocytes, d_locations, d_location_partition_matrix);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_type));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_dir));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_motility));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_range));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_iteration));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_keratinocyte_default_count+h_xmachine_memory_keratinocyte_count > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of differentiate agents in state default will be exceeded moving working agents to next state in function differentiate\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  keratinocytes_default_temp = d_keratinocytes;
  d_keratinocytes = d_keratinocytes_default;
  d_keratinocytes_default = keratinocytes_default_temp;
        
	//update new state agent size
	h_xmachine_memory_keratinocyte_default_count += h_xmachine_memory_keratinocyte_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int keratinocyte_death_signal_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** keratinocyte_death_signal
 * Agent function prototype for death_signal function of keratinocyte agent
 */
void keratinocyte_death_signal(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_keratinocyte_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_keratinocyte_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_keratinocyte_list* keratinocytes_default_temp = d_keratinocytes;
	d_keratinocytes = d_keratinocytes_default;
	d_keratinocytes_default = keratinocytes_default_temp;
	//set working count to current state count
	h_xmachine_memory_keratinocyte_count = h_xmachine_memory_keratinocyte_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_count, &h_xmachine_memory_keratinocyte_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_keratinocyte_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_death_signal, keratinocyte_death_signal_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = keratinocyte_death_signal_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_location_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_id_byte_offset, tex_xmachine_message_location_id, d_locations->id, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_id_offset = (int)tex_xmachine_message_location_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_id_offset, &h_tex_xmachine_message_location_id_offset, sizeof(int)));
	size_t tex_xmachine_message_location_type_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_type_byte_offset, tex_xmachine_message_location_type, d_locations->type, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_type_offset = (int)tex_xmachine_message_location_type_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_type_offset, &h_tex_xmachine_message_location_type_offset, sizeof(int)));
	size_t tex_xmachine_message_location_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_x_byte_offset, tex_xmachine_message_location_x, d_locations->x, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_x_offset = (int)tex_xmachine_message_location_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_x_offset, &h_tex_xmachine_message_location_x_offset, sizeof(int)));
	size_t tex_xmachine_message_location_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_y_byte_offset, tex_xmachine_message_location_y, d_locations->y, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_y_offset = (int)tex_xmachine_message_location_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_y_offset, &h_tex_xmachine_message_location_y_offset, sizeof(int)));
	size_t tex_xmachine_message_location_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_z_byte_offset, tex_xmachine_message_location_z, d_locations->z, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_z_offset = (int)tex_xmachine_message_location_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_z_offset, &h_tex_xmachine_message_location_z_offset, sizeof(int)));
	size_t tex_xmachine_message_location_dir_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_dir_byte_offset, tex_xmachine_message_location_dir, d_locations->dir, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_dir_offset = (int)tex_xmachine_message_location_dir_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_dir_offset, &h_tex_xmachine_message_location_dir_offset, sizeof(int)));
	size_t tex_xmachine_message_location_motility_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_motility_byte_offset, tex_xmachine_message_location_motility, d_locations->motility, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_motility_offset = (int)tex_xmachine_message_location_motility_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_motility_offset, &h_tex_xmachine_message_location_motility_offset, sizeof(int)));
	size_t tex_xmachine_message_location_range_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_range_byte_offset, tex_xmachine_message_location_range, d_locations->range, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_range_offset = (int)tex_xmachine_message_location_range_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_range_offset, &h_tex_xmachine_message_location_range_offset, sizeof(int)));
	size_t tex_xmachine_message_location_iteration_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_iteration_byte_offset, tex_xmachine_message_location_iteration, d_locations->iteration, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_iteration_offset = (int)tex_xmachine_message_location_iteration_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_iteration_offset, &h_tex_xmachine_message_location_iteration_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_location_pbm_start_byte_offset;
	size_t tex_xmachine_message_location_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_pbm_start_byte_offset, tex_xmachine_message_location_pbm_start, d_location_partition_matrix->start, sizeof(int)*xmachine_message_location_grid_size));
	h_tex_xmachine_message_location_pbm_start_offset = (int)tex_xmachine_message_location_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_pbm_start_offset, &h_tex_xmachine_message_location_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_pbm_end_or_count_byte_offset, tex_xmachine_message_location_pbm_end_or_count, d_location_partition_matrix->end_or_count, sizeof(int)*xmachine_message_location_grid_size));
  h_tex_xmachine_message_location_pbm_end_or_count_offset = (int)tex_xmachine_message_location_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_pbm_end_or_count_offset, &h_tex_xmachine_message_location_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (death_signal)
	//Reallocate   : false
	//Input        : location
	//Output       : 
	//Agent Output : 
	GPUFLAME_death_signal<<<g, b, sm_size, stream>>>(d_keratinocytes, d_locations, d_location_partition_matrix, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_type));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_dir));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_motility));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_range));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_iteration));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_keratinocyte_default_count+h_xmachine_memory_keratinocyte_count > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of death_signal agents in state default will be exceeded moving working agents to next state in function death_signal\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  keratinocytes_default_temp = d_keratinocytes;
  d_keratinocytes = d_keratinocytes_default;
  d_keratinocytes_default = keratinocytes_default_temp;
        
	//update new state agent size
	h_xmachine_memory_keratinocyte_default_count += h_xmachine_memory_keratinocyte_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int keratinocyte_migrate_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** keratinocyte_migrate
 * Agent function prototype for migrate function of keratinocyte agent
 */
void keratinocyte_migrate(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_keratinocyte_default_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_keratinocyte_default_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_keratinocyte_list* keratinocytes_default_temp = d_keratinocytes;
	d_keratinocytes = d_keratinocytes_default;
	d_keratinocytes_default = keratinocytes_default_temp;
	//set working count to current state count
	h_xmachine_memory_keratinocyte_count = h_xmachine_memory_keratinocyte_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_count, &h_xmachine_memory_keratinocyte_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_keratinocyte_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_default_count, &h_xmachine_memory_keratinocyte_default_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_migrate, keratinocyte_migrate_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = keratinocyte_migrate_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_location_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_id_byte_offset, tex_xmachine_message_location_id, d_locations->id, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_id_offset = (int)tex_xmachine_message_location_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_id_offset, &h_tex_xmachine_message_location_id_offset, sizeof(int)));
	size_t tex_xmachine_message_location_type_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_type_byte_offset, tex_xmachine_message_location_type, d_locations->type, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_type_offset = (int)tex_xmachine_message_location_type_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_type_offset, &h_tex_xmachine_message_location_type_offset, sizeof(int)));
	size_t tex_xmachine_message_location_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_x_byte_offset, tex_xmachine_message_location_x, d_locations->x, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_x_offset = (int)tex_xmachine_message_location_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_x_offset, &h_tex_xmachine_message_location_x_offset, sizeof(int)));
	size_t tex_xmachine_message_location_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_y_byte_offset, tex_xmachine_message_location_y, d_locations->y, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_y_offset = (int)tex_xmachine_message_location_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_y_offset, &h_tex_xmachine_message_location_y_offset, sizeof(int)));
	size_t tex_xmachine_message_location_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_z_byte_offset, tex_xmachine_message_location_z, d_locations->z, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_z_offset = (int)tex_xmachine_message_location_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_z_offset, &h_tex_xmachine_message_location_z_offset, sizeof(int)));
	size_t tex_xmachine_message_location_dir_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_dir_byte_offset, tex_xmachine_message_location_dir, d_locations->dir, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_dir_offset = (int)tex_xmachine_message_location_dir_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_dir_offset, &h_tex_xmachine_message_location_dir_offset, sizeof(int)));
	size_t tex_xmachine_message_location_motility_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_motility_byte_offset, tex_xmachine_message_location_motility, d_locations->motility, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_motility_offset = (int)tex_xmachine_message_location_motility_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_motility_offset, &h_tex_xmachine_message_location_motility_offset, sizeof(int)));
	size_t tex_xmachine_message_location_range_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_range_byte_offset, tex_xmachine_message_location_range, d_locations->range, sizeof(float)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_range_offset = (int)tex_xmachine_message_location_range_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_range_offset, &h_tex_xmachine_message_location_range_offset, sizeof(int)));
	size_t tex_xmachine_message_location_iteration_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_iteration_byte_offset, tex_xmachine_message_location_iteration, d_locations->iteration, sizeof(int)*xmachine_message_location_MAX));
	h_tex_xmachine_message_location_iteration_offset = (int)tex_xmachine_message_location_iteration_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_iteration_offset, &h_tex_xmachine_message_location_iteration_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_location_pbm_start_byte_offset;
	size_t tex_xmachine_message_location_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_pbm_start_byte_offset, tex_xmachine_message_location_pbm_start, d_location_partition_matrix->start, sizeof(int)*xmachine_message_location_grid_size));
	h_tex_xmachine_message_location_pbm_start_offset = (int)tex_xmachine_message_location_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_pbm_start_offset, &h_tex_xmachine_message_location_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_location_pbm_end_or_count_byte_offset, tex_xmachine_message_location_pbm_end_or_count, d_location_partition_matrix->end_or_count, sizeof(int)*xmachine_message_location_grid_size));
  h_tex_xmachine_message_location_pbm_end_or_count_offset = (int)tex_xmachine_message_location_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_location_pbm_end_or_count_offset, &h_tex_xmachine_message_location_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (migrate)
	//Reallocate   : false
	//Input        : location
	//Output       : 
	//Agent Output : 
	GPUFLAME_migrate<<<g, b, sm_size, stream>>>(d_keratinocytes, d_locations, d_location_partition_matrix, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_id));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_type));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_dir));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_motility));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_range));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_iteration));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_location_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_keratinocyte_resolve_count+h_xmachine_memory_keratinocyte_count > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of migrate agents in state resolve will be exceeded moving working agents to next state in function migrate\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_keratinocyte_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_keratinocyte_Agents<<<gridSize, blockSize, 0, stream>>>(d_keratinocytes_resolve, d_keratinocytes, h_xmachine_memory_keratinocyte_resolve_count, h_xmachine_memory_keratinocyte_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_keratinocyte_resolve_count += h_xmachine_memory_keratinocyte_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_resolve_count, &h_xmachine_memory_keratinocyte_resolve_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int keratinocyte_force_resolution_output_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** keratinocyte_force_resolution_output
 * Agent function prototype for force_resolution_output function of keratinocyte agent
 */
void keratinocyte_force_resolution_output(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_keratinocyte_resolve_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_keratinocyte_resolve_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_keratinocyte_list* keratinocytes_resolve_temp = d_keratinocytes;
	d_keratinocytes = d_keratinocytes_resolve;
	d_keratinocytes_resolve = keratinocytes_resolve_temp;
	//set working count to current state count
	h_xmachine_memory_keratinocyte_count = h_xmachine_memory_keratinocyte_resolve_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_count, &h_xmachine_memory_keratinocyte_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_keratinocyte_resolve_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_resolve_count, &h_xmachine_memory_keratinocyte_resolve_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_force_count + h_xmachine_memory_keratinocyte_count > xmachine_message_force_MAX){
		printf("Error: Buffer size of force message will be exceeded in function force_resolution_output\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_force_resolution_output, keratinocyte_force_resolution_output_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = keratinocyte_force_resolution_output_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_force_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_force_output_type, &h_message_force_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (force_resolution_output)
	//Reallocate   : false
	//Input        : 
	//Output       : force
	//Agent Output : 
	GPUFLAME_force_resolution_output<<<g, b, sm_size, stream>>>(d_keratinocytes, d_forces);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_force_count += h_xmachine_memory_keratinocyte_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_force_count, &h_message_force_count, sizeof(int)));	
	
	//reset partition matrix
	gpuErrchk( cudaMemset( (void*) d_force_partition_matrix, 0, sizeof(xmachine_message_force_PBM)));
    //PR Bug fix: Second fix. This should prevent future problems when multiple agents write the same message as now the message structure is completely rebuilt after an output.
    if (h_message_force_count > 0){
#ifdef FAST_ATOMIC_SORTING
      //USE ATOMICS TO BUILD PARTITION BOUNDARY
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hist_force_messages, no_sm, h_message_force_count); 
	  gridSize = (h_message_force_count + blockSize - 1) / blockSize;
	  hist_force_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_force_local_bin_index, d_xmachine_message_force_unsorted_index, d_force_partition_matrix->end_or_count, d_forces, h_message_force_count);
	  gpuErrchkLaunch();
	
      // Scan
      cub::DeviceScan::ExclusiveSum(
          d_temp_scan_storage_xmachine_message_force, 
          temp_scan_bytes_xmachine_message_force, 
          d_force_partition_matrix->end_or_count,
          d_force_partition_matrix->start,
          xmachine_message_force_grid_size, 
          stream
      );
	
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_force_messages, no_sm, h_message_force_count); 
	  gridSize = (h_message_force_count + blockSize - 1) / blockSize; 	// Round up according to array size 
	  reorder_force_messages <<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_force_local_bin_index, d_xmachine_message_force_unsorted_index, d_force_partition_matrix->start, d_forces, d_forces_swap, h_message_force_count);
	  gpuErrchkLaunch();
#else
	  //HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
	  //Get message hash values for sorting
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hash_force_messages, no_sm, h_message_force_count); 
	  gridSize = (h_message_force_count + blockSize - 1) / blockSize;
	  hash_force_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_force_keys, d_xmachine_message_force_values, d_forces);
	  gpuErrchkLaunch();
	  //Sort
	  thrust::sort_by_key(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_xmachine_message_force_keys),  thrust::device_pointer_cast(d_xmachine_message_force_keys) + h_message_force_count,  thrust::device_pointer_cast(d_xmachine_message_force_values));
	  gpuErrchkLaunch();
	  //reorder and build pcb
	  gpuErrchk(cudaMemset(d_force_partition_matrix->start, 0xffffffff, xmachine_message_force_grid_size* sizeof(int)));
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_force_messages, reorder_messages_sm_size, h_message_force_count); 
	  gridSize = (h_message_force_count + blockSize - 1) / blockSize;
	  int reorder_sm_size = reorder_messages_sm_size(blockSize);
	  reorder_force_messages<<<gridSize, blockSize, reorder_sm_size, stream>>>(d_xmachine_message_force_keys, d_xmachine_message_force_values, d_force_partition_matrix, d_forces, d_forces_swap);
	  gpuErrchkLaunch();
#endif
  }
	//swap ordered list
	xmachine_message_force_list* d_forces_temp = d_forces;
	d_forces = d_forces_swap;
	d_forces_swap = d_forces_temp;
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_keratinocyte_resolve_count+h_xmachine_memory_keratinocyte_count > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of force_resolution_output agents in state resolve will be exceeded moving working agents to next state in function force_resolution_output\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  keratinocytes_resolve_temp = d_keratinocytes;
  d_keratinocytes = d_keratinocytes_resolve;
  d_keratinocytes_resolve = keratinocytes_resolve_temp;
        
	//update new state agent size
	h_xmachine_memory_keratinocyte_resolve_count += h_xmachine_memory_keratinocyte_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_resolve_count, &h_xmachine_memory_keratinocyte_resolve_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int keratinocyte_resolve_forces_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_force));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** keratinocyte_resolve_forces
 * Agent function prototype for resolve_forces function of keratinocyte agent
 */
void keratinocyte_resolve_forces(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_keratinocyte_resolve_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_keratinocyte_resolve_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_keratinocyte_list* keratinocytes_resolve_temp = d_keratinocytes;
	d_keratinocytes = d_keratinocytes_resolve;
	d_keratinocytes_resolve = keratinocytes_resolve_temp;
	//set working count to current state count
	h_xmachine_memory_keratinocyte_count = h_xmachine_memory_keratinocyte_resolve_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_count, &h_xmachine_memory_keratinocyte_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_keratinocyte_resolve_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_resolve_count, &h_xmachine_memory_keratinocyte_resolve_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_resolve_forces, keratinocyte_resolve_forces_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = keratinocyte_resolve_forces_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_force_type_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_force_type_byte_offset, tex_xmachine_message_force_type, d_forces->type, sizeof(int)*xmachine_message_force_MAX));
	h_tex_xmachine_message_force_type_offset = (int)tex_xmachine_message_force_type_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_force_type_offset, &h_tex_xmachine_message_force_type_offset, sizeof(int)));
	size_t tex_xmachine_message_force_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_force_x_byte_offset, tex_xmachine_message_force_x, d_forces->x, sizeof(float)*xmachine_message_force_MAX));
	h_tex_xmachine_message_force_x_offset = (int)tex_xmachine_message_force_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_force_x_offset, &h_tex_xmachine_message_force_x_offset, sizeof(int)));
	size_t tex_xmachine_message_force_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_force_y_byte_offset, tex_xmachine_message_force_y, d_forces->y, sizeof(float)*xmachine_message_force_MAX));
	h_tex_xmachine_message_force_y_offset = (int)tex_xmachine_message_force_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_force_y_offset, &h_tex_xmachine_message_force_y_offset, sizeof(int)));
	size_t tex_xmachine_message_force_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_force_z_byte_offset, tex_xmachine_message_force_z, d_forces->z, sizeof(float)*xmachine_message_force_MAX));
	h_tex_xmachine_message_force_z_offset = (int)tex_xmachine_message_force_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_force_z_offset, &h_tex_xmachine_message_force_z_offset, sizeof(int)));
	size_t tex_xmachine_message_force_id_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_force_id_byte_offset, tex_xmachine_message_force_id, d_forces->id, sizeof(int)*xmachine_message_force_MAX));
	h_tex_xmachine_message_force_id_offset = (int)tex_xmachine_message_force_id_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_force_id_offset, &h_tex_xmachine_message_force_id_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_force_pbm_start_byte_offset;
	size_t tex_xmachine_message_force_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_force_pbm_start_byte_offset, tex_xmachine_message_force_pbm_start, d_force_partition_matrix->start, sizeof(int)*xmachine_message_force_grid_size));
	h_tex_xmachine_message_force_pbm_start_offset = (int)tex_xmachine_message_force_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_force_pbm_start_offset, &h_tex_xmachine_message_force_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_force_pbm_end_or_count_byte_offset, tex_xmachine_message_force_pbm_end_or_count, d_force_partition_matrix->end_or_count, sizeof(int)*xmachine_message_force_grid_size));
  h_tex_xmachine_message_force_pbm_end_or_count_offset = (int)tex_xmachine_message_force_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_force_pbm_end_or_count_offset, &h_tex_xmachine_message_force_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (resolve_forces)
	//Reallocate   : false
	//Input        : force
	//Output       : 
	//Agent Output : 
	GPUFLAME_resolve_forces<<<g, b, sm_size, stream>>>(d_keratinocytes, d_forces, d_force_partition_matrix);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_force_type));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_force_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_force_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_force_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_force_id));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_force_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_force_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_keratinocyte_resolve_count+h_xmachine_memory_keratinocyte_count > xmachine_memory_keratinocyte_MAX){
		printf("Error: Buffer size of resolve_forces agents in state resolve will be exceeded moving working agents to next state in function resolve_forces\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  keratinocytes_resolve_temp = d_keratinocytes;
  d_keratinocytes = d_keratinocytes_resolve;
  d_keratinocytes_resolve = keratinocytes_resolve_temp;
        
	//update new state agent size
	h_xmachine_memory_keratinocyte_resolve_count += h_xmachine_memory_keratinocyte_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_keratinocyte_resolve_count, &h_xmachine_memory_keratinocyte_resolve_count, sizeof(int)));	
	
	
}


 
extern void reset_keratinocyte_default_count()
{
    h_xmachine_memory_keratinocyte_default_count = 0;
}
 
extern void reset_keratinocyte_resolve_count()
{
    h_xmachine_memory_keratinocyte_resolve_count = 0;
}
