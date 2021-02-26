
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
uint * d_xmachine_memory_agent_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_agent_values;  /**< Agent sort identifiers value */

/* agent state variables */
xmachine_memory_agent_list* h_agents_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_agent_list* d_agents_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_agent_default_count;   /**< Agent population size counter */ 

/* medic Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_medic_list* d_medics;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_medic_list* d_medics_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_medic_list* d_medics_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_medic_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_medic_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_medic_values;  /**< Agent sort identifiers value */

/* medic state variables */
xmachine_memory_medic_list* h_medics_default2;      /**< Pointer to agent list (population) on host*/
xmachine_memory_medic_list* d_medics_default2;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_medic_default2_count;   /**< Agent population size counter */ 

/* receptionist Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_receptionist_list* d_receptionists;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_receptionist_list* d_receptionists_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_receptionist_list* d_receptionists_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_receptionist_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_receptionist_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_receptionist_values;  /**< Agent sort identifiers value */

/* receptionist state variables */
xmachine_memory_receptionist_list* h_receptionists_defaultReceptionist;      /**< Pointer to agent list (population) on host*/
xmachine_memory_receptionist_list* d_receptionists_defaultReceptionist;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_receptionist_defaultReceptionist_count;   /**< Agent population size counter */ 

/* chair_admin Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_chair_admin_list* d_chair_admins;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_chair_admin_list* d_chair_admins_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_chair_admin_list* d_chair_admins_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_chair_admin_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_chair_admin_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_chair_admin_values;  /**< Agent sort identifiers value */

/* chair_admin state variables */
xmachine_memory_chair_admin_list* h_chair_admins_defaultAdmin;      /**< Pointer to agent list (population) on host*/
xmachine_memory_chair_admin_list* d_chair_admins_defaultAdmin;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_chair_admin_defaultAdmin_count;   /**< Agent population size counter */ 

/* navmap Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_navmap_list* d_navmaps;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_navmap_list* d_navmaps_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_navmap_list* d_navmaps_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_navmap_count;   /**< Agent population size counter */ 
int h_xmachine_memory_navmap_pop_width;   /**< Agent population width */
uint * d_xmachine_memory_navmap_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_navmap_values;  /**< Agent sort identifiers value */

/* navmap state variables */
xmachine_memory_navmap_list* h_navmaps_static;      /**< Pointer to agent list (population) on host*/
xmachine_memory_navmap_list* d_navmaps_static;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_navmap_static_count;   /**< Agent population size counter */ 


/* Variables to track the state of host copies of state lists, for the purposes of host agent data access.
 * @future - if the host data is current it may be possible to avoid duplicating memcpy in xml output.
 */
unsigned int h_agents_default_variable_id_data_iteration;
unsigned int h_agents_default_variable_x_data_iteration;
unsigned int h_agents_default_variable_y_data_iteration;
unsigned int h_agents_default_variable_velx_data_iteration;
unsigned int h_agents_default_variable_vely_data_iteration;
unsigned int h_agents_default_variable_steer_x_data_iteration;
unsigned int h_agents_default_variable_steer_y_data_iteration;
unsigned int h_agents_default_variable_height_data_iteration;
unsigned int h_agents_default_variable_exit_no_data_iteration;
unsigned int h_agents_default_variable_speed_data_iteration;
unsigned int h_agents_default_variable_lod_data_iteration;
unsigned int h_agents_default_variable_animate_data_iteration;
unsigned int h_agents_default_variable_animate_dir_data_iteration;
unsigned int h_agents_default_variable_estado_data_iteration;
unsigned int h_agents_default_variable_tick_data_iteration;
unsigned int h_agents_default_variable_estado_movimiento_data_iteration;
unsigned int h_agents_default_variable_go_to_x_data_iteration;
unsigned int h_agents_default_variable_go_to_y_data_iteration;
unsigned int h_medics_default2_variable_x_data_iteration;
unsigned int h_medics_default2_variable_y_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_x_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_y_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_colaPacientes_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_front_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_rear_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_size_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_tick_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_current_patient_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_attend_patient_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_estado_data_iteration;
unsigned int h_chair_admins_defaultAdmin_variable_id_data_iteration;
unsigned int h_chair_admins_defaultAdmin_variable_chairArray_data_iteration;
unsigned int h_navmaps_static_variable_x_data_iteration;
unsigned int h_navmaps_static_variable_y_data_iteration;
unsigned int h_navmaps_static_variable_exit_no_data_iteration;
unsigned int h_navmaps_static_variable_height_data_iteration;
unsigned int h_navmaps_static_variable_collision_x_data_iteration;
unsigned int h_navmaps_static_variable_collision_y_data_iteration;
unsigned int h_navmaps_static_variable_exit0_x_data_iteration;
unsigned int h_navmaps_static_variable_exit0_y_data_iteration;
unsigned int h_navmaps_static_variable_exit1_x_data_iteration;
unsigned int h_navmaps_static_variable_exit1_y_data_iteration;
unsigned int h_navmaps_static_variable_exit2_x_data_iteration;
unsigned int h_navmaps_static_variable_exit2_y_data_iteration;
unsigned int h_navmaps_static_variable_exit3_x_data_iteration;
unsigned int h_navmaps_static_variable_exit3_y_data_iteration;
unsigned int h_navmaps_static_variable_exit4_x_data_iteration;
unsigned int h_navmaps_static_variable_exit4_y_data_iteration;
unsigned int h_navmaps_static_variable_exit5_x_data_iteration;
unsigned int h_navmaps_static_variable_exit5_y_data_iteration;
unsigned int h_navmaps_static_variable_exit6_x_data_iteration;
unsigned int h_navmaps_static_variable_exit6_y_data_iteration;
unsigned int h_navmaps_static_variable_cant_generados_data_iteration;


/* Message Memory */

/* pedestrian_location Message variables */
xmachine_message_pedestrian_location_list* h_pedestrian_locations;         /**< Pointer to message list on host*/
xmachine_message_pedestrian_location_list* d_pedestrian_locations;         /**< Pointer to message list on device*/
xmachine_message_pedestrian_location_list* d_pedestrian_locations_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_pedestrian_location_count;         /**< message list counter*/
int h_message_pedestrian_location_output_type;   /**< message output type (single or optional)*/
/* Spatial Partitioning Variables*/
#ifdef FAST_ATOMIC_SORTING
	uint * d_xmachine_message_pedestrian_location_local_bin_index;	  /**< index offset within the assigned bin */
	uint * d_xmachine_message_pedestrian_location_unsorted_index;		/**< unsorted index (hash) value for message */
    // Values for CUB exclusive scan of spatially partitioned variables
    void * d_temp_scan_storage_xmachine_message_pedestrian_location;
    size_t temp_scan_bytes_xmachine_message_pedestrian_location;
#else
	uint * d_xmachine_message_pedestrian_location_keys;	  /**< message sort identifier keys*/
	uint * d_xmachine_message_pedestrian_location_values;  /**< message sort identifier values */
#endif
xmachine_message_pedestrian_location_PBM * d_pedestrian_location_partition_matrix;  /**< Pointer to PCB matrix */
glm::vec3 h_message_pedestrian_location_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
glm::vec3 h_message_pedestrian_location_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
glm::ivec3 h_message_pedestrian_location_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
float h_message_pedestrian_location_radius;                 /**< partition radius (used to determin the size of the partitions) */
/* Texture offset values for host */
int h_tex_xmachine_message_pedestrian_location_x_offset;
int h_tex_xmachine_message_pedestrian_location_y_offset;
int h_tex_xmachine_message_pedestrian_location_z_offset;
int h_tex_xmachine_message_pedestrian_location_estado_offset;
int h_tex_xmachine_message_pedestrian_location_pbm_start_offset;
int h_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset;

/* pedestrian_state Message variables */
xmachine_message_pedestrian_state_list* h_pedestrian_states;         /**< Pointer to message list on host*/
xmachine_message_pedestrian_state_list* d_pedestrian_states;         /**< Pointer to message list on device*/
xmachine_message_pedestrian_state_list* d_pedestrian_states_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_pedestrian_state_count;         /**< message list counter*/
int h_message_pedestrian_state_output_type;   /**< message output type (single or optional)*/
/* Spatial Partitioning Variables*/
#ifdef FAST_ATOMIC_SORTING
	uint * d_xmachine_message_pedestrian_state_local_bin_index;	  /**< index offset within the assigned bin */
	uint * d_xmachine_message_pedestrian_state_unsorted_index;		/**< unsorted index (hash) value for message */
    // Values for CUB exclusive scan of spatially partitioned variables
    void * d_temp_scan_storage_xmachine_message_pedestrian_state;
    size_t temp_scan_bytes_xmachine_message_pedestrian_state;
#else
	uint * d_xmachine_message_pedestrian_state_keys;	  /**< message sort identifier keys*/
	uint * d_xmachine_message_pedestrian_state_values;  /**< message sort identifier values */
#endif
xmachine_message_pedestrian_state_PBM * d_pedestrian_state_partition_matrix;  /**< Pointer to PCB matrix */
glm::vec3 h_message_pedestrian_state_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
glm::vec3 h_message_pedestrian_state_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
glm::ivec3 h_message_pedestrian_state_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
float h_message_pedestrian_state_radius;                 /**< partition radius (used to determin the size of the partitions) */
/* Texture offset values for host */
int h_tex_xmachine_message_pedestrian_state_x_offset;
int h_tex_xmachine_message_pedestrian_state_y_offset;
int h_tex_xmachine_message_pedestrian_state_z_offset;
int h_tex_xmachine_message_pedestrian_state_estado_offset;
int h_tex_xmachine_message_pedestrian_state_pbm_start_offset;
int h_tex_xmachine_message_pedestrian_state_pbm_end_or_count_offset;

/* navmap_cell Message variables */
xmachine_message_navmap_cell_list* h_navmap_cells;         /**< Pointer to message list on host*/
xmachine_message_navmap_cell_list* d_navmap_cells;         /**< Pointer to message list on device*/
xmachine_message_navmap_cell_list* d_navmap_cells_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Discrete Partitioning Variables*/
int h_message_navmap_cell_range;     /**< range of the discrete message*/
int h_message_navmap_cell_width;     /**< with of the message grid*/
/* Texture offset values for host */
int h_tex_xmachine_message_navmap_cell_x_offset;
int h_tex_xmachine_message_navmap_cell_y_offset;
int h_tex_xmachine_message_navmap_cell_exit_no_offset;
int h_tex_xmachine_message_navmap_cell_height_offset;
int h_tex_xmachine_message_navmap_cell_collision_x_offset;
int h_tex_xmachine_message_navmap_cell_collision_y_offset;
/* check_in Message variables */
xmachine_message_check_in_list* h_check_ins;         /**< Pointer to message list on host*/
xmachine_message_check_in_list* d_check_ins;         /**< Pointer to message list on device*/
xmachine_message_check_in_list* d_check_ins_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_check_in_count;         /**< message list counter*/
int h_message_check_in_output_type;   /**< message output type (single or optional)*/

/* check_in_done Message variables */
xmachine_message_check_in_done_list* h_check_in_dones;         /**< Pointer to message list on host*/
xmachine_message_check_in_done_list* d_check_in_dones;         /**< Pointer to message list on device*/
xmachine_message_check_in_done_list* d_check_in_dones_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_check_in_done_count;         /**< message list counter*/
int h_message_check_in_done_output_type;   /**< message output type (single or optional)*/

/* chair_petition Message variables */
xmachine_message_chair_petition_list* h_chair_petitions;         /**< Pointer to message list on host*/
xmachine_message_chair_petition_list* d_chair_petitions;         /**< Pointer to message list on device*/
xmachine_message_chair_petition_list* d_chair_petitions_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_chair_petition_count;         /**< message list counter*/
int h_message_chair_petition_output_type;   /**< message output type (single or optional)*/

/* chair_response Message variables */
xmachine_message_chair_response_list* h_chair_responses;         /**< Pointer to message list on host*/
xmachine_message_chair_response_list* d_chair_responses;         /**< Pointer to message list on device*/
xmachine_message_chair_response_list* d_chair_responses_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_chair_response_count;         /**< message list counter*/
int h_message_chair_response_output_type;   /**< message output type (single or optional)*/

  
/* CUDA Streams for function layers */
cudaStream_t stream1;
cudaStream_t stream2;
cudaStream_t stream3;
cudaStream_t stream4;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_agent;
size_t temp_scan_storage_bytes_agent;

void * d_temp_scan_storage_medic;
size_t temp_scan_storage_bytes_medic;

void * d_temp_scan_storage_receptionist;
size_t temp_scan_storage_bytes_receptionist;

void * d_temp_scan_storage_chair_admin;
size_t temp_scan_storage_bytes_chair_admin;

void * d_temp_scan_storage_navmap;
size_t temp_scan_storage_bytes_navmap;


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

/** agent_output_pedestrian_location
 * Agent function prototype for output_pedestrian_location function of agent agent
 */
void agent_output_pedestrian_location(cudaStream_t &stream);

/** agent_avoid_pedestrians
 * Agent function prototype for avoid_pedestrians function of agent agent
 */
void agent_avoid_pedestrians(cudaStream_t &stream);

/** agent_output_pedestrian_state
 * Agent function prototype for output_pedestrian_state function of agent agent
 */
void agent_output_pedestrian_state(cudaStream_t &stream);

/** agent_infect_pedestrians
 * Agent function prototype for infect_pedestrians function of agent agent
 */
void agent_infect_pedestrians(cudaStream_t &stream);

/** agent_move
 * Agent function prototype for move function of agent agent
 */
void agent_move(cudaStream_t &stream);

/** agent_output_chair_petition
 * Agent function prototype for output_chair_petition function of agent agent
 */
void agent_output_chair_petition(cudaStream_t &stream);

/** agent_check_check_in_done
 * Agent function prototype for check_check_in_done function of agent agent
 */
void agent_check_check_in_done(cudaStream_t &stream);

/** agent_check_chair_response
 * Agent function prototype for check_chair_response function of agent agent
 */
void agent_check_chair_response(cudaStream_t &stream);

/** medic_prueba
 * Agent function prototype for prueba function of medic agent
 */
void medic_prueba(cudaStream_t &stream);

/** receptionist_receptionServer
 * Agent function prototype for receptionServer function of receptionist agent
 */
void receptionist_receptionServer(cudaStream_t &stream);

/** receptionist_infect_receptionist
 * Agent function prototype for infect_receptionist function of receptionist agent
 */
void receptionist_infect_receptionist(cudaStream_t &stream);

/** chair_admin_attend_chair_petitions
 * Agent function prototype for attend_chair_petitions function of chair_admin agent
 */
void chair_admin_attend_chair_petitions(cudaStream_t &stream);

/** navmap_output_navmap_cells
 * Agent function prototype for output_navmap_cells function of navmap agent
 */
void navmap_output_navmap_cells(cudaStream_t &stream);

/** navmap_generate_pedestrians
 * Agent function prototype for generate_pedestrians function of navmap agent
 */
void navmap_generate_pedestrians(cudaStream_t &stream);

/** navmap_generate_medics
 * Agent function prototype for generate_medics function of navmap agent
 */
void navmap_generate_medics(cudaStream_t &stream);

  
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
    h_agents_default_variable_id_data_iteration = 0;
    h_agents_default_variable_x_data_iteration = 0;
    h_agents_default_variable_y_data_iteration = 0;
    h_agents_default_variable_velx_data_iteration = 0;
    h_agents_default_variable_vely_data_iteration = 0;
    h_agents_default_variable_steer_x_data_iteration = 0;
    h_agents_default_variable_steer_y_data_iteration = 0;
    h_agents_default_variable_height_data_iteration = 0;
    h_agents_default_variable_exit_no_data_iteration = 0;
    h_agents_default_variable_speed_data_iteration = 0;
    h_agents_default_variable_lod_data_iteration = 0;
    h_agents_default_variable_animate_data_iteration = 0;
    h_agents_default_variable_animate_dir_data_iteration = 0;
    h_agents_default_variable_estado_data_iteration = 0;
    h_agents_default_variable_tick_data_iteration = 0;
    h_agents_default_variable_estado_movimiento_data_iteration = 0;
    h_agents_default_variable_go_to_x_data_iteration = 0;
    h_agents_default_variable_go_to_y_data_iteration = 0;
    h_medics_default2_variable_x_data_iteration = 0;
    h_medics_default2_variable_y_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_x_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_y_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_colaPacientes_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_front_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_rear_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_size_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_tick_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_current_patient_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_attend_patient_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_estado_data_iteration = 0;
    h_chair_admins_defaultAdmin_variable_id_data_iteration = 0;
    h_chair_admins_defaultAdmin_variable_chairArray_data_iteration = 0;
    h_navmaps_static_variable_x_data_iteration = 0;
    h_navmaps_static_variable_y_data_iteration = 0;
    h_navmaps_static_variable_exit_no_data_iteration = 0;
    h_navmaps_static_variable_height_data_iteration = 0;
    h_navmaps_static_variable_collision_x_data_iteration = 0;
    h_navmaps_static_variable_collision_y_data_iteration = 0;
    h_navmaps_static_variable_exit0_x_data_iteration = 0;
    h_navmaps_static_variable_exit0_y_data_iteration = 0;
    h_navmaps_static_variable_exit1_x_data_iteration = 0;
    h_navmaps_static_variable_exit1_y_data_iteration = 0;
    h_navmaps_static_variable_exit2_x_data_iteration = 0;
    h_navmaps_static_variable_exit2_y_data_iteration = 0;
    h_navmaps_static_variable_exit3_x_data_iteration = 0;
    h_navmaps_static_variable_exit3_y_data_iteration = 0;
    h_navmaps_static_variable_exit4_x_data_iteration = 0;
    h_navmaps_static_variable_exit4_y_data_iteration = 0;
    h_navmaps_static_variable_exit5_x_data_iteration = 0;
    h_navmaps_static_variable_exit5_y_data_iteration = 0;
    h_navmaps_static_variable_exit6_x_data_iteration = 0;
    h_navmaps_static_variable_exit6_y_data_iteration = 0;
    h_navmaps_static_variable_cant_generados_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_agent_SoA_size = sizeof(xmachine_memory_agent_list);
	h_agents_default = (xmachine_memory_agent_list*)malloc(xmachine_agent_SoA_size);
	int xmachine_medic_SoA_size = sizeof(xmachine_memory_medic_list);
	h_medics_default2 = (xmachine_memory_medic_list*)malloc(xmachine_medic_SoA_size);
	int xmachine_receptionist_SoA_size = sizeof(xmachine_memory_receptionist_list);
	h_receptionists_defaultReceptionist = (xmachine_memory_receptionist_list*)malloc(xmachine_receptionist_SoA_size);
	int xmachine_chair_admin_SoA_size = sizeof(xmachine_memory_chair_admin_list);
	h_chair_admins_defaultAdmin = (xmachine_memory_chair_admin_list*)malloc(xmachine_chair_admin_SoA_size);
	int xmachine_navmap_SoA_size = sizeof(xmachine_memory_navmap_list);
	h_navmaps_static = (xmachine_memory_navmap_list*)malloc(xmachine_navmap_SoA_size);

	/* Message memory allocation (CPU) */
	int message_pedestrian_location_SoA_size = sizeof(xmachine_message_pedestrian_location_list);
	h_pedestrian_locations = (xmachine_message_pedestrian_location_list*)malloc(message_pedestrian_location_SoA_size);
	int message_pedestrian_state_SoA_size = sizeof(xmachine_message_pedestrian_state_list);
	h_pedestrian_states = (xmachine_message_pedestrian_state_list*)malloc(message_pedestrian_state_SoA_size);
	int message_navmap_cell_SoA_size = sizeof(xmachine_message_navmap_cell_list);
	h_navmap_cells = (xmachine_message_navmap_cell_list*)malloc(message_navmap_cell_SoA_size);
	int message_check_in_SoA_size = sizeof(xmachine_message_check_in_list);
	h_check_ins = (xmachine_message_check_in_list*)malloc(message_check_in_SoA_size);
	int message_check_in_done_SoA_size = sizeof(xmachine_message_check_in_done_list);
	h_check_in_dones = (xmachine_message_check_in_done_list*)malloc(message_check_in_done_SoA_size);
	int message_chair_petition_SoA_size = sizeof(xmachine_message_chair_petition_list);
	h_chair_petitions = (xmachine_message_chair_petition_list*)malloc(message_chair_petition_SoA_size);
	int message_chair_response_SoA_size = sizeof(xmachine_message_chair_response_list);
	h_chair_responses = (xmachine_message_chair_response_list*)malloc(message_chair_response_SoA_size);

	//Exit if agent or message buffer sizes are to small for function outputs

  /* Graph memory allocation (CPU) */
  

    PROFILE_POP_RANGE(); //"allocate host"
	
			
	/* Set spatial partitioning pedestrian_location message variables (min_bounds, max_bounds)*/
	h_message_pedestrian_location_radius = (float)0.025;
	gpuErrchk(cudaMemcpyToSymbol( d_message_pedestrian_location_radius, &h_message_pedestrian_location_radius, sizeof(float)));	
	    h_message_pedestrian_location_min_bounds = glm::vec3((float)-1.0, (float)-1.0, (float)0.0);
	gpuErrchk(cudaMemcpyToSymbol( d_message_pedestrian_location_min_bounds, &h_message_pedestrian_location_min_bounds, sizeof(glm::vec3)));	
	h_message_pedestrian_location_max_bounds = glm::vec3((float)1.0, (float)1.0, (float)0.025);
	gpuErrchk(cudaMemcpyToSymbol( d_message_pedestrian_location_max_bounds, &h_message_pedestrian_location_max_bounds, sizeof(glm::vec3)));	
	h_message_pedestrian_location_partitionDim.x = (int)ceil((h_message_pedestrian_location_max_bounds.x - h_message_pedestrian_location_min_bounds.x)/h_message_pedestrian_location_radius);
	h_message_pedestrian_location_partitionDim.y = (int)ceil((h_message_pedestrian_location_max_bounds.y - h_message_pedestrian_location_min_bounds.y)/h_message_pedestrian_location_radius);
	h_message_pedestrian_location_partitionDim.z = (int)ceil((h_message_pedestrian_location_max_bounds.z - h_message_pedestrian_location_min_bounds.z)/h_message_pedestrian_location_radius);
	gpuErrchk(cudaMemcpyToSymbol( d_message_pedestrian_location_partitionDim, &h_message_pedestrian_location_partitionDim, sizeof(glm::ivec3)));	
	
			
	/* Set spatial partitioning pedestrian_state message variables (min_bounds, max_bounds)*/
	h_message_pedestrian_state_radius = (float)0.015625;
	gpuErrchk(cudaMemcpyToSymbol( d_message_pedestrian_state_radius, &h_message_pedestrian_state_radius, sizeof(float)));	
	    h_message_pedestrian_state_min_bounds = glm::vec3((float)-1.0, (float)-1.0, (float)0.0);
	gpuErrchk(cudaMemcpyToSymbol( d_message_pedestrian_state_min_bounds, &h_message_pedestrian_state_min_bounds, sizeof(glm::vec3)));	
	h_message_pedestrian_state_max_bounds = glm::vec3((float)1.0, (float)1.0, (float)0.015625);
	gpuErrchk(cudaMemcpyToSymbol( d_message_pedestrian_state_max_bounds, &h_message_pedestrian_state_max_bounds, sizeof(glm::vec3)));	
	h_message_pedestrian_state_partitionDim.x = (int)ceil((h_message_pedestrian_state_max_bounds.x - h_message_pedestrian_state_min_bounds.x)/h_message_pedestrian_state_radius);
	h_message_pedestrian_state_partitionDim.y = (int)ceil((h_message_pedestrian_state_max_bounds.y - h_message_pedestrian_state_min_bounds.y)/h_message_pedestrian_state_radius);
	h_message_pedestrian_state_partitionDim.z = (int)ceil((h_message_pedestrian_state_max_bounds.z - h_message_pedestrian_state_min_bounds.z)/h_message_pedestrian_state_radius);
	gpuErrchk(cudaMemcpyToSymbol( d_message_pedestrian_state_partitionDim, &h_message_pedestrian_state_partitionDim, sizeof(glm::ivec3)));	
	
	
	/* Set discrete navmap_cell message variables (range, width)*/
	h_message_navmap_cell_range = 0; //from xml
	h_message_navmap_cell_width = (int)floor(sqrt((float)xmachine_message_navmap_cell_MAX));
	//check the width
	if (!is_sqr_pow2(xmachine_message_navmap_cell_MAX)){
		printf("ERROR: navmap_cell message max must be a square power of 2 for a 2D discrete message grid!\n");
		exit(EXIT_FAILURE);
	}
	gpuErrchk(cudaMemcpyToSymbol( d_message_navmap_cell_range, &h_message_navmap_cell_range, sizeof(int)));	
	gpuErrchk(cudaMemcpyToSymbol( d_message_navmap_cell_width, &h_message_navmap_cell_width, sizeof(int)));
	
	/* Check that population size is a square power of 2*/
	if (!is_sqr_pow2(xmachine_memory_navmap_MAX)){
		printf("ERROR: navmaps agent count must be a square power of 2!\n");
		exit(EXIT_FAILURE);
	}
	h_xmachine_memory_navmap_pop_width = (int)sqrt(xmachine_memory_navmap_MAX);
	

	//read initial states
	readInitialStates(inputfile, h_agents_default, &h_xmachine_memory_agent_default_count, h_medics_default2, &h_xmachine_memory_medic_default2_count, h_receptionists_defaultReceptionist, &h_xmachine_memory_receptionist_defaultReceptionist_count, h_chair_admins_defaultAdmin, &h_xmachine_memory_chair_admin_defaultAdmin_count, h_navmaps_static, &h_xmachine_memory_navmap_static_count);

  // Read graphs from disk
  

  PROFILE_PUSH_RANGE("allocate device");
	
	/* agent Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_agents, xmachine_agent_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_agents_swap, xmachine_agent_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_agents_new, xmachine_agent_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_agent_keys, xmachine_memory_agent_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_agent_values, xmachine_memory_agent_MAX* sizeof(uint)));
	/* default memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_agents_default, xmachine_agent_SoA_size));
	gpuErrchk( cudaMemcpy( d_agents_default, h_agents_default, xmachine_agent_SoA_size, cudaMemcpyHostToDevice));
    
	/* medic Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_medics, xmachine_medic_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_medics_swap, xmachine_medic_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_medics_new, xmachine_medic_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_medic_keys, xmachine_memory_medic_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_medic_values, xmachine_memory_medic_MAX* sizeof(uint)));
	/* default2 memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_medics_default2, xmachine_medic_SoA_size));
	gpuErrchk( cudaMemcpy( d_medics_default2, h_medics_default2, xmachine_medic_SoA_size, cudaMemcpyHostToDevice));
    
	/* receptionist Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_receptionists, xmachine_receptionist_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_receptionists_swap, xmachine_receptionist_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_receptionists_new, xmachine_receptionist_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_receptionist_keys, xmachine_memory_receptionist_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_receptionist_values, xmachine_memory_receptionist_MAX* sizeof(uint)));
	/* defaultReceptionist memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_receptionists_defaultReceptionist, xmachine_receptionist_SoA_size));
	gpuErrchk( cudaMemcpy( d_receptionists_defaultReceptionist, h_receptionists_defaultReceptionist, xmachine_receptionist_SoA_size, cudaMemcpyHostToDevice));
    
	/* chair_admin Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_chair_admins, xmachine_chair_admin_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_chair_admins_swap, xmachine_chair_admin_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_chair_admins_new, xmachine_chair_admin_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_chair_admin_keys, xmachine_memory_chair_admin_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_chair_admin_values, xmachine_memory_chair_admin_MAX* sizeof(uint)));
	/* defaultAdmin memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_chair_admins_defaultAdmin, xmachine_chair_admin_SoA_size));
	gpuErrchk( cudaMemcpy( d_chair_admins_defaultAdmin, h_chair_admins_defaultAdmin, xmachine_chair_admin_SoA_size, cudaMemcpyHostToDevice));
    
	/* navmap Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_navmaps, xmachine_navmap_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_navmaps_swap, xmachine_navmap_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_navmaps_new, xmachine_navmap_SoA_size));
    
	/* static memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_navmaps_static, xmachine_navmap_SoA_size));
	gpuErrchk( cudaMemcpy( d_navmaps_static, h_navmaps_static, xmachine_navmap_SoA_size, cudaMemcpyHostToDevice));
    
	/* pedestrian_location Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_pedestrian_locations, message_pedestrian_location_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_pedestrian_locations_swap, message_pedestrian_location_SoA_size));
	gpuErrchk( cudaMemcpy( d_pedestrian_locations, h_pedestrian_locations, message_pedestrian_location_SoA_size, cudaMemcpyHostToDevice));
	gpuErrchk( cudaMalloc( (void**) &d_pedestrian_location_partition_matrix, sizeof(xmachine_message_pedestrian_location_PBM)));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_pedestrian_location_local_bin_index, xmachine_message_pedestrian_location_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_pedestrian_location_unsorted_index, xmachine_message_pedestrian_location_MAX* sizeof(uint)));
    /* Calculate and allocate CUB temporary memory for exclusive scans */
    d_temp_scan_storage_xmachine_message_pedestrian_location = nullptr;
    temp_scan_bytes_xmachine_message_pedestrian_location = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_xmachine_message_pedestrian_location, 
        temp_scan_bytes_xmachine_message_pedestrian_location, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_message_pedestrian_location_grid_size
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_xmachine_message_pedestrian_location, temp_scan_bytes_xmachine_message_pedestrian_location));
#else
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_pedestrian_location_keys, xmachine_message_pedestrian_location_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_pedestrian_location_values, xmachine_message_pedestrian_location_MAX* sizeof(uint)));
#endif
	
	/* pedestrian_state Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_pedestrian_states, message_pedestrian_state_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_pedestrian_states_swap, message_pedestrian_state_SoA_size));
	gpuErrchk( cudaMemcpy( d_pedestrian_states, h_pedestrian_states, message_pedestrian_state_SoA_size, cudaMemcpyHostToDevice));
	gpuErrchk( cudaMalloc( (void**) &d_pedestrian_state_partition_matrix, sizeof(xmachine_message_pedestrian_state_PBM)));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_pedestrian_state_local_bin_index, xmachine_message_pedestrian_state_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_pedestrian_state_unsorted_index, xmachine_message_pedestrian_state_MAX* sizeof(uint)));
    /* Calculate and allocate CUB temporary memory for exclusive scans */
    d_temp_scan_storage_xmachine_message_pedestrian_state = nullptr;
    temp_scan_bytes_xmachine_message_pedestrian_state = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_xmachine_message_pedestrian_state, 
        temp_scan_bytes_xmachine_message_pedestrian_state, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_message_pedestrian_state_grid_size
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_xmachine_message_pedestrian_state, temp_scan_bytes_xmachine_message_pedestrian_state));
#else
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_pedestrian_state_keys, xmachine_message_pedestrian_state_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_message_pedestrian_state_values, xmachine_message_pedestrian_state_MAX* sizeof(uint)));
#endif
	
	/* navmap_cell Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_navmap_cells, message_navmap_cell_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_navmap_cells_swap, message_navmap_cell_SoA_size));
	gpuErrchk( cudaMemcpy( d_navmap_cells, h_navmap_cells, message_navmap_cell_SoA_size, cudaMemcpyHostToDevice));
	
	/* check_in Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_check_ins, message_check_in_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_check_ins_swap, message_check_in_SoA_size));
	gpuErrchk( cudaMemcpy( d_check_ins, h_check_ins, message_check_in_SoA_size, cudaMemcpyHostToDevice));
	
	/* check_in_done Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_check_in_dones, message_check_in_done_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_check_in_dones_swap, message_check_in_done_SoA_size));
	gpuErrchk( cudaMemcpy( d_check_in_dones, h_check_in_dones, message_check_in_done_SoA_size, cudaMemcpyHostToDevice));
	
	/* chair_petition Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_chair_petitions, message_chair_petition_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_chair_petitions_swap, message_chair_petition_SoA_size));
	gpuErrchk( cudaMemcpy( d_chair_petitions, h_chair_petitions, message_chair_petition_SoA_size, cudaMemcpyHostToDevice));
	
	/* chair_response Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_chair_responses, message_chair_response_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_chair_responses_swap, message_chair_response_SoA_size));
	gpuErrchk( cudaMemcpy( d_chair_responses, h_chair_responses, message_chair_response_SoA_size, cudaMemcpyHostToDevice));
		


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
    
    d_temp_scan_storage_medic = nullptr;
    temp_scan_storage_bytes_medic = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_medic, 
        temp_scan_storage_bytes_medic, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_medic_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_medic, temp_scan_storage_bytes_medic));
    
    d_temp_scan_storage_receptionist = nullptr;
    temp_scan_storage_bytes_receptionist = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_receptionist, 
        temp_scan_storage_bytes_receptionist, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_receptionist_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_receptionist, temp_scan_storage_bytes_receptionist));
    
    d_temp_scan_storage_chair_admin = nullptr;
    temp_scan_storage_bytes_chair_admin = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_chair_admin, 
        temp_scan_storage_bytes_chair_admin, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_chair_admin_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_chair_admin, temp_scan_storage_bytes_chair_admin));
    
    d_temp_scan_storage_navmap = nullptr;
    temp_scan_storage_bytes_navmap = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_navmap, 
        temp_scan_storage_bytes_navmap, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_navmap_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_navmap, temp_scan_storage_bytes_navmap));
    

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
    inicializarMapa();
    PROFILE_PUSH_RANGE("inicializarMapa");
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_INIT_FUNCTIONS) && INSTRUMENT_INIT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: inicializarMapa = %f (ms)\n", instrument_milliseconds);
#endif
	
  
  /* Init CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamCreate(&stream1));
  gpuErrchk(cudaStreamCreate(&stream2));
  gpuErrchk(cudaStreamCreate(&stream3));
  gpuErrchk(cudaStreamCreate(&stream4));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_agent_default_count: %u\n",get_agent_agent_default_count());
	
		printf("Init agent_medic_default2_count: %u\n",get_agent_medic_default2_count());
	
		printf("Init agent_receptionist_defaultReceptionist_count: %u\n",get_agent_receptionist_defaultReceptionist_count());
	
		printf("Init agent_chair_admin_defaultAdmin_count: %u\n",get_agent_chair_admin_defaultAdmin_count());
	
		printf("Init agent_navmap_static_count: %u\n",get_agent_navmap_static_count());
	
#endif
} 


void sort_agents_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_agent_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_agent_default_count); 
	gridSize = (h_xmachine_memory_agent_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_agent_keys, d_xmachine_memory_agent_values, d_agents_default);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_agent_keys),  thrust::device_pointer_cast(d_xmachine_memory_agent_keys) + h_xmachine_memory_agent_default_count,  thrust::device_pointer_cast(d_xmachine_memory_agent_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_agent_agents, no_sm, h_xmachine_memory_agent_default_count); 
	gridSize = (h_xmachine_memory_agent_default_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_agent_agents<<<gridSize, blockSize>>>(d_xmachine_memory_agent_values, d_agents_default, d_agents_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_agent_list* d_agents_temp = d_agents_default;
	d_agents_default = d_agents_swap;
	d_agents_swap = d_agents_temp;	
}

void sort_medics_default2(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_medic_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_medic_default2_count); 
	gridSize = (h_xmachine_memory_medic_default2_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_medic_keys, d_xmachine_memory_medic_values, d_medics_default2);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_medic_keys),  thrust::device_pointer_cast(d_xmachine_memory_medic_keys) + h_xmachine_memory_medic_default2_count,  thrust::device_pointer_cast(d_xmachine_memory_medic_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_medic_agents, no_sm, h_xmachine_memory_medic_default2_count); 
	gridSize = (h_xmachine_memory_medic_default2_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_medic_agents<<<gridSize, blockSize>>>(d_xmachine_memory_medic_values, d_medics_default2, d_medics_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_medic_list* d_medics_temp = d_medics_default2;
	d_medics_default2 = d_medics_swap;
	d_medics_swap = d_medics_temp;	
}

void sort_receptionists_defaultReceptionist(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_receptionist_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_receptionist_defaultReceptionist_count); 
	gridSize = (h_xmachine_memory_receptionist_defaultReceptionist_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_receptionist_keys, d_xmachine_memory_receptionist_values, d_receptionists_defaultReceptionist);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_receptionist_keys),  thrust::device_pointer_cast(d_xmachine_memory_receptionist_keys) + h_xmachine_memory_receptionist_defaultReceptionist_count,  thrust::device_pointer_cast(d_xmachine_memory_receptionist_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_receptionist_agents, no_sm, h_xmachine_memory_receptionist_defaultReceptionist_count); 
	gridSize = (h_xmachine_memory_receptionist_defaultReceptionist_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_receptionist_agents<<<gridSize, blockSize>>>(d_xmachine_memory_receptionist_values, d_receptionists_defaultReceptionist, d_receptionists_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_receptionist_list* d_receptionists_temp = d_receptionists_defaultReceptionist;
	d_receptionists_defaultReceptionist = d_receptionists_swap;
	d_receptionists_swap = d_receptionists_temp;	
}

void sort_chair_admins_defaultAdmin(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_chair_admin_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_chair_admin_defaultAdmin_count); 
	gridSize = (h_xmachine_memory_chair_admin_defaultAdmin_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_chair_admin_keys, d_xmachine_memory_chair_admin_values, d_chair_admins_defaultAdmin);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_chair_admin_keys),  thrust::device_pointer_cast(d_xmachine_memory_chair_admin_keys) + h_xmachine_memory_chair_admin_defaultAdmin_count,  thrust::device_pointer_cast(d_xmachine_memory_chair_admin_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_chair_admin_agents, no_sm, h_xmachine_memory_chair_admin_defaultAdmin_count); 
	gridSize = (h_xmachine_memory_chair_admin_defaultAdmin_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_chair_admin_agents<<<gridSize, blockSize>>>(d_xmachine_memory_chair_admin_values, d_chair_admins_defaultAdmin, d_chair_admins_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_chair_admin_list* d_chair_admins_temp = d_chair_admins_defaultAdmin;
	d_chair_admins_defaultAdmin = d_chair_admins_swap;
	d_chair_admins_swap = d_chair_admins_temp;	
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
	
	/* medic Agent variables */
	gpuErrchk(cudaFree(d_medics));
	gpuErrchk(cudaFree(d_medics_swap));
	gpuErrchk(cudaFree(d_medics_new));
	
	free( h_medics_default2);
	gpuErrchk(cudaFree(d_medics_default2));
	
	/* receptionist Agent variables */
	gpuErrchk(cudaFree(d_receptionists));
	gpuErrchk(cudaFree(d_receptionists_swap));
	gpuErrchk(cudaFree(d_receptionists_new));
	
	free( h_receptionists_defaultReceptionist);
	gpuErrchk(cudaFree(d_receptionists_defaultReceptionist));
	
	/* chair_admin Agent variables */
	gpuErrchk(cudaFree(d_chair_admins));
	gpuErrchk(cudaFree(d_chair_admins_swap));
	gpuErrchk(cudaFree(d_chair_admins_new));
	
	free( h_chair_admins_defaultAdmin);
	gpuErrchk(cudaFree(d_chair_admins_defaultAdmin));
	
	/* navmap Agent variables */
	gpuErrchk(cudaFree(d_navmaps));
	gpuErrchk(cudaFree(d_navmaps_swap));
	gpuErrchk(cudaFree(d_navmaps_new));
	
	free( h_navmaps_static);
	gpuErrchk(cudaFree(d_navmaps_static));
	

	/* Message data free */
	
	/* pedestrian_location Message variables */
	free( h_pedestrian_locations);
	gpuErrchk(cudaFree(d_pedestrian_locations));
	gpuErrchk(cudaFree(d_pedestrian_locations_swap));
	gpuErrchk(cudaFree(d_pedestrian_location_partition_matrix));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaFree(d_xmachine_message_pedestrian_location_local_bin_index));
	gpuErrchk(cudaFree(d_xmachine_message_pedestrian_location_unsorted_index));
  gpuErrchk(cudaFree(d_temp_scan_storage_xmachine_message_pedestrian_location));
  d_temp_scan_storage_xmachine_message_pedestrian_location = nullptr;
  temp_scan_bytes_xmachine_message_pedestrian_location = 0;
#else
	gpuErrchk(cudaFree(d_xmachine_message_pedestrian_location_keys));
	gpuErrchk(cudaFree(d_xmachine_message_pedestrian_location_values));
#endif
	
	/* pedestrian_state Message variables */
	free( h_pedestrian_states);
	gpuErrchk(cudaFree(d_pedestrian_states));
	gpuErrchk(cudaFree(d_pedestrian_states_swap));
	gpuErrchk(cudaFree(d_pedestrian_state_partition_matrix));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaFree(d_xmachine_message_pedestrian_state_local_bin_index));
	gpuErrchk(cudaFree(d_xmachine_message_pedestrian_state_unsorted_index));
  gpuErrchk(cudaFree(d_temp_scan_storage_xmachine_message_pedestrian_state));
  d_temp_scan_storage_xmachine_message_pedestrian_state = nullptr;
  temp_scan_bytes_xmachine_message_pedestrian_state = 0;
#else
	gpuErrchk(cudaFree(d_xmachine_message_pedestrian_state_keys));
	gpuErrchk(cudaFree(d_xmachine_message_pedestrian_state_values));
#endif
	
	/* navmap_cell Message variables */
	free( h_navmap_cells);
	gpuErrchk(cudaFree(d_navmap_cells));
	gpuErrchk(cudaFree(d_navmap_cells_swap));
	
	/* check_in Message variables */
	free( h_check_ins);
	gpuErrchk(cudaFree(d_check_ins));
	gpuErrchk(cudaFree(d_check_ins_swap));
	
	/* check_in_done Message variables */
	free( h_check_in_dones);
	gpuErrchk(cudaFree(d_check_in_dones));
	gpuErrchk(cudaFree(d_check_in_dones_swap));
	
	/* chair_petition Message variables */
	free( h_chair_petitions);
	gpuErrchk(cudaFree(d_chair_petitions));
	gpuErrchk(cudaFree(d_chair_petitions_swap));
	
	/* chair_response Message variables */
	free( h_chair_responses);
	gpuErrchk(cudaFree(d_chair_responses));
	gpuErrchk(cudaFree(d_chair_responses_swap));
	

    /* Free temporary CUB memory if required. */
    
    if(d_temp_scan_storage_agent != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_agent));
      d_temp_scan_storage_agent = nullptr;
      temp_scan_storage_bytes_agent = 0;
    }
    
    if(d_temp_scan_storage_medic != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_medic));
      d_temp_scan_storage_medic = nullptr;
      temp_scan_storage_bytes_medic = 0;
    }
    
    if(d_temp_scan_storage_receptionist != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_receptionist));
      d_temp_scan_storage_receptionist = nullptr;
      temp_scan_storage_bytes_receptionist = 0;
    }
    
    if(d_temp_scan_storage_chair_admin != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_chair_admin));
      d_temp_scan_storage_chair_admin = nullptr;
      temp_scan_storage_bytes_chair_admin = 0;
    }
    
    if(d_temp_scan_storage_navmap != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_navmap));
      d_temp_scan_storage_navmap = nullptr;
      temp_scan_storage_bytes_navmap = 0;
    }
    

  /* Graph data free */
  
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));
  gpuErrchk(cudaStreamDestroy(stream2));
  gpuErrchk(cudaStreamDestroy(stream3));
  gpuErrchk(cudaStreamDestroy(stream4));

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
	h_message_pedestrian_location_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_pedestrian_location_count, &h_message_pedestrian_location_count, sizeof(int)));
	
	h_message_pedestrian_state_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_pedestrian_state_count, &h_message_pedestrian_state_count, sizeof(int)));
	
	h_message_check_in_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_check_in_count, &h_message_check_in_count, sizeof(int)));
	
	h_message_check_in_done_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_check_in_done_count, &h_message_check_in_done_count, sizeof(int)));
	
	h_message_chair_petition_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_chair_petition_count, &h_message_chair_petition_count, sizeof(int)));
	
	h_message_chair_response_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_chair_response_count, &h_message_chair_response_count, sizeof(int)));
	

	/* Call agent functions in order iterating through the layer functions */
	
	/* Layer 1*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("navmap_generate_pedestrians");
	navmap_generate_pedestrians(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: navmap_generate_pedestrians = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("navmap_generate_medics");
	navmap_generate_medics(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: navmap_generate_medics = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 2*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_output_pedestrian_location");
	agent_output_pedestrian_location(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_output_pedestrian_location = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("navmap_output_navmap_cells");
	navmap_output_navmap_cells(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: navmap_output_navmap_cells = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_output_pedestrian_state");
	agent_output_pedestrian_state(stream3);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_output_pedestrian_state = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_output_chair_petition");
	agent_output_chair_petition(stream4);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_output_chair_petition = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_avoid_pedestrians");
	agent_avoid_pedestrians(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_avoid_pedestrians = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 4*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_infect_pedestrians");
	agent_infect_pedestrians(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_infect_pedestrians = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("receptionist_infect_receptionist");
	receptionist_infect_receptionist(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: receptionist_infect_receptionist = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 5*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_move");
	agent_move(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_move = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 6*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("receptionist_receptionServer");
	receptionist_receptionServer(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: receptionist_receptionServer = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 7*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("chair_admin_attend_chair_petitions");
	chair_admin_attend_chair_petitions(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: chair_admin_attend_chair_petitions = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 8*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_check_check_in_done");
	agent_check_check_in_done(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_check_check_in_done = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_check_chair_response");
	agent_check_chair_response(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_check_chair_response = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_agent_default_count: %u\n",get_agent_agent_default_count());
	
		printf("agent_medic_default2_count: %u\n",get_agent_medic_default2_count());
	
		printf("agent_receptionist_defaultReceptionist_count: %u\n",get_agent_receptionist_defaultReceptionist_count());
	
		printf("agent_chair_admin_defaultAdmin_count: %u\n",get_agent_chair_admin_defaultAdmin_count());
	
		printf("agent_navmap_static_count: %u\n",get_agent_navmap_static_count());
	
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
float h_env_EMMISION_RATE_EXIT1;
float h_env_EMMISION_RATE_EXIT2;
float h_env_EMMISION_RATE_EXIT3;
float h_env_EMMISION_RATE_EXIT4;
float h_env_EMMISION_RATE_EXIT5;
float h_env_EMMISION_RATE_EXIT6;
float h_env_EMMISION_RATE_EXIT7;
int h_env_EXIT1_PROBABILITY;
int h_env_EXIT2_PROBABILITY;
int h_env_EXIT3_PROBABILITY;
int h_env_EXIT4_PROBABILITY;
int h_env_EXIT5_PROBABILITY;
int h_env_EXIT6_PROBABILITY;
int h_env_EXIT7_PROBABILITY;
int h_env_EXIT1_STATE;
int h_env_EXIT2_STATE;
int h_env_EXIT3_STATE;
int h_env_EXIT4_STATE;
int h_env_EXIT5_STATE;
int h_env_EXIT6_STATE;
int h_env_EXIT7_STATE;
int h_env_EXIT1_CELL_COUNT;
int h_env_EXIT2_CELL_COUNT;
int h_env_EXIT3_CELL_COUNT;
int h_env_EXIT4_CELL_COUNT;
int h_env_EXIT5_CELL_COUNT;
int h_env_EXIT6_CELL_COUNT;
int h_env_EXIT7_CELL_COUNT;
float h_env_TIME_SCALER;
float h_env_STEER_WEIGHT;
float h_env_AVOID_WEIGHT;
float h_env_COLLISION_WEIGHT;
float h_env_GOAL_WEIGHT;


//constant setter
void set_EMMISION_RATE_EXIT1(float* h_EMMISION_RATE_EXIT1){
    gpuErrchk(cudaMemcpyToSymbol(EMMISION_RATE_EXIT1, h_EMMISION_RATE_EXIT1, sizeof(float)));
    memcpy(&h_env_EMMISION_RATE_EXIT1, h_EMMISION_RATE_EXIT1,sizeof(float));
}

//constant getter
const float* get_EMMISION_RATE_EXIT1(){
    return &h_env_EMMISION_RATE_EXIT1;
}



//constant setter
void set_EMMISION_RATE_EXIT2(float* h_EMMISION_RATE_EXIT2){
    gpuErrchk(cudaMemcpyToSymbol(EMMISION_RATE_EXIT2, h_EMMISION_RATE_EXIT2, sizeof(float)));
    memcpy(&h_env_EMMISION_RATE_EXIT2, h_EMMISION_RATE_EXIT2,sizeof(float));
}

//constant getter
const float* get_EMMISION_RATE_EXIT2(){
    return &h_env_EMMISION_RATE_EXIT2;
}



//constant setter
void set_EMMISION_RATE_EXIT3(float* h_EMMISION_RATE_EXIT3){
    gpuErrchk(cudaMemcpyToSymbol(EMMISION_RATE_EXIT3, h_EMMISION_RATE_EXIT3, sizeof(float)));
    memcpy(&h_env_EMMISION_RATE_EXIT3, h_EMMISION_RATE_EXIT3,sizeof(float));
}

//constant getter
const float* get_EMMISION_RATE_EXIT3(){
    return &h_env_EMMISION_RATE_EXIT3;
}



//constant setter
void set_EMMISION_RATE_EXIT4(float* h_EMMISION_RATE_EXIT4){
    gpuErrchk(cudaMemcpyToSymbol(EMMISION_RATE_EXIT4, h_EMMISION_RATE_EXIT4, sizeof(float)));
    memcpy(&h_env_EMMISION_RATE_EXIT4, h_EMMISION_RATE_EXIT4,sizeof(float));
}

//constant getter
const float* get_EMMISION_RATE_EXIT4(){
    return &h_env_EMMISION_RATE_EXIT4;
}



//constant setter
void set_EMMISION_RATE_EXIT5(float* h_EMMISION_RATE_EXIT5){
    gpuErrchk(cudaMemcpyToSymbol(EMMISION_RATE_EXIT5, h_EMMISION_RATE_EXIT5, sizeof(float)));
    memcpy(&h_env_EMMISION_RATE_EXIT5, h_EMMISION_RATE_EXIT5,sizeof(float));
}

//constant getter
const float* get_EMMISION_RATE_EXIT5(){
    return &h_env_EMMISION_RATE_EXIT5;
}



//constant setter
void set_EMMISION_RATE_EXIT6(float* h_EMMISION_RATE_EXIT6){
    gpuErrchk(cudaMemcpyToSymbol(EMMISION_RATE_EXIT6, h_EMMISION_RATE_EXIT6, sizeof(float)));
    memcpy(&h_env_EMMISION_RATE_EXIT6, h_EMMISION_RATE_EXIT6,sizeof(float));
}

//constant getter
const float* get_EMMISION_RATE_EXIT6(){
    return &h_env_EMMISION_RATE_EXIT6;
}



//constant setter
void set_EMMISION_RATE_EXIT7(float* h_EMMISION_RATE_EXIT7){
    gpuErrchk(cudaMemcpyToSymbol(EMMISION_RATE_EXIT7, h_EMMISION_RATE_EXIT7, sizeof(float)));
    memcpy(&h_env_EMMISION_RATE_EXIT7, h_EMMISION_RATE_EXIT7,sizeof(float));
}

//constant getter
const float* get_EMMISION_RATE_EXIT7(){
    return &h_env_EMMISION_RATE_EXIT7;
}



//constant setter
void set_EXIT1_PROBABILITY(int* h_EXIT1_PROBABILITY){
    gpuErrchk(cudaMemcpyToSymbol(EXIT1_PROBABILITY, h_EXIT1_PROBABILITY, sizeof(int)));
    memcpy(&h_env_EXIT1_PROBABILITY, h_EXIT1_PROBABILITY,sizeof(int));
}

//constant getter
const int* get_EXIT1_PROBABILITY(){
    return &h_env_EXIT1_PROBABILITY;
}



//constant setter
void set_EXIT2_PROBABILITY(int* h_EXIT2_PROBABILITY){
    gpuErrchk(cudaMemcpyToSymbol(EXIT2_PROBABILITY, h_EXIT2_PROBABILITY, sizeof(int)));
    memcpy(&h_env_EXIT2_PROBABILITY, h_EXIT2_PROBABILITY,sizeof(int));
}

//constant getter
const int* get_EXIT2_PROBABILITY(){
    return &h_env_EXIT2_PROBABILITY;
}



//constant setter
void set_EXIT3_PROBABILITY(int* h_EXIT3_PROBABILITY){
    gpuErrchk(cudaMemcpyToSymbol(EXIT3_PROBABILITY, h_EXIT3_PROBABILITY, sizeof(int)));
    memcpy(&h_env_EXIT3_PROBABILITY, h_EXIT3_PROBABILITY,sizeof(int));
}

//constant getter
const int* get_EXIT3_PROBABILITY(){
    return &h_env_EXIT3_PROBABILITY;
}



//constant setter
void set_EXIT4_PROBABILITY(int* h_EXIT4_PROBABILITY){
    gpuErrchk(cudaMemcpyToSymbol(EXIT4_PROBABILITY, h_EXIT4_PROBABILITY, sizeof(int)));
    memcpy(&h_env_EXIT4_PROBABILITY, h_EXIT4_PROBABILITY,sizeof(int));
}

//constant getter
const int* get_EXIT4_PROBABILITY(){
    return &h_env_EXIT4_PROBABILITY;
}



//constant setter
void set_EXIT5_PROBABILITY(int* h_EXIT5_PROBABILITY){
    gpuErrchk(cudaMemcpyToSymbol(EXIT5_PROBABILITY, h_EXIT5_PROBABILITY, sizeof(int)));
    memcpy(&h_env_EXIT5_PROBABILITY, h_EXIT5_PROBABILITY,sizeof(int));
}

//constant getter
const int* get_EXIT5_PROBABILITY(){
    return &h_env_EXIT5_PROBABILITY;
}



//constant setter
void set_EXIT6_PROBABILITY(int* h_EXIT6_PROBABILITY){
    gpuErrchk(cudaMemcpyToSymbol(EXIT6_PROBABILITY, h_EXIT6_PROBABILITY, sizeof(int)));
    memcpy(&h_env_EXIT6_PROBABILITY, h_EXIT6_PROBABILITY,sizeof(int));
}

//constant getter
const int* get_EXIT6_PROBABILITY(){
    return &h_env_EXIT6_PROBABILITY;
}



//constant setter
void set_EXIT7_PROBABILITY(int* h_EXIT7_PROBABILITY){
    gpuErrchk(cudaMemcpyToSymbol(EXIT7_PROBABILITY, h_EXIT7_PROBABILITY, sizeof(int)));
    memcpy(&h_env_EXIT7_PROBABILITY, h_EXIT7_PROBABILITY,sizeof(int));
}

//constant getter
const int* get_EXIT7_PROBABILITY(){
    return &h_env_EXIT7_PROBABILITY;
}



//constant setter
void set_EXIT1_STATE(int* h_EXIT1_STATE){
    gpuErrchk(cudaMemcpyToSymbol(EXIT1_STATE, h_EXIT1_STATE, sizeof(int)));
    memcpy(&h_env_EXIT1_STATE, h_EXIT1_STATE,sizeof(int));
}

//constant getter
const int* get_EXIT1_STATE(){
    return &h_env_EXIT1_STATE;
}



//constant setter
void set_EXIT2_STATE(int* h_EXIT2_STATE){
    gpuErrchk(cudaMemcpyToSymbol(EXIT2_STATE, h_EXIT2_STATE, sizeof(int)));
    memcpy(&h_env_EXIT2_STATE, h_EXIT2_STATE,sizeof(int));
}

//constant getter
const int* get_EXIT2_STATE(){
    return &h_env_EXIT2_STATE;
}



//constant setter
void set_EXIT3_STATE(int* h_EXIT3_STATE){
    gpuErrchk(cudaMemcpyToSymbol(EXIT3_STATE, h_EXIT3_STATE, sizeof(int)));
    memcpy(&h_env_EXIT3_STATE, h_EXIT3_STATE,sizeof(int));
}

//constant getter
const int* get_EXIT3_STATE(){
    return &h_env_EXIT3_STATE;
}



//constant setter
void set_EXIT4_STATE(int* h_EXIT4_STATE){
    gpuErrchk(cudaMemcpyToSymbol(EXIT4_STATE, h_EXIT4_STATE, sizeof(int)));
    memcpy(&h_env_EXIT4_STATE, h_EXIT4_STATE,sizeof(int));
}

//constant getter
const int* get_EXIT4_STATE(){
    return &h_env_EXIT4_STATE;
}



//constant setter
void set_EXIT5_STATE(int* h_EXIT5_STATE){
    gpuErrchk(cudaMemcpyToSymbol(EXIT5_STATE, h_EXIT5_STATE, sizeof(int)));
    memcpy(&h_env_EXIT5_STATE, h_EXIT5_STATE,sizeof(int));
}

//constant getter
const int* get_EXIT5_STATE(){
    return &h_env_EXIT5_STATE;
}



//constant setter
void set_EXIT6_STATE(int* h_EXIT6_STATE){
    gpuErrchk(cudaMemcpyToSymbol(EXIT6_STATE, h_EXIT6_STATE, sizeof(int)));
    memcpy(&h_env_EXIT6_STATE, h_EXIT6_STATE,sizeof(int));
}

//constant getter
const int* get_EXIT6_STATE(){
    return &h_env_EXIT6_STATE;
}



//constant setter
void set_EXIT7_STATE(int* h_EXIT7_STATE){
    gpuErrchk(cudaMemcpyToSymbol(EXIT7_STATE, h_EXIT7_STATE, sizeof(int)));
    memcpy(&h_env_EXIT7_STATE, h_EXIT7_STATE,sizeof(int));
}

//constant getter
const int* get_EXIT7_STATE(){
    return &h_env_EXIT7_STATE;
}



//constant setter
void set_EXIT1_CELL_COUNT(int* h_EXIT1_CELL_COUNT){
    gpuErrchk(cudaMemcpyToSymbol(EXIT1_CELL_COUNT, h_EXIT1_CELL_COUNT, sizeof(int)));
    memcpy(&h_env_EXIT1_CELL_COUNT, h_EXIT1_CELL_COUNT,sizeof(int));
}

//constant getter
const int* get_EXIT1_CELL_COUNT(){
    return &h_env_EXIT1_CELL_COUNT;
}



//constant setter
void set_EXIT2_CELL_COUNT(int* h_EXIT2_CELL_COUNT){
    gpuErrchk(cudaMemcpyToSymbol(EXIT2_CELL_COUNT, h_EXIT2_CELL_COUNT, sizeof(int)));
    memcpy(&h_env_EXIT2_CELL_COUNT, h_EXIT2_CELL_COUNT,sizeof(int));
}

//constant getter
const int* get_EXIT2_CELL_COUNT(){
    return &h_env_EXIT2_CELL_COUNT;
}



//constant setter
void set_EXIT3_CELL_COUNT(int* h_EXIT3_CELL_COUNT){
    gpuErrchk(cudaMemcpyToSymbol(EXIT3_CELL_COUNT, h_EXIT3_CELL_COUNT, sizeof(int)));
    memcpy(&h_env_EXIT3_CELL_COUNT, h_EXIT3_CELL_COUNT,sizeof(int));
}

//constant getter
const int* get_EXIT3_CELL_COUNT(){
    return &h_env_EXIT3_CELL_COUNT;
}



//constant setter
void set_EXIT4_CELL_COUNT(int* h_EXIT4_CELL_COUNT){
    gpuErrchk(cudaMemcpyToSymbol(EXIT4_CELL_COUNT, h_EXIT4_CELL_COUNT, sizeof(int)));
    memcpy(&h_env_EXIT4_CELL_COUNT, h_EXIT4_CELL_COUNT,sizeof(int));
}

//constant getter
const int* get_EXIT4_CELL_COUNT(){
    return &h_env_EXIT4_CELL_COUNT;
}



//constant setter
void set_EXIT5_CELL_COUNT(int* h_EXIT5_CELL_COUNT){
    gpuErrchk(cudaMemcpyToSymbol(EXIT5_CELL_COUNT, h_EXIT5_CELL_COUNT, sizeof(int)));
    memcpy(&h_env_EXIT5_CELL_COUNT, h_EXIT5_CELL_COUNT,sizeof(int));
}

//constant getter
const int* get_EXIT5_CELL_COUNT(){
    return &h_env_EXIT5_CELL_COUNT;
}



//constant setter
void set_EXIT6_CELL_COUNT(int* h_EXIT6_CELL_COUNT){
    gpuErrchk(cudaMemcpyToSymbol(EXIT6_CELL_COUNT, h_EXIT6_CELL_COUNT, sizeof(int)));
    memcpy(&h_env_EXIT6_CELL_COUNT, h_EXIT6_CELL_COUNT,sizeof(int));
}

//constant getter
const int* get_EXIT6_CELL_COUNT(){
    return &h_env_EXIT6_CELL_COUNT;
}



//constant setter
void set_EXIT7_CELL_COUNT(int* h_EXIT7_CELL_COUNT){
    gpuErrchk(cudaMemcpyToSymbol(EXIT7_CELL_COUNT, h_EXIT7_CELL_COUNT, sizeof(int)));
    memcpy(&h_env_EXIT7_CELL_COUNT, h_EXIT7_CELL_COUNT,sizeof(int));
}

//constant getter
const int* get_EXIT7_CELL_COUNT(){
    return &h_env_EXIT7_CELL_COUNT;
}



//constant setter
void set_TIME_SCALER(float* h_TIME_SCALER){
    gpuErrchk(cudaMemcpyToSymbol(TIME_SCALER, h_TIME_SCALER, sizeof(float)));
    memcpy(&h_env_TIME_SCALER, h_TIME_SCALER,sizeof(float));
}

//constant getter
const float* get_TIME_SCALER(){
    return &h_env_TIME_SCALER;
}



//constant setter
void set_STEER_WEIGHT(float* h_STEER_WEIGHT){
    gpuErrchk(cudaMemcpyToSymbol(STEER_WEIGHT, h_STEER_WEIGHT, sizeof(float)));
    memcpy(&h_env_STEER_WEIGHT, h_STEER_WEIGHT,sizeof(float));
}

//constant getter
const float* get_STEER_WEIGHT(){
    return &h_env_STEER_WEIGHT;
}



//constant setter
void set_AVOID_WEIGHT(float* h_AVOID_WEIGHT){
    gpuErrchk(cudaMemcpyToSymbol(AVOID_WEIGHT, h_AVOID_WEIGHT, sizeof(float)));
    memcpy(&h_env_AVOID_WEIGHT, h_AVOID_WEIGHT,sizeof(float));
}

//constant getter
const float* get_AVOID_WEIGHT(){
    return &h_env_AVOID_WEIGHT;
}



//constant setter
void set_COLLISION_WEIGHT(float* h_COLLISION_WEIGHT){
    gpuErrchk(cudaMemcpyToSymbol(COLLISION_WEIGHT, h_COLLISION_WEIGHT, sizeof(float)));
    memcpy(&h_env_COLLISION_WEIGHT, h_COLLISION_WEIGHT,sizeof(float));
}

//constant getter
const float* get_COLLISION_WEIGHT(){
    return &h_env_COLLISION_WEIGHT;
}



//constant setter
void set_GOAL_WEIGHT(float* h_GOAL_WEIGHT){
    gpuErrchk(cudaMemcpyToSymbol(GOAL_WEIGHT, h_GOAL_WEIGHT, sizeof(float)));
    memcpy(&h_env_GOAL_WEIGHT, h_GOAL_WEIGHT,sizeof(float));
}

//constant getter
const float* get_GOAL_WEIGHT(){
    return &h_env_GOAL_WEIGHT;
}




/* Agent data access functions*/

    
int get_agent_agent_MAX_count(){
    return xmachine_memory_agent_MAX;
}


int get_agent_agent_default_count(){
	//continuous agent
	return h_xmachine_memory_agent_default_count;
	
}

xmachine_memory_agent_list* get_device_agent_default_agents(){
	return d_agents_default;
}

xmachine_memory_agent_list* get_host_agent_default_agents(){
	return h_agents_default;
}

    
int get_agent_medic_MAX_count(){
    return xmachine_memory_medic_MAX;
}


int get_agent_medic_default2_count(){
	//continuous agent
	return h_xmachine_memory_medic_default2_count;
	
}

xmachine_memory_medic_list* get_device_medic_default2_agents(){
	return d_medics_default2;
}

xmachine_memory_medic_list* get_host_medic_default2_agents(){
	return h_medics_default2;
}

    
int get_agent_receptionist_MAX_count(){
    return xmachine_memory_receptionist_MAX;
}


int get_agent_receptionist_defaultReceptionist_count(){
	//continuous agent
	return h_xmachine_memory_receptionist_defaultReceptionist_count;
	
}

xmachine_memory_receptionist_list* get_device_receptionist_defaultReceptionist_agents(){
	return d_receptionists_defaultReceptionist;
}

xmachine_memory_receptionist_list* get_host_receptionist_defaultReceptionist_agents(){
	return h_receptionists_defaultReceptionist;
}

    
int get_agent_chair_admin_MAX_count(){
    return xmachine_memory_chair_admin_MAX;
}


int get_agent_chair_admin_defaultAdmin_count(){
	//continuous agent
	return h_xmachine_memory_chair_admin_defaultAdmin_count;
	
}

xmachine_memory_chair_admin_list* get_device_chair_admin_defaultAdmin_agents(){
	return d_chair_admins_defaultAdmin;
}

xmachine_memory_chair_admin_list* get_host_chair_admin_defaultAdmin_agents(){
	return h_chair_admins_defaultAdmin;
}

    
int get_agent_navmap_MAX_count(){
    return xmachine_memory_navmap_MAX;
}


int get_agent_navmap_static_count(){
	//discrete agent 
	return xmachine_memory_navmap_MAX;
}

xmachine_memory_navmap_list* get_device_navmap_static_agents(){
	return d_navmaps_static;
}

xmachine_memory_navmap_list* get_host_navmap_static_agents(){
	return h_navmaps_static;
}

int get_navmap_population_width(){
  return h_xmachine_memory_navmap_pop_width;
}



/* Host based access of agent variables*/

/** unsigned int get_agent_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_agent_default_variable_id(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->id,
                    d_agents_default->id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_agent_default_variable_x(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->x,
                    d_agents_default->x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_agent_default_variable_y(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->y,
                    d_agents_default->y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_velx(unsigned int index)
 * Gets the value of the velx variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable velx
 */
__host__ float get_agent_default_variable_velx(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_velx_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->velx,
                    d_agents_default->velx,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_velx_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->velx[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access velx for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_vely(unsigned int index)
 * Gets the value of the vely variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable vely
 */
__host__ float get_agent_default_variable_vely(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_vely_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->vely,
                    d_agents_default->vely,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_vely_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->vely[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access vely for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_steer_x(unsigned int index)
 * Gets the value of the steer_x variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable steer_x
 */
__host__ float get_agent_default_variable_steer_x(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_steer_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->steer_x,
                    d_agents_default->steer_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_steer_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->steer_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access steer_x for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_steer_y(unsigned int index)
 * Gets the value of the steer_y variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable steer_y
 */
__host__ float get_agent_default_variable_steer_y(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_steer_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->steer_y,
                    d_agents_default->steer_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_steer_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->steer_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access steer_y for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_height(unsigned int index)
 * Gets the value of the height variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable height
 */
__host__ float get_agent_default_variable_height(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_height_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->height,
                    d_agents_default->height,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_height_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->height[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access height for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_default_variable_exit_no(unsigned int index)
 * Gets the value of the exit_no variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit_no
 */
__host__ int get_agent_default_variable_exit_no(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_exit_no_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->exit_no,
                    d_agents_default->exit_no,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_exit_no_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->exit_no[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit_no for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_speed(unsigned int index)
 * Gets the value of the speed variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable speed
 */
__host__ float get_agent_default_variable_speed(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_speed_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->speed,
                    d_agents_default->speed,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_speed_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->speed[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access speed for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_default_variable_lod(unsigned int index)
 * Gets the value of the lod variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lod
 */
__host__ int get_agent_default_variable_lod(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_lod_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->lod,
                    d_agents_default->lod,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_lod_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->lod[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access lod for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_agent_default_variable_animate(unsigned int index)
 * Gets the value of the animate variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable animate
 */
__host__ float get_agent_default_variable_animate(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_animate_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->animate,
                    d_agents_default->animate,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_animate_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->animate[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access animate for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_default_variable_animate_dir(unsigned int index)
 * Gets the value of the animate_dir variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable animate_dir
 */
__host__ int get_agent_default_variable_animate_dir(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_animate_dir_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->animate_dir,
                    d_agents_default->animate_dir,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_animate_dir_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->animate_dir[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access animate_dir for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_default_variable_estado(unsigned int index)
 * Gets the value of the estado variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable estado
 */
__host__ int get_agent_default_variable_estado(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_estado_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->estado,
                    d_agents_default->estado,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_estado_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->estado[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access estado for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_default_variable_tick(unsigned int index)
 * Gets the value of the tick variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable tick
 */
__host__ int get_agent_default_variable_tick(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_tick_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->tick,
                    d_agents_default->tick,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_tick_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->tick[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access tick for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_agent_default_variable_estado_movimiento(unsigned int index)
 * Gets the value of the estado_movimiento variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable estado_movimiento
 */
__host__ unsigned int get_agent_default_variable_estado_movimiento(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_estado_movimiento_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->estado_movimiento,
                    d_agents_default->estado_movimiento,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_estado_movimiento_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->estado_movimiento[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access estado_movimiento for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_agent_default_variable_go_to_x(unsigned int index)
 * Gets the value of the go_to_x variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable go_to_x
 */
__host__ unsigned int get_agent_default_variable_go_to_x(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_go_to_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->go_to_x,
                    d_agents_default->go_to_x,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_go_to_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->go_to_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access go_to_x for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_agent_default_variable_go_to_y(unsigned int index)
 * Gets the value of the go_to_y variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable go_to_y
 */
__host__ unsigned int get_agent_default_variable_go_to_y(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_go_to_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->go_to_y,
                    d_agents_default->go_to_y,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_go_to_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->go_to_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access go_to_y for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_medic_default2_variable_x(unsigned int index)
 * Gets the value of the x variable of an medic agent in the default2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_medic_default2_variable_x(unsigned int index){
    unsigned int count = get_agent_medic_default2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_medics_default2_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_medics_default2->x,
                    d_medics_default2->x,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_medics_default2_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_medics_default2->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of medic_default2. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_medic_default2_variable_y(unsigned int index)
 * Gets the value of the y variable of an medic agent in the default2 state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_medic_default2_variable_y(unsigned int index){
    unsigned int count = get_agent_medic_default2_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_medics_default2_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_medics_default2->y,
                    d_medics_default2->y,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_medics_default2_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_medics_default2->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of medic_default2. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_receptionist_defaultReceptionist_variable_x(unsigned int index)
 * Gets the value of the x variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_receptionist_defaultReceptionist_variable_x(unsigned int index){
    unsigned int count = get_agent_receptionist_defaultReceptionist_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_receptionists_defaultReceptionist_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_receptionists_defaultReceptionist->x,
                    d_receptionists_defaultReceptionist->x,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_receptionists_defaultReceptionist_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_receptionists_defaultReceptionist->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of receptionist_defaultReceptionist. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_receptionist_defaultReceptionist_variable_y(unsigned int index)
 * Gets the value of the y variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_receptionist_defaultReceptionist_variable_y(unsigned int index){
    unsigned int count = get_agent_receptionist_defaultReceptionist_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_receptionists_defaultReceptionist_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_receptionists_defaultReceptionist->y,
                    d_receptionists_defaultReceptionist->y,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_receptionists_defaultReceptionist_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_receptionists_defaultReceptionist->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of receptionist_defaultReceptionist. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_receptionist_defaultReceptionist_variable_colaPacientes(unsigned int index, unsigned int element)
 * Gets the element-th value of the colaPacientes variable array of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable colaPacientes
 */
__host__ unsigned int get_receptionist_defaultReceptionist_variable_colaPacientes(unsigned int index, unsigned int element){
    unsigned int count = get_agent_receptionist_defaultReceptionist_count();
    unsigned int numElements = 2000;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_receptionists_defaultReceptionist_variable_colaPacientes_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_receptionists_defaultReceptionist->colaPacientes + (e * xmachine_memory_receptionist_MAX),
                        d_receptionists_defaultReceptionist->colaPacientes + (e * xmachine_memory_receptionist_MAX), 
                        count * sizeof(unsigned int), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_receptionists_defaultReceptionist_variable_colaPacientes_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_receptionists_defaultReceptionist->colaPacientes[index + (element * xmachine_memory_receptionist_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of colaPacientes for the %u th member of receptionist_defaultReceptionist. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_receptionist_defaultReceptionist_variable_front(unsigned int index)
 * Gets the value of the front variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable front
 */
__host__ unsigned int get_receptionist_defaultReceptionist_variable_front(unsigned int index){
    unsigned int count = get_agent_receptionist_defaultReceptionist_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_receptionists_defaultReceptionist_variable_front_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_receptionists_defaultReceptionist->front,
                    d_receptionists_defaultReceptionist->front,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_receptionists_defaultReceptionist_variable_front_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_receptionists_defaultReceptionist->front[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access front for the %u th member of receptionist_defaultReceptionist. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_receptionist_defaultReceptionist_variable_rear(unsigned int index)
 * Gets the value of the rear variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable rear
 */
__host__ unsigned int get_receptionist_defaultReceptionist_variable_rear(unsigned int index){
    unsigned int count = get_agent_receptionist_defaultReceptionist_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_receptionists_defaultReceptionist_variable_rear_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_receptionists_defaultReceptionist->rear,
                    d_receptionists_defaultReceptionist->rear,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_receptionists_defaultReceptionist_variable_rear_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_receptionists_defaultReceptionist->rear[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access rear for the %u th member of receptionist_defaultReceptionist. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_receptionist_defaultReceptionist_variable_size(unsigned int index)
 * Gets the value of the size variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable size
 */
__host__ unsigned int get_receptionist_defaultReceptionist_variable_size(unsigned int index){
    unsigned int count = get_agent_receptionist_defaultReceptionist_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_receptionists_defaultReceptionist_variable_size_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_receptionists_defaultReceptionist->size,
                    d_receptionists_defaultReceptionist->size,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_receptionists_defaultReceptionist_variable_size_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_receptionists_defaultReceptionist->size[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access size for the %u th member of receptionist_defaultReceptionist. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_receptionist_defaultReceptionist_variable_tick(unsigned int index)
 * Gets the value of the tick variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable tick
 */
__host__ unsigned int get_receptionist_defaultReceptionist_variable_tick(unsigned int index){
    unsigned int count = get_agent_receptionist_defaultReceptionist_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_receptionists_defaultReceptionist_variable_tick_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_receptionists_defaultReceptionist->tick,
                    d_receptionists_defaultReceptionist->tick,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_receptionists_defaultReceptionist_variable_tick_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_receptionists_defaultReceptionist->tick[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access tick for the %u th member of receptionist_defaultReceptionist. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_receptionist_defaultReceptionist_variable_current_patient(unsigned int index)
 * Gets the value of the current_patient variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_patient
 */
__host__ int get_receptionist_defaultReceptionist_variable_current_patient(unsigned int index){
    unsigned int count = get_agent_receptionist_defaultReceptionist_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_receptionists_defaultReceptionist_variable_current_patient_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_receptionists_defaultReceptionist->current_patient,
                    d_receptionists_defaultReceptionist->current_patient,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_receptionists_defaultReceptionist_variable_current_patient_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_receptionists_defaultReceptionist->current_patient[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access current_patient for the %u th member of receptionist_defaultReceptionist. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_receptionist_defaultReceptionist_variable_attend_patient(unsigned int index)
 * Gets the value of the attend_patient variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable attend_patient
 */
__host__ int get_receptionist_defaultReceptionist_variable_attend_patient(unsigned int index){
    unsigned int count = get_agent_receptionist_defaultReceptionist_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_receptionists_defaultReceptionist_variable_attend_patient_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_receptionists_defaultReceptionist->attend_patient,
                    d_receptionists_defaultReceptionist->attend_patient,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_receptionists_defaultReceptionist_variable_attend_patient_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_receptionists_defaultReceptionist->attend_patient[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access attend_patient for the %u th member of receptionist_defaultReceptionist. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_receptionist_defaultReceptionist_variable_estado(unsigned int index)
 * Gets the value of the estado variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable estado
 */
__host__ int get_receptionist_defaultReceptionist_variable_estado(unsigned int index){
    unsigned int count = get_agent_receptionist_defaultReceptionist_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_receptionists_defaultReceptionist_variable_estado_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_receptionists_defaultReceptionist->estado,
                    d_receptionists_defaultReceptionist->estado,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_receptionists_defaultReceptionist_variable_estado_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_receptionists_defaultReceptionist->estado[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access estado for the %u th member of receptionist_defaultReceptionist. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_chair_admin_defaultAdmin_variable_id(unsigned int index)
 * Gets the value of the id variable of an chair_admin agent in the defaultAdmin state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_chair_admin_defaultAdmin_variable_id(unsigned int index){
    unsigned int count = get_agent_chair_admin_defaultAdmin_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_chair_admins_defaultAdmin_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_chair_admins_defaultAdmin->id,
                    d_chair_admins_defaultAdmin->id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_chair_admins_defaultAdmin_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_chair_admins_defaultAdmin->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of chair_admin_defaultAdmin. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_chair_admin_defaultAdmin_variable_chairArray(unsigned int index, unsigned int element)
 * Gets the element-th value of the chairArray variable array of an chair_admin agent in the defaultAdmin state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable chairArray
 */
__host__ unsigned int get_chair_admin_defaultAdmin_variable_chairArray(unsigned int index, unsigned int element){
    unsigned int count = get_agent_chair_admin_defaultAdmin_count();
    unsigned int numElements = 35;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_chair_admins_defaultAdmin_variable_chairArray_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_chair_admins_defaultAdmin->chairArray + (e * xmachine_memory_chair_admin_MAX),
                        d_chair_admins_defaultAdmin->chairArray + (e * xmachine_memory_chair_admin_MAX), 
                        count * sizeof(unsigned int), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_chair_admins_defaultAdmin_variable_chairArray_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_chair_admins_defaultAdmin->chairArray[index + (element * xmachine_memory_chair_admin_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of chairArray for the %u th member of chair_admin_defaultAdmin. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_navmap_static_variable_x(unsigned int index)
 * Gets the value of the x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_navmap_static_variable_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->x,
                    d_navmaps_static->x,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_navmap_static_variable_y(unsigned int index)
 * Gets the value of the y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_navmap_static_variable_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->y,
                    d_navmaps_static->y,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_navmap_static_variable_exit_no(unsigned int index)
 * Gets the value of the exit_no variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit_no
 */
__host__ int get_navmap_static_variable_exit_no(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit_no_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit_no,
                    d_navmaps_static->exit_no,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit_no_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit_no[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit_no for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_height(unsigned int index)
 * Gets the value of the height variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable height
 */
__host__ float get_navmap_static_variable_height(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_height_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->height,
                    d_navmaps_static->height,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_height_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->height[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access height for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_collision_x(unsigned int index)
 * Gets the value of the collision_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable collision_x
 */
__host__ float get_navmap_static_variable_collision_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_collision_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->collision_x,
                    d_navmaps_static->collision_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_collision_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->collision_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access collision_x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_collision_y(unsigned int index)
 * Gets the value of the collision_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable collision_y
 */
__host__ float get_navmap_static_variable_collision_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_collision_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->collision_y,
                    d_navmaps_static->collision_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_collision_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->collision_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access collision_y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit0_x(unsigned int index)
 * Gets the value of the exit0_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit0_x
 */
__host__ float get_navmap_static_variable_exit0_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit0_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit0_x,
                    d_navmaps_static->exit0_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit0_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit0_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit0_x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit0_y(unsigned int index)
 * Gets the value of the exit0_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit0_y
 */
__host__ float get_navmap_static_variable_exit0_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit0_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit0_y,
                    d_navmaps_static->exit0_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit0_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit0_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit0_y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit1_x(unsigned int index)
 * Gets the value of the exit1_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit1_x
 */
__host__ float get_navmap_static_variable_exit1_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit1_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit1_x,
                    d_navmaps_static->exit1_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit1_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit1_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit1_x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit1_y(unsigned int index)
 * Gets the value of the exit1_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit1_y
 */
__host__ float get_navmap_static_variable_exit1_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit1_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit1_y,
                    d_navmaps_static->exit1_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit1_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit1_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit1_y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit2_x(unsigned int index)
 * Gets the value of the exit2_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit2_x
 */
__host__ float get_navmap_static_variable_exit2_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit2_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit2_x,
                    d_navmaps_static->exit2_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit2_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit2_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit2_x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit2_y(unsigned int index)
 * Gets the value of the exit2_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit2_y
 */
__host__ float get_navmap_static_variable_exit2_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit2_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit2_y,
                    d_navmaps_static->exit2_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit2_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit2_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit2_y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit3_x(unsigned int index)
 * Gets the value of the exit3_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit3_x
 */
__host__ float get_navmap_static_variable_exit3_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit3_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit3_x,
                    d_navmaps_static->exit3_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit3_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit3_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit3_x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit3_y(unsigned int index)
 * Gets the value of the exit3_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit3_y
 */
__host__ float get_navmap_static_variable_exit3_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit3_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit3_y,
                    d_navmaps_static->exit3_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit3_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit3_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit3_y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit4_x(unsigned int index)
 * Gets the value of the exit4_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit4_x
 */
__host__ float get_navmap_static_variable_exit4_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit4_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit4_x,
                    d_navmaps_static->exit4_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit4_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit4_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit4_x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit4_y(unsigned int index)
 * Gets the value of the exit4_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit4_y
 */
__host__ float get_navmap_static_variable_exit4_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit4_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit4_y,
                    d_navmaps_static->exit4_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit4_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit4_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit4_y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit5_x(unsigned int index)
 * Gets the value of the exit5_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit5_x
 */
__host__ float get_navmap_static_variable_exit5_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit5_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit5_x,
                    d_navmaps_static->exit5_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit5_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit5_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit5_x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit5_y(unsigned int index)
 * Gets the value of the exit5_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit5_y
 */
__host__ float get_navmap_static_variable_exit5_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit5_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit5_y,
                    d_navmaps_static->exit5_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit5_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit5_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit5_y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit6_x(unsigned int index)
 * Gets the value of the exit6_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit6_x
 */
__host__ float get_navmap_static_variable_exit6_x(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit6_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit6_x,
                    d_navmaps_static->exit6_x,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit6_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit6_x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit6_x for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** float get_navmap_static_variable_exit6_y(unsigned int index)
 * Gets the value of the exit6_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit6_y
 */
__host__ float get_navmap_static_variable_exit6_y(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_exit6_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->exit6_y,
                    d_navmaps_static->exit6_y,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_exit6_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->exit6_y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access exit6_y for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_navmap_static_variable_cant_generados(unsigned int index)
 * Gets the value of the cant_generados variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable cant_generados
 */
__host__ unsigned int get_navmap_static_variable_cant_generados(unsigned int index){
    unsigned int count = get_agent_navmap_static_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_navmaps_static_variable_cant_generados_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_navmaps_static->cant_generados,
                    d_navmaps_static->cant_generados,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_navmaps_static_variable_cant_generados_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_navmaps_static->cant_generados[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access cant_generados for the %u th member of navmap_static. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}



/* Host based agent creation functions */
// These are only available for continuous agents.



/* copy_single_xmachine_memory_agent_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_agent_hostToDevice(xmachine_memory_agent_list * d_dst, xmachine_memory_agent * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, &h_agent->x, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, &h_agent->y, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->velx, &h_agent->velx, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->vely, &h_agent->vely, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->steer_x, &h_agent->steer_x, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->steer_y, &h_agent->steer_y, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->height, &h_agent->height, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->exit_no, &h_agent->exit_no, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->speed, &h_agent->speed, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lod, &h_agent->lod, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->animate, &h_agent->animate, sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->animate_dir, &h_agent->animate_dir, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->estado, &h_agent->estado, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->tick, &h_agent->tick, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->estado_movimiento, &h_agent->estado_movimiento, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->go_to_x, &h_agent->go_to_x, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->go_to_y, &h_agent->go_to_y, sizeof(unsigned int), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_agent_hostToDevice(xmachine_memory_agent_list * d_dst, xmachine_memory_agent_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, h_src->x, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, h_src->y, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->velx, h_src->velx, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->vely, h_src->vely, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->steer_x, h_src->steer_x, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->steer_y, h_src->steer_y, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->height, h_src->height, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->exit_no, h_src->exit_no, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->speed, h_src->speed, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->lod, h_src->lod, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->animate, h_src->animate, count * sizeof(float), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->animate_dir, h_src->animate_dir, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->estado, h_src->estado, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->tick, h_src->tick, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->estado_movimiento, h_src->estado_movimiento, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->go_to_x, h_src->go_to_x, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->go_to_y, h_src->go_to_y, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_medic_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_medic_hostToDevice(xmachine_memory_medic_list * d_dst, xmachine_memory_medic * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->x, &h_agent->x, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, &h_agent->y, sizeof(int), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_medic_hostToDevice(xmachine_memory_medic_list * d_dst, xmachine_memory_medic_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->x, h_src->x, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, h_src->y, count * sizeof(int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_receptionist_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_receptionist_hostToDevice(xmachine_memory_receptionist_list * d_dst, xmachine_memory_receptionist * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->x, &h_agent->x, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, &h_agent->y, sizeof(int), cudaMemcpyHostToDevice));
 
	for(unsigned int i = 0; i < 2000; i++){
		gpuErrchk(cudaMemcpy(d_dst->colaPacientes + (i * xmachine_memory_receptionist_MAX), h_agent->colaPacientes + i, sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
 
		gpuErrchk(cudaMemcpy(d_dst->front, &h_agent->front, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->rear, &h_agent->rear, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->size, &h_agent->size, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->tick, &h_agent->tick, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->current_patient, &h_agent->current_patient, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->attend_patient, &h_agent->attend_patient, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->estado, &h_agent->estado, sizeof(int), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_receptionist_hostToDevice(xmachine_memory_receptionist_list * d_dst, xmachine_memory_receptionist_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->x, h_src->x, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, h_src->y, count * sizeof(int), cudaMemcpyHostToDevice));
 
		for(unsigned int i = 0; i < 2000; i++){
			gpuErrchk(cudaMemcpy(d_dst->colaPacientes + (i * xmachine_memory_receptionist_MAX), h_src->colaPacientes + (i * xmachine_memory_receptionist_MAX), count * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }

 
		gpuErrchk(cudaMemcpy(d_dst->front, h_src->front, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->rear, h_src->rear, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->size, h_src->size, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->tick, h_src->tick, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->current_patient, h_src->current_patient, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->attend_patient, h_src->attend_patient, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->estado, h_src->estado, count * sizeof(int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_chair_admin_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_chair_admin_hostToDevice(xmachine_memory_chair_admin_list * d_dst, xmachine_memory_chair_admin * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
	for(unsigned int i = 0; i < 35; i++){
		gpuErrchk(cudaMemcpy(d_dst->chairArray + (i * xmachine_memory_chair_admin_MAX), h_agent->chairArray + i, sizeof(unsigned int), cudaMemcpyHostToDevice));
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
void copy_partial_xmachine_memory_chair_admin_hostToDevice(xmachine_memory_chair_admin_list * d_dst, xmachine_memory_chair_admin_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		for(unsigned int i = 0; i < 35; i++){
			gpuErrchk(cudaMemcpy(d_dst->chairArray + (i * xmachine_memory_chair_admin_MAX), h_src->chairArray + (i * xmachine_memory_chair_admin_MAX), count * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }


    }
}

xmachine_memory_agent* h_allocate_agent_agent(){
	xmachine_memory_agent* agent = (xmachine_memory_agent*)malloc(sizeof(xmachine_memory_agent));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_agent));

    agent->estado_movimiento = 0;

	return agent;
}
void h_free_agent_agent(xmachine_memory_agent** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_agent** h_allocate_agent_agent_array(unsigned int count){
	xmachine_memory_agent ** agents = (xmachine_memory_agent**)malloc(count * sizeof(xmachine_memory_agent*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_agent();
	}
	return agents;
}
void h_free_agent_agent_array(xmachine_memory_agent*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_agent(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_agent_AoS_to_SoA(xmachine_memory_agent_list * dst, xmachine_memory_agent** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->x[i] = src[i]->x;
			 
			dst->y[i] = src[i]->y;
			 
			dst->velx[i] = src[i]->velx;
			 
			dst->vely[i] = src[i]->vely;
			 
			dst->steer_x[i] = src[i]->steer_x;
			 
			dst->steer_y[i] = src[i]->steer_y;
			 
			dst->height[i] = src[i]->height;
			 
			dst->exit_no[i] = src[i]->exit_no;
			 
			dst->speed[i] = src[i]->speed;
			 
			dst->lod[i] = src[i]->lod;
			 
			dst->animate[i] = src[i]->animate;
			 
			dst->animate_dir[i] = src[i]->animate_dir;
			 
			dst->estado[i] = src[i]->estado;
			 
			dst->tick[i] = src[i]->tick;
			 
			dst->estado_movimiento[i] = src[i]->estado_movimiento;
			 
			dst->go_to_x[i] = src[i]->go_to_x;
			 
			dst->go_to_y[i] = src[i]->go_to_y;
			
		}
	}
}


void h_add_agent_agent_default(xmachine_memory_agent* agent){
	if (h_xmachine_memory_agent_count + 1 > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of agent agents in state default will be exceeded by h_add_agent_agent_default\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_agent_hostToDevice(d_agents_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_agent_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_agent_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_agents_default, d_agents_new, h_xmachine_memory_agent_default_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_agent_default_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_agents_default_variable_id_data_iteration = 0;
    h_agents_default_variable_x_data_iteration = 0;
    h_agents_default_variable_y_data_iteration = 0;
    h_agents_default_variable_velx_data_iteration = 0;
    h_agents_default_variable_vely_data_iteration = 0;
    h_agents_default_variable_steer_x_data_iteration = 0;
    h_agents_default_variable_steer_y_data_iteration = 0;
    h_agents_default_variable_height_data_iteration = 0;
    h_agents_default_variable_exit_no_data_iteration = 0;
    h_agents_default_variable_speed_data_iteration = 0;
    h_agents_default_variable_lod_data_iteration = 0;
    h_agents_default_variable_animate_data_iteration = 0;
    h_agents_default_variable_animate_dir_data_iteration = 0;
    h_agents_default_variable_estado_data_iteration = 0;
    h_agents_default_variable_tick_data_iteration = 0;
    h_agents_default_variable_estado_movimiento_data_iteration = 0;
    h_agents_default_variable_go_to_x_data_iteration = 0;
    h_agents_default_variable_go_to_y_data_iteration = 0;
    

}
void h_add_agents_agent_default(xmachine_memory_agent** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_agent_count + count > xmachine_memory_agent_MAX){
			printf("Error: Buffer size of agent agents in state default will be exceeded by h_add_agents_agent_default\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_agent_AoS_to_SoA(h_agents_default, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_agent_hostToDevice(d_agents_new, h_agents_default, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_agent_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_agent_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_agents_default, d_agents_new, h_xmachine_memory_agent_default_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_agent_default_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_agents_default_variable_id_data_iteration = 0;
        h_agents_default_variable_x_data_iteration = 0;
        h_agents_default_variable_y_data_iteration = 0;
        h_agents_default_variable_velx_data_iteration = 0;
        h_agents_default_variable_vely_data_iteration = 0;
        h_agents_default_variable_steer_x_data_iteration = 0;
        h_agents_default_variable_steer_y_data_iteration = 0;
        h_agents_default_variable_height_data_iteration = 0;
        h_agents_default_variable_exit_no_data_iteration = 0;
        h_agents_default_variable_speed_data_iteration = 0;
        h_agents_default_variable_lod_data_iteration = 0;
        h_agents_default_variable_animate_data_iteration = 0;
        h_agents_default_variable_animate_dir_data_iteration = 0;
        h_agents_default_variable_estado_data_iteration = 0;
        h_agents_default_variable_tick_data_iteration = 0;
        h_agents_default_variable_estado_movimiento_data_iteration = 0;
        h_agents_default_variable_go_to_x_data_iteration = 0;
        h_agents_default_variable_go_to_y_data_iteration = 0;
        

	}
}

xmachine_memory_medic* h_allocate_agent_medic(){
	xmachine_memory_medic* agent = (xmachine_memory_medic*)malloc(sizeof(xmachine_memory_medic));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_medic));

	return agent;
}
void h_free_agent_medic(xmachine_memory_medic** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_medic** h_allocate_agent_medic_array(unsigned int count){
	xmachine_memory_medic ** agents = (xmachine_memory_medic**)malloc(count * sizeof(xmachine_memory_medic*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_medic();
	}
	return agents;
}
void h_free_agent_medic_array(xmachine_memory_medic*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_medic(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_medic_AoS_to_SoA(xmachine_memory_medic_list * dst, xmachine_memory_medic** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->x[i] = src[i]->x;
			 
			dst->y[i] = src[i]->y;
			
		}
	}
}


void h_add_agent_medic_default2(xmachine_memory_medic* agent){
	if (h_xmachine_memory_medic_count + 1 > xmachine_memory_medic_MAX){
		printf("Error: Buffer size of medic agents in state default2 will be exceeded by h_add_agent_medic_default2\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_medic_hostToDevice(d_medics_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_medic_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_medic_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_medics_default2, d_medics_new, h_xmachine_memory_medic_default2_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_medic_default2_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_medic_default2_count, &h_xmachine_memory_medic_default2_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_medics_default2_variable_x_data_iteration = 0;
    h_medics_default2_variable_y_data_iteration = 0;
    

}
void h_add_agents_medic_default2(xmachine_memory_medic** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_medic_count + count > xmachine_memory_medic_MAX){
			printf("Error: Buffer size of medic agents in state default2 will be exceeded by h_add_agents_medic_default2\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_medic_AoS_to_SoA(h_medics_default2, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_medic_hostToDevice(d_medics_new, h_medics_default2, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_medic_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_medic_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_medics_default2, d_medics_new, h_xmachine_memory_medic_default2_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_medic_default2_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_medic_default2_count, &h_xmachine_memory_medic_default2_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_medics_default2_variable_x_data_iteration = 0;
        h_medics_default2_variable_y_data_iteration = 0;
        

	}
}

xmachine_memory_receptionist* h_allocate_agent_receptionist(){
	xmachine_memory_receptionist* agent = (xmachine_memory_receptionist*)malloc(sizeof(xmachine_memory_receptionist));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_receptionist));

    agent->x = 0.093750;

    agent->y = -0.375000;
	// Agent variable arrays must be allocated
    agent->colaPacientes = (unsigned int*)malloc(2000 * sizeof(unsigned int));
	
    // If there is no default value, memset to 0.
    memset(agent->colaPacientes, 0, sizeof(unsigned int)*2000);
    agent->tick = 0;

    agent->current_patient = -1;

    agent->attend_patient = 0;

    agent->estado = 0;

	return agent;
}
void h_free_agent_receptionist(xmachine_memory_receptionist** agent){

    free((*agent)->colaPacientes);
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_receptionist** h_allocate_agent_receptionist_array(unsigned int count){
	xmachine_memory_receptionist ** agents = (xmachine_memory_receptionist**)malloc(count * sizeof(xmachine_memory_receptionist*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_receptionist();
	}
	return agents;
}
void h_free_agent_receptionist_array(xmachine_memory_receptionist*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_receptionist(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_receptionist_AoS_to_SoA(xmachine_memory_receptionist_list * dst, xmachine_memory_receptionist** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->x[i] = src[i]->x;
			 
			dst->y[i] = src[i]->y;
			 
			for(unsigned int j = 0; j < 2000; j++){
				dst->colaPacientes[(j * xmachine_memory_receptionist_MAX) + i] = src[i]->colaPacientes[j];
			}
			 
			dst->front[i] = src[i]->front;
			 
			dst->rear[i] = src[i]->rear;
			 
			dst->size[i] = src[i]->size;
			 
			dst->tick[i] = src[i]->tick;
			 
			dst->current_patient[i] = src[i]->current_patient;
			 
			dst->attend_patient[i] = src[i]->attend_patient;
			 
			dst->estado[i] = src[i]->estado;
			
		}
	}
}


void h_add_agent_receptionist_defaultReceptionist(xmachine_memory_receptionist* agent){
	if (h_xmachine_memory_receptionist_count + 1 > xmachine_memory_receptionist_MAX){
		printf("Error: Buffer size of receptionist agents in state defaultReceptionist will be exceeded by h_add_agent_receptionist_defaultReceptionist\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_receptionist_hostToDevice(d_receptionists_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_receptionist_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_receptionist_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_receptionists_defaultReceptionist, d_receptionists_new, h_xmachine_memory_receptionist_defaultReceptionist_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_receptionist_defaultReceptionist_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_receptionist_defaultReceptionist_count, &h_xmachine_memory_receptionist_defaultReceptionist_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_receptionists_defaultReceptionist_variable_x_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_y_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_colaPacientes_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_front_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_rear_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_size_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_tick_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_current_patient_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_attend_patient_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_estado_data_iteration = 0;
    

}
void h_add_agents_receptionist_defaultReceptionist(xmachine_memory_receptionist** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_receptionist_count + count > xmachine_memory_receptionist_MAX){
			printf("Error: Buffer size of receptionist agents in state defaultReceptionist will be exceeded by h_add_agents_receptionist_defaultReceptionist\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_receptionist_AoS_to_SoA(h_receptionists_defaultReceptionist, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_receptionist_hostToDevice(d_receptionists_new, h_receptionists_defaultReceptionist, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_receptionist_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_receptionist_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_receptionists_defaultReceptionist, d_receptionists_new, h_xmachine_memory_receptionist_defaultReceptionist_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_receptionist_defaultReceptionist_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_receptionist_defaultReceptionist_count, &h_xmachine_memory_receptionist_defaultReceptionist_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_receptionists_defaultReceptionist_variable_x_data_iteration = 0;
        h_receptionists_defaultReceptionist_variable_y_data_iteration = 0;
        h_receptionists_defaultReceptionist_variable_colaPacientes_data_iteration = 0;
        h_receptionists_defaultReceptionist_variable_front_data_iteration = 0;
        h_receptionists_defaultReceptionist_variable_rear_data_iteration = 0;
        h_receptionists_defaultReceptionist_variable_size_data_iteration = 0;
        h_receptionists_defaultReceptionist_variable_tick_data_iteration = 0;
        h_receptionists_defaultReceptionist_variable_current_patient_data_iteration = 0;
        h_receptionists_defaultReceptionist_variable_attend_patient_data_iteration = 0;
        h_receptionists_defaultReceptionist_variable_estado_data_iteration = 0;
        

	}
}

xmachine_memory_chair_admin* h_allocate_agent_chair_admin(){
	xmachine_memory_chair_admin* agent = (xmachine_memory_chair_admin*)malloc(sizeof(xmachine_memory_chair_admin));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_chair_admin));

    agent->id = 0;
	// Agent variable arrays must be allocated
    agent->chairArray = (unsigned int*)malloc(35 * sizeof(unsigned int));
	// If we have a default value, set each element correctly.
	for(unsigned int index = 0; index < 35; index++){
		agent->chairArray[index] = 0;
	}
	return agent;
}
void h_free_agent_chair_admin(xmachine_memory_chair_admin** agent){

    free((*agent)->chairArray);
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_chair_admin** h_allocate_agent_chair_admin_array(unsigned int count){
	xmachine_memory_chair_admin ** agents = (xmachine_memory_chair_admin**)malloc(count * sizeof(xmachine_memory_chair_admin*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_chair_admin();
	}
	return agents;
}
void h_free_agent_chair_admin_array(xmachine_memory_chair_admin*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_chair_admin(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_chair_admin_AoS_to_SoA(xmachine_memory_chair_admin_list * dst, xmachine_memory_chair_admin** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			for(unsigned int j = 0; j < 35; j++){
				dst->chairArray[(j * xmachine_memory_chair_admin_MAX) + i] = src[i]->chairArray[j];
			}
			
		}
	}
}


void h_add_agent_chair_admin_defaultAdmin(xmachine_memory_chair_admin* agent){
	if (h_xmachine_memory_chair_admin_count + 1 > xmachine_memory_chair_admin_MAX){
		printf("Error: Buffer size of chair_admin agents in state defaultAdmin will be exceeded by h_add_agent_chair_admin_defaultAdmin\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_chair_admin_hostToDevice(d_chair_admins_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_chair_admin_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_chair_admin_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_chair_admins_defaultAdmin, d_chair_admins_new, h_xmachine_memory_chair_admin_defaultAdmin_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_chair_admin_defaultAdmin_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_chair_admin_defaultAdmin_count, &h_xmachine_memory_chair_admin_defaultAdmin_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_chair_admins_defaultAdmin_variable_id_data_iteration = 0;
    h_chair_admins_defaultAdmin_variable_chairArray_data_iteration = 0;
    

}
void h_add_agents_chair_admin_defaultAdmin(xmachine_memory_chair_admin** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_chair_admin_count + count > xmachine_memory_chair_admin_MAX){
			printf("Error: Buffer size of chair_admin agents in state defaultAdmin will be exceeded by h_add_agents_chair_admin_defaultAdmin\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_chair_admin_AoS_to_SoA(h_chair_admins_defaultAdmin, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_chair_admin_hostToDevice(d_chair_admins_new, h_chair_admins_defaultAdmin, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_chair_admin_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_chair_admin_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_chair_admins_defaultAdmin, d_chair_admins_new, h_xmachine_memory_chair_admin_defaultAdmin_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_chair_admin_defaultAdmin_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_chair_admin_defaultAdmin_count, &h_xmachine_memory_chair_admin_defaultAdmin_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_chair_admins_defaultAdmin_variable_id_data_iteration = 0;
        h_chair_admins_defaultAdmin_variable_chairArray_data_iteration = 0;
        

	}
}


/*  Analytics Functions */

unsigned int reduce_agent_default_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->id),  thrust::device_pointer_cast(d_agents_default->id) + h_xmachine_memory_agent_default_count);
}

unsigned int count_agent_default_id_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_agents_default->id),  thrust::device_pointer_cast(d_agents_default->id) + h_xmachine_memory_agent_default_count, count_value);
}
unsigned int min_agent_default_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_agent_default_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->x),  thrust::device_pointer_cast(d_agents_default->x) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->y),  thrust::device_pointer_cast(d_agents_default->y) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_velx_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->velx),  thrust::device_pointer_cast(d_agents_default->velx) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_velx_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->velx);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_velx_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->velx);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_vely_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->vely),  thrust::device_pointer_cast(d_agents_default->vely) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_vely_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->vely);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_vely_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->vely);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_steer_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->steer_x),  thrust::device_pointer_cast(d_agents_default->steer_x) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_steer_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->steer_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_steer_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->steer_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_steer_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->steer_y),  thrust::device_pointer_cast(d_agents_default->steer_y) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_steer_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->steer_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_steer_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->steer_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_height_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->height),  thrust::device_pointer_cast(d_agents_default->height) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_height_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->height);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_height_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->height);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_default_exit_no_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->exit_no),  thrust::device_pointer_cast(d_agents_default->exit_no) + h_xmachine_memory_agent_default_count);
}

int count_agent_default_exit_no_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agents_default->exit_no),  thrust::device_pointer_cast(d_agents_default->exit_no) + h_xmachine_memory_agent_default_count, count_value);
}
int min_agent_default_exit_no_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->exit_no);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_default_exit_no_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->exit_no);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_speed_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->speed),  thrust::device_pointer_cast(d_agents_default->speed) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_speed_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->speed);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_speed_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->speed);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_default_lod_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->lod),  thrust::device_pointer_cast(d_agents_default->lod) + h_xmachine_memory_agent_default_count);
}

int count_agent_default_lod_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agents_default->lod),  thrust::device_pointer_cast(d_agents_default->lod) + h_xmachine_memory_agent_default_count, count_value);
}
int min_agent_default_lod_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->lod);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_default_lod_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->lod);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_agent_default_animate_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->animate),  thrust::device_pointer_cast(d_agents_default->animate) + h_xmachine_memory_agent_default_count);
}

float min_agent_default_animate_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->animate);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_agent_default_animate_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_agents_default->animate);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_default_animate_dir_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->animate_dir),  thrust::device_pointer_cast(d_agents_default->animate_dir) + h_xmachine_memory_agent_default_count);
}

int count_agent_default_animate_dir_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agents_default->animate_dir),  thrust::device_pointer_cast(d_agents_default->animate_dir) + h_xmachine_memory_agent_default_count, count_value);
}
int min_agent_default_animate_dir_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->animate_dir);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_default_animate_dir_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->animate_dir);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_default_estado_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->estado),  thrust::device_pointer_cast(d_agents_default->estado) + h_xmachine_memory_agent_default_count);
}

int count_agent_default_estado_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agents_default->estado),  thrust::device_pointer_cast(d_agents_default->estado) + h_xmachine_memory_agent_default_count, count_value);
}
int min_agent_default_estado_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->estado);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_default_estado_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->estado);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_default_tick_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->tick),  thrust::device_pointer_cast(d_agents_default->tick) + h_xmachine_memory_agent_default_count);
}

int count_agent_default_tick_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agents_default->tick),  thrust::device_pointer_cast(d_agents_default->tick) + h_xmachine_memory_agent_default_count, count_value);
}
int min_agent_default_tick_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->tick);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_default_tick_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->tick);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_agent_default_estado_movimiento_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->estado_movimiento),  thrust::device_pointer_cast(d_agents_default->estado_movimiento) + h_xmachine_memory_agent_default_count);
}

unsigned int count_agent_default_estado_movimiento_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_agents_default->estado_movimiento),  thrust::device_pointer_cast(d_agents_default->estado_movimiento) + h_xmachine_memory_agent_default_count, count_value);
}
unsigned int min_agent_default_estado_movimiento_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->estado_movimiento);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_agent_default_estado_movimiento_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->estado_movimiento);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_agent_default_go_to_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->go_to_x),  thrust::device_pointer_cast(d_agents_default->go_to_x) + h_xmachine_memory_agent_default_count);
}

unsigned int count_agent_default_go_to_x_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_agents_default->go_to_x),  thrust::device_pointer_cast(d_agents_default->go_to_x) + h_xmachine_memory_agent_default_count, count_value);
}
unsigned int min_agent_default_go_to_x_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->go_to_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_agent_default_go_to_x_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->go_to_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_agent_default_go_to_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->go_to_y),  thrust::device_pointer_cast(d_agents_default->go_to_y) + h_xmachine_memory_agent_default_count);
}

unsigned int count_agent_default_go_to_y_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_agents_default->go_to_y),  thrust::device_pointer_cast(d_agents_default->go_to_y) + h_xmachine_memory_agent_default_count, count_value);
}
unsigned int min_agent_default_go_to_y_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->go_to_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_agent_default_go_to_y_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->go_to_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_medic_default2_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_medics_default2->x),  thrust::device_pointer_cast(d_medics_default2->x) + h_xmachine_memory_medic_default2_count);
}

int count_medic_default2_x_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_medics_default2->x),  thrust::device_pointer_cast(d_medics_default2->x) + h_xmachine_memory_medic_default2_count, count_value);
}
int min_medic_default2_x_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_medics_default2->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_medic_default2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_medic_default2_x_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_medics_default2->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_medic_default2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_medic_default2_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_medics_default2->y),  thrust::device_pointer_cast(d_medics_default2->y) + h_xmachine_memory_medic_default2_count);
}

int count_medic_default2_y_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_medics_default2->y),  thrust::device_pointer_cast(d_medics_default2->y) + h_xmachine_memory_medic_default2_count, count_value);
}
int min_medic_default2_y_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_medics_default2->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_medic_default2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_medic_default2_y_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_medics_default2->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_medic_default2_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_receptionist_defaultReceptionist_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->x),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->x) + h_xmachine_memory_receptionist_defaultReceptionist_count);
}

int count_receptionist_defaultReceptionist_x_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->x),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->x) + h_xmachine_memory_receptionist_defaultReceptionist_count, count_value);
}
int min_receptionist_defaultReceptionist_x_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_receptionist_defaultReceptionist_x_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_receptionist_defaultReceptionist_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->y),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->y) + h_xmachine_memory_receptionist_defaultReceptionist_count);
}

int count_receptionist_defaultReceptionist_y_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->y),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->y) + h_xmachine_memory_receptionist_defaultReceptionist_count, count_value);
}
int min_receptionist_defaultReceptionist_y_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_receptionist_defaultReceptionist_y_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_receptionist_defaultReceptionist_front_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->front),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->front) + h_xmachine_memory_receptionist_defaultReceptionist_count);
}

unsigned int count_receptionist_defaultReceptionist_front_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->front),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->front) + h_xmachine_memory_receptionist_defaultReceptionist_count, count_value);
}
unsigned int min_receptionist_defaultReceptionist_front_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->front);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_receptionist_defaultReceptionist_front_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->front);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_receptionist_defaultReceptionist_rear_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->rear),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->rear) + h_xmachine_memory_receptionist_defaultReceptionist_count);
}

unsigned int count_receptionist_defaultReceptionist_rear_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->rear),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->rear) + h_xmachine_memory_receptionist_defaultReceptionist_count, count_value);
}
unsigned int min_receptionist_defaultReceptionist_rear_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->rear);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_receptionist_defaultReceptionist_rear_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->rear);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_receptionist_defaultReceptionist_size_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->size),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->size) + h_xmachine_memory_receptionist_defaultReceptionist_count);
}

unsigned int count_receptionist_defaultReceptionist_size_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->size),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->size) + h_xmachine_memory_receptionist_defaultReceptionist_count, count_value);
}
unsigned int min_receptionist_defaultReceptionist_size_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->size);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_receptionist_defaultReceptionist_size_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->size);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_receptionist_defaultReceptionist_tick_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->tick),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->tick) + h_xmachine_memory_receptionist_defaultReceptionist_count);
}

unsigned int count_receptionist_defaultReceptionist_tick_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->tick),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->tick) + h_xmachine_memory_receptionist_defaultReceptionist_count, count_value);
}
unsigned int min_receptionist_defaultReceptionist_tick_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->tick);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_receptionist_defaultReceptionist_tick_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->tick);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_receptionist_defaultReceptionist_current_patient_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->current_patient),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->current_patient) + h_xmachine_memory_receptionist_defaultReceptionist_count);
}

int count_receptionist_defaultReceptionist_current_patient_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->current_patient),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->current_patient) + h_xmachine_memory_receptionist_defaultReceptionist_count, count_value);
}
int min_receptionist_defaultReceptionist_current_patient_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->current_patient);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_receptionist_defaultReceptionist_current_patient_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->current_patient);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_receptionist_defaultReceptionist_attend_patient_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->attend_patient),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->attend_patient) + h_xmachine_memory_receptionist_defaultReceptionist_count);
}

int count_receptionist_defaultReceptionist_attend_patient_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->attend_patient),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->attend_patient) + h_xmachine_memory_receptionist_defaultReceptionist_count, count_value);
}
int min_receptionist_defaultReceptionist_attend_patient_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->attend_patient);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_receptionist_defaultReceptionist_attend_patient_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->attend_patient);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_receptionist_defaultReceptionist_estado_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->estado),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->estado) + h_xmachine_memory_receptionist_defaultReceptionist_count);
}

int count_receptionist_defaultReceptionist_estado_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_receptionists_defaultReceptionist->estado),  thrust::device_pointer_cast(d_receptionists_defaultReceptionist->estado) + h_xmachine_memory_receptionist_defaultReceptionist_count, count_value);
}
int min_receptionist_defaultReceptionist_estado_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->estado);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_receptionist_defaultReceptionist_estado_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_receptionists_defaultReceptionist->estado);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_receptionist_defaultReceptionist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_chair_admin_defaultAdmin_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_chair_admins_defaultAdmin->id),  thrust::device_pointer_cast(d_chair_admins_defaultAdmin->id) + h_xmachine_memory_chair_admin_defaultAdmin_count);
}

unsigned int count_chair_admin_defaultAdmin_id_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_chair_admins_defaultAdmin->id),  thrust::device_pointer_cast(d_chair_admins_defaultAdmin->id) + h_xmachine_memory_chair_admin_defaultAdmin_count, count_value);
}
unsigned int min_chair_admin_defaultAdmin_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_chair_admins_defaultAdmin->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_chair_admin_defaultAdmin_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_chair_admin_defaultAdmin_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_chair_admins_defaultAdmin->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_chair_admin_defaultAdmin_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_navmap_static_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->x),  thrust::device_pointer_cast(d_navmaps_static->x) + h_xmachine_memory_navmap_static_count);
}

int count_navmap_static_x_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_navmaps_static->x),  thrust::device_pointer_cast(d_navmaps_static->x) + h_xmachine_memory_navmap_static_count, count_value);
}
int min_navmap_static_x_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_navmap_static_x_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_navmap_static_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->y),  thrust::device_pointer_cast(d_navmaps_static->y) + h_xmachine_memory_navmap_static_count);
}

int count_navmap_static_y_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_navmaps_static->y),  thrust::device_pointer_cast(d_navmaps_static->y) + h_xmachine_memory_navmap_static_count, count_value);
}
int min_navmap_static_y_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_navmap_static_y_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_navmap_static_exit_no_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit_no),  thrust::device_pointer_cast(d_navmaps_static->exit_no) + h_xmachine_memory_navmap_static_count);
}

int count_navmap_static_exit_no_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_navmaps_static->exit_no),  thrust::device_pointer_cast(d_navmaps_static->exit_no) + h_xmachine_memory_navmap_static_count, count_value);
}
int min_navmap_static_exit_no_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit_no);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_navmap_static_exit_no_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit_no);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_height_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->height),  thrust::device_pointer_cast(d_navmaps_static->height) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_height_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->height);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_height_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->height);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_collision_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->collision_x),  thrust::device_pointer_cast(d_navmaps_static->collision_x) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_collision_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->collision_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_collision_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->collision_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_collision_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->collision_y),  thrust::device_pointer_cast(d_navmaps_static->collision_y) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_collision_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->collision_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_collision_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->collision_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit0_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit0_x),  thrust::device_pointer_cast(d_navmaps_static->exit0_x) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit0_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit0_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit0_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit0_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit0_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit0_y),  thrust::device_pointer_cast(d_navmaps_static->exit0_y) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit0_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit0_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit0_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit0_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit1_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit1_x),  thrust::device_pointer_cast(d_navmaps_static->exit1_x) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit1_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit1_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit1_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit1_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit1_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit1_y),  thrust::device_pointer_cast(d_navmaps_static->exit1_y) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit1_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit1_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit1_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit1_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit2_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit2_x),  thrust::device_pointer_cast(d_navmaps_static->exit2_x) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit2_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit2_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit2_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit2_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit2_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit2_y),  thrust::device_pointer_cast(d_navmaps_static->exit2_y) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit2_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit2_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit2_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit2_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit3_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit3_x),  thrust::device_pointer_cast(d_navmaps_static->exit3_x) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit3_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit3_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit3_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit3_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit3_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit3_y),  thrust::device_pointer_cast(d_navmaps_static->exit3_y) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit3_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit3_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit3_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit3_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit4_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit4_x),  thrust::device_pointer_cast(d_navmaps_static->exit4_x) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit4_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit4_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit4_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit4_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit4_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit4_y),  thrust::device_pointer_cast(d_navmaps_static->exit4_y) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit4_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit4_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit4_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit4_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit5_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit5_x),  thrust::device_pointer_cast(d_navmaps_static->exit5_x) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit5_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit5_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit5_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit5_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit5_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit5_y),  thrust::device_pointer_cast(d_navmaps_static->exit5_y) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit5_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit5_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit5_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit5_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit6_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit6_x),  thrust::device_pointer_cast(d_navmaps_static->exit6_x) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit6_x_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit6_x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit6_x_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit6_x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float reduce_navmap_static_exit6_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->exit6_y),  thrust::device_pointer_cast(d_navmaps_static->exit6_y) + h_xmachine_memory_navmap_static_count);
}

float min_navmap_static_exit6_y_variable(){
    //min in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit6_y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
float max_navmap_static_exit6_y_variable(){
    //max in default stream
    thrust::device_ptr<float> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->exit6_y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_navmap_static_cant_generados_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_navmaps_static->cant_generados),  thrust::device_pointer_cast(d_navmaps_static->cant_generados) + h_xmachine_memory_navmap_static_count);
}

unsigned int count_navmap_static_cant_generados_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_navmaps_static->cant_generados),  thrust::device_pointer_cast(d_navmaps_static->cant_generados) + h_xmachine_memory_navmap_static_count, count_value);
}
unsigned int min_navmap_static_cant_generados_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->cant_generados);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_navmap_static_cant_generados_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_navmaps_static->cant_generados);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_navmap_static_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}



/* Agent functions */


	
/* Shared memory size calculator for agent function */
int agent_output_pedestrian_location_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_output_pedestrian_location
 * Agent function prototype for output_pedestrian_location function of agent agent
 */
void agent_output_pedestrian_location(cudaStream_t &stream){

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

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_pedestrian_location_count + h_xmachine_memory_agent_count > xmachine_message_pedestrian_location_MAX){
		printf("Error: Buffer size of pedestrian_location message will be exceeded in function output_pedestrian_location\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_pedestrian_location, agent_output_pedestrian_location_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_output_pedestrian_location_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_pedestrian_location_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_pedestrian_location_output_type, &h_message_pedestrian_location_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (output_pedestrian_location)
	//Reallocate   : false
	//Input        : 
	//Output       : pedestrian_location
	//Agent Output : 
	GPUFLAME_output_pedestrian_location<<<g, b, sm_size, stream>>>(d_agents, d_pedestrian_locations);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_pedestrian_location_count += h_xmachine_memory_agent_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_pedestrian_location_count, &h_message_pedestrian_location_count, sizeof(int)));	
	
	//reset partition matrix
	gpuErrchk( cudaMemset( (void*) d_pedestrian_location_partition_matrix, 0, sizeof(xmachine_message_pedestrian_location_PBM)));
    //PR Bug fix: Second fix. This should prevent future problems when multiple agents write the same message as now the message structure is completely rebuilt after an output.
    if (h_message_pedestrian_location_count > 0){
#ifdef FAST_ATOMIC_SORTING
      //USE ATOMICS TO BUILD PARTITION BOUNDARY
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hist_pedestrian_location_messages, no_sm, h_message_pedestrian_location_count); 
	  gridSize = (h_message_pedestrian_location_count + blockSize - 1) / blockSize;
	  hist_pedestrian_location_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_pedestrian_location_local_bin_index, d_xmachine_message_pedestrian_location_unsorted_index, d_pedestrian_location_partition_matrix->end_or_count, d_pedestrian_locations, h_message_pedestrian_location_count);
	  gpuErrchkLaunch();
	
      // Scan
      cub::DeviceScan::ExclusiveSum(
          d_temp_scan_storage_xmachine_message_pedestrian_location, 
          temp_scan_bytes_xmachine_message_pedestrian_location, 
          d_pedestrian_location_partition_matrix->end_or_count,
          d_pedestrian_location_partition_matrix->start,
          xmachine_message_pedestrian_location_grid_size, 
          stream
      );
	
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_pedestrian_location_messages, no_sm, h_message_pedestrian_location_count); 
	  gridSize = (h_message_pedestrian_location_count + blockSize - 1) / blockSize; 	// Round up according to array size 
	  reorder_pedestrian_location_messages <<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_pedestrian_location_local_bin_index, d_xmachine_message_pedestrian_location_unsorted_index, d_pedestrian_location_partition_matrix->start, d_pedestrian_locations, d_pedestrian_locations_swap, h_message_pedestrian_location_count);
	  gpuErrchkLaunch();
#else
	  //HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
	  //Get message hash values for sorting
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hash_pedestrian_location_messages, no_sm, h_message_pedestrian_location_count); 
	  gridSize = (h_message_pedestrian_location_count + blockSize - 1) / blockSize;
	  hash_pedestrian_location_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_pedestrian_location_keys, d_xmachine_message_pedestrian_location_values, d_pedestrian_locations);
	  gpuErrchkLaunch();
	  //Sort
	  thrust::sort_by_key(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_xmachine_message_pedestrian_location_keys),  thrust::device_pointer_cast(d_xmachine_message_pedestrian_location_keys) + h_message_pedestrian_location_count,  thrust::device_pointer_cast(d_xmachine_message_pedestrian_location_values));
	  gpuErrchkLaunch();
	  //reorder and build pcb
	  gpuErrchk(cudaMemset(d_pedestrian_location_partition_matrix->start, 0xffffffff, xmachine_message_pedestrian_location_grid_size* sizeof(int)));
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_pedestrian_location_messages, reorder_messages_sm_size, h_message_pedestrian_location_count); 
	  gridSize = (h_message_pedestrian_location_count + blockSize - 1) / blockSize;
	  int reorder_sm_size = reorder_messages_sm_size(blockSize);
	  reorder_pedestrian_location_messages<<<gridSize, blockSize, reorder_sm_size, stream>>>(d_xmachine_message_pedestrian_location_keys, d_xmachine_message_pedestrian_location_values, d_pedestrian_location_partition_matrix, d_pedestrian_locations, d_pedestrian_locations_swap);
	  gpuErrchkLaunch();
#endif
  }
	//swap ordered list
	xmachine_message_pedestrian_location_list* d_pedestrian_locations_temp = d_pedestrian_locations;
	d_pedestrian_locations = d_pedestrian_locations_swap;
	d_pedestrian_locations_swap = d_pedestrian_locations_temp;
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of output_pedestrian_location agents in state default will be exceeded moving working agents to next state in function output_pedestrian_location\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  agents_default_temp = d_agents;
  d_agents = d_agents_default;
  d_agents_default = agents_default_temp;
        
	//update new state agent size
	h_xmachine_memory_agent_default_count += h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_avoid_pedestrians_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_pedestrian_location));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** agent_avoid_pedestrians
 * Agent function prototype for avoid_pedestrians function of agent agent
 */
void agent_avoid_pedestrians(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_avoid_pedestrians, agent_avoid_pedestrians_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_avoid_pedestrians_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_pedestrian_location_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_location_x_byte_offset, tex_xmachine_message_pedestrian_location_x, d_pedestrian_locations->x, sizeof(float)*xmachine_message_pedestrian_location_MAX));
	h_tex_xmachine_message_pedestrian_location_x_offset = (int)tex_xmachine_message_pedestrian_location_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_location_x_offset, &h_tex_xmachine_message_pedestrian_location_x_offset, sizeof(int)));
	size_t tex_xmachine_message_pedestrian_location_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_location_y_byte_offset, tex_xmachine_message_pedestrian_location_y, d_pedestrian_locations->y, sizeof(float)*xmachine_message_pedestrian_location_MAX));
	h_tex_xmachine_message_pedestrian_location_y_offset = (int)tex_xmachine_message_pedestrian_location_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_location_y_offset, &h_tex_xmachine_message_pedestrian_location_y_offset, sizeof(int)));
	size_t tex_xmachine_message_pedestrian_location_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_location_z_byte_offset, tex_xmachine_message_pedestrian_location_z, d_pedestrian_locations->z, sizeof(float)*xmachine_message_pedestrian_location_MAX));
	h_tex_xmachine_message_pedestrian_location_z_offset = (int)tex_xmachine_message_pedestrian_location_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_location_z_offset, &h_tex_xmachine_message_pedestrian_location_z_offset, sizeof(int)));
	size_t tex_xmachine_message_pedestrian_location_estado_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_location_estado_byte_offset, tex_xmachine_message_pedestrian_location_estado, d_pedestrian_locations->estado, sizeof(int)*xmachine_message_pedestrian_location_MAX));
	h_tex_xmachine_message_pedestrian_location_estado_offset = (int)tex_xmachine_message_pedestrian_location_estado_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_location_estado_offset, &h_tex_xmachine_message_pedestrian_location_estado_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_pedestrian_location_pbm_start_byte_offset;
	size_t tex_xmachine_message_pedestrian_location_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_location_pbm_start_byte_offset, tex_xmachine_message_pedestrian_location_pbm_start, d_pedestrian_location_partition_matrix->start, sizeof(int)*xmachine_message_pedestrian_location_grid_size));
	h_tex_xmachine_message_pedestrian_location_pbm_start_offset = (int)tex_xmachine_message_pedestrian_location_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_location_pbm_start_offset, &h_tex_xmachine_message_pedestrian_location_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_location_pbm_end_or_count_byte_offset, tex_xmachine_message_pedestrian_location_pbm_end_or_count, d_pedestrian_location_partition_matrix->end_or_count, sizeof(int)*xmachine_message_pedestrian_location_grid_size));
  h_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset = (int)tex_xmachine_message_pedestrian_location_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset, &h_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (avoid_pedestrians)
	//Reallocate   : false
	//Input        : pedestrian_location
	//Output       : 
	//Agent Output : 
	GPUFLAME_avoid_pedestrians<<<g, b, sm_size, stream>>>(d_agents, d_pedestrian_locations, d_pedestrian_location_partition_matrix, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_location_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_location_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_location_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_location_estado));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_location_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_location_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of avoid_pedestrians agents in state default will be exceeded moving working agents to next state in function avoid_pedestrians\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  agents_default_temp = d_agents;
  d_agents = d_agents_default;
  d_agents_default = agents_default_temp;
        
	//update new state agent size
	h_xmachine_memory_agent_default_count += h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_output_pedestrian_state_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_output_pedestrian_state
 * Agent function prototype for output_pedestrian_state function of agent agent
 */
void agent_output_pedestrian_state(cudaStream_t &stream){

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

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_pedestrian_state_count + h_xmachine_memory_agent_count > xmachine_message_pedestrian_state_MAX){
		printf("Error: Buffer size of pedestrian_state message will be exceeded in function output_pedestrian_state\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_pedestrian_state, agent_output_pedestrian_state_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_output_pedestrian_state_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_pedestrian_state_output_type = single_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_pedestrian_state_output_type, &h_message_pedestrian_state_output_type, sizeof(int)));
	
	
	//MAIN XMACHINE FUNCTION CALL (output_pedestrian_state)
	//Reallocate   : false
	//Input        : 
	//Output       : pedestrian_state
	//Agent Output : 
	GPUFLAME_output_pedestrian_state<<<g, b, sm_size, stream>>>(d_agents, d_pedestrian_states);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	h_message_pedestrian_state_count += h_xmachine_memory_agent_count;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_pedestrian_state_count, &h_message_pedestrian_state_count, sizeof(int)));	
	
	//reset partition matrix
	gpuErrchk( cudaMemset( (void*) d_pedestrian_state_partition_matrix, 0, sizeof(xmachine_message_pedestrian_state_PBM)));
    //PR Bug fix: Second fix. This should prevent future problems when multiple agents write the same message as now the message structure is completely rebuilt after an output.
    if (h_message_pedestrian_state_count > 0){
#ifdef FAST_ATOMIC_SORTING
      //USE ATOMICS TO BUILD PARTITION BOUNDARY
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hist_pedestrian_state_messages, no_sm, h_message_pedestrian_state_count); 
	  gridSize = (h_message_pedestrian_state_count + blockSize - 1) / blockSize;
	  hist_pedestrian_state_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_pedestrian_state_local_bin_index, d_xmachine_message_pedestrian_state_unsorted_index, d_pedestrian_state_partition_matrix->end_or_count, d_pedestrian_states, h_message_pedestrian_state_count);
	  gpuErrchkLaunch();
	
      // Scan
      cub::DeviceScan::ExclusiveSum(
          d_temp_scan_storage_xmachine_message_pedestrian_state, 
          temp_scan_bytes_xmachine_message_pedestrian_state, 
          d_pedestrian_state_partition_matrix->end_or_count,
          d_pedestrian_state_partition_matrix->start,
          xmachine_message_pedestrian_state_grid_size, 
          stream
      );
	
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_pedestrian_state_messages, no_sm, h_message_pedestrian_state_count); 
	  gridSize = (h_message_pedestrian_state_count + blockSize - 1) / blockSize; 	// Round up according to array size 
	  reorder_pedestrian_state_messages <<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_pedestrian_state_local_bin_index, d_xmachine_message_pedestrian_state_unsorted_index, d_pedestrian_state_partition_matrix->start, d_pedestrian_states, d_pedestrian_states_swap, h_message_pedestrian_state_count);
	  gpuErrchkLaunch();
#else
	  //HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
	  //Get message hash values for sorting
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, hash_pedestrian_state_messages, no_sm, h_message_pedestrian_state_count); 
	  gridSize = (h_message_pedestrian_state_count + blockSize - 1) / blockSize;
	  hash_pedestrian_state_messages<<<gridSize, blockSize, 0, stream>>>(d_xmachine_message_pedestrian_state_keys, d_xmachine_message_pedestrian_state_values, d_pedestrian_states);
	  gpuErrchkLaunch();
	  //Sort
	  thrust::sort_by_key(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_xmachine_message_pedestrian_state_keys),  thrust::device_pointer_cast(d_xmachine_message_pedestrian_state_keys) + h_message_pedestrian_state_count,  thrust::device_pointer_cast(d_xmachine_message_pedestrian_state_values));
	  gpuErrchkLaunch();
	  //reorder and build pcb
	  gpuErrchk(cudaMemset(d_pedestrian_state_partition_matrix->start, 0xffffffff, xmachine_message_pedestrian_state_grid_size* sizeof(int)));
	  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_pedestrian_state_messages, reorder_messages_sm_size, h_message_pedestrian_state_count); 
	  gridSize = (h_message_pedestrian_state_count + blockSize - 1) / blockSize;
	  int reorder_sm_size = reorder_messages_sm_size(blockSize);
	  reorder_pedestrian_state_messages<<<gridSize, blockSize, reorder_sm_size, stream>>>(d_xmachine_message_pedestrian_state_keys, d_xmachine_message_pedestrian_state_values, d_pedestrian_state_partition_matrix, d_pedestrian_states, d_pedestrian_states_swap);
	  gpuErrchkLaunch();
#endif
  }
	//swap ordered list
	xmachine_message_pedestrian_state_list* d_pedestrian_states_temp = d_pedestrian_states;
	d_pedestrian_states = d_pedestrian_states_swap;
	d_pedestrian_states_swap = d_pedestrian_states_temp;
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of output_pedestrian_state agents in state default will be exceeded moving working agents to next state in function output_pedestrian_state\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  agents_default_temp = d_agents;
  d_agents = d_agents_default;
  d_agents_default = agents_default_temp;
        
	//update new state agent size
	h_xmachine_memory_agent_default_count += h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_infect_pedestrians_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_pedestrian_state));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** agent_infect_pedestrians
 * Agent function prototype for infect_pedestrians function of agent agent
 */
void agent_infect_pedestrians(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_infect_pedestrians, agent_infect_pedestrians_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_infect_pedestrians_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_pedestrian_state_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_state_x_byte_offset, tex_xmachine_message_pedestrian_state_x, d_pedestrian_states->x, sizeof(float)*xmachine_message_pedestrian_state_MAX));
	h_tex_xmachine_message_pedestrian_state_x_offset = (int)tex_xmachine_message_pedestrian_state_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_state_x_offset, &h_tex_xmachine_message_pedestrian_state_x_offset, sizeof(int)));
	size_t tex_xmachine_message_pedestrian_state_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_state_y_byte_offset, tex_xmachine_message_pedestrian_state_y, d_pedestrian_states->y, sizeof(float)*xmachine_message_pedestrian_state_MAX));
	h_tex_xmachine_message_pedestrian_state_y_offset = (int)tex_xmachine_message_pedestrian_state_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_state_y_offset, &h_tex_xmachine_message_pedestrian_state_y_offset, sizeof(int)));
	size_t tex_xmachine_message_pedestrian_state_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_state_z_byte_offset, tex_xmachine_message_pedestrian_state_z, d_pedestrian_states->z, sizeof(float)*xmachine_message_pedestrian_state_MAX));
	h_tex_xmachine_message_pedestrian_state_z_offset = (int)tex_xmachine_message_pedestrian_state_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_state_z_offset, &h_tex_xmachine_message_pedestrian_state_z_offset, sizeof(int)));
	size_t tex_xmachine_message_pedestrian_state_estado_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_state_estado_byte_offset, tex_xmachine_message_pedestrian_state_estado, d_pedestrian_states->estado, sizeof(int)*xmachine_message_pedestrian_state_MAX));
	h_tex_xmachine_message_pedestrian_state_estado_offset = (int)tex_xmachine_message_pedestrian_state_estado_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_state_estado_offset, &h_tex_xmachine_message_pedestrian_state_estado_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_pedestrian_state_pbm_start_byte_offset;
	size_t tex_xmachine_message_pedestrian_state_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_state_pbm_start_byte_offset, tex_xmachine_message_pedestrian_state_pbm_start, d_pedestrian_state_partition_matrix->start, sizeof(int)*xmachine_message_pedestrian_state_grid_size));
	h_tex_xmachine_message_pedestrian_state_pbm_start_offset = (int)tex_xmachine_message_pedestrian_state_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_state_pbm_start_offset, &h_tex_xmachine_message_pedestrian_state_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_state_pbm_end_or_count_byte_offset, tex_xmachine_message_pedestrian_state_pbm_end_or_count, d_pedestrian_state_partition_matrix->end_or_count, sizeof(int)*xmachine_message_pedestrian_state_grid_size));
  h_tex_xmachine_message_pedestrian_state_pbm_end_or_count_offset = (int)tex_xmachine_message_pedestrian_state_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_state_pbm_end_or_count_offset, &h_tex_xmachine_message_pedestrian_state_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (infect_pedestrians)
	//Reallocate   : false
	//Input        : pedestrian_state
	//Output       : 
	//Agent Output : 
	GPUFLAME_infect_pedestrians<<<g, b, sm_size, stream>>>(d_agents, d_pedestrian_states, d_pedestrian_state_partition_matrix, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_state_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_state_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_state_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_state_estado));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_state_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_state_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of infect_pedestrians agents in state default will be exceeded moving working agents to next state in function infect_pedestrians\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  agents_default_temp = d_agents;
  d_agents = d_agents_default;
  d_agents_default = agents_default_temp;
        
	//update new state agent size
	h_xmachine_memory_agent_default_count += h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_move_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_move
 * Agent function prototype for move function of agent agent
 */
void agent_move(cudaStream_t &stream){

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

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_check_in_count + h_xmachine_memory_agent_count > xmachine_message_check_in_MAX){
		printf("Error: Buffer size of check_in message will be exceeded in function move\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_move, agent_move_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_move_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_check_in_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_check_in_output_type, &h_message_check_in_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_check_in_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_check_in_swaps<<<gridSize, blockSize, 0, stream>>>(d_check_ins); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (move)
	//Reallocate   : false
	//Input        : 
	//Output       : check_in
	//Agent Output : 
	GPUFLAME_move<<<g, b, sm_size, stream>>>(d_agents, d_check_ins);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//check_in Message Type Prefix Sum
	
	//swap output
	xmachine_message_check_in_list* d_check_ins_scanswap_temp = d_check_ins;
	d_check_ins = d_check_ins_swap;
	d_check_ins_swap = d_check_ins_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_check_ins_swap->_scan_input,
        d_check_ins_swap->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_check_in_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_check_in_messages<<<gridSize, blockSize, 0, stream>>>(d_check_ins, d_check_ins_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_check_ins_swap->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_check_ins_swap->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_check_in_count += scan_last_sum+1;
	}else{
		h_message_check_in_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_check_in_count, &h_message_check_in_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of move agents in state default will be exceeded moving working agents to next state in function move\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  agents_default_temp = d_agents;
  d_agents = d_agents_default;
  d_agents_default = agents_default_temp;
        
	//update new state agent size
	h_xmachine_memory_agent_default_count += h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_output_chair_petition_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_output_chair_petition
 * Agent function prototype for output_chair_petition function of agent agent
 */
void agent_output_chair_petition(cudaStream_t &stream){

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
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_agent_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agents_default);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agents);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, output_chair_petition_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	output_chair_petition_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_agent_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
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
	if (scan_last_included == 1)
		h_xmachine_memory_agent_default_count = scan_last_sum+1;
	else		
		h_xmachine_memory_agent_default_count = scan_last_sum;
	//Scatter into swap
	scatter_agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_agents_swap, d_agents_default, 0, h_xmachine_memory_agent_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_agent_list* agents_default_temp = d_agents_default;
	d_agents_default = d_agents_swap;
	d_agents_swap = agents_default_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_agents->_scan_input,
        d_agents->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agents->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agents->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_agents_swap, d_agents, 0, h_xmachine_memory_agent_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_agent_count = scan_last_sum+1;
	else		
		h_xmachine_memory_agent_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_agent_list* agents_temp = d_agents;
	d_agents = d_agents_swap;
	d_agents_swap = agents_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_agent_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_agent_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_chair_petition_count + h_xmachine_memory_agent_count > xmachine_message_chair_petition_MAX){
		printf("Error: Buffer size of chair_petition message will be exceeded in function output_chair_petition\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_chair_petition, agent_output_chair_petition_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_output_chair_petition_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_chair_petition_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_chair_petition_output_type, &h_message_chair_petition_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_chair_petition_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_chair_petition_swaps<<<gridSize, blockSize, 0, stream>>>(d_chair_petitions); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (output_chair_petition)
	//Reallocate   : false
	//Input        : 
	//Output       : chair_petition
	//Agent Output : 
	GPUFLAME_output_chair_petition<<<g, b, sm_size, stream>>>(d_agents, d_chair_petitions);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//chair_petition Message Type Prefix Sum
	
	//swap output
	xmachine_message_chair_petition_list* d_chair_petitions_scanswap_temp = d_chair_petitions;
	d_chair_petitions = d_chair_petitions_swap;
	d_chair_petitions_swap = d_chair_petitions_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_chair_petitions_swap->_scan_input,
        d_chair_petitions_swap->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_chair_petition_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_chair_petition_messages<<<gridSize, blockSize, 0, stream>>>(d_chair_petitions, d_chair_petitions_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_chair_petitions_swap->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_chair_petitions_swap->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_chair_petition_count += scan_last_sum+1;
	}else{
		h_message_chair_petition_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_chair_petition_count, &h_message_chair_petition_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of output_chair_petition agents in state default will be exceeded moving working agents to next state in function output_chair_petition\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_agent_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents, h_xmachine_memory_agent_default_count, h_xmachine_memory_agent_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_agent_default_count += h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_check_check_in_done_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_check_in_done));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** agent_check_check_in_done
 * Agent function prototype for check_check_in_done function of agent agent
 */
void agent_check_check_in_done(cudaStream_t &stream){

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
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_agent_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agents_default);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agents);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, check_check_in_done_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	check_check_in_done_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_agent_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
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
	if (scan_last_included == 1)
		h_xmachine_memory_agent_default_count = scan_last_sum+1;
	else		
		h_xmachine_memory_agent_default_count = scan_last_sum;
	//Scatter into swap
	scatter_agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_agents_swap, d_agents_default, 0, h_xmachine_memory_agent_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_agent_list* agents_default_temp = d_agents_default;
	d_agents_default = d_agents_swap;
	d_agents_swap = agents_default_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_agents->_scan_input,
        d_agents->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agents->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agents->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_agents_swap, d_agents, 0, h_xmachine_memory_agent_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_agent_count = scan_last_sum+1;
	else		
		h_xmachine_memory_agent_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_agent_list* agents_temp = d_agents;
	d_agents = d_agents_swap;
	d_agents_swap = agents_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_agent_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_agent_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_check_check_in_done, agent_check_check_in_done_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_check_check_in_done_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (check_check_in_done)
	//Reallocate   : false
	//Input        : check_in_done
	//Output       : 
	//Agent Output : 
	GPUFLAME_check_check_in_done<<<g, b, sm_size, stream>>>(d_agents, d_check_in_dones);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of check_check_in_done agents in state default will be exceeded moving working agents to next state in function check_check_in_done\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_agent_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents, h_xmachine_memory_agent_default_count, h_xmachine_memory_agent_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_agent_default_count += h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_check_chair_response_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_chair_response));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** agent_check_chair_response
 * Agent function prototype for check_chair_response function of agent agent
 */
void agent_check_chair_response(cudaStream_t &stream){

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
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_agent_count = h_xmachine_memory_agent_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_agent_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agents_default);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agents);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, check_chair_response_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	check_chair_response_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_agent_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
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
	if (scan_last_included == 1)
		h_xmachine_memory_agent_default_count = scan_last_sum+1;
	else		
		h_xmachine_memory_agent_default_count = scan_last_sum;
	//Scatter into swap
	scatter_agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_agents_swap, d_agents_default, 0, h_xmachine_memory_agent_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_agent_list* agents_default_temp = d_agents_default;
	d_agents_default = d_agents_swap;
	d_agents_swap = agents_default_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_agents->_scan_input,
        d_agents->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agents->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agents->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_agents_swap, d_agents, 0, h_xmachine_memory_agent_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_agent_count = scan_last_sum+1;
	else		
		h_xmachine_memory_agent_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_agent_list* agents_temp = d_agents;
	d_agents = d_agents_swap;
	d_agents_swap = agents_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_agent_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_agent_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_check_chair_response, agent_check_chair_response_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_check_chair_response_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_agent_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agents);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (check_chair_response)
	//Reallocate   : true
	//Input        : chair_response
	//Output       : 
	//Agent Output : 
	GPUFLAME_check_chair_response<<<g, b, sm_size, stream>>>(d_agents, d_chair_responses);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_agents->_scan_input,
        d_agents->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_agent_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_agents_swap, d_agents, 0, h_xmachine_memory_agent_count);
	gpuErrchkLaunch();
	//use a temp pointer to make swap default
	xmachine_memory_agent_list* check_chair_response_agents_temp = d_agents;
	d_agents = d_agents_swap;
	d_agents_swap = check_chair_response_agents_temp;
	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agents_swap->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agents_swap->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_agent_count = scan_last_sum+1;
	else
		h_xmachine_memory_agent_count = scan_last_sum;
	//Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_count, &h_xmachine_memory_agent_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of check_chair_response agents in state default will be exceeded moving working agents to next state in function check_chair_response\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_agent_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents, h_xmachine_memory_agent_default_count, h_xmachine_memory_agent_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_agent_default_count += h_xmachine_memory_agent_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int medic_prueba_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** medic_prueba
 * Agent function prototype for prueba function of medic agent
 */
void medic_prueba(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_medic_default2_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_medic_default2_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_medic_list* medics_default2_temp = d_medics;
	d_medics = d_medics_default2;
	d_medics_default2 = medics_default2_temp;
	//set working count to current state count
	h_xmachine_memory_medic_count = h_xmachine_memory_medic_default2_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_medic_count, &h_xmachine_memory_medic_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_medic_default2_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_medic_default2_count, &h_xmachine_memory_medic_default2_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_prueba, medic_prueba_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = medic_prueba_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (prueba)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_prueba<<<g, b, sm_size, stream>>>(d_medics);
	gpuErrchkLaunch();
	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_medic_default2_count+h_xmachine_memory_medic_count > xmachine_memory_medic_MAX){
		printf("Error: Buffer size of prueba agents in state default2 will be exceeded moving working agents to next state in function prueba\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  medics_default2_temp = d_medics;
  d_medics = d_medics_default2;
  d_medics_default2 = medics_default2_temp;
        
	//update new state agent size
	h_xmachine_memory_medic_default2_count += h_xmachine_memory_medic_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_medic_default2_count, &h_xmachine_memory_medic_default2_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int receptionist_receptionServer_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_check_in));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** receptionist_receptionServer
 * Agent function prototype for receptionServer function of receptionist agent
 */
void receptionist_receptionServer(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_receptionist_defaultReceptionist_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_receptionist_defaultReceptionist_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_receptionist_list* receptionists_defaultReceptionist_temp = d_receptionists;
	d_receptionists = d_receptionists_defaultReceptionist;
	d_receptionists_defaultReceptionist = receptionists_defaultReceptionist_temp;
	//set working count to current state count
	h_xmachine_memory_receptionist_count = h_xmachine_memory_receptionist_defaultReceptionist_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_receptionist_count, &h_xmachine_memory_receptionist_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_receptionist_defaultReceptionist_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_receptionist_defaultReceptionist_count, &h_xmachine_memory_receptionist_defaultReceptionist_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_check_in_done_count + h_xmachine_memory_receptionist_count > xmachine_message_check_in_done_MAX){
		printf("Error: Buffer size of check_in_done message will be exceeded in function receptionServer\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_receptionServer, receptionist_receptionServer_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = receptionist_receptionServer_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_check_in_done_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_check_in_done_output_type, &h_message_check_in_done_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_check_in_done_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_check_in_done_swaps<<<gridSize, blockSize, 0, stream>>>(d_check_in_dones); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (receptionServer)
	//Reallocate   : false
	//Input        : check_in
	//Output       : check_in_done
	//Agent Output : 
	GPUFLAME_receptionServer<<<g, b, sm_size, stream>>>(d_receptionists, d_check_ins, d_check_in_dones);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//check_in_done Message Type Prefix Sum
	
	//swap output
	xmachine_message_check_in_done_list* d_check_in_dones_scanswap_temp = d_check_in_dones;
	d_check_in_dones = d_check_in_dones_swap;
	d_check_in_dones_swap = d_check_in_dones_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_receptionist, 
        temp_scan_storage_bytes_receptionist, 
        d_check_in_dones_swap->_scan_input,
        d_check_in_dones_swap->_position,
        h_xmachine_memory_receptionist_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_check_in_done_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_check_in_done_messages<<<gridSize, blockSize, 0, stream>>>(d_check_in_dones, d_check_in_dones_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_check_in_dones_swap->_position[h_xmachine_memory_receptionist_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_check_in_dones_swap->_scan_input[h_xmachine_memory_receptionist_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_check_in_done_count += scan_last_sum+1;
	}else{
		h_message_check_in_done_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_check_in_done_count, &h_message_check_in_done_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_receptionist_defaultReceptionist_count+h_xmachine_memory_receptionist_count > xmachine_memory_receptionist_MAX){
		printf("Error: Buffer size of receptionServer agents in state defaultReceptionist will be exceeded moving working agents to next state in function receptionServer\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  receptionists_defaultReceptionist_temp = d_receptionists;
  d_receptionists = d_receptionists_defaultReceptionist;
  d_receptionists_defaultReceptionist = receptionists_defaultReceptionist_temp;
        
	//update new state agent size
	h_xmachine_memory_receptionist_defaultReceptionist_count += h_xmachine_memory_receptionist_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_receptionist_defaultReceptionist_count, &h_xmachine_memory_receptionist_defaultReceptionist_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int receptionist_infect_receptionist_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_pedestrian_state));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** receptionist_infect_receptionist
 * Agent function prototype for infect_receptionist function of receptionist agent
 */
void receptionist_infect_receptionist(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_receptionist_defaultReceptionist_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_receptionist_defaultReceptionist_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_receptionist_list* receptionists_defaultReceptionist_temp = d_receptionists;
	d_receptionists = d_receptionists_defaultReceptionist;
	d_receptionists_defaultReceptionist = receptionists_defaultReceptionist_temp;
	//set working count to current state count
	h_xmachine_memory_receptionist_count = h_xmachine_memory_receptionist_defaultReceptionist_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_receptionist_count, &h_xmachine_memory_receptionist_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_receptionist_defaultReceptionist_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_receptionist_defaultReceptionist_count, &h_xmachine_memory_receptionist_defaultReceptionist_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_infect_receptionist, receptionist_infect_receptionist_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = receptionist_infect_receptionist_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_pedestrian_state_x_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_state_x_byte_offset, tex_xmachine_message_pedestrian_state_x, d_pedestrian_states->x, sizeof(float)*xmachine_message_pedestrian_state_MAX));
	h_tex_xmachine_message_pedestrian_state_x_offset = (int)tex_xmachine_message_pedestrian_state_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_state_x_offset, &h_tex_xmachine_message_pedestrian_state_x_offset, sizeof(int)));
	size_t tex_xmachine_message_pedestrian_state_y_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_state_y_byte_offset, tex_xmachine_message_pedestrian_state_y, d_pedestrian_states->y, sizeof(float)*xmachine_message_pedestrian_state_MAX));
	h_tex_xmachine_message_pedestrian_state_y_offset = (int)tex_xmachine_message_pedestrian_state_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_state_y_offset, &h_tex_xmachine_message_pedestrian_state_y_offset, sizeof(int)));
	size_t tex_xmachine_message_pedestrian_state_z_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_state_z_byte_offset, tex_xmachine_message_pedestrian_state_z, d_pedestrian_states->z, sizeof(float)*xmachine_message_pedestrian_state_MAX));
	h_tex_xmachine_message_pedestrian_state_z_offset = (int)tex_xmachine_message_pedestrian_state_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_state_z_offset, &h_tex_xmachine_message_pedestrian_state_z_offset, sizeof(int)));
	size_t tex_xmachine_message_pedestrian_state_estado_byte_offset;    
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_state_estado_byte_offset, tex_xmachine_message_pedestrian_state_estado, d_pedestrian_states->estado, sizeof(int)*xmachine_message_pedestrian_state_MAX));
	h_tex_xmachine_message_pedestrian_state_estado_offset = (int)tex_xmachine_message_pedestrian_state_estado_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_state_estado_offset, &h_tex_xmachine_message_pedestrian_state_estado_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_pedestrian_state_pbm_start_byte_offset;
	size_t tex_xmachine_message_pedestrian_state_pbm_end_or_count_byte_offset;
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_state_pbm_start_byte_offset, tex_xmachine_message_pedestrian_state_pbm_start, d_pedestrian_state_partition_matrix->start, sizeof(int)*xmachine_message_pedestrian_state_grid_size));
	h_tex_xmachine_message_pedestrian_state_pbm_start_offset = (int)tex_xmachine_message_pedestrian_state_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_state_pbm_start_offset, &h_tex_xmachine_message_pedestrian_state_pbm_start_offset, sizeof(int)));
	gpuErrchk( cudaBindTexture(&tex_xmachine_message_pedestrian_state_pbm_end_or_count_byte_offset, tex_xmachine_message_pedestrian_state_pbm_end_or_count, d_pedestrian_state_partition_matrix->end_or_count, sizeof(int)*xmachine_message_pedestrian_state_grid_size));
  h_tex_xmachine_message_pedestrian_state_pbm_end_or_count_offset = (int)tex_xmachine_message_pedestrian_state_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol( d_tex_xmachine_message_pedestrian_state_pbm_end_or_count_offset, &h_tex_xmachine_message_pedestrian_state_pbm_end_or_count_offset, sizeof(int)));

	
	
	//MAIN XMACHINE FUNCTION CALL (infect_receptionist)
	//Reallocate   : false
	//Input        : pedestrian_state
	//Output       : 
	//Agent Output : 
	GPUFLAME_infect_receptionist<<<g, b, sm_size, stream>>>(d_receptionists, d_pedestrian_states, d_pedestrian_state_partition_matrix, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_state_x));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_state_y));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_state_z));
	gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_state_estado));
	//unbind pbm indices
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_state_pbm_start));
    gpuErrchk( cudaUnbindTexture(tex_xmachine_message_pedestrian_state_pbm_end_or_count));
    
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_receptionist_defaultReceptionist_count+h_xmachine_memory_receptionist_count > xmachine_memory_receptionist_MAX){
		printf("Error: Buffer size of infect_receptionist agents in state defaultReceptionist will be exceeded moving working agents to next state in function infect_receptionist\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  receptionists_defaultReceptionist_temp = d_receptionists;
  d_receptionists = d_receptionists_defaultReceptionist;
  d_receptionists_defaultReceptionist = receptionists_defaultReceptionist_temp;
        
	//update new state agent size
	h_xmachine_memory_receptionist_defaultReceptionist_count += h_xmachine_memory_receptionist_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_receptionist_defaultReceptionist_count, &h_xmachine_memory_receptionist_defaultReceptionist_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int chair_admin_attend_chair_petitions_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_chair_petition));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** chair_admin_attend_chair_petitions
 * Agent function prototype for attend_chair_petitions function of chair_admin agent
 */
void chair_admin_attend_chair_petitions(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_chair_admin_defaultAdmin_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_chair_admin_defaultAdmin_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_chair_admin_list* chair_admins_defaultAdmin_temp = d_chair_admins;
	d_chair_admins = d_chair_admins_defaultAdmin;
	d_chair_admins_defaultAdmin = chair_admins_defaultAdmin_temp;
	//set working count to current state count
	h_xmachine_memory_chair_admin_count = h_xmachine_memory_chair_admin_defaultAdmin_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_chair_admin_count, &h_xmachine_memory_chair_admin_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_chair_admin_defaultAdmin_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_chair_admin_defaultAdmin_count, &h_xmachine_memory_chair_admin_defaultAdmin_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_chair_response_count + h_xmachine_memory_chair_admin_count > xmachine_message_chair_response_MAX){
		printf("Error: Buffer size of chair_response message will be exceeded in function attend_chair_petitions\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_attend_chair_petitions, chair_admin_attend_chair_petitions_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = chair_admin_attend_chair_petitions_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_chair_response_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_chair_response_output_type, &h_message_chair_response_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_chair_response_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_chair_response_swaps<<<gridSize, blockSize, 0, stream>>>(d_chair_responses); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (attend_chair_petitions)
	//Reallocate   : false
	//Input        : chair_petition
	//Output       : chair_response
	//Agent Output : 
	GPUFLAME_attend_chair_petitions<<<g, b, sm_size, stream>>>(d_chair_admins, d_chair_petitions, d_chair_responses, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//chair_response Message Type Prefix Sum
	
	//swap output
	xmachine_message_chair_response_list* d_chair_responses_scanswap_temp = d_chair_responses;
	d_chair_responses = d_chair_responses_swap;
	d_chair_responses_swap = d_chair_responses_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_chair_admin, 
        temp_scan_storage_bytes_chair_admin, 
        d_chair_responses_swap->_scan_input,
        d_chair_responses_swap->_position,
        h_xmachine_memory_chair_admin_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_chair_response_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_chair_response_messages<<<gridSize, blockSize, 0, stream>>>(d_chair_responses, d_chair_responses_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_chair_responses_swap->_position[h_xmachine_memory_chair_admin_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_chair_responses_swap->_scan_input[h_xmachine_memory_chair_admin_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_chair_response_count += scan_last_sum+1;
	}else{
		h_message_chair_response_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_chair_response_count, &h_message_chair_response_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_chair_admin_defaultAdmin_count+h_xmachine_memory_chair_admin_count > xmachine_memory_chair_admin_MAX){
		printf("Error: Buffer size of attend_chair_petitions agents in state defaultAdmin will be exceeded moving working agents to next state in function attend_chair_petitions\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  chair_admins_defaultAdmin_temp = d_chair_admins;
  d_chair_admins = d_chair_admins_defaultAdmin;
  d_chair_admins_defaultAdmin = chair_admins_defaultAdmin_temp;
        
	//update new state agent size
	h_xmachine_memory_chair_admin_defaultAdmin_count += h_xmachine_memory_chair_admin_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_chair_admin_defaultAdmin_count, &h_xmachine_memory_chair_admin_defaultAdmin_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int navmap_output_navmap_cells_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** navmap_output_navmap_cells
 * Agent function prototype for output_navmap_cells function of navmap agent
 */
void navmap_output_navmap_cells(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_navmap_static_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_navmap_static_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_navmap_list* navmaps_static_temp = d_navmaps;
	d_navmaps = d_navmaps_static;
	d_navmaps_static = navmaps_static_temp;
	//set working count to current state count
	h_xmachine_memory_navmap_count = h_xmachine_memory_navmap_static_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_navmap_count, &h_xmachine_memory_navmap_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_navmap_static_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_navmap_static_count, &h_xmachine_memory_navmap_static_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_navmap_cells, navmap_output_navmap_cells_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = navmap_output_navmap_cells_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	
	
	//MAIN XMACHINE FUNCTION CALL (output_navmap_cells)
	//Reallocate   : false
	//Input        : 
	//Output       : navmap_cell
	//Agent Output : 
	GPUFLAME_output_navmap_cells<<<g, b, sm_size, stream>>>(d_navmaps, d_navmap_cells);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	navmaps_static_temp = d_navmaps_static;
	d_navmaps_static = d_navmaps;
	d_navmaps = navmaps_static_temp;
    //set current state count
	h_xmachine_memory_navmap_static_count = h_xmachine_memory_navmap_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_navmap_static_count, &h_xmachine_memory_navmap_static_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int navmap_generate_pedestrians_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** navmap_generate_pedestrians
 * Agent function prototype for generate_pedestrians function of navmap agent
 */
void navmap_generate_pedestrians(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_navmap_static_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_navmap_static_count;

	
	//FOR agent AGENT OUTPUT, RESET THE AGENT NEW LIST SCAN INPUT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_agent_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agents_new);
	gpuErrchkLaunch();
	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_navmap_list* navmaps_static_temp = d_navmaps;
	d_navmaps = d_navmaps_static;
	d_navmaps_static = navmaps_static_temp;
	//set working count to current state count
	h_xmachine_memory_navmap_count = h_xmachine_memory_navmap_static_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_navmap_count, &h_xmachine_memory_navmap_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_navmap_static_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_navmap_static_count, &h_xmachine_memory_navmap_static_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_generate_pedestrians, navmap_generate_pedestrians_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = navmap_generate_pedestrians_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (generate_pedestrians)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : agent
	GPUFLAME_generate_pedestrians<<<g, b, sm_size, stream>>>(d_navmaps, d_agents_new, d_rand48);
	gpuErrchkLaunch();
	
	
    //COPY ANY AGENT COUNT BEFORE navmap AGENTS ARE KILLED (needed for scatter)
	int navmaps_pre_death_count = h_xmachine_memory_navmap_count;
	
	//FOR agent AGENT OUTPUT SCATTER AGENTS 

    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_agents_new->_scan_input, 
        d_agents_new->_position, 
        navmaps_pre_death_count,
        stream
    );

	//reset agent count
	int agent_after_birth_count;
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agents_new->_position[navmaps_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agents_new->_scan_input[navmaps_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		agent_after_birth_count = h_xmachine_memory_agent_default_count + scan_last_sum+1;
	else
		agent_after_birth_count = h_xmachine_memory_agent_default_count + scan_last_sum;
	//check buffer is not exceeded
	if (agent_after_birth_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of agent agents in state default will be exceeded writing new agents in function generate_pedestrians\n");
		exit(EXIT_FAILURE);
	}
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_agent_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_agent_Agents<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents_new, h_xmachine_memory_agent_default_count, navmaps_pre_death_count);
	gpuErrchkLaunch();
	//Copy count to device
	h_xmachine_memory_agent_default_count = agent_after_birth_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_default_count, &h_xmachine_memory_agent_default_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	navmaps_static_temp = d_navmaps_static;
	d_navmaps_static = d_navmaps;
	d_navmaps = navmaps_static_temp;
    //set current state count
	h_xmachine_memory_navmap_static_count = h_xmachine_memory_navmap_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_navmap_static_count, &h_xmachine_memory_navmap_static_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int navmap_generate_medics_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** navmap_generate_medics
 * Agent function prototype for generate_medics function of navmap agent
 */
void navmap_generate_medics(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_navmap_static_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_navmap_static_count;

	
	//FOR medic AGENT OUTPUT, RESET THE AGENT NEW LIST SCAN INPUT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_medic_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_medic_scan_input<<<gridSize, blockSize, 0, stream>>>(d_medics_new);
	gpuErrchkLaunch();
	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_navmap_list* navmaps_static_temp = d_navmaps;
	d_navmaps = d_navmaps_static;
	d_navmaps_static = navmaps_static_temp;
	//set working count to current state count
	h_xmachine_memory_navmap_count = h_xmachine_memory_navmap_static_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_navmap_count, &h_xmachine_memory_navmap_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_navmap_static_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_navmap_static_count, &h_xmachine_memory_navmap_static_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_generate_medics, navmap_generate_medics_sm_size, state_list_size);
	blockSize = lowest_sqr_pow2(blockSize); //For discrete agents the block size must be a square power of 2
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = (int)sqrt(blockSize);
	b.y = b.x;
	g.x = (int)sqrt(gridSize);
	g.y = g.x;
	sm_size = navmap_generate_medics_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (generate_medics)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : medic
	GPUFLAME_generate_medics<<<g, b, sm_size, stream>>>(d_navmaps, d_medics_new, d_rand48);
	gpuErrchkLaunch();
	
	
    //COPY ANY AGENT COUNT BEFORE navmap AGENTS ARE KILLED (needed for scatter)
	int navmaps_pre_death_count = h_xmachine_memory_navmap_count;
	
	//FOR medic AGENT OUTPUT SCATTER AGENTS 

    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_medic, 
        temp_scan_storage_bytes_medic, 
        d_medics_new->_scan_input, 
        d_medics_new->_position, 
        navmaps_pre_death_count,
        stream
    );

	//reset agent count
	int medic_after_birth_count;
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_medics_new->_position[navmaps_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_medics_new->_scan_input[navmaps_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		medic_after_birth_count = h_xmachine_memory_medic_default2_count + scan_last_sum+1;
	else
		medic_after_birth_count = h_xmachine_memory_medic_default2_count + scan_last_sum;
	//check buffer is not exceeded
	if (medic_after_birth_count > xmachine_memory_medic_MAX){
		printf("Error: Buffer size of medic agents in state default2 will be exceeded writing new agents in function generate_medics\n");
		exit(EXIT_FAILURE);
	}
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_medic_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_medic_Agents<<<gridSize, blockSize, 0, stream>>>(d_medics_default2, d_medics_new, h_xmachine_memory_medic_default2_count, navmaps_pre_death_count);
	gpuErrchkLaunch();
	//Copy count to device
	h_xmachine_memory_medic_default2_count = medic_after_birth_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_medic_default2_count, &h_xmachine_memory_medic_default2_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
    //currentState maps to working list
	navmaps_static_temp = d_navmaps_static;
	d_navmaps_static = d_navmaps;
	d_navmaps = navmaps_static_temp;
    //set current state count
	h_xmachine_memory_navmap_static_count = h_xmachine_memory_navmap_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_navmap_static_count, &h_xmachine_memory_navmap_static_count, sizeof(int)));	
	
	
}


 
extern void reset_agent_default_count()
{
    h_xmachine_memory_agent_default_count = 0;
}
 
extern void reset_medic_default2_count()
{
    h_xmachine_memory_medic_default2_count = 0;
}
 
extern void reset_receptionist_defaultReceptionist_count()
{
    h_xmachine_memory_receptionist_defaultReceptionist_count = 0;
}
 
extern void reset_chair_admin_defaultAdmin_count()
{
    h_xmachine_memory_chair_admin_defaultAdmin_count = 0;
}
 
extern void reset_navmap_static_count()
{
    h_xmachine_memory_navmap_static_count = 0;
}
