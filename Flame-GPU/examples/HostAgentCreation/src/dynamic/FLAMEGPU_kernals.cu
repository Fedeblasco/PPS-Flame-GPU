
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


#ifndef _FLAMEGPU_KERNELS_H_
#define _FLAMEGPU_KERNELS_H_

#include "header.h"


/* Agent count constants */

__constant__ int d_xmachine_memory_Agent_count;

/* Agent state count constants */

__constant__ int d_xmachine_memory_Agent_default_count;

__constant__ int d_xmachine_memory_Agent_s2_count;


/* Message constants */

	

/* Graph Constants */


/* Graph device array pointer(s) */


/* Graph host array pointer(s) */

    
//include each function file

#include "functions.c"
    
/* Texture bindings */
    
#define WRAP(x,m) (((x)<m)?(x):(x%m)) /**< Simple wrap */
#define sWRAP(x,m) (((x)<m)?(((x)<0)?(m+(x)):(x)):(m-(x))) /**<signed integer wrap (no modulus) for negatives where 2m > |x| > m */

//PADDING WILL ONLY AVOID SM CONFLICTS FOR 32BIT
//SM_OFFSET REQUIRED AS FERMI STARTS INDEXING MEMORY FROM LOCATION 0 (i.e. NULL)??
__constant__ int d_SM_START;
__constant__ int d_PADDING;

//SM addressing macro to avoid conflicts (32 bit only)
#define SHARE_INDEX(i, s) ((((s) + d_PADDING)* (i))+d_SM_START) /**<offset struct size by padding to avoid bank conflicts */

//if doubel support is needed then define the following function which requires sm_13 or later
#ifdef _DOUBLE_SUPPORT_REQUIRED_
__inline__ __device__ double tex1DfetchDouble(texture<int2, 1, cudaReadModeElementType> tex, int i)
{
	int2 v = tex1Dfetch(tex, i);
  //IF YOU HAVE AN ERROR HERE THEN YOU ARE USING DOUBLE VALUES IN AGENT MEMORY AND NOT COMPILING FOR DOUBLE SUPPORTED HARDWARE
  //To compile for double supported hardware change the CUDA Build rule property "Use sm_13 Architecture (double support)" on the CUDA-Specific Propert Page of the CUDA Build Rule for simulation.cu
	return __hiloint2double(v.y, v.x);
}
#endif

/* Helper functions */
/** next_cell
 * Function used for finding the next cell when using spatial partitioning
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1,1
 */
__device__ bool next_cell3D(glm::ivec3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	if (relative_cell->z < 1)
	{
		relative_cell->z++;
		return true;
	}
	relative_cell->z = -1;
	
	return false;
}

/** next_cell2D
 * Function used for finding the next cell when using spatial partitioning. Z component is ignored
 * Upddates the relative cell variable which can have value of -1, 0 or +1
 * @param relative_cell pointer to the relative cell position
 * @return boolean if there is a next cell. True unless relative_Cell value was 1,1
 */
__device__ bool next_cell2D(glm::ivec3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;
	
	return false;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created Agent agent functions */

/** reset_Agent_scan_input
 * Agent agent reset scan input function
 * @param agents The xmachine_memory_Agent_list agent list
 */
__global__ void reset_Agent_scan_input(xmachine_memory_Agent_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_Agent_Agents
 * Agent scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Agent_list agent list destination
 * @param agents_src xmachine_memory_Agent_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_Agent_Agents(xmachine_memory_Agent_list* agents_dst, xmachine_memory_Agent_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->id[output_index] = agents_src->id[index];        
		agents_dst->time_alive[output_index] = agents_src->time_alive[index];
	    for (int i=0; i<4; i++){
	      agents_dst->example_array[(i*xmachine_memory_Agent_MAX)+output_index] = agents_src->example_array[(i*xmachine_memory_Agent_MAX)+index];
	    }        
		agents_dst->example_vector[output_index] = agents_src->example_vector[index];
	}
}

/** append_Agent_Agents
 * Agent scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_Agent_list agent list destination
 * @param agents_src xmachine_memory_Agent_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_Agent_Agents(xmachine_memory_Agent_list* agents_dst, xmachine_memory_Agent_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->time_alive[output_index] = agents_src->time_alive[index];
	    for (int i=0; i<4; i++){
	      agents_dst->example_array[(i*xmachine_memory_Agent_MAX)+output_index] = agents_src->example_array[(i*xmachine_memory_Agent_MAX)+index];
	    }
	    agents_dst->example_vector[output_index] = agents_src->example_vector[index];
    }
}

/** add_Agent_agent
 * Continuous Agent agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Agent_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param time_alive agent variable of type unsigned int
 * @param example_array agent variable of type float
 * @param example_vector agent variable of type ivec4
 */
template <int AGENT_TYPE>
__device__ void add_Agent_agent(xmachine_memory_Agent_list* agents, unsigned int id, unsigned int time_alive, ivec4 example_vector){
	
	int index;
    
    //calculate the agents index in global agent list (depends on agent type)
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x* gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x*blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y*blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y* width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	//for prefix sum
	agents->_position[index] = 0;
	agents->_scan_input[index] = 1;

	//write data to new buffer
	agents->id[index] = id;
	agents->time_alive[index] = time_alive;
	agents->example_vector[index] = example_vector;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Agent_agent(xmachine_memory_Agent_list* agents, unsigned int id, unsigned int time_alive, ivec4 example_vector){
    add_Agent_agent<DISCRETE_2D>(agents, id, time_alive, example_vector);
}

/** reorder_Agent_agents
 * Continuous Agent agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_Agent_agents(unsigned int* values, xmachine_memory_Agent_list* unordered_agents, xmachine_memory_Agent_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->time_alive[index] = unordered_agents->time_alive[old_pos];
	for (int i=0; i<4; i++){
	  ordered_agents->example_array[(i*xmachine_memory_Agent_MAX)+index] = unordered_agents->example_array[(i*xmachine_memory_Agent_MAX)+old_pos];
	}
	ordered_agents->example_vector[index] = unordered_agents->example_vector[old_pos];
}

/** get_Agent_agent_array_value
 *  Template function for accessing Agent agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_Agent_agent_array_value(T *array, uint index){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    return array[index*xmachine_memory_Agent_MAX];
    } else {
    	// Return the default value for this data type 
	    return 0;
    }
}

/** set_Agent_agent_array_value
 *  Template function for setting Agent agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_Agent_agent_array_value(T *array, uint index, T value){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    array[index*xmachine_memory_Agent_MAX] = value;
    }
}

	
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created GPU kernels  */



/**
 *
 */
__global__ void GPUFLAME_update(xmachine_memory_Agent_list* agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_update Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.time_alive = agents->time_alive[index];
    agent.example_array = &(agents->example_array[index]);
	agent.example_vector = agents->example_vector[index];

	//FLAME function call
	int dead = !update(&agent);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_update Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->time_alive[index] = agent.time_alive;
	agents->example_vector[index] = agent.example_vector;
}

	
	
/* Graph utility functions */



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Rand48 functions */

__device__ static glm::uvec2 RNG_rand48_iterate_single(glm::uvec2 Xn, glm::uvec2 A, glm::uvec2 C)
{
	unsigned int R0, R1;

	// low 24-bit multiplication
	const unsigned int lo00 = __umul24(Xn.x, A.x);
	const unsigned int hi00 = __umulhi(Xn.x, A.x);

	// 24bit distribution of 32bit multiplication results
	R0 = (lo00 & 0xFFFFFF);
	R1 = (lo00 >> 24) | (hi00 << 8);

	R0 += C.x; R1 += C.y;

	// transfer overflows
	R1 += (R0 >> 24);
	R0 &= 0xFFFFFF;

	// cross-terms, low/hi 24-bit multiplication
	R1 += __umul24(Xn.y, A.x);
	R1 += __umul24(Xn.x, A.y);

	R1 &= 0xFFFFFF;

	return glm::uvec2(R0, R1);
}

//Templated function
template <int AGENT_TYPE>
__device__ float rnd(RNG_rand48* rand48){

	int index;
	
	//calculate the agents index in global agent list
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x * gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y * width);
	}else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	glm::uvec2 state = rand48->seeds[index];
	glm::uvec2 A = rand48->A;
	glm::uvec2 C = rand48->C;

	int rand = ( state.x >> 17 ) | ( state.y << 7);

	// this actually iterates the RNG
	state = RNG_rand48_iterate_single(state, A, C);

	rand48->seeds[index] = state;

	return (float)rand/2147483647;
}

__device__ float rnd(RNG_rand48* rand48){
	return rnd<DISCRETE_2D>(rand48);
}

#endif //_FLAMEGPU_KERNELS_H_