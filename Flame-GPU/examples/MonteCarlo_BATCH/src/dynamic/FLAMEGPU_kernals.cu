
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

__constant__ int d_xmachine_memory_crystal_count;

/* Agent state count constants */

__constant__ int d_xmachine_memory_crystal_default_count;


/* Message constants */

/* internal_coord Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_internal_coord_count;         /**< message list counter*/
__constant__ int d_message_internal_coord_output_type;   /**< message output type (single or optional)*/

	

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
/* Dynamically created crystal agent functions */

/** reset_crystal_scan_input
 * crystal agent reset scan input function
 * @param agents The xmachine_memory_crystal_list agent list
 */
__global__ void reset_crystal_scan_input(xmachine_memory_crystal_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_crystal_Agents
 * crystal scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_crystal_list agent list destination
 * @param agents_src xmachine_memory_crystal_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_crystal_Agents(xmachine_memory_crystal_list* agents_dst, xmachine_memory_crystal_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->rank[output_index] = agents_src->rank[index];        
		agents_dst->l[output_index] = agents_src->l[index];        
		agents_dst->bin[output_index] = agents_src->bin[index];
	}
}

/** append_crystal_Agents
 * crystal scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_crystal_list agent list destination
 * @param agents_src xmachine_memory_crystal_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_crystal_Agents(xmachine_memory_crystal_list* agents_dst, xmachine_memory_crystal_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->rank[output_index] = agents_src->rank[index];
	    agents_dst->l[output_index] = agents_src->l[index];
	    agents_dst->bin[output_index] = agents_src->bin[index];
    }
}

/** add_crystal_agent
 * Continuous crystal agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_crystal_list to add agents to 
 * @param rank agent variable of type float
 * @param l agent variable of type float
 * @param bin agent variable of type int
 */
template <int AGENT_TYPE>
__device__ void add_crystal_agent(xmachine_memory_crystal_list* agents, float rank, float l, int bin){
	
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
	agents->rank[index] = rank;
	agents->l[index] = l;
	agents->bin[index] = bin;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_crystal_agent(xmachine_memory_crystal_list* agents, float rank, float l, int bin){
    add_crystal_agent<DISCRETE_2D>(agents, rank, l, bin);
}

/** reorder_crystal_agents
 * Continuous crystal agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_crystal_agents(unsigned int* values, xmachine_memory_crystal_list* unordered_agents, xmachine_memory_crystal_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->rank[index] = unordered_agents->rank[old_pos];
	ordered_agents->l[index] = unordered_agents->l[old_pos];
	ordered_agents->bin[index] = unordered_agents->bin[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created internal_coord message functions */


/** add_internal_coord_message
 * Add non partitioned or spatially partitioned internal_coord message
 * @param messages xmachine_message_internal_coord_list message list to add too
 * @param rank agent variable of type float
 * @param l agent variable of type float
 */
__device__ void add_internal_coord_message(xmachine_message_internal_coord_list* messages, float rank, float l){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_internal_coord_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_internal_coord_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_internal_coord_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_internal_coord Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->rank[index] = rank;
	messages->l[index] = l;

}

/**
 * Scatter non partitioned or spatially partitioned internal_coord message (for optional messages)
 * @param messages scatter_optional_internal_coord_messages Sparse xmachine_message_internal_coord_list message list
 * @param message_swap temp xmachine_message_internal_coord_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_internal_coord_messages(xmachine_message_internal_coord_list* messages, xmachine_message_internal_coord_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_internal_coord_count;

		//AoS - xmachine_message_internal_coord Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->rank[output_index] = messages_swap->rank[index];
		messages->l[output_index] = messages_swap->l[index];				
	}
}

/** reset_internal_coord_swaps
 * Reset non partitioned or spatially partitioned internal_coord message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_internal_coord_swaps(xmachine_message_internal_coord_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_internal_coord* get_first_internal_coord_message(xmachine_message_internal_coord_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_internal_coord_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_internal_coord Coalesced memory read
	xmachine_message_internal_coord temp_message;
	temp_message._position = messages->_position[index];
	temp_message.rank = messages->rank[index];
	temp_message.l = messages->l[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_internal_coord));
	xmachine_message_internal_coord* sm_message = ((xmachine_message_internal_coord*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_internal_coord*)&message_share[d_SM_START]);
}

__device__ xmachine_message_internal_coord* get_next_internal_coord_message(xmachine_message_internal_coord* message, xmachine_message_internal_coord_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_internal_coord_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_internal_coord_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_internal_coord Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_internal_coord temp_message;
		temp_message._position = messages->_position[index];
		temp_message.rank = messages->rank[index];
		temp_message.l = messages->l[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_internal_coord));
		xmachine_message_internal_coord* sm_message = ((xmachine_message_internal_coord*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_internal_coord));
	return ((xmachine_message_internal_coord*)&message_share[message_index]);
}

	
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created GPU kernels  */



/**
 *
 */
__global__ void GPUFLAME_create_ranks(xmachine_memory_crystal_list* agents, xmachine_message_internal_coord_list* internal_coord_messages, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_crystal_count)
        return;
    

	//SoA to AoS - xmachine_memory_create_ranks Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_crystal agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.rank = agents->rank[index];
	agent.l = agents->l[index];
	agent.bin = agents->bin[index];

	//FLAME function call
	int dead = !create_ranks(&agent, internal_coord_messages	, rand48);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_create_ranks Coalesced memory write (ignore arrays)
	agents->rank[index] = agent.rank;
	agents->l[index] = agent.l;
	agents->bin[index] = agent.bin;
}

/**
 *
 */
__global__ void GPUFLAME_simulate(xmachine_memory_crystal_list* agents, xmachine_message_internal_coord_list* internal_coord_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_simulate Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_crystal agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_crystal_count){
    
	agent.rank = agents->rank[index];
	agent.l = agents->l[index];
	agent.bin = agents->bin[index];
	} else {
	
	agent.rank = 0;
	agent.l = 0;
	agent.bin = 0;
	}

	//FLAME function call
	int dead = !simulate(&agent, internal_coord_messages);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_crystal_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_simulate Coalesced memory write (ignore arrays)
	agents->rank[index] = agent.rank;
	agents->l[index] = agent.l;
	agents->bin[index] = agent.bin;
	}
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
