
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


/* Message constants */

/* location Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_location_count;         /**< message list counter*/
__constant__ int d_message_location_output_type;   /**< message output type (single or optional)*/

/* intent Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_intent_count;         /**< message list counter*/
__constant__ int d_message_intent_output_type;   /**< message output type (single or optional)*/

	

/* Graph Constants */

__constant__ staticGraph_memory_network* d_staticGraph_memory_network_ptr;


/* Graph device array pointer(s) */

staticGraph_memory_network* d_staticGraph_memory_network = nullptr;


/* Graph host array pointer(s) */

staticGraph_memory_network* h_staticGraph_memory_network = nullptr;

    
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


/** resolve_intent_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_Agent_list representing agent i the current state
 * @param nextState xmachine_memory_Agent_list representing agent i the next state
 */
 __global__ void resolve_intent_function_filter(xmachine_memory_Agent_list* currentState, xmachine_memory_Agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_Agent_count){
	
		//apply the filter
		if (currentState->hasIntent[index]==true)
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->currentEdge[index] = currentState->currentEdge[index];
			nextState->nextEdge[index] = currentState->nextEdge[index];
			nextState->nextEdgeRemainingCapacity[index] = currentState->nextEdgeRemainingCapacity[index];
			nextState->hasIntent[index] = currentState->hasIntent[index];
			nextState->position[index] = currentState->position[index];
			nextState->distanceTravelled[index] = currentState->distanceTravelled[index];
			nextState->blockedIterationCount[index] = currentState->blockedIterationCount[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->z[index] = currentState->z[index];
			nextState->colour[index] = currentState->colour[index];
			//set scan input flag to 1
			nextState->_scan_input[index] = 1;
		}
		else
		{
			//set scan input flag of current state to 1 (keep agent)
			currentState->_scan_input[index] = 1;
		}
	
	}
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
		agents_dst->currentEdge[output_index] = agents_src->currentEdge[index];        
		agents_dst->nextEdge[output_index] = agents_src->nextEdge[index];        
		agents_dst->nextEdgeRemainingCapacity[output_index] = agents_src->nextEdgeRemainingCapacity[index];        
		agents_dst->hasIntent[output_index] = agents_src->hasIntent[index];        
		agents_dst->position[output_index] = agents_src->position[index];        
		agents_dst->distanceTravelled[output_index] = agents_src->distanceTravelled[index];        
		agents_dst->blockedIterationCount[output_index] = agents_src->blockedIterationCount[index];        
		agents_dst->speed[output_index] = agents_src->speed[index];        
		agents_dst->x[output_index] = agents_src->x[index];        
		agents_dst->y[output_index] = agents_src->y[index];        
		agents_dst->z[output_index] = agents_src->z[index];        
		agents_dst->colour[output_index] = agents_src->colour[index];
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
	    agents_dst->currentEdge[output_index] = agents_src->currentEdge[index];
	    agents_dst->nextEdge[output_index] = agents_src->nextEdge[index];
	    agents_dst->nextEdgeRemainingCapacity[output_index] = agents_src->nextEdgeRemainingCapacity[index];
	    agents_dst->hasIntent[output_index] = agents_src->hasIntent[index];
	    agents_dst->position[output_index] = agents_src->position[index];
	    agents_dst->distanceTravelled[output_index] = agents_src->distanceTravelled[index];
	    agents_dst->blockedIterationCount[output_index] = agents_src->blockedIterationCount[index];
	    agents_dst->speed[output_index] = agents_src->speed[index];
	    agents_dst->x[output_index] = agents_src->x[index];
	    agents_dst->y[output_index] = agents_src->y[index];
	    agents_dst->z[output_index] = agents_src->z[index];
	    agents_dst->colour[output_index] = agents_src->colour[index];
    }
}

/** add_Agent_agent
 * Continuous Agent agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_Agent_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param currentEdge agent variable of type unsigned int
 * @param nextEdge agent variable of type unsigned int
 * @param nextEdgeRemainingCapacity agent variable of type unsigned int
 * @param hasIntent agent variable of type bool
 * @param position agent variable of type float
 * @param distanceTravelled agent variable of type float
 * @param blockedIterationCount agent variable of type unsigned int
 * @param speed agent variable of type float
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 * @param colour agent variable of type float
 */
template <int AGENT_TYPE>
__device__ void add_Agent_agent(xmachine_memory_Agent_list* agents, unsigned int id, unsigned int currentEdge, unsigned int nextEdge, unsigned int nextEdgeRemainingCapacity, bool hasIntent, float position, float distanceTravelled, unsigned int blockedIterationCount, float speed, float x, float y, float z, float colour){
	
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
	agents->currentEdge[index] = currentEdge;
	agents->nextEdge[index] = nextEdge;
	agents->nextEdgeRemainingCapacity[index] = nextEdgeRemainingCapacity;
	agents->hasIntent[index] = hasIntent;
	agents->position[index] = position;
	agents->distanceTravelled[index] = distanceTravelled;
	agents->blockedIterationCount[index] = blockedIterationCount;
	agents->speed[index] = speed;
	agents->x[index] = x;
	agents->y[index] = y;
	agents->z[index] = z;
	agents->colour[index] = colour;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_Agent_agent(xmachine_memory_Agent_list* agents, unsigned int id, unsigned int currentEdge, unsigned int nextEdge, unsigned int nextEdgeRemainingCapacity, bool hasIntent, float position, float distanceTravelled, unsigned int blockedIterationCount, float speed, float x, float y, float z, float colour){
    add_Agent_agent<DISCRETE_2D>(agents, id, currentEdge, nextEdge, nextEdgeRemainingCapacity, hasIntent, position, distanceTravelled, blockedIterationCount, speed, x, y, z, colour);
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
	ordered_agents->currentEdge[index] = unordered_agents->currentEdge[old_pos];
	ordered_agents->nextEdge[index] = unordered_agents->nextEdge[old_pos];
	ordered_agents->nextEdgeRemainingCapacity[index] = unordered_agents->nextEdgeRemainingCapacity[old_pos];
	ordered_agents->hasIntent[index] = unordered_agents->hasIntent[old_pos];
	ordered_agents->position[index] = unordered_agents->position[old_pos];
	ordered_agents->distanceTravelled[index] = unordered_agents->distanceTravelled[old_pos];
	ordered_agents->blockedIterationCount[index] = unordered_agents->blockedIterationCount[old_pos];
	ordered_agents->speed[index] = unordered_agents->speed[old_pos];
	ordered_agents->x[index] = unordered_agents->x[old_pos];
	ordered_agents->y[index] = unordered_agents->y[old_pos];
	ordered_agents->z[index] = unordered_agents->z[old_pos];
	ordered_agents->colour[index] = unordered_agents->colour[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created location message functions */


/** add_location_message
 * Add non partitioned or spatially partitioned location message
 * @param messages xmachine_message_location_list message list to add too
 * @param id agent variable of type unsigned int
 * @param edge_id agent variable of type unsigned int
 * @param position agent variable of type float
 */
__device__ void add_location_message(xmachine_message_location_list* messages, unsigned int id, unsigned int edge_id, float position){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_location_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_location_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_location_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_location Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->edge_id[index] = edge_id;
	messages->position[index] = position;

}

/**
 * Scatter non partitioned or spatially partitioned location message (for optional messages)
 * @param messages scatter_optional_location_messages Sparse xmachine_message_location_list message list
 * @param message_swap temp xmachine_message_location_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_location_messages(xmachine_message_location_list* messages, xmachine_message_location_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_location_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->edge_id[output_index] = messages_swap->edge_id[index];
		messages->position[output_index] = messages_swap->position[index];				
	}
}

/** reset_location_swaps
 * Reset non partitioned or spatially partitioned location message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_location_swaps(xmachine_message_location_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

/*
 * Load the next graph edge partitioned location messag (either from SM or next batch load)
 * @param messages message list 
 * @param message_bounds edge graph messaging data structure
 * @param edge_id target edge index
 * @param messageIndex index of the message
 * @return boolean indicating if a message was loaded or not.
 */
__device__ bool load_location_message(xmachine_message_location_list* messages, xmachine_message_location_bounds* message_bounds, unsigned int edge_id, unsigned int messageIndex){
	// Define smem stuff
	extern __shared__ int sm_data[];
	char* message_share = (char*)&sm_data[0];

	// If the taget message is greater than the number of messages return false.
	if (messageIndex >= d_message_location_count){
		return false;
	}
	
	// Load the max value from the boundary struct.
	unsigned int firstMessageForEdge = message_bounds->start[edge_id];
	
	unsigned int messageForNextEdge = firstMessageForEdge + message_bounds->count[edge_id];


	// If there are no other messages return false
	if (messageIndex < firstMessageForEdge || messageIndex >= messageForNextEdge){
		return false;
	}

	// Get the message data for the target message
	xmachine_message_location temp_message;
	temp_message._position = messages->_position[messageIndex];
	temp_message.id = messages->id[messageIndex];
	temp_message.edge_id = messages->edge_id[messageIndex];
	temp_message.position = messages->position[messageIndex];
	

	// Load the message into shared memory. No sync?
	
	int message_index = SHARE_INDEX(threadIdx.y * blockDim.x + threadIdx.x, sizeof(xmachine_message_location));
	xmachine_message_location* sm_message = ((xmachine_message_location*)&message_share[message_index]);
	sm_message[0] = temp_message;
	
	return true;
}

/**
 * Get the first message from the location edge partitioned message list
 * @param messages  the message list
 * @param message_bounds boundary data structure for edge partitioned messages
 * @param edge_id target edge for messages
 * @return pointer to the message.
 */
__device__ xmachine_message_location* get_first_location_message(xmachine_message_location_list* messages, xmachine_message_location_bounds* message_bounds, unsigned int edge_id){

	extern __shared__ int sm_data[];
	char* message_share = (char*)&sm_data[0];

	// Get the first index for the target edge.
	unsigned int firstMessageIndex = message_bounds->start[edge_id];

	if (load_location_message(messages, message_bounds, edge_id, firstMessageIndex))
	{
		unsigned int message_index = SHARE_INDEX(threadIdx.y*blockDim.x + threadIdx.x, sizeof(xmachine_message_location));
		return ((xmachine_message_location*)&message_share[message_index]);
	}
	else
	{
		return nullptr;
	}
}

/**
 * Get the next message from the location edge partitioned message list
 * @param messages  the message list
 * @param message_bounds boundary data structure for edge partitioned messages
 * @return pointer to the message.
 */
__device__ xmachine_message_location* get_next_location_message(xmachine_message_location* message, xmachine_message_location_list* messages, xmachine_message_location_bounds* message_bounds){
	extern __shared__ int sm_data[];
	char* message_share = (char*)&sm_data[0];

	if (load_location_message(messages, message_bounds, message->edge_id, message->_position + 1))
	{
		//get conflict free address of 
		unsigned int message_index = SHARE_INDEX(threadIdx.y*blockDim.x + threadIdx.x, sizeof(xmachine_message_location));
		return ((xmachine_message_location*)&message_share[message_index]);
	}
	else {
		return nullptr;
	}
	
}

/**
 * Generate a histogram of location messages per edge index
 * @param local_index
 * @param unsorted_index
 * @param message_counts
 * @param messages
 * @param agent_count
 */
__global__ void hist_location_messages(unsigned int* local_index, unsigned int * unsorted_index, unsigned int* message_counts, xmachine_message_location_list * messages, unsigned int agent_count){
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= agent_count){
		return;
	}
	unsigned int edge_id = messages->edge_id[index];
	unsigned int bin_index = atomicInc((unsigned int*)&message_counts[edge_id], 0xFFFFFFFF);
	local_index[index] = bin_index;
	unsorted_index[index] = edge_id;
}

/**
 * Reorder location messages for edge partitioned communication
 * @param local_index
 * @param unsorted_index
 * @param start_index
 * @param unordered_messages
 * @param ordered_messages
 * @param agent_count
 */
__global__ void reorder_location_messages(unsigned int* local_index, unsigned int* unsorted_index, unsigned int* start_index, xmachine_message_location_list* unordered_messages, xmachine_message_location_list* ordered_messages, unsigned int agent_count){
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= agent_count){
		return;
	}

	unsigned int sorted_index = local_index[index] + start_index[unsorted_index[index]];
	
	// The position value should be updated to reflect the new position.
	ordered_messages->_position[sorted_index] = sorted_index;
	ordered_messages->_scan_input[sorted_index] = unordered_messages->_scan_input[index];

	ordered_messages->id[sorted_index] = unordered_messages->id[index];
	ordered_messages->edge_id[sorted_index] = unordered_messages->edge_id[index];
	ordered_messages->position[sorted_index] = unordered_messages->position[index];
	
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created intent message functions */


/** add_intent_message
 * Add non partitioned or spatially partitioned intent message
 * @param messages xmachine_message_intent_list message list to add too
 * @param id agent variable of type unsigned int
 * @param edge_id agent variable of type unsigned int
 */
__device__ void add_intent_message(xmachine_message_intent_list* messages, unsigned int id, unsigned int edge_id){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_intent_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_intent_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_intent_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_intent Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->edge_id[index] = edge_id;

}

/**
 * Scatter non partitioned or spatially partitioned intent message (for optional messages)
 * @param messages scatter_optional_intent_messages Sparse xmachine_message_intent_list message list
 * @param message_swap temp xmachine_message_intent_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_intent_messages(xmachine_message_intent_list* messages, xmachine_message_intent_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_intent_count;

		//AoS - xmachine_message_intent Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->edge_id[output_index] = messages_swap->edge_id[index];				
	}
}

/** reset_intent_swaps
 * Reset non partitioned or spatially partitioned intent message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_intent_swaps(xmachine_message_intent_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

/*
 * Load the next graph edge partitioned intent messag (either from SM or next batch load)
 * @param messages message list 
 * @param message_bounds edge graph messaging data structure
 * @param edge_id target edge index
 * @param messageIndex index of the message
 * @return boolean indicating if a message was loaded or not.
 */
__device__ bool load_intent_message(xmachine_message_intent_list* messages, xmachine_message_intent_bounds* message_bounds, unsigned int edge_id, unsigned int messageIndex){
	// Define smem stuff
	extern __shared__ int sm_data[];
	char* message_share = (char*)&sm_data[0];

	// If the taget message is greater than the number of messages return false.
	if (messageIndex >= d_message_intent_count){
		return false;
	}
	
	// Load the max value from the boundary struct.
	unsigned int firstMessageForEdge = message_bounds->start[edge_id];
	
	unsigned int messageForNextEdge = firstMessageForEdge + message_bounds->count[edge_id];


	// If there are no other messages return false
	if (messageIndex < firstMessageForEdge || messageIndex >= messageForNextEdge){
		return false;
	}

	// Get the message data for the target message
	xmachine_message_intent temp_message;
	temp_message._position = messages->_position[messageIndex];
	temp_message.id = messages->id[messageIndex];
	temp_message.edge_id = messages->edge_id[messageIndex];
	

	// Load the message into shared memory. No sync?
	
	int message_index = SHARE_INDEX(threadIdx.y * blockDim.x + threadIdx.x, sizeof(xmachine_message_intent));
	xmachine_message_intent* sm_message = ((xmachine_message_intent*)&message_share[message_index]);
	sm_message[0] = temp_message;
	
	return true;
}

/**
 * Get the first message from the intent edge partitioned message list
 * @param messages  the message list
 * @param message_bounds boundary data structure for edge partitioned messages
 * @param edge_id target edge for messages
 * @return pointer to the message.
 */
__device__ xmachine_message_intent* get_first_intent_message(xmachine_message_intent_list* messages, xmachine_message_intent_bounds* message_bounds, unsigned int edge_id){

	extern __shared__ int sm_data[];
	char* message_share = (char*)&sm_data[0];

	// Get the first index for the target edge.
	unsigned int firstMessageIndex = message_bounds->start[edge_id];

	if (load_intent_message(messages, message_bounds, edge_id, firstMessageIndex))
	{
		unsigned int message_index = SHARE_INDEX(threadIdx.y*blockDim.x + threadIdx.x, sizeof(xmachine_message_intent));
		return ((xmachine_message_intent*)&message_share[message_index]);
	}
	else
	{
		return nullptr;
	}
}

/**
 * Get the next message from the intent edge partitioned message list
 * @param messages  the message list
 * @param message_bounds boundary data structure for edge partitioned messages
 * @return pointer to the message.
 */
__device__ xmachine_message_intent* get_next_intent_message(xmachine_message_intent* message, xmachine_message_intent_list* messages, xmachine_message_intent_bounds* message_bounds){
	extern __shared__ int sm_data[];
	char* message_share = (char*)&sm_data[0];

	if (load_intent_message(messages, message_bounds, message->edge_id, message->_position + 1))
	{
		//get conflict free address of 
		unsigned int message_index = SHARE_INDEX(threadIdx.y*blockDim.x + threadIdx.x, sizeof(xmachine_message_intent));
		return ((xmachine_message_intent*)&message_share[message_index]);
	}
	else {
		return nullptr;
	}
	
}

/**
 * Generate a histogram of intent messages per edge index
 * @param local_index
 * @param unsorted_index
 * @param message_counts
 * @param messages
 * @param agent_count
 */
__global__ void hist_intent_messages(unsigned int* local_index, unsigned int * unsorted_index, unsigned int* message_counts, xmachine_message_intent_list * messages, unsigned int agent_count){
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= agent_count){
		return;
	}
	unsigned int edge_id = messages->edge_id[index];
	unsigned int bin_index = atomicInc((unsigned int*)&message_counts[edge_id], 0xFFFFFFFF);
	local_index[index] = bin_index;
	unsorted_index[index] = edge_id;
}

/**
 * Reorder intent messages for edge partitioned communication
 * @param local_index
 * @param unsorted_index
 * @param start_index
 * @param unordered_messages
 * @param ordered_messages
 * @param agent_count
 */
__global__ void reorder_intent_messages(unsigned int* local_index, unsigned int* unsorted_index, unsigned int* start_index, xmachine_message_intent_list* unordered_messages, xmachine_message_intent_list* ordered_messages, unsigned int agent_count){
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index >= agent_count){
		return;
	}

	unsigned int sorted_index = local_index[index] + start_index[unsorted_index[index]];
	
	// The position value should be updated to reflect the new position.
	ordered_messages->_position[sorted_index] = sorted_index;
	ordered_messages->_scan_input[sorted_index] = unordered_messages->_scan_input[index];

	ordered_messages->id[sorted_index] = unordered_messages->id[index];
	ordered_messages->edge_id[sorted_index] = unordered_messages->edge_id[index];
	
}


	
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created GPU kernels  */



/**
 *
 */
__global__ void GPUFLAME_output_location(xmachine_memory_Agent_list* agents, xmachine_message_location_list* location_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_location Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.currentEdge = agents->currentEdge[index];
	agent.nextEdge = agents->nextEdge[index];
	agent.nextEdgeRemainingCapacity = agents->nextEdgeRemainingCapacity[index];
	agent.hasIntent = agents->hasIntent[index];
	agent.position = agents->position[index];
	agent.distanceTravelled = agents->distanceTravelled[index];
	agent.blockedIterationCount = agents->blockedIterationCount[index];
	agent.speed = agents->speed[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.colour = agents->colour[index];

	//FLAME function call
	int dead = !output_location(&agent, location_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_location Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->currentEdge[index] = agent.currentEdge;
	agents->nextEdge[index] = agent.nextEdge;
	agents->nextEdgeRemainingCapacity[index] = agent.nextEdgeRemainingCapacity;
	agents->hasIntent[index] = agent.hasIntent;
	agents->position[index] = agent.position;
	agents->distanceTravelled[index] = agent.distanceTravelled;
	agents->blockedIterationCount[index] = agent.blockedIterationCount;
	agents->speed[index] = agent.speed;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->colour[index] = agent.colour;
}

/**
 *
 */
__global__ void GPUFLAME_read_locations(xmachine_memory_Agent_list* agents, xmachine_message_location_list* location_messages, xmachine_message_location_bounds* message_bounds, xmachine_message_intent_list* intent_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_read_locations Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.currentEdge = agents->currentEdge[index];
	agent.nextEdge = agents->nextEdge[index];
	agent.nextEdgeRemainingCapacity = agents->nextEdgeRemainingCapacity[index];
	agent.hasIntent = agents->hasIntent[index];
	agent.position = agents->position[index];
	agent.distanceTravelled = agents->distanceTravelled[index];
	agent.blockedIterationCount = agents->blockedIterationCount[index];
	agent.speed = agents->speed[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.colour = agents->colour[index];

	//FLAME function call
	int dead = !read_locations(&agent, location_messages, message_bounds, intent_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_read_locations Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->currentEdge[index] = agent.currentEdge;
	agents->nextEdge[index] = agent.nextEdge;
	agents->nextEdgeRemainingCapacity[index] = agent.nextEdgeRemainingCapacity;
	agents->hasIntent[index] = agent.hasIntent;
	agents->position[index] = agent.position;
	agents->distanceTravelled[index] = agent.distanceTravelled;
	agents->blockedIterationCount[index] = agent.blockedIterationCount;
	agents->speed[index] = agent.speed;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->colour[index] = agent.colour;
}

/**
 *
 */
__global__ void GPUFLAME_resolve_intent(xmachine_memory_Agent_list* agents, xmachine_message_intent_list* intent_messages, xmachine_message_intent_bounds* message_bounds, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_resolve_intent Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.currentEdge = agents->currentEdge[index];
	agent.nextEdge = agents->nextEdge[index];
	agent.nextEdgeRemainingCapacity = agents->nextEdgeRemainingCapacity[index];
	agent.hasIntent = agents->hasIntent[index];
	agent.position = agents->position[index];
	agent.distanceTravelled = agents->distanceTravelled[index];
	agent.blockedIterationCount = agents->blockedIterationCount[index];
	agent.speed = agents->speed[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.colour = agents->colour[index];

	//FLAME function call
	int dead = !resolve_intent(&agent, intent_messages, message_bounds, rand48);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_resolve_intent Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->currentEdge[index] = agent.currentEdge;
	agents->nextEdge[index] = agent.nextEdge;
	agents->nextEdgeRemainingCapacity[index] = agent.nextEdgeRemainingCapacity;
	agents->hasIntent[index] = agent.hasIntent;
	agents->position[index] = agent.position;
	agents->distanceTravelled[index] = agent.distanceTravelled;
	agents->blockedIterationCount[index] = agent.blockedIterationCount;
	agents->speed[index] = agent.speed;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->colour[index] = agent.colour;
}

/**
 *
 */
__global__ void GPUFLAME_move(xmachine_memory_Agent_list* agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_Agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_move Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_Agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.currentEdge = agents->currentEdge[index];
	agent.nextEdge = agents->nextEdge[index];
	agent.nextEdgeRemainingCapacity = agents->nextEdgeRemainingCapacity[index];
	agent.hasIntent = agents->hasIntent[index];
	agent.position = agents->position[index];
	agent.distanceTravelled = agents->distanceTravelled[index];
	agent.blockedIterationCount = agents->blockedIterationCount[index];
	agent.speed = agents->speed[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.z = agents->z[index];
	agent.colour = agents->colour[index];

	//FLAME function call
	int dead = !move(&agent);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_move Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->currentEdge[index] = agent.currentEdge;
	agents->nextEdge[index] = agent.nextEdge;
	agents->nextEdgeRemainingCapacity[index] = agent.nextEdgeRemainingCapacity;
	agents->hasIntent[index] = agent.hasIntent;
	agents->position[index] = agent.position;
	agents->distanceTravelled[index] = agent.distanceTravelled;
	agents->blockedIterationCount[index] = agent.blockedIterationCount;
	agents->speed[index] = agent.speed;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->z[index] = agent.z;
	agents->colour[index] = agent.colour;
}

	
	
/* Graph utility functions */

__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_vertex_count(){
#if defined(__CUDA_ARCH__)
	return d_staticGraph_memory_network_ptr->vertex.count;
#else
	return h_staticGraph_memory_network->vertex.count;
#endif 
}
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_edge_count(){
#if defined(__CUDA_ARCH__)
	return d_staticGraph_memory_network_ptr->edge.count;
	#else
	return h_staticGraph_memory_network->edge.count;
#endif 
}
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_vertex_first_edge_index(unsigned int vertexIndex){
	if(vertexIndex <= get_staticGraph_network_vertex_count()){
	#if defined(__CUDA_ARCH__)
		return d_staticGraph_memory_network_ptr->vertex.first_edge_index[vertexIndex];
	#else
		return h_staticGraph_memory_network->vertex.first_edge_index[vertexIndex];
	#endif 
	} else {
		// Return the buffer size, i.e. no messages can start here.
		return staticGraph_network_edge_bufferSize;
	}
}
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_vertex_num_edges(unsigned int vertexIndex){
if(vertexIndex <= get_staticGraph_network_vertex_count()){
	#if defined(__CUDA_ARCH__)
		return d_staticGraph_memory_network_ptr->vertex.first_edge_index[vertexIndex + 1] - d_staticGraph_memory_network_ptr->vertex.first_edge_index[vertexIndex];
	#else
		return h_staticGraph_memory_network->vertex.first_edge_index[vertexIndex + 1] - h_staticGraph_memory_network->vertex.first_edge_index[vertexIndex];
	#endif 
	} else {
		return 0;
	}
}


__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_vertex_id(unsigned int vertexIndex){
	if(vertexIndex < get_staticGraph_network_vertex_count()){
	#if defined(__CUDA_ARCH__)
		return d_staticGraph_memory_network_ptr->vertex.id[ + vertexIndex];
	#else
		return h_staticGraph_memory_network->vertex.id[ + vertexIndex];
	#endif 
	} else {
		return 0;
	}
}
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ float get_staticGraph_network_vertex_x(unsigned int vertexIndex){
	if(vertexIndex < get_staticGraph_network_vertex_count()){
	#if defined(__CUDA_ARCH__)
		return d_staticGraph_memory_network_ptr->vertex.x[ + vertexIndex];
	#else
		return h_staticGraph_memory_network->vertex.x[ + vertexIndex];
	#endif 
	} else {
		return 1.0f;
	}
}
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ float get_staticGraph_network_vertex_y(unsigned int vertexIndex){
	if(vertexIndex < get_staticGraph_network_vertex_count()){
	#if defined(__CUDA_ARCH__)
		return d_staticGraph_memory_network_ptr->vertex.y[ + vertexIndex];
	#else
		return h_staticGraph_memory_network->vertex.y[ + vertexIndex];
	#endif 
	} else {
		return 1.0f;
	}
}
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ float get_staticGraph_network_vertex_z(unsigned int vertexIndex){
	if(vertexIndex < get_staticGraph_network_vertex_count()){
	#if defined(__CUDA_ARCH__)
		return d_staticGraph_memory_network_ptr->vertex.z[ + vertexIndex];
	#else
		return h_staticGraph_memory_network->vertex.z[ + vertexIndex];
	#endif 
	} else {
		return 1.0f;
	}
}
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_edge_id(unsigned int edgeIndex){
	if(edgeIndex < get_staticGraph_network_edge_count()){
	#if defined(__CUDA_ARCH__)
		return d_staticGraph_memory_network_ptr->edge.id[ + edgeIndex];
	#else
		return h_staticGraph_memory_network->edge.id[ + edgeIndex];
	#endif 
	} else {
		return 0;
	}
}
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_edge_source(unsigned int edgeIndex){
	if(edgeIndex < get_staticGraph_network_edge_count()){
	#if defined(__CUDA_ARCH__)
		return d_staticGraph_memory_network_ptr->edge.source[ + edgeIndex];
	#else
		return h_staticGraph_memory_network->edge.source[ + edgeIndex];
	#endif 
	} else {
		return 0;
	}
}
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_edge_destination(unsigned int edgeIndex){
	if(edgeIndex < get_staticGraph_network_edge_count()){
	#if defined(__CUDA_ARCH__)
		return d_staticGraph_memory_network_ptr->edge.destination[ + edgeIndex];
	#else
		return h_staticGraph_memory_network->edge.destination[ + edgeIndex];
	#endif 
	} else {
		return 0;
	}
}
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ float get_staticGraph_network_edge_length(unsigned int edgeIndex){
	if(edgeIndex < get_staticGraph_network_edge_count()){
	#if defined(__CUDA_ARCH__)
		return d_staticGraph_memory_network_ptr->edge.length[ + edgeIndex];
	#else
		return h_staticGraph_memory_network->edge.length[ + edgeIndex];
	#endif 
	} else {
		return 1;
	}
}
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_edge_capacity(unsigned int edgeIndex){
	if(edgeIndex < get_staticGraph_network_edge_count()){
	#if defined(__CUDA_ARCH__)
		return d_staticGraph_memory_network_ptr->edge.capacity[ + edgeIndex];
	#else
		return h_staticGraph_memory_network->edge.capacity[ + edgeIndex];
	#endif 
	} else {
		return 1;
	}
}



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
