
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

__constant__ int d_xmachine_memory_agent_count;

__constant__ int d_xmachine_memory_navmap_count;

__constant__ int d_xmachine_memory_chair_count;

__constant__ int d_xmachine_memory_bed_count;

__constant__ int d_xmachine_memory_doctor_manager_count;

__constant__ int d_xmachine_memory_specialist_manager_count;

__constant__ int d_xmachine_memory_specialist_count;

__constant__ int d_xmachine_memory_receptionist_count;

__constant__ int d_xmachine_memory_agent_generator_count;

__constant__ int d_xmachine_memory_chair_admin_count;

__constant__ int d_xmachine_memory_uci_count;

__constant__ int d_xmachine_memory_box_count;

__constant__ int d_xmachine_memory_doctor_count;

__constant__ int d_xmachine_memory_triage_count;

/* Agent state count constants */

__constant__ int d_xmachine_memory_agent_default_count;

__constant__ int d_xmachine_memory_navmap_static_count;

__constant__ int d_xmachine_memory_chair_defaultChair_count;

__constant__ int d_xmachine_memory_bed_defaultBed_count;

__constant__ int d_xmachine_memory_doctor_manager_defaultDoctorManager_count;

__constant__ int d_xmachine_memory_specialist_manager_defaultSpecialistManager_count;

__constant__ int d_xmachine_memory_specialist_defaultSpecialist_count;

__constant__ int d_xmachine_memory_receptionist_defaultReceptionist_count;

__constant__ int d_xmachine_memory_agent_generator_defaultGenerator_count;

__constant__ int d_xmachine_memory_chair_admin_defaultAdmin_count;

__constant__ int d_xmachine_memory_uci_defaultUci_count;

__constant__ int d_xmachine_memory_box_defaultBox_count;

__constant__ int d_xmachine_memory_doctor_defaultDoctor_count;

__constant__ int d_xmachine_memory_triage_defaultTriage_count;


/* Message constants */

/* pedestrian_location Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_pedestrian_location_count;         /**< message list counter*/
__constant__ int d_message_pedestrian_location_output_type;   /**< message output type (single or optional)*/
//Spatial Partitioning Variables
__constant__ glm::vec3 d_message_pedestrian_location_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
__constant__ glm::vec3 d_message_pedestrian_location_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
__constant__ glm::ivec3 d_message_pedestrian_location_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
__constant__ float d_message_pedestrian_location_radius;                 /**< partition radius (used to determin the size of the partitions) */

/* pedestrian_state Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_pedestrian_state_count;         /**< message list counter*/
__constant__ int d_message_pedestrian_state_output_type;   /**< message output type (single or optional)*/
//Spatial Partitioning Variables
__constant__ glm::vec3 d_message_pedestrian_state_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
__constant__ glm::vec3 d_message_pedestrian_state_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
__constant__ glm::ivec3 d_message_pedestrian_state_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
__constant__ float d_message_pedestrian_state_radius;                 /**< partition radius (used to determin the size of the partitions) */

/* navmap_cell Message variables */
//Discrete Partitioning Variables
__constant__ int d_message_navmap_cell_range;     /**< range of the discrete message*/
__constant__ int d_message_navmap_cell_width;     /**< with of the message grid*/

/* check_in Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_check_in_count;         /**< message list counter*/
__constant__ int d_message_check_in_output_type;   /**< message output type (single or optional)*/

/* check_in_response Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_check_in_response_count;         /**< message list counter*/
__constant__ int d_message_check_in_response_output_type;   /**< message output type (single or optional)*/

/* chair_petition Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_chair_petition_count;         /**< message list counter*/
__constant__ int d_message_chair_petition_output_type;   /**< message output type (single or optional)*/

/* chair_response Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_chair_response_count;         /**< message list counter*/
__constant__ int d_message_chair_response_output_type;   /**< message output type (single or optional)*/

/* free_chair Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_free_chair_count;         /**< message list counter*/
__constant__ int d_message_free_chair_output_type;   /**< message output type (single or optional)*/

/* chair_state Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_chair_state_count;         /**< message list counter*/
__constant__ int d_message_chair_state_output_type;   /**< message output type (single or optional)*/

/* chair_contact Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_chair_contact_count;         /**< message list counter*/
__constant__ int d_message_chair_contact_output_type;   /**< message output type (single or optional)*/

/* bed_state Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_bed_state_count;         /**< message list counter*/
__constant__ int d_message_bed_state_output_type;   /**< message output type (single or optional)*/

/* bed_contact Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_bed_contact_count;         /**< message list counter*/
__constant__ int d_message_bed_contact_output_type;   /**< message output type (single or optional)*/

/* box_petition Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_box_petition_count;         /**< message list counter*/
__constant__ int d_message_box_petition_output_type;   /**< message output type (single or optional)*/

/* box_response Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_box_response_count;         /**< message list counter*/
__constant__ int d_message_box_response_output_type;   /**< message output type (single or optional)*/

/* specialist_reached Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_specialist_reached_count;         /**< message list counter*/
__constant__ int d_message_specialist_reached_output_type;   /**< message output type (single or optional)*/

/* specialist_terminated Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_specialist_terminated_count;         /**< message list counter*/
__constant__ int d_message_specialist_terminated_output_type;   /**< message output type (single or optional)*/

/* free_specialist Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_free_specialist_count;         /**< message list counter*/
__constant__ int d_message_free_specialist_output_type;   /**< message output type (single or optional)*/

/* specialist_petition Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_specialist_petition_count;         /**< message list counter*/
__constant__ int d_message_specialist_petition_output_type;   /**< message output type (single or optional)*/

/* specialist_response Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_specialist_response_count;         /**< message list counter*/
__constant__ int d_message_specialist_response_output_type;   /**< message output type (single or optional)*/

/* doctor_reached Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_doctor_reached_count;         /**< message list counter*/
__constant__ int d_message_doctor_reached_output_type;   /**< message output type (single or optional)*/

/* free_doctor Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_free_doctor_count;         /**< message list counter*/
__constant__ int d_message_free_doctor_output_type;   /**< message output type (single or optional)*/

/* attention_terminated Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_attention_terminated_count;         /**< message list counter*/
__constant__ int d_message_attention_terminated_output_type;   /**< message output type (single or optional)*/

/* doctor_petition Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_doctor_petition_count;         /**< message list counter*/
__constant__ int d_message_doctor_petition_output_type;   /**< message output type (single or optional)*/

/* doctor_response Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_doctor_response_count;         /**< message list counter*/
__constant__ int d_message_doctor_response_output_type;   /**< message output type (single or optional)*/

/* bed_petition Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_bed_petition_count;         /**< message list counter*/
__constant__ int d_message_bed_petition_output_type;   /**< message output type (single or optional)*/

/* bed_response Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_bed_response_count;         /**< message list counter*/
__constant__ int d_message_bed_response_output_type;   /**< message output type (single or optional)*/

/* triage_petition Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_triage_petition_count;         /**< message list counter*/
__constant__ int d_message_triage_petition_output_type;   /**< message output type (single or optional)*/

/* triage_response Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_triage_response_count;         /**< message list counter*/
__constant__ int d_message_triage_response_output_type;   /**< message output type (single or optional)*/

/* free_box Message variables */
/* Non partitioned, spatial partitioned and on-graph partitioned message variables  */
__constant__ int d_message_free_box_count;         /**< message list counter*/
__constant__ int d_message_free_box_output_type;   /**< message output type (single or optional)*/

	

/* Graph Constants */


/* Graph device array pointer(s) */


/* Graph host array pointer(s) */

    
//include each function file

#include "functions.c"
    
/* Texture bindings */
/* pedestrian_location Message Bindings */texture<float, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_x;
__constant__ int d_tex_xmachine_message_pedestrian_location_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_y;
__constant__ int d_tex_xmachine_message_pedestrian_location_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_z;
__constant__ int d_tex_xmachine_message_pedestrian_location_z_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_estado;
__constant__ int d_tex_xmachine_message_pedestrian_location_estado_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_pbm_start;
__constant__ int d_tex_xmachine_message_pedestrian_location_pbm_start_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_pbm_end_or_count;
__constant__ int d_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset;


/* pedestrian_state Message Bindings */texture<float, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_state_x;
__constant__ int d_tex_xmachine_message_pedestrian_state_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_state_y;
__constant__ int d_tex_xmachine_message_pedestrian_state_y_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_state_z;
__constant__ int d_tex_xmachine_message_pedestrian_state_z_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_state_estado;
__constant__ int d_tex_xmachine_message_pedestrian_state_estado_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_state_pbm_start;
__constant__ int d_tex_xmachine_message_pedestrian_state_pbm_start_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_state_pbm_end_or_count;
__constant__ int d_tex_xmachine_message_pedestrian_state_pbm_end_or_count_offset;


/* navmap_cell Message Bindings */texture<int, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_x;
__constant__ int d_tex_xmachine_message_navmap_cell_x_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_y;
__constant__ int d_tex_xmachine_message_navmap_cell_y_offset;texture<int, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_exit_no;
__constant__ int d_tex_xmachine_message_navmap_cell_exit_no_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_height;
__constant__ int d_tex_xmachine_message_navmap_cell_height_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_collision_x;
__constant__ int d_tex_xmachine_message_navmap_cell_collision_x_offset;texture<float, 1, cudaReadModeElementType> tex_xmachine_message_navmap_cell_collision_y;
__constant__ int d_tex_xmachine_message_navmap_cell_collision_y_offset;



























    
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


/** infect_patients_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void infect_patients_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if ((currentState->estado_movimiento[index]!=35)and(currentState->estado[index]==0))
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** infect_patients_UCI_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void infect_patients_UCI_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if ((currentState->estado_movimiento[index]==35)and(currentState->estado[index]==0))
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** output_chair_contact_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void output_chair_contact_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if (currentState->estado_movimiento[index]==4)
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** output_free_chair_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void output_free_chair_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if (currentState->estado_movimiento[index]==39)
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** output_chair_petition_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void output_chair_petition_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if (((currentState->estado_movimiento[index]==1)||(currentState->estado_movimiento[index]==11))||((currentState->estado_movimiento[index]==24)||(currentState->estado_movimiento[index]==29)))
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** receive_chair_response_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void receive_chair_response_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if (((currentState->estado_movimiento[index]==2)||(currentState->estado_movimiento[index]==12))||(currentState->estado_movimiento[index]==25))
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** receive_check_in_response_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void receive_check_in_response_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if ((currentState->estado_movimiento[index]==4)||(currentState->estado_movimiento[index]==8))
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** output_box_petition_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void output_box_petition_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if (currentState->estado_movimiento[index]==19)
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** output_doctor_petition_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void output_doctor_petition_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if ((currentState->estado_movimiento[index]==27)and(currentState->specialist_no[index]==0))
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** receive_doctor_response_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void receive_doctor_response_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if ((currentState->estado_movimiento[index]==28)and(currentState->specialist_no[index]==0))
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** receive_attention_terminated_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void receive_attention_terminated_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if ((currentState->estado_movimiento[index]==32)and(currentState->specialist_no[index]==0))
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** receive_specialist_terminated_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void receive_specialist_terminated_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if ((currentState->estado_movimiento[index]==32)and((currentState->specialist_no[index]>0)and(currentState->specialist_no[index]<6)))
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** output_doctor_reached_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void output_doctor_reached_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if ((currentState->estado_movimiento[index]==31)and(currentState->specialist_no[index]==0))
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** receive_specialist_response_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void receive_specialist_response_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if ((currentState->estado_movimiento[index]==28)and((currentState->specialist_no[index]>0)and(currentState->specialist_no[index]<6)))
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** output_specialist_petition_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void output_specialist_petition_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if ((currentState->estado_movimiento[index]==27)and((currentState->specialist_no[index]>0)and(currentState->specialist_no[index]<6)))
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** output_specialist_reached_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void output_specialist_reached_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if ((currentState->estado_movimiento[index]==31)and((currentState->specialist_no[index]>0)and(currentState->specialist_no[index]<6)))
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** output_bed_petition_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void output_bed_petition_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if (currentState->estado_movimiento[index]==33)
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** receive_bed_response_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void receive_bed_response_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if ((currentState->estado_movimiento[index]==34)||(currentState->estado_movimiento[index]==35))
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** output_bed_contact_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void output_bed_contact_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if (currentState->estado_movimiento[index]==35)
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** output_triage_petition_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void output_triage_petition_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if (currentState->estado_movimiento[index]==14)
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** receive_triage_response_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_list representing agent i the current state
 * @param nextState xmachine_memory_agent_list representing agent i the next state
 */
 __global__ void receive_triage_response_function_filter(xmachine_memory_agent_list* currentState, xmachine_memory_agent_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_count){
	
		//apply the filter
		if (currentState->estado_movimiento[index]==15)
		{	//copy agent data to newstate list
			nextState->id[index] = currentState->id[index];
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->velx[index] = currentState->velx[index];
			nextState->vely[index] = currentState->vely[index];
			nextState->steer_x[index] = currentState->steer_x[index];
			nextState->steer_y[index] = currentState->steer_y[index];
			nextState->height[index] = currentState->height[index];
			nextState->exit_no[index] = currentState->exit_no[index];
			nextState->speed[index] = currentState->speed[index];
			nextState->lod[index] = currentState->lod[index];
			nextState->animate[index] = currentState->animate[index];
			nextState->animate_dir[index] = currentState->animate_dir[index];
			nextState->estado[index] = currentState->estado[index];
			nextState->tick[index] = currentState->tick[index];
			nextState->estado_movimiento[index] = currentState->estado_movimiento[index];
			nextState->go_to_x[index] = currentState->go_to_x[index];
			nextState->go_to_y[index] = currentState->go_to_y[index];
			nextState->checkpoint[index] = currentState->checkpoint[index];
			nextState->chair_no[index] = currentState->chair_no[index];
			nextState->box_no[index] = currentState->box_no[index];
			nextState->doctor_no[index] = currentState->doctor_no[index];
			nextState->specialist_no[index] = currentState->specialist_no[index];
			nextState->bed_no[index] = currentState->bed_no[index];
			nextState->priority[index] = currentState->priority[index];
			nextState->vaccine[index] = currentState->vaccine[index];
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

/** generate_chairs_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_generator_list representing agent i the current state
 * @param nextState xmachine_memory_agent_generator_list representing agent i the next state
 */
 __global__ void generate_chairs_function_filter(xmachine_memory_agent_generator_list* currentState, xmachine_memory_agent_generator_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_generator_count){
	
		//apply the filter
		if (currentState->chairs_generated[index]<35)
		{	//copy agent data to newstate list
			nextState->chairs_generated[index] = currentState->chairs_generated[index];
			nextState->beds_generated[index] = currentState->beds_generated[index];
			nextState->boxes_generated[index] = currentState->boxes_generated[index];
			nextState->doctors_generated[index] = currentState->doctors_generated[index];
			nextState->specialists_generated[index] = currentState->specialists_generated[index];
			nextState->personal_generated[index] = currentState->personal_generated[index];
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

/** generate_beds_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_generator_list representing agent i the current state
 * @param nextState xmachine_memory_agent_generator_list representing agent i the next state
 */
 __global__ void generate_beds_function_filter(xmachine_memory_agent_generator_list* currentState, xmachine_memory_agent_generator_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_generator_count){
	
		//apply the filter
		if (currentState->beds_generated[index]<NUMBER_OF_BEDS)
		{	//copy agent data to newstate list
			nextState->chairs_generated[index] = currentState->chairs_generated[index];
			nextState->beds_generated[index] = currentState->beds_generated[index];
			nextState->boxes_generated[index] = currentState->boxes_generated[index];
			nextState->doctors_generated[index] = currentState->doctors_generated[index];
			nextState->specialists_generated[index] = currentState->specialists_generated[index];
			nextState->personal_generated[index] = currentState->personal_generated[index];
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

/** generate_boxes_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_generator_list representing agent i the current state
 * @param nextState xmachine_memory_agent_generator_list representing agent i the next state
 */
 __global__ void generate_boxes_function_filter(xmachine_memory_agent_generator_list* currentState, xmachine_memory_agent_generator_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_generator_count){
	
		//apply the filter
		if (currentState->boxes_generated[index]<3)
		{	//copy agent data to newstate list
			nextState->chairs_generated[index] = currentState->chairs_generated[index];
			nextState->beds_generated[index] = currentState->beds_generated[index];
			nextState->boxes_generated[index] = currentState->boxes_generated[index];
			nextState->doctors_generated[index] = currentState->doctors_generated[index];
			nextState->specialists_generated[index] = currentState->specialists_generated[index];
			nextState->personal_generated[index] = currentState->personal_generated[index];
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

/** generate_doctors_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_generator_list representing agent i the current state
 * @param nextState xmachine_memory_agent_generator_list representing agent i the next state
 */
 __global__ void generate_doctors_function_filter(xmachine_memory_agent_generator_list* currentState, xmachine_memory_agent_generator_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_generator_count){
	
		//apply the filter
		if (currentState->doctors_generated[index]<4)
		{	//copy agent data to newstate list
			nextState->chairs_generated[index] = currentState->chairs_generated[index];
			nextState->beds_generated[index] = currentState->beds_generated[index];
			nextState->boxes_generated[index] = currentState->boxes_generated[index];
			nextState->doctors_generated[index] = currentState->doctors_generated[index];
			nextState->specialists_generated[index] = currentState->specialists_generated[index];
			nextState->personal_generated[index] = currentState->personal_generated[index];
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

/** generate_specialists_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_generator_list representing agent i the current state
 * @param nextState xmachine_memory_agent_generator_list representing agent i the next state
 */
 __global__ void generate_specialists_function_filter(xmachine_memory_agent_generator_list* currentState, xmachine_memory_agent_generator_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_generator_count){
	
		//apply the filter
		if (currentState->specialists_generated[index]<5)
		{	//copy agent data to newstate list
			nextState->chairs_generated[index] = currentState->chairs_generated[index];
			nextState->beds_generated[index] = currentState->beds_generated[index];
			nextState->boxes_generated[index] = currentState->boxes_generated[index];
			nextState->doctors_generated[index] = currentState->doctors_generated[index];
			nextState->specialists_generated[index] = currentState->specialists_generated[index];
			nextState->personal_generated[index] = currentState->personal_generated[index];
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

/** generate_personal_function_filter
 *	Standard agent condition function. Filters agents from one state list to the next depending on the condition
 * @param currentState xmachine_memory_agent_generator_list representing agent i the current state
 * @param nextState xmachine_memory_agent_generator_list representing agent i the next state
 */
 __global__ void generate_personal_function_filter(xmachine_memory_agent_generator_list* currentState, xmachine_memory_agent_generator_list* nextState)
 {
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//check thread max
	if (index < d_xmachine_memory_agent_generator_count){
	
		//apply the filter
		if (currentState->personal_generated[index]<11)
		{	//copy agent data to newstate list
			nextState->chairs_generated[index] = currentState->chairs_generated[index];
			nextState->beds_generated[index] = currentState->beds_generated[index];
			nextState->boxes_generated[index] = currentState->boxes_generated[index];
			nextState->doctors_generated[index] = currentState->doctors_generated[index];
			nextState->specialists_generated[index] = currentState->specialists_generated[index];
			nextState->personal_generated[index] = currentState->personal_generated[index];
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
/* Dynamically created agent agent functions */

/** reset_agent_scan_input
 * agent agent reset scan input function
 * @param agents The xmachine_memory_agent_list agent list
 */
__global__ void reset_agent_scan_input(xmachine_memory_agent_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_agent_Agents
 * agent scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_agent_list agent list destination
 * @param agents_src xmachine_memory_agent_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_agent_Agents(xmachine_memory_agent_list* agents_dst, xmachine_memory_agent_list* agents_src, int dst_agent_count, int number_to_scatter){
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
		agents_dst->x[output_index] = agents_src->x[index];        
		agents_dst->y[output_index] = agents_src->y[index];        
		agents_dst->velx[output_index] = agents_src->velx[index];        
		agents_dst->vely[output_index] = agents_src->vely[index];        
		agents_dst->steer_x[output_index] = agents_src->steer_x[index];        
		agents_dst->steer_y[output_index] = agents_src->steer_y[index];        
		agents_dst->height[output_index] = agents_src->height[index];        
		agents_dst->exit_no[output_index] = agents_src->exit_no[index];        
		agents_dst->speed[output_index] = agents_src->speed[index];        
		agents_dst->lod[output_index] = agents_src->lod[index];        
		agents_dst->animate[output_index] = agents_src->animate[index];        
		agents_dst->animate_dir[output_index] = agents_src->animate_dir[index];        
		agents_dst->estado[output_index] = agents_src->estado[index];        
		agents_dst->tick[output_index] = agents_src->tick[index];        
		agents_dst->estado_movimiento[output_index] = agents_src->estado_movimiento[index];        
		agents_dst->go_to_x[output_index] = agents_src->go_to_x[index];        
		agents_dst->go_to_y[output_index] = agents_src->go_to_y[index];        
		agents_dst->checkpoint[output_index] = agents_src->checkpoint[index];        
		agents_dst->chair_no[output_index] = agents_src->chair_no[index];        
		agents_dst->box_no[output_index] = agents_src->box_no[index];        
		agents_dst->doctor_no[output_index] = agents_src->doctor_no[index];        
		agents_dst->specialist_no[output_index] = agents_src->specialist_no[index];        
		agents_dst->bed_no[output_index] = agents_src->bed_no[index];        
		agents_dst->priority[output_index] = agents_src->priority[index];        
		agents_dst->vaccine[output_index] = agents_src->vaccine[index];
	}
}

/** append_agent_Agents
 * agent scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_agent_list agent list destination
 * @param agents_src xmachine_memory_agent_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_agent_Agents(xmachine_memory_agent_list* agents_dst, xmachine_memory_agent_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->x[output_index] = agents_src->x[index];
	    agents_dst->y[output_index] = agents_src->y[index];
	    agents_dst->velx[output_index] = agents_src->velx[index];
	    agents_dst->vely[output_index] = agents_src->vely[index];
	    agents_dst->steer_x[output_index] = agents_src->steer_x[index];
	    agents_dst->steer_y[output_index] = agents_src->steer_y[index];
	    agents_dst->height[output_index] = agents_src->height[index];
	    agents_dst->exit_no[output_index] = agents_src->exit_no[index];
	    agents_dst->speed[output_index] = agents_src->speed[index];
	    agents_dst->lod[output_index] = agents_src->lod[index];
	    agents_dst->animate[output_index] = agents_src->animate[index];
	    agents_dst->animate_dir[output_index] = agents_src->animate_dir[index];
	    agents_dst->estado[output_index] = agents_src->estado[index];
	    agents_dst->tick[output_index] = agents_src->tick[index];
	    agents_dst->estado_movimiento[output_index] = agents_src->estado_movimiento[index];
	    agents_dst->go_to_x[output_index] = agents_src->go_to_x[index];
	    agents_dst->go_to_y[output_index] = agents_src->go_to_y[index];
	    agents_dst->checkpoint[output_index] = agents_src->checkpoint[index];
	    agents_dst->chair_no[output_index] = agents_src->chair_no[index];
	    agents_dst->box_no[output_index] = agents_src->box_no[index];
	    agents_dst->doctor_no[output_index] = agents_src->doctor_no[index];
	    agents_dst->specialist_no[output_index] = agents_src->specialist_no[index];
	    agents_dst->bed_no[output_index] = agents_src->bed_no[index];
	    agents_dst->priority[output_index] = agents_src->priority[index];
	    agents_dst->vaccine[output_index] = agents_src->vaccine[index];
    }
}

/** add_agent_agent
 * Continuous agent agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_agent_list to add agents to 
 * @param id agent variable of type int
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param velx agent variable of type float
 * @param vely agent variable of type float
 * @param steer_x agent variable of type float
 * @param steer_y agent variable of type float
 * @param height agent variable of type float
 * @param exit_no agent variable of type int
 * @param speed agent variable of type float
 * @param lod agent variable of type int
 * @param animate agent variable of type float
 * @param animate_dir agent variable of type int
 * @param estado agent variable of type int
 * @param tick agent variable of type int
 * @param estado_movimiento agent variable of type unsigned int
 * @param go_to_x agent variable of type unsigned int
 * @param go_to_y agent variable of type unsigned int
 * @param checkpoint agent variable of type unsigned int
 * @param chair_no agent variable of type int
 * @param box_no agent variable of type unsigned int
 * @param doctor_no agent variable of type unsigned int
 * @param specialist_no agent variable of type unsigned int
 * @param bed_no agent variable of type unsigned int
 * @param priority agent variable of type unsigned int
 * @param vaccine agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_agent_agent(xmachine_memory_agent_list* agents, int id, float x, float y, float velx, float vely, float steer_x, float steer_y, float height, int exit_no, float speed, int lod, float animate, int animate_dir, int estado, int tick, unsigned int estado_movimiento, unsigned int go_to_x, unsigned int go_to_y, unsigned int checkpoint, int chair_no, unsigned int box_no, unsigned int doctor_no, unsigned int specialist_no, unsigned int bed_no, unsigned int priority, unsigned int vaccine){
	
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
	agents->x[index] = x;
	agents->y[index] = y;
	agents->velx[index] = velx;
	agents->vely[index] = vely;
	agents->steer_x[index] = steer_x;
	agents->steer_y[index] = steer_y;
	agents->height[index] = height;
	agents->exit_no[index] = exit_no;
	agents->speed[index] = speed;
	agents->lod[index] = lod;
	agents->animate[index] = animate;
	agents->animate_dir[index] = animate_dir;
	agents->estado[index] = estado;
	agents->tick[index] = tick;
	agents->estado_movimiento[index] = estado_movimiento;
	agents->go_to_x[index] = go_to_x;
	agents->go_to_y[index] = go_to_y;
	agents->checkpoint[index] = checkpoint;
	agents->chair_no[index] = chair_no;
	agents->box_no[index] = box_no;
	agents->doctor_no[index] = doctor_no;
	agents->specialist_no[index] = specialist_no;
	agents->bed_no[index] = bed_no;
	agents->priority[index] = priority;
	agents->vaccine[index] = vaccine;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_agent_agent(xmachine_memory_agent_list* agents, int id, float x, float y, float velx, float vely, float steer_x, float steer_y, float height, int exit_no, float speed, int lod, float animate, int animate_dir, int estado, int tick, unsigned int estado_movimiento, unsigned int go_to_x, unsigned int go_to_y, unsigned int checkpoint, int chair_no, unsigned int box_no, unsigned int doctor_no, unsigned int specialist_no, unsigned int bed_no, unsigned int priority, unsigned int vaccine){
    add_agent_agent<DISCRETE_2D>(agents, id, x, y, velx, vely, steer_x, steer_y, height, exit_no, speed, lod, animate, animate_dir, estado, tick, estado_movimiento, go_to_x, go_to_y, checkpoint, chair_no, box_no, doctor_no, specialist_no, bed_no, priority, vaccine);
}

/** reorder_agent_agents
 * Continuous agent agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_agent_agents(unsigned int* values, xmachine_memory_agent_list* unordered_agents, xmachine_memory_agent_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->x[index] = unordered_agents->x[old_pos];
	ordered_agents->y[index] = unordered_agents->y[old_pos];
	ordered_agents->velx[index] = unordered_agents->velx[old_pos];
	ordered_agents->vely[index] = unordered_agents->vely[old_pos];
	ordered_agents->steer_x[index] = unordered_agents->steer_x[old_pos];
	ordered_agents->steer_y[index] = unordered_agents->steer_y[old_pos];
	ordered_agents->height[index] = unordered_agents->height[old_pos];
	ordered_agents->exit_no[index] = unordered_agents->exit_no[old_pos];
	ordered_agents->speed[index] = unordered_agents->speed[old_pos];
	ordered_agents->lod[index] = unordered_agents->lod[old_pos];
	ordered_agents->animate[index] = unordered_agents->animate[old_pos];
	ordered_agents->animate_dir[index] = unordered_agents->animate_dir[old_pos];
	ordered_agents->estado[index] = unordered_agents->estado[old_pos];
	ordered_agents->tick[index] = unordered_agents->tick[old_pos];
	ordered_agents->estado_movimiento[index] = unordered_agents->estado_movimiento[old_pos];
	ordered_agents->go_to_x[index] = unordered_agents->go_to_x[old_pos];
	ordered_agents->go_to_y[index] = unordered_agents->go_to_y[old_pos];
	ordered_agents->checkpoint[index] = unordered_agents->checkpoint[old_pos];
	ordered_agents->chair_no[index] = unordered_agents->chair_no[old_pos];
	ordered_agents->box_no[index] = unordered_agents->box_no[old_pos];
	ordered_agents->doctor_no[index] = unordered_agents->doctor_no[old_pos];
	ordered_agents->specialist_no[index] = unordered_agents->specialist_no[old_pos];
	ordered_agents->bed_no[index] = unordered_agents->bed_no[old_pos];
	ordered_agents->priority[index] = unordered_agents->priority[old_pos];
	ordered_agents->vaccine[index] = unordered_agents->vaccine[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created navmap agent functions */

/** reset_navmap_scan_input
 * navmap agent reset scan input function
 * @param agents The xmachine_memory_navmap_list agent list
 */
__global__ void reset_navmap_scan_input(xmachine_memory_navmap_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created chair agent functions */

/** reset_chair_scan_input
 * chair agent reset scan input function
 * @param agents The xmachine_memory_chair_list agent list
 */
__global__ void reset_chair_scan_input(xmachine_memory_chair_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_chair_Agents
 * chair scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_chair_list agent list destination
 * @param agents_src xmachine_memory_chair_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_chair_Agents(xmachine_memory_chair_list* agents_dst, xmachine_memory_chair_list* agents_src, int dst_agent_count, int number_to_scatter){
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
		agents_dst->tick[output_index] = agents_src->tick[index];        
		agents_dst->x[output_index] = agents_src->x[index];        
		agents_dst->y[output_index] = agents_src->y[index];        
		agents_dst->state[output_index] = agents_src->state[index];
	}
}

/** append_chair_Agents
 * chair scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_chair_list agent list destination
 * @param agents_src xmachine_memory_chair_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_chair_Agents(xmachine_memory_chair_list* agents_dst, xmachine_memory_chair_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->tick[output_index] = agents_src->tick[index];
	    agents_dst->x[output_index] = agents_src->x[index];
	    agents_dst->y[output_index] = agents_src->y[index];
	    agents_dst->state[output_index] = agents_src->state[index];
    }
}

/** add_chair_agent
 * Continuous chair agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_chair_list to add agents to 
 * @param id agent variable of type int
 * @param tick agent variable of type int
 * @param x agent variable of type int
 * @param y agent variable of type int
 * @param state agent variable of type int
 */
template <int AGENT_TYPE>
__device__ void add_chair_agent(xmachine_memory_chair_list* agents, int id, int tick, int x, int y, int state){
	
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
	agents->tick[index] = tick;
	agents->x[index] = x;
	agents->y[index] = y;
	agents->state[index] = state;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_chair_agent(xmachine_memory_chair_list* agents, int id, int tick, int x, int y, int state){
    add_chair_agent<DISCRETE_2D>(agents, id, tick, x, y, state);
}

/** reorder_chair_agents
 * Continuous chair agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_chair_agents(unsigned int* values, xmachine_memory_chair_list* unordered_agents, xmachine_memory_chair_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->tick[index] = unordered_agents->tick[old_pos];
	ordered_agents->x[index] = unordered_agents->x[old_pos];
	ordered_agents->y[index] = unordered_agents->y[old_pos];
	ordered_agents->state[index] = unordered_agents->state[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created bed agent functions */

/** reset_bed_scan_input
 * bed agent reset scan input function
 * @param agents The xmachine_memory_bed_list agent list
 */
__global__ void reset_bed_scan_input(xmachine_memory_bed_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_bed_Agents
 * bed scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_bed_list agent list destination
 * @param agents_src xmachine_memory_bed_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_bed_Agents(xmachine_memory_bed_list* agents_dst, xmachine_memory_bed_list* agents_src, int dst_agent_count, int number_to_scatter){
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
		agents_dst->tick[output_index] = agents_src->tick[index];        
		agents_dst->state[output_index] = agents_src->state[index];
	}
}

/** append_bed_Agents
 * bed scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_bed_list agent list destination
 * @param agents_src xmachine_memory_bed_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_bed_Agents(xmachine_memory_bed_list* agents_dst, xmachine_memory_bed_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->tick[output_index] = agents_src->tick[index];
	    agents_dst->state[output_index] = agents_src->state[index];
    }
}

/** add_bed_agent
 * Continuous bed agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_bed_list to add agents to 
 * @param id agent variable of type int
 * @param tick agent variable of type int
 * @param state agent variable of type int
 */
template <int AGENT_TYPE>
__device__ void add_bed_agent(xmachine_memory_bed_list* agents, int id, int tick, int state){
	
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
	agents->tick[index] = tick;
	agents->state[index] = state;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_bed_agent(xmachine_memory_bed_list* agents, int id, int tick, int state){
    add_bed_agent<DISCRETE_2D>(agents, id, tick, state);
}

/** reorder_bed_agents
 * Continuous bed agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_bed_agents(unsigned int* values, xmachine_memory_bed_list* unordered_agents, xmachine_memory_bed_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->tick[index] = unordered_agents->tick[old_pos];
	ordered_agents->state[index] = unordered_agents->state[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created doctor_manager agent functions */

/** reset_doctor_manager_scan_input
 * doctor_manager agent reset scan input function
 * @param agents The xmachine_memory_doctor_manager_list agent list
 */
__global__ void reset_doctor_manager_scan_input(xmachine_memory_doctor_manager_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_doctor_manager_Agents
 * doctor_manager scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_doctor_manager_list agent list destination
 * @param agents_src xmachine_memory_doctor_manager_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_doctor_manager_Agents(xmachine_memory_doctor_manager_list* agents_dst, xmachine_memory_doctor_manager_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->tick[output_index] = agents_src->tick[index];        
		agents_dst->rear[output_index] = agents_src->rear[index];        
		agents_dst->size[output_index] = agents_src->size[index];
	    for (int i=0; i<4; i++){
	      agents_dst->doctors_occupied[(i*xmachine_memory_doctor_manager_MAX)+output_index] = agents_src->doctors_occupied[(i*xmachine_memory_doctor_manager_MAX)+index];
	    }        
		agents_dst->free_doctors[output_index] = agents_src->free_doctors[index];
	    for (int i=0; i<35; i++){
	      agents_dst->patientQueue[(i*xmachine_memory_doctor_manager_MAX)+output_index] = agents_src->patientQueue[(i*xmachine_memory_doctor_manager_MAX)+index];
	    }
	}
}

/** append_doctor_manager_Agents
 * doctor_manager scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_doctor_manager_list agent list destination
 * @param agents_src xmachine_memory_doctor_manager_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_doctor_manager_Agents(xmachine_memory_doctor_manager_list* agents_dst, xmachine_memory_doctor_manager_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->tick[output_index] = agents_src->tick[index];
	    agents_dst->rear[output_index] = agents_src->rear[index];
	    agents_dst->size[output_index] = agents_src->size[index];
	    for (int i=0; i<4; i++){
	      agents_dst->doctors_occupied[(i*xmachine_memory_doctor_manager_MAX)+output_index] = agents_src->doctors_occupied[(i*xmachine_memory_doctor_manager_MAX)+index];
	    }
	    agents_dst->free_doctors[output_index] = agents_src->free_doctors[index];
	    for (int i=0; i<35; i++){
	      agents_dst->patientQueue[(i*xmachine_memory_doctor_manager_MAX)+output_index] = agents_src->patientQueue[(i*xmachine_memory_doctor_manager_MAX)+index];
	    }
    }
}

/** add_doctor_manager_agent
 * Continuous doctor_manager agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_doctor_manager_list to add agents to 
 * @param tick agent variable of type unsigned int
 * @param rear agent variable of type unsigned int
 * @param size agent variable of type unsigned int
 * @param doctors_occupied agent variable of type int
 * @param free_doctors agent variable of type unsigned int
 * @param patientQueue agent variable of type ivec2
 */
template <int AGENT_TYPE>
__device__ void add_doctor_manager_agent(xmachine_memory_doctor_manager_list* agents, unsigned int tick, unsigned int rear, unsigned int size, unsigned int free_doctors){
	
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
	agents->tick[index] = tick;
	agents->rear[index] = rear;
	agents->size[index] = size;
	agents->free_doctors[index] = free_doctors;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_doctor_manager_agent(xmachine_memory_doctor_manager_list* agents, unsigned int tick, unsigned int rear, unsigned int size, unsigned int free_doctors){
    add_doctor_manager_agent<DISCRETE_2D>(agents, tick, rear, size, free_doctors);
}

/** reorder_doctor_manager_agents
 * Continuous doctor_manager agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_doctor_manager_agents(unsigned int* values, xmachine_memory_doctor_manager_list* unordered_agents, xmachine_memory_doctor_manager_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->tick[index] = unordered_agents->tick[old_pos];
	ordered_agents->rear[index] = unordered_agents->rear[old_pos];
	ordered_agents->size[index] = unordered_agents->size[old_pos];
	for (int i=0; i<4; i++){
	  ordered_agents->doctors_occupied[(i*xmachine_memory_doctor_manager_MAX)+index] = unordered_agents->doctors_occupied[(i*xmachine_memory_doctor_manager_MAX)+old_pos];
	}
	ordered_agents->free_doctors[index] = unordered_agents->free_doctors[old_pos];
	for (int i=0; i<35; i++){
	  ordered_agents->patientQueue[(i*xmachine_memory_doctor_manager_MAX)+index] = unordered_agents->patientQueue[(i*xmachine_memory_doctor_manager_MAX)+old_pos];
	}
}

/** get_doctor_manager_agent_array_value
 *  Template function for accessing doctor_manager agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_doctor_manager_agent_array_value(T *array, uint index){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    return array[index*xmachine_memory_doctor_manager_MAX];
    } else {
    	// Return the default value for this data type 
	    return 0;
    }
}

/** set_doctor_manager_agent_array_value
 *  Template function for setting doctor_manager agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_doctor_manager_agent_array_value(T *array, uint index, T value){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    array[index*xmachine_memory_doctor_manager_MAX] = value;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created specialist_manager agent functions */

/** reset_specialist_manager_scan_input
 * specialist_manager agent reset scan input function
 * @param agents The xmachine_memory_specialist_manager_list agent list
 */
__global__ void reset_specialist_manager_scan_input(xmachine_memory_specialist_manager_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_specialist_manager_Agents
 * specialist_manager scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_specialist_manager_list agent list destination
 * @param agents_src xmachine_memory_specialist_manager_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_specialist_manager_Agents(xmachine_memory_specialist_manager_list* agents_dst, xmachine_memory_specialist_manager_list* agents_src, int dst_agent_count, int number_to_scatter){
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
	    for (int i=0; i<5; i++){
	      agents_dst->tick[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->tick[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	    for (int i=0; i<5; i++){
	      agents_dst->free_specialist[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->free_specialist[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	    for (int i=0; i<5; i++){
	      agents_dst->rear[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->rear[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	    for (int i=0; i<5; i++){
	      agents_dst->size[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->size[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	    for (int i=0; i<35; i++){
	      agents_dst->surgicalQueue[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->surgicalQueue[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	    for (int i=0; i<35; i++){
	      agents_dst->pediatricsQueue[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->pediatricsQueue[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	    for (int i=0; i<35; i++){
	      agents_dst->gynecologistQueue[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->gynecologistQueue[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	    for (int i=0; i<35; i++){
	      agents_dst->geriatricsQueue[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->geriatricsQueue[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	    for (int i=0; i<35; i++){
	      agents_dst->psychiatristQueue[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->psychiatristQueue[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	}
}

/** append_specialist_manager_Agents
 * specialist_manager scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_specialist_manager_list agent list destination
 * @param agents_src xmachine_memory_specialist_manager_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_specialist_manager_Agents(xmachine_memory_specialist_manager_list* agents_dst, xmachine_memory_specialist_manager_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    for (int i=0; i<5; i++){
	      agents_dst->tick[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->tick[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	    for (int i=0; i<5; i++){
	      agents_dst->free_specialist[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->free_specialist[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	    for (int i=0; i<5; i++){
	      agents_dst->rear[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->rear[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	    for (int i=0; i<5; i++){
	      agents_dst->size[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->size[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	    for (int i=0; i<35; i++){
	      agents_dst->surgicalQueue[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->surgicalQueue[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	    for (int i=0; i<35; i++){
	      agents_dst->pediatricsQueue[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->pediatricsQueue[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	    for (int i=0; i<35; i++){
	      agents_dst->gynecologistQueue[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->gynecologistQueue[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	    for (int i=0; i<35; i++){
	      agents_dst->geriatricsQueue[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->geriatricsQueue[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
	    for (int i=0; i<35; i++){
	      agents_dst->psychiatristQueue[(i*xmachine_memory_specialist_manager_MAX)+output_index] = agents_src->psychiatristQueue[(i*xmachine_memory_specialist_manager_MAX)+index];
	    }
    }
}

/** add_specialist_manager_agent
 * Continuous specialist_manager agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_specialist_manager_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param tick agent variable of type unsigned int
 * @param free_specialist agent variable of type unsigned int
 * @param rear agent variable of type unsigned int
 * @param size agent variable of type unsigned int
 * @param surgicalQueue agent variable of type ivec2
 * @param pediatricsQueue agent variable of type ivec2
 * @param gynecologistQueue agent variable of type ivec2
 * @param geriatricsQueue agent variable of type ivec2
 * @param psychiatristQueue agent variable of type ivec2
 */
template <int AGENT_TYPE>
__device__ void add_specialist_manager_agent(xmachine_memory_specialist_manager_list* agents, unsigned int id){
	
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

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_specialist_manager_agent(xmachine_memory_specialist_manager_list* agents, unsigned int id){
    add_specialist_manager_agent<DISCRETE_2D>(agents, id);
}

/** reorder_specialist_manager_agents
 * Continuous specialist_manager agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_specialist_manager_agents(unsigned int* values, xmachine_memory_specialist_manager_list* unordered_agents, xmachine_memory_specialist_manager_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	for (int i=0; i<5; i++){
	  ordered_agents->tick[(i*xmachine_memory_specialist_manager_MAX)+index] = unordered_agents->tick[(i*xmachine_memory_specialist_manager_MAX)+old_pos];
	}
	for (int i=0; i<5; i++){
	  ordered_agents->free_specialist[(i*xmachine_memory_specialist_manager_MAX)+index] = unordered_agents->free_specialist[(i*xmachine_memory_specialist_manager_MAX)+old_pos];
	}
	for (int i=0; i<5; i++){
	  ordered_agents->rear[(i*xmachine_memory_specialist_manager_MAX)+index] = unordered_agents->rear[(i*xmachine_memory_specialist_manager_MAX)+old_pos];
	}
	for (int i=0; i<5; i++){
	  ordered_agents->size[(i*xmachine_memory_specialist_manager_MAX)+index] = unordered_agents->size[(i*xmachine_memory_specialist_manager_MAX)+old_pos];
	}
	for (int i=0; i<35; i++){
	  ordered_agents->surgicalQueue[(i*xmachine_memory_specialist_manager_MAX)+index] = unordered_agents->surgicalQueue[(i*xmachine_memory_specialist_manager_MAX)+old_pos];
	}
	for (int i=0; i<35; i++){
	  ordered_agents->pediatricsQueue[(i*xmachine_memory_specialist_manager_MAX)+index] = unordered_agents->pediatricsQueue[(i*xmachine_memory_specialist_manager_MAX)+old_pos];
	}
	for (int i=0; i<35; i++){
	  ordered_agents->gynecologistQueue[(i*xmachine_memory_specialist_manager_MAX)+index] = unordered_agents->gynecologistQueue[(i*xmachine_memory_specialist_manager_MAX)+old_pos];
	}
	for (int i=0; i<35; i++){
	  ordered_agents->geriatricsQueue[(i*xmachine_memory_specialist_manager_MAX)+index] = unordered_agents->geriatricsQueue[(i*xmachine_memory_specialist_manager_MAX)+old_pos];
	}
	for (int i=0; i<35; i++){
	  ordered_agents->psychiatristQueue[(i*xmachine_memory_specialist_manager_MAX)+index] = unordered_agents->psychiatristQueue[(i*xmachine_memory_specialist_manager_MAX)+old_pos];
	}
}

/** get_specialist_manager_agent_array_value
 *  Template function for accessing specialist_manager agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_specialist_manager_agent_array_value(T *array, uint index){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    return array[index*xmachine_memory_specialist_manager_MAX];
    } else {
    	// Return the default value for this data type 
	    return 0;
    }
}

/** set_specialist_manager_agent_array_value
 *  Template function for setting specialist_manager agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_specialist_manager_agent_array_value(T *array, uint index, T value){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    array[index*xmachine_memory_specialist_manager_MAX] = value;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created specialist agent functions */

/** reset_specialist_scan_input
 * specialist agent reset scan input function
 * @param agents The xmachine_memory_specialist_list agent list
 */
__global__ void reset_specialist_scan_input(xmachine_memory_specialist_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_specialist_Agents
 * specialist scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_specialist_list agent list destination
 * @param agents_src xmachine_memory_specialist_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_specialist_Agents(xmachine_memory_specialist_list* agents_dst, xmachine_memory_specialist_list* agents_src, int dst_agent_count, int number_to_scatter){
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
		agents_dst->current_patient[output_index] = agents_src->current_patient[index];        
		agents_dst->tick[output_index] = agents_src->tick[index];
	}
}

/** append_specialist_Agents
 * specialist scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_specialist_list agent list destination
 * @param agents_src xmachine_memory_specialist_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_specialist_Agents(xmachine_memory_specialist_list* agents_dst, xmachine_memory_specialist_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->current_patient[output_index] = agents_src->current_patient[index];
	    agents_dst->tick[output_index] = agents_src->tick[index];
    }
}

/** add_specialist_agent
 * Continuous specialist agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_specialist_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param current_patient agent variable of type unsigned int
 * @param tick agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_specialist_agent(xmachine_memory_specialist_list* agents, unsigned int id, unsigned int current_patient, unsigned int tick){
	
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
	agents->current_patient[index] = current_patient;
	agents->tick[index] = tick;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_specialist_agent(xmachine_memory_specialist_list* agents, unsigned int id, unsigned int current_patient, unsigned int tick){
    add_specialist_agent<DISCRETE_2D>(agents, id, current_patient, tick);
}

/** reorder_specialist_agents
 * Continuous specialist agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_specialist_agents(unsigned int* values, xmachine_memory_specialist_list* unordered_agents, xmachine_memory_specialist_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->current_patient[index] = unordered_agents->current_patient[old_pos];
	ordered_agents->tick[index] = unordered_agents->tick[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created receptionist agent functions */

/** reset_receptionist_scan_input
 * receptionist agent reset scan input function
 * @param agents The xmachine_memory_receptionist_list agent list
 */
__global__ void reset_receptionist_scan_input(xmachine_memory_receptionist_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_receptionist_Agents
 * receptionist scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_receptionist_list agent list destination
 * @param agents_src xmachine_memory_receptionist_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_receptionist_Agents(xmachine_memory_receptionist_list* agents_dst, xmachine_memory_receptionist_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;
	    for (int i=0; i<35; i++){
	      agents_dst->patientQueue[(i*xmachine_memory_receptionist_MAX)+output_index] = agents_src->patientQueue[(i*xmachine_memory_receptionist_MAX)+index];
	    }        
		agents_dst->front[output_index] = agents_src->front[index];        
		agents_dst->rear[output_index] = agents_src->rear[index];        
		agents_dst->size[output_index] = agents_src->size[index];        
		agents_dst->tick[output_index] = agents_src->tick[index];        
		agents_dst->current_patient[output_index] = agents_src->current_patient[index];        
		agents_dst->attend_patient[output_index] = agents_src->attend_patient[index];
	}
}

/** append_receptionist_Agents
 * receptionist scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_receptionist_list agent list destination
 * @param agents_src xmachine_memory_receptionist_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_receptionist_Agents(xmachine_memory_receptionist_list* agents_dst, xmachine_memory_receptionist_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    for (int i=0; i<35; i++){
	      agents_dst->patientQueue[(i*xmachine_memory_receptionist_MAX)+output_index] = agents_src->patientQueue[(i*xmachine_memory_receptionist_MAX)+index];
	    }
	    agents_dst->front[output_index] = agents_src->front[index];
	    agents_dst->rear[output_index] = agents_src->rear[index];
	    agents_dst->size[output_index] = agents_src->size[index];
	    agents_dst->tick[output_index] = agents_src->tick[index];
	    agents_dst->current_patient[output_index] = agents_src->current_patient[index];
	    agents_dst->attend_patient[output_index] = agents_src->attend_patient[index];
    }
}

/** add_receptionist_agent
 * Continuous receptionist agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_receptionist_list to add agents to 
 * @param patientQueue agent variable of type unsigned int
 * @param front agent variable of type unsigned int
 * @param rear agent variable of type unsigned int
 * @param size agent variable of type unsigned int
 * @param tick agent variable of type unsigned int
 * @param current_patient agent variable of type int
 * @param attend_patient agent variable of type int
 */
template <int AGENT_TYPE>
__device__ void add_receptionist_agent(xmachine_memory_receptionist_list* agents, unsigned int front, unsigned int rear, unsigned int size, unsigned int tick, int current_patient, int attend_patient){
	
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
	agents->front[index] = front;
	agents->rear[index] = rear;
	agents->size[index] = size;
	agents->tick[index] = tick;
	agents->current_patient[index] = current_patient;
	agents->attend_patient[index] = attend_patient;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_receptionist_agent(xmachine_memory_receptionist_list* agents, unsigned int front, unsigned int rear, unsigned int size, unsigned int tick, int current_patient, int attend_patient){
    add_receptionist_agent<DISCRETE_2D>(agents, front, rear, size, tick, current_patient, attend_patient);
}

/** reorder_receptionist_agents
 * Continuous receptionist agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_receptionist_agents(unsigned int* values, xmachine_memory_receptionist_list* unordered_agents, xmachine_memory_receptionist_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	for (int i=0; i<35; i++){
	  ordered_agents->patientQueue[(i*xmachine_memory_receptionist_MAX)+index] = unordered_agents->patientQueue[(i*xmachine_memory_receptionist_MAX)+old_pos];
	}
	ordered_agents->front[index] = unordered_agents->front[old_pos];
	ordered_agents->rear[index] = unordered_agents->rear[old_pos];
	ordered_agents->size[index] = unordered_agents->size[old_pos];
	ordered_agents->tick[index] = unordered_agents->tick[old_pos];
	ordered_agents->current_patient[index] = unordered_agents->current_patient[old_pos];
	ordered_agents->attend_patient[index] = unordered_agents->attend_patient[old_pos];
}

/** get_receptionist_agent_array_value
 *  Template function for accessing receptionist agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_receptionist_agent_array_value(T *array, uint index){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    return array[index*xmachine_memory_receptionist_MAX];
    } else {
    	// Return the default value for this data type 
	    return 0;
    }
}

/** set_receptionist_agent_array_value
 *  Template function for setting receptionist agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_receptionist_agent_array_value(T *array, uint index, T value){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    array[index*xmachine_memory_receptionist_MAX] = value;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created agent_generator agent functions */

/** reset_agent_generator_scan_input
 * agent_generator agent reset scan input function
 * @param agents The xmachine_memory_agent_generator_list agent list
 */
__global__ void reset_agent_generator_scan_input(xmachine_memory_agent_generator_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_agent_generator_Agents
 * agent_generator scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_agent_generator_list agent list destination
 * @param agents_src xmachine_memory_agent_generator_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_agent_generator_Agents(xmachine_memory_agent_generator_list* agents_dst, xmachine_memory_agent_generator_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->chairs_generated[output_index] = agents_src->chairs_generated[index];        
		agents_dst->beds_generated[output_index] = agents_src->beds_generated[index];        
		agents_dst->boxes_generated[output_index] = agents_src->boxes_generated[index];        
		agents_dst->doctors_generated[output_index] = agents_src->doctors_generated[index];        
		agents_dst->specialists_generated[output_index] = agents_src->specialists_generated[index];        
		agents_dst->personal_generated[output_index] = agents_src->personal_generated[index];
	}
}

/** append_agent_generator_Agents
 * agent_generator scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_agent_generator_list agent list destination
 * @param agents_src xmachine_memory_agent_generator_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_agent_generator_Agents(xmachine_memory_agent_generator_list* agents_dst, xmachine_memory_agent_generator_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->chairs_generated[output_index] = agents_src->chairs_generated[index];
	    agents_dst->beds_generated[output_index] = agents_src->beds_generated[index];
	    agents_dst->boxes_generated[output_index] = agents_src->boxes_generated[index];
	    agents_dst->doctors_generated[output_index] = agents_src->doctors_generated[index];
	    agents_dst->specialists_generated[output_index] = agents_src->specialists_generated[index];
	    agents_dst->personal_generated[output_index] = agents_src->personal_generated[index];
    }
}

/** add_agent_generator_agent
 * Continuous agent_generator agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_agent_generator_list to add agents to 
 * @param chairs_generated agent variable of type int
 * @param beds_generated agent variable of type int
 * @param boxes_generated agent variable of type int
 * @param doctors_generated agent variable of type int
 * @param specialists_generated agent variable of type int
 * @param personal_generated agent variable of type int
 */
template <int AGENT_TYPE>
__device__ void add_agent_generator_agent(xmachine_memory_agent_generator_list* agents, int chairs_generated, int beds_generated, int boxes_generated, int doctors_generated, int specialists_generated, int personal_generated){
	
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
	agents->chairs_generated[index] = chairs_generated;
	agents->beds_generated[index] = beds_generated;
	agents->boxes_generated[index] = boxes_generated;
	agents->doctors_generated[index] = doctors_generated;
	agents->specialists_generated[index] = specialists_generated;
	agents->personal_generated[index] = personal_generated;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_agent_generator_agent(xmachine_memory_agent_generator_list* agents, int chairs_generated, int beds_generated, int boxes_generated, int doctors_generated, int specialists_generated, int personal_generated){
    add_agent_generator_agent<DISCRETE_2D>(agents, chairs_generated, beds_generated, boxes_generated, doctors_generated, specialists_generated, personal_generated);
}

/** reorder_agent_generator_agents
 * Continuous agent_generator agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_agent_generator_agents(unsigned int* values, xmachine_memory_agent_generator_list* unordered_agents, xmachine_memory_agent_generator_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->chairs_generated[index] = unordered_agents->chairs_generated[old_pos];
	ordered_agents->beds_generated[index] = unordered_agents->beds_generated[old_pos];
	ordered_agents->boxes_generated[index] = unordered_agents->boxes_generated[old_pos];
	ordered_agents->doctors_generated[index] = unordered_agents->doctors_generated[old_pos];
	ordered_agents->specialists_generated[index] = unordered_agents->specialists_generated[old_pos];
	ordered_agents->personal_generated[index] = unordered_agents->personal_generated[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created chair_admin agent functions */

/** reset_chair_admin_scan_input
 * chair_admin agent reset scan input function
 * @param agents The xmachine_memory_chair_admin_list agent list
 */
__global__ void reset_chair_admin_scan_input(xmachine_memory_chair_admin_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_chair_admin_Agents
 * chair_admin scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_chair_admin_list agent list destination
 * @param agents_src xmachine_memory_chair_admin_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_chair_admin_Agents(xmachine_memory_chair_admin_list* agents_dst, xmachine_memory_chair_admin_list* agents_src, int dst_agent_count, int number_to_scatter){
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
	    for (int i=0; i<35; i++){
	      agents_dst->chairArray[(i*xmachine_memory_chair_admin_MAX)+output_index] = agents_src->chairArray[(i*xmachine_memory_chair_admin_MAX)+index];
	    }
	}
}

/** append_chair_admin_Agents
 * chair_admin scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_chair_admin_list agent list destination
 * @param agents_src xmachine_memory_chair_admin_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_chair_admin_Agents(xmachine_memory_chair_admin_list* agents_dst, xmachine_memory_chair_admin_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    for (int i=0; i<35; i++){
	      agents_dst->chairArray[(i*xmachine_memory_chair_admin_MAX)+output_index] = agents_src->chairArray[(i*xmachine_memory_chair_admin_MAX)+index];
	    }
    }
}

/** add_chair_admin_agent
 * Continuous chair_admin agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_chair_admin_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param chairArray agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_chair_admin_agent(xmachine_memory_chair_admin_list* agents, unsigned int id){
	
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

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_chair_admin_agent(xmachine_memory_chair_admin_list* agents, unsigned int id){
    add_chair_admin_agent<DISCRETE_2D>(agents, id);
}

/** reorder_chair_admin_agents
 * Continuous chair_admin agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_chair_admin_agents(unsigned int* values, xmachine_memory_chair_admin_list* unordered_agents, xmachine_memory_chair_admin_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	for (int i=0; i<35; i++){
	  ordered_agents->chairArray[(i*xmachine_memory_chair_admin_MAX)+index] = unordered_agents->chairArray[(i*xmachine_memory_chair_admin_MAX)+old_pos];
	}
}

/** get_chair_admin_agent_array_value
 *  Template function for accessing chair_admin agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_chair_admin_agent_array_value(T *array, uint index){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    return array[index*xmachine_memory_chair_admin_MAX];
    } else {
    	// Return the default value for this data type 
	    return 0;
    }
}

/** set_chair_admin_agent_array_value
 *  Template function for setting chair_admin agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_chair_admin_agent_array_value(T *array, uint index, T value){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    array[index*xmachine_memory_chair_admin_MAX] = value;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created uci agent functions */

/** reset_uci_scan_input
 * uci agent reset scan input function
 * @param agents The xmachine_memory_uci_list agent list
 */
__global__ void reset_uci_scan_input(xmachine_memory_uci_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_uci_Agents
 * uci scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_uci_list agent list destination
 * @param agents_src xmachine_memory_uci_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_uci_Agents(xmachine_memory_uci_list* agents_dst, xmachine_memory_uci_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->tick[output_index] = agents_src->tick[index];
	    for (int i=0; i<100; i++){
	      agents_dst->bedArray[(i*xmachine_memory_uci_MAX)+output_index] = agents_src->bedArray[(i*xmachine_memory_uci_MAX)+index];
	    }
	}
}

/** append_uci_Agents
 * uci scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_uci_list agent list destination
 * @param agents_src xmachine_memory_uci_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_uci_Agents(xmachine_memory_uci_list* agents_dst, xmachine_memory_uci_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->tick[output_index] = agents_src->tick[index];
	    for (int i=0; i<100; i++){
	      agents_dst->bedArray[(i*xmachine_memory_uci_MAX)+output_index] = agents_src->bedArray[(i*xmachine_memory_uci_MAX)+index];
	    }
    }
}

/** add_uci_agent
 * Continuous uci agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_uci_list to add agents to 
 * @param tick agent variable of type unsigned int
 * @param bedArray agent variable of type ivec2
 */
template <int AGENT_TYPE>
__device__ void add_uci_agent(xmachine_memory_uci_list* agents, unsigned int tick){
	
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
	agents->tick[index] = tick;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_uci_agent(xmachine_memory_uci_list* agents, unsigned int tick){
    add_uci_agent<DISCRETE_2D>(agents, tick);
}

/** reorder_uci_agents
 * Continuous uci agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_uci_agents(unsigned int* values, xmachine_memory_uci_list* unordered_agents, xmachine_memory_uci_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->tick[index] = unordered_agents->tick[old_pos];
	for (int i=0; i<100; i++){
	  ordered_agents->bedArray[(i*xmachine_memory_uci_MAX)+index] = unordered_agents->bedArray[(i*xmachine_memory_uci_MAX)+old_pos];
	}
}

/** get_uci_agent_array_value
 *  Template function for accessing uci agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_uci_agent_array_value(T *array, uint index){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    return array[index*xmachine_memory_uci_MAX];
    } else {
    	// Return the default value for this data type 
	    return 0;
    }
}

/** set_uci_agent_array_value
 *  Template function for setting uci agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_uci_agent_array_value(T *array, uint index, T value){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    array[index*xmachine_memory_uci_MAX] = value;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created box agent functions */

/** reset_box_scan_input
 * box agent reset scan input function
 * @param agents The xmachine_memory_box_list agent list
 */
__global__ void reset_box_scan_input(xmachine_memory_box_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_box_Agents
 * box scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_box_list agent list destination
 * @param agents_src xmachine_memory_box_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_box_Agents(xmachine_memory_box_list* agents_dst, xmachine_memory_box_list* agents_src, int dst_agent_count, int number_to_scatter){
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
		agents_dst->current_patient[output_index] = agents_src->current_patient[index];        
		agents_dst->tick[output_index] = agents_src->tick[index];
	}
}

/** append_box_Agents
 * box scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_box_list agent list destination
 * @param agents_src xmachine_memory_box_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_box_Agents(xmachine_memory_box_list* agents_dst, xmachine_memory_box_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->current_patient[output_index] = agents_src->current_patient[index];
	    agents_dst->tick[output_index] = agents_src->tick[index];
    }
}

/** add_box_agent
 * Continuous box agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_box_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param current_patient agent variable of type unsigned int
 * @param tick agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_box_agent(xmachine_memory_box_list* agents, unsigned int id, unsigned int current_patient, unsigned int tick){
	
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
	agents->current_patient[index] = current_patient;
	agents->tick[index] = tick;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_box_agent(xmachine_memory_box_list* agents, unsigned int id, unsigned int current_patient, unsigned int tick){
    add_box_agent<DISCRETE_2D>(agents, id, current_patient, tick);
}

/** reorder_box_agents
 * Continuous box agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_box_agents(unsigned int* values, xmachine_memory_box_list* unordered_agents, xmachine_memory_box_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->current_patient[index] = unordered_agents->current_patient[old_pos];
	ordered_agents->tick[index] = unordered_agents->tick[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created doctor agent functions */

/** reset_doctor_scan_input
 * doctor agent reset scan input function
 * @param agents The xmachine_memory_doctor_list agent list
 */
__global__ void reset_doctor_scan_input(xmachine_memory_doctor_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_doctor_Agents
 * doctor scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_doctor_list agent list destination
 * @param agents_src xmachine_memory_doctor_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_doctor_Agents(xmachine_memory_doctor_list* agents_dst, xmachine_memory_doctor_list* agents_src, int dst_agent_count, int number_to_scatter){
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
		agents_dst->current_patient[output_index] = agents_src->current_patient[index];        
		agents_dst->tick[output_index] = agents_src->tick[index];
	}
}

/** append_doctor_Agents
 * doctor scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_doctor_list agent list destination
 * @param agents_src xmachine_memory_doctor_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_doctor_Agents(xmachine_memory_doctor_list* agents_dst, xmachine_memory_doctor_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->id[output_index] = agents_src->id[index];
	    agents_dst->current_patient[output_index] = agents_src->current_patient[index];
	    agents_dst->tick[output_index] = agents_src->tick[index];
    }
}

/** add_doctor_agent
 * Continuous doctor agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_doctor_list to add agents to 
 * @param id agent variable of type unsigned int
 * @param current_patient agent variable of type int
 * @param tick agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_doctor_agent(xmachine_memory_doctor_list* agents, unsigned int id, int current_patient, unsigned int tick){
	
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
	agents->current_patient[index] = current_patient;
	agents->tick[index] = tick;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_doctor_agent(xmachine_memory_doctor_list* agents, unsigned int id, int current_patient, unsigned int tick){
    add_doctor_agent<DISCRETE_2D>(agents, id, current_patient, tick);
}

/** reorder_doctor_agents
 * Continuous doctor agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_doctor_agents(unsigned int* values, xmachine_memory_doctor_list* unordered_agents, xmachine_memory_doctor_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->id[index] = unordered_agents->id[old_pos];
	ordered_agents->current_patient[index] = unordered_agents->current_patient[old_pos];
	ordered_agents->tick[index] = unordered_agents->tick[old_pos];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created triage agent functions */

/** reset_triage_scan_input
 * triage agent reset scan input function
 * @param agents The xmachine_memory_triage_list agent list
 */
__global__ void reset_triage_scan_input(xmachine_memory_triage_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_triage_Agents
 * triage scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_triage_list agent list destination
 * @param agents_src xmachine_memory_triage_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void scatter_triage_Agents(xmachine_memory_triage_list* agents_dst, xmachine_memory_triage_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1)&&(index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
        agents_dst->_position[output_index] = output_index;        
		agents_dst->front[output_index] = agents_src->front[index];        
		agents_dst->rear[output_index] = agents_src->rear[index];        
		agents_dst->size[output_index] = agents_src->size[index];        
		agents_dst->tick[output_index] = agents_src->tick[index];
	    for (int i=0; i<3; i++){
	      agents_dst->free_boxes[(i*xmachine_memory_triage_MAX)+output_index] = agents_src->free_boxes[(i*xmachine_memory_triage_MAX)+index];
	    }
	    for (int i=0; i<35; i++){
	      agents_dst->patientQueue[(i*xmachine_memory_triage_MAX)+output_index] = agents_src->patientQueue[(i*xmachine_memory_triage_MAX)+index];
	    }
	}
}

/** append_triage_Agents
 * triage scatter agents function (used after agent birth/death)
 * @param agents_dst xmachine_memory_triage_list agent list destination
 * @param agents_src xmachine_memory_triage_list agent list source
 * @param dst_agent_count index to start scattering agents from
 */
__global__ void append_triage_Agents(xmachine_memory_triage_list* agents_dst, xmachine_memory_triage_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
    if (index < number_to_append){
	    int output_index = index + dst_agent_count;

	    //AoS - xmachine_message_location Un-Coalesced scattered memory write
	    agents_dst->_position[output_index] = output_index;
	    agents_dst->front[output_index] = agents_src->front[index];
	    agents_dst->rear[output_index] = agents_src->rear[index];
	    agents_dst->size[output_index] = agents_src->size[index];
	    agents_dst->tick[output_index] = agents_src->tick[index];
	    for (int i=0; i<3; i++){
	      agents_dst->free_boxes[(i*xmachine_memory_triage_MAX)+output_index] = agents_src->free_boxes[(i*xmachine_memory_triage_MAX)+index];
	    }
	    for (int i=0; i<35; i++){
	      agents_dst->patientQueue[(i*xmachine_memory_triage_MAX)+output_index] = agents_src->patientQueue[(i*xmachine_memory_triage_MAX)+index];
	    }
    }
}

/** add_triage_agent
 * Continuous triage agent add agent function writes agent data to agent swap
 * @param agents xmachine_memory_triage_list to add agents to 
 * @param front agent variable of type unsigned int
 * @param rear agent variable of type unsigned int
 * @param size agent variable of type unsigned int
 * @param tick agent variable of type unsigned int
 * @param free_boxes agent variable of type unsigned int
 * @param patientQueue agent variable of type unsigned int
 */
template <int AGENT_TYPE>
__device__ void add_triage_agent(xmachine_memory_triage_list* agents, unsigned int front, unsigned int rear, unsigned int size, unsigned int tick){
	
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
	agents->front[index] = front;
	agents->rear[index] = rear;
	agents->size[index] = size;
	agents->tick[index] = tick;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_triage_agent(xmachine_memory_triage_list* agents, unsigned int front, unsigned int rear, unsigned int size, unsigned int tick){
    add_triage_agent<DISCRETE_2D>(agents, front, rear, size, tick);
}

/** reorder_triage_agents
 * Continuous triage agent areorder function used after key value pairs have been sorted
 * @param values sorted index values
 * @param unordered_agents list of unordered agents
 * @ param ordered_agents list used to output ordered agents
 */
__global__ void reorder_triage_agents(unsigned int* values, xmachine_memory_triage_list* unordered_agents, xmachine_memory_triage_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->front[index] = unordered_agents->front[old_pos];
	ordered_agents->rear[index] = unordered_agents->rear[old_pos];
	ordered_agents->size[index] = unordered_agents->size[old_pos];
	ordered_agents->tick[index] = unordered_agents->tick[old_pos];
	for (int i=0; i<3; i++){
	  ordered_agents->free_boxes[(i*xmachine_memory_triage_MAX)+index] = unordered_agents->free_boxes[(i*xmachine_memory_triage_MAX)+old_pos];
	}
	for (int i=0; i<35; i++){
	  ordered_agents->patientQueue[(i*xmachine_memory_triage_MAX)+index] = unordered_agents->patientQueue[(i*xmachine_memory_triage_MAX)+old_pos];
	}
}

/** get_triage_agent_array_value
 *  Template function for accessing triage agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_triage_agent_array_value(T *array, uint index){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    return array[index*xmachine_memory_triage_MAX];
    } else {
    	// Return the default value for this data type 
	    return 0;
    }
}

/** set_triage_agent_array_value
 *  Template function for setting triage agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_triage_agent_array_value(T *array, uint index, T value){
	// Null check for out of bounds agents (brute force communication. )
	if(array != nullptr){
	    array[index*xmachine_memory_triage_MAX] = value;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created pedestrian_location message functions */


/** add_pedestrian_location_message
 * Add non partitioned or spatially partitioned pedestrian_location message
 * @param messages xmachine_message_pedestrian_location_list message list to add too
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 * @param estado agent variable of type int
 */
__device__ void add_pedestrian_location_message(xmachine_message_pedestrian_location_list* messages, float x, float y, float z, int estado){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_pedestrian_location_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_pedestrian_location_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_pedestrian_location_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_pedestrian_location Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->x[index] = x;
	messages->y[index] = y;
	messages->z[index] = z;
	messages->estado[index] = estado;

}

/**
 * Scatter non partitioned or spatially partitioned pedestrian_location message (for optional messages)
 * @param messages scatter_optional_pedestrian_location_messages Sparse xmachine_message_pedestrian_location_list message list
 * @param message_swap temp xmachine_message_pedestrian_location_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_pedestrian_location_messages(xmachine_message_pedestrian_location_list* messages, xmachine_message_pedestrian_location_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_pedestrian_location_count;

		//AoS - xmachine_message_pedestrian_location Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->x[output_index] = messages_swap->x[index];
		messages->y[output_index] = messages_swap->y[index];
		messages->z[output_index] = messages_swap->z[index];
		messages->estado[output_index] = messages_swap->estado[index];				
	}
}

/** reset_pedestrian_location_swaps
 * Reset non partitioned or spatially partitioned pedestrian_location message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_pedestrian_location_swaps(xmachine_message_pedestrian_location_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

/** message_pedestrian_location_grid_position
 * Calculates the grid cell position given an glm::vec3 vector
 * @param position glm::vec3 vector representing a position
 */
__device__ glm::ivec3 message_pedestrian_location_grid_position(glm::vec3 position)
{
    glm::ivec3 gridPos;
    gridPos.x = floor((position.x - d_message_pedestrian_location_min_bounds.x) * (float)d_message_pedestrian_location_partitionDim.x / (d_message_pedestrian_location_max_bounds.x - d_message_pedestrian_location_min_bounds.x));
    gridPos.y = floor((position.y - d_message_pedestrian_location_min_bounds.y) * (float)d_message_pedestrian_location_partitionDim.y / (d_message_pedestrian_location_max_bounds.y - d_message_pedestrian_location_min_bounds.y));
    gridPos.z = floor((position.z - d_message_pedestrian_location_min_bounds.z) * (float)d_message_pedestrian_location_partitionDim.z / (d_message_pedestrian_location_max_bounds.z - d_message_pedestrian_location_min_bounds.z));

	//do wrapping or bounding
	

    return gridPos;
}

/** message_pedestrian_location_hash
 * Given the grid position in partition space this function calculates a hash value
 * @param gridPos The position in partition space
 */
__device__ unsigned int message_pedestrian_location_hash(glm::ivec3 gridPos)
{
	//cheap bounding without mod (within range +- partition dimension)
	gridPos.x = (gridPos.x<0)? d_message_pedestrian_location_partitionDim.x-1: gridPos.x; 
	gridPos.x = (gridPos.x>=d_message_pedestrian_location_partitionDim.x)? 0 : gridPos.x; 
	gridPos.y = (gridPos.y<0)? d_message_pedestrian_location_partitionDim.y-1 : gridPos.y; 
	gridPos.y = (gridPos.y>=d_message_pedestrian_location_partitionDim.y)? 0 : gridPos.y; 
	gridPos.z = (gridPos.z<0)? d_message_pedestrian_location_partitionDim.z-1: gridPos.z; 
	gridPos.z = (gridPos.z>=d_message_pedestrian_location_partitionDim.z)? 0 : gridPos.z; 

	//unique id
	return ((gridPos.z * d_message_pedestrian_location_partitionDim.y) * d_message_pedestrian_location_partitionDim.x) + (gridPos.y * d_message_pedestrian_location_partitionDim.x) + gridPos.x;
}

#ifdef FAST_ATOMIC_SORTING
	/** hist_pedestrian_location_messages
		 * Kernal function for performing a histogram (count) on each partition bin and saving the hash and index of a message within that bin
		 * @param local_bin_index output index of the message within the calculated bin
		 * @param unsorted_index output bin index (hash) value
		 * @param messages the message list used to generate the hash value outputs
		 * @param agent_count the current number of agents outputting messages
		 */
	__global__ void hist_pedestrian_location_messages(uint* local_bin_index, uint* unsorted_index, int* global_bin_count, xmachine_message_pedestrian_location_list* messages, int agent_count)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;
        glm::vec3 position = glm::vec3(messages->x[index], messages->y[index], messages->z[index]);
		glm::ivec3 grid_position = message_pedestrian_location_grid_position(position);
		unsigned int hash = message_pedestrian_location_hash(grid_position);
		unsigned int bin_idx = atomicInc((unsigned int*) &global_bin_count[hash], 0xFFFFFFFF);
		local_bin_index[index] = bin_idx;
		unsorted_index[index] = hash;
	}
	
	/** reorder_pedestrian_location_messages
	 * Reorders the messages accoring to the partition boundary matrix start indices of each bin
	 * @param local_bin_index index of the message within the desired bin
	 * @param unsorted_index bin index (hash) value
	 * @param pbm_start_index the start indices of the partition boundary matrix
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	  @param agent_count the current number of agents outputting messages
	 */
	 __global__ void reorder_pedestrian_location_messages(uint* local_bin_index, uint* unsorted_index, int* pbm_start_index, xmachine_message_pedestrian_location_list* unordered_messages, xmachine_message_pedestrian_location_list* ordered_messages, int agent_count)
	{
		int index = (blockIdx.x *blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;

		uint i = unsorted_index[index];
		unsigned int sorted_index = local_bin_index[index] + pbm_start_index[i];

		//finally reorder agent data
		ordered_messages->x[sorted_index] = unordered_messages->x[index];
		ordered_messages->y[sorted_index] = unordered_messages->y[index];
		ordered_messages->z[sorted_index] = unordered_messages->z[index];
		ordered_messages->estado[sorted_index] = unordered_messages->estado[index];
	}
	 
#else

	/** hash_pedestrian_location_messages
	 * Kernal function for calculating a hash value for each messahe depending on its position
	 * @param keys output for the hash key
	 * @param values output for the index value
	 * @param messages the message list used to generate the hash value outputs
	 */
	__global__ void hash_pedestrian_location_messages(uint* keys, uint* values, xmachine_message_pedestrian_location_list* messages)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        glm::vec3 position = glm::vec3(messages->x[index], messages->y[index], messages->z[index]);
		glm::ivec3 grid_position = message_pedestrian_location_grid_position(position);
		unsigned int hash = message_pedestrian_location_hash(grid_position);

		keys[index] = hash;
		values[index] = index;
	}

	/** reorder_pedestrian_location_messages
	 * Reorders the messages accoring to the ordered sort identifiers and builds a Partition Boundary Matrix by looking at the previosu threads sort id.
	 * @param keys the sorted hash keys
	 * @param values the sorted index values
	 * @param matrix the PBM
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	 */
	__global__ void reorder_pedestrian_location_messages(uint* keys, uint* values, xmachine_message_pedestrian_location_PBM* matrix, xmachine_message_pedestrian_location_list* unordered_messages, xmachine_message_pedestrian_location_list* ordered_messages)
	{
		extern __shared__ int sm_data [];

		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		//load threads sort key into sm
		uint key = keys[index];
		uint old_pos = values[index];

		sm_data[threadIdx.x] = key;
		__syncthreads();
	
		unsigned int prev_key;

		//if first thread then no prev sm value so get prev from global memory 
		if (threadIdx.x == 0)
		{
			//first thread has no prev value so ignore
			if (index != 0)
				prev_key = keys[index-1];
		}
		//get previous ident from sm
		else	
		{
			prev_key = sm_data[threadIdx.x-1];
		}

		//TODO: Check key is not out of bounds

		//set partition boundaries
		if (index < d_message_pedestrian_location_count)
		{
			//if first thread then set first partition cell start
			if (index == 0)
			{
				matrix->start[key] = index;
			}

			//if edge of a boundr update start and end of partition
			else if (prev_key != key)
			{
				//set start for key
				matrix->start[key] = index;

				//set end for key -1
				matrix->end_or_count[prev_key] = index;
			}

			//if last thread then set final partition cell end
			if (index == d_message_pedestrian_location_count-1)
			{
				matrix->end_or_count[key] = index+1;
			}
		}
	
		//finally reorder agent data
		ordered_messages->x[index] = unordered_messages->x[old_pos];
		ordered_messages->y[index] = unordered_messages->y[old_pos];
		ordered_messages->z[index] = unordered_messages->z[old_pos];
		ordered_messages->estado[index] = unordered_messages->estado[old_pos];
	}

#endif

/** load_next_pedestrian_location_message
 * Used to load the next message data to shared memory
 * Idea is check the current cell index to see if we can simply get a message from the current cell
 * If we are at the end of the current cell then loop till we find the next cell with messages (this way we ignore cells with no messages)
 * @param messages the message list
 * @param partition_matrix the PBM
 * @param relative_cell the relative partition cell position from the agent position
 * @param cell_index_max the maximum index of the current partition cell
 * @param agent_grid_cell the agents partition cell position
 * @param cell_index the current cell index in agent_grid_cell+relative_cell
 * @return true if a message has been loaded into sm false otherwise
 */
__device__ bool load_next_pedestrian_location_message(xmachine_message_pedestrian_location_list* messages, xmachine_message_pedestrian_location_PBM* partition_matrix, glm::ivec3 relative_cell, int cell_index_max, glm::ivec3 agent_grid_cell, int cell_index)
{
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	int move_cell = true;
	cell_index ++;

	//see if we need to move to a new partition cell
	if(cell_index < cell_index_max)
		move_cell = false;

	while(move_cell)
	{
		//get the next relative grid position 
        if (next_cell2D(&relative_cell))
		{
			//calculate the next cells grid position and hash
			glm::ivec3 next_cell_position = agent_grid_cell + relative_cell;
			int next_cell_hash = message_pedestrian_location_hash(next_cell_position);
			//use the hash to calculate the start index
			int cell_index_min = tex1Dfetch(tex_xmachine_message_pedestrian_location_pbm_start, next_cell_hash + d_tex_xmachine_message_pedestrian_location_pbm_start_offset);
			cell_index_max = tex1Dfetch(tex_xmachine_message_pedestrian_location_pbm_end_or_count, next_cell_hash + d_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset);
			//check for messages in the cell (cell index max is the count for atomic sorting)
#ifdef FAST_ATOMIC_SORTING
			if (cell_index_max > 0)
			{
				//when using fast atomics value represents bin count not last index!
				cell_index_max += cell_index_min; //when using fast atomics value represents bin count not last index!
#else
			if (cell_index_min != 0xffffffff)
			{
#endif
				//start from the cell index min
				cell_index = cell_index_min;
				//exit the loop as we have found a valid cell with message data
				move_cell = false;
			}
		}
		else
		{
			//we have exhausted all the neighbouring cells so there are no more messages
			return false;
		}
	}
	
	//get message data using texture fetch
	xmachine_message_pedestrian_location temp_message;
	temp_message._relative_cell = relative_cell;
	temp_message._cell_index_max = cell_index_max;
	temp_message._cell_index = cell_index;
	temp_message._agent_grid_cell = agent_grid_cell;

	//Using texture cache
  temp_message.x = tex1Dfetch(tex_xmachine_message_pedestrian_location_x, cell_index + d_tex_xmachine_message_pedestrian_location_x_offset); temp_message.y = tex1Dfetch(tex_xmachine_message_pedestrian_location_y, cell_index + d_tex_xmachine_message_pedestrian_location_y_offset); temp_message.z = tex1Dfetch(tex_xmachine_message_pedestrian_location_z, cell_index + d_tex_xmachine_message_pedestrian_location_z_offset); temp_message.estado = tex1Dfetch(tex_xmachine_message_pedestrian_location_estado, cell_index + d_tex_xmachine_message_pedestrian_location_estado_offset); 

	//load it into shared memory (no sync as no sharing between threads)
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_pedestrian_location));
	xmachine_message_pedestrian_location* sm_message = ((xmachine_message_pedestrian_location*)&message_share[message_index]);
	sm_message[0] = temp_message;

	return true;
}


/*
 * get first spatial partitioned pedestrian_location message (first batch load into shared memory)
 */
__device__ xmachine_message_pedestrian_location* get_first_pedestrian_location_message(xmachine_message_pedestrian_location_list* messages, xmachine_message_pedestrian_location_PBM* partition_matrix, float x, float y, float z){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	// If there are no messages, do not load any messages
	if(d_message_pedestrian_location_count == 0){
		return nullptr;
	}

	glm::ivec3 relative_cell = glm::ivec3(-2, -1, -1);
	int cell_index_max = 0;
	int cell_index = 0;
	glm::vec3 position = glm::vec3(x, y, z);
	glm::ivec3 agent_grid_cell = message_pedestrian_location_grid_position(position);
	
	if (load_next_pedestrian_location_message(messages, partition_matrix, relative_cell, cell_index_max, agent_grid_cell, cell_index))
	{
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_pedestrian_location));
		return ((xmachine_message_pedestrian_location*)&message_share[message_index]);
	}
	else
	{
		return nullptr;
	}
}

/*
 * get next spatial partitioned pedestrian_location message (either from SM or next batch load)
 */
__device__ xmachine_message_pedestrian_location* get_next_pedestrian_location_message(xmachine_message_pedestrian_location* message, xmachine_message_pedestrian_location_list* messages, xmachine_message_pedestrian_location_PBM* partition_matrix){
	
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	// If there are no messages, do not load any messages
	if(d_message_pedestrian_location_count == 0){
		return nullptr;
	}
	
	if (load_next_pedestrian_location_message(messages, partition_matrix, message->_relative_cell, message->_cell_index_max, message->_agent_grid_cell, message->_cell_index))
	{
		//get conflict free address of 
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_pedestrian_location));
		return ((xmachine_message_pedestrian_location*)&message_share[message_index]);
	}
	else
		return nullptr;
	
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created pedestrian_state message functions */


/** add_pedestrian_state_message
 * Add non partitioned or spatially partitioned pedestrian_state message
 * @param messages xmachine_message_pedestrian_state_list message list to add too
 * @param x agent variable of type float
 * @param y agent variable of type float
 * @param z agent variable of type float
 * @param estado agent variable of type int
 */
__device__ void add_pedestrian_state_message(xmachine_message_pedestrian_state_list* messages, float x, float y, float z, int estado){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_pedestrian_state_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_pedestrian_state_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_pedestrian_state_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_pedestrian_state Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->x[index] = x;
	messages->y[index] = y;
	messages->z[index] = z;
	messages->estado[index] = estado;

}

/**
 * Scatter non partitioned or spatially partitioned pedestrian_state message (for optional messages)
 * @param messages scatter_optional_pedestrian_state_messages Sparse xmachine_message_pedestrian_state_list message list
 * @param message_swap temp xmachine_message_pedestrian_state_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_pedestrian_state_messages(xmachine_message_pedestrian_state_list* messages, xmachine_message_pedestrian_state_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_pedestrian_state_count;

		//AoS - xmachine_message_pedestrian_state Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->x[output_index] = messages_swap->x[index];
		messages->y[output_index] = messages_swap->y[index];
		messages->z[output_index] = messages_swap->z[index];
		messages->estado[output_index] = messages_swap->estado[index];				
	}
}

/** reset_pedestrian_state_swaps
 * Reset non partitioned or spatially partitioned pedestrian_state message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_pedestrian_state_swaps(xmachine_message_pedestrian_state_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

/** message_pedestrian_state_grid_position
 * Calculates the grid cell position given an glm::vec3 vector
 * @param position glm::vec3 vector representing a position
 */
__device__ glm::ivec3 message_pedestrian_state_grid_position(glm::vec3 position)
{
    glm::ivec3 gridPos;
    gridPos.x = floor((position.x - d_message_pedestrian_state_min_bounds.x) * (float)d_message_pedestrian_state_partitionDim.x / (d_message_pedestrian_state_max_bounds.x - d_message_pedestrian_state_min_bounds.x));
    gridPos.y = floor((position.y - d_message_pedestrian_state_min_bounds.y) * (float)d_message_pedestrian_state_partitionDim.y / (d_message_pedestrian_state_max_bounds.y - d_message_pedestrian_state_min_bounds.y));
    gridPos.z = floor((position.z - d_message_pedestrian_state_min_bounds.z) * (float)d_message_pedestrian_state_partitionDim.z / (d_message_pedestrian_state_max_bounds.z - d_message_pedestrian_state_min_bounds.z));

	//do wrapping or bounding
	

    return gridPos;
}

/** message_pedestrian_state_hash
 * Given the grid position in partition space this function calculates a hash value
 * @param gridPos The position in partition space
 */
__device__ unsigned int message_pedestrian_state_hash(glm::ivec3 gridPos)
{
	//cheap bounding without mod (within range +- partition dimension)
	gridPos.x = (gridPos.x<0)? d_message_pedestrian_state_partitionDim.x-1: gridPos.x; 
	gridPos.x = (gridPos.x>=d_message_pedestrian_state_partitionDim.x)? 0 : gridPos.x; 
	gridPos.y = (gridPos.y<0)? d_message_pedestrian_state_partitionDim.y-1 : gridPos.y; 
	gridPos.y = (gridPos.y>=d_message_pedestrian_state_partitionDim.y)? 0 : gridPos.y; 
	gridPos.z = (gridPos.z<0)? d_message_pedestrian_state_partitionDim.z-1: gridPos.z; 
	gridPos.z = (gridPos.z>=d_message_pedestrian_state_partitionDim.z)? 0 : gridPos.z; 

	//unique id
	return ((gridPos.z * d_message_pedestrian_state_partitionDim.y) * d_message_pedestrian_state_partitionDim.x) + (gridPos.y * d_message_pedestrian_state_partitionDim.x) + gridPos.x;
}

#ifdef FAST_ATOMIC_SORTING
	/** hist_pedestrian_state_messages
		 * Kernal function for performing a histogram (count) on each partition bin and saving the hash and index of a message within that bin
		 * @param local_bin_index output index of the message within the calculated bin
		 * @param unsorted_index output bin index (hash) value
		 * @param messages the message list used to generate the hash value outputs
		 * @param agent_count the current number of agents outputting messages
		 */
	__global__ void hist_pedestrian_state_messages(uint* local_bin_index, uint* unsorted_index, int* global_bin_count, xmachine_message_pedestrian_state_list* messages, int agent_count)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;
        glm::vec3 position = glm::vec3(messages->x[index], messages->y[index], messages->z[index]);
		glm::ivec3 grid_position = message_pedestrian_state_grid_position(position);
		unsigned int hash = message_pedestrian_state_hash(grid_position);
		unsigned int bin_idx = atomicInc((unsigned int*) &global_bin_count[hash], 0xFFFFFFFF);
		local_bin_index[index] = bin_idx;
		unsorted_index[index] = hash;
	}
	
	/** reorder_pedestrian_state_messages
	 * Reorders the messages accoring to the partition boundary matrix start indices of each bin
	 * @param local_bin_index index of the message within the desired bin
	 * @param unsorted_index bin index (hash) value
	 * @param pbm_start_index the start indices of the partition boundary matrix
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	  @param agent_count the current number of agents outputting messages
	 */
	 __global__ void reorder_pedestrian_state_messages(uint* local_bin_index, uint* unsorted_index, int* pbm_start_index, xmachine_message_pedestrian_state_list* unordered_messages, xmachine_message_pedestrian_state_list* ordered_messages, int agent_count)
	{
		int index = (blockIdx.x *blockDim.x) + threadIdx.x;

		if (index >= agent_count)
			return;

		uint i = unsorted_index[index];
		unsigned int sorted_index = local_bin_index[index] + pbm_start_index[i];

		//finally reorder agent data
		ordered_messages->x[sorted_index] = unordered_messages->x[index];
		ordered_messages->y[sorted_index] = unordered_messages->y[index];
		ordered_messages->z[sorted_index] = unordered_messages->z[index];
		ordered_messages->estado[sorted_index] = unordered_messages->estado[index];
	}
	 
#else

	/** hash_pedestrian_state_messages
	 * Kernal function for calculating a hash value for each messahe depending on its position
	 * @param keys output for the hash key
	 * @param values output for the index value
	 * @param messages the message list used to generate the hash value outputs
	 */
	__global__ void hash_pedestrian_state_messages(uint* keys, uint* values, xmachine_message_pedestrian_state_list* messages)
	{
		unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        glm::vec3 position = glm::vec3(messages->x[index], messages->y[index], messages->z[index]);
		glm::ivec3 grid_position = message_pedestrian_state_grid_position(position);
		unsigned int hash = message_pedestrian_state_hash(grid_position);

		keys[index] = hash;
		values[index] = index;
	}

	/** reorder_pedestrian_state_messages
	 * Reorders the messages accoring to the ordered sort identifiers and builds a Partition Boundary Matrix by looking at the previosu threads sort id.
	 * @param keys the sorted hash keys
	 * @param values the sorted index values
	 * @param matrix the PBM
	 * @param unordered_messages the original unordered message data
	 * @param ordered_messages buffer used to scatter messages into the correct order
	 */
	__global__ void reorder_pedestrian_state_messages(uint* keys, uint* values, xmachine_message_pedestrian_state_PBM* matrix, xmachine_message_pedestrian_state_list* unordered_messages, xmachine_message_pedestrian_state_list* ordered_messages)
	{
		extern __shared__ int sm_data [];

		int index = (blockIdx.x * blockDim.x) + threadIdx.x;

		//load threads sort key into sm
		uint key = keys[index];
		uint old_pos = values[index];

		sm_data[threadIdx.x] = key;
		__syncthreads();
	
		unsigned int prev_key;

		//if first thread then no prev sm value so get prev from global memory 
		if (threadIdx.x == 0)
		{
			//first thread has no prev value so ignore
			if (index != 0)
				prev_key = keys[index-1];
		}
		//get previous ident from sm
		else	
		{
			prev_key = sm_data[threadIdx.x-1];
		}

		//TODO: Check key is not out of bounds

		//set partition boundaries
		if (index < d_message_pedestrian_state_count)
		{
			//if first thread then set first partition cell start
			if (index == 0)
			{
				matrix->start[key] = index;
			}

			//if edge of a boundr update start and end of partition
			else if (prev_key != key)
			{
				//set start for key
				matrix->start[key] = index;

				//set end for key -1
				matrix->end_or_count[prev_key] = index;
			}

			//if last thread then set final partition cell end
			if (index == d_message_pedestrian_state_count-1)
			{
				matrix->end_or_count[key] = index+1;
			}
		}
	
		//finally reorder agent data
		ordered_messages->x[index] = unordered_messages->x[old_pos];
		ordered_messages->y[index] = unordered_messages->y[old_pos];
		ordered_messages->z[index] = unordered_messages->z[old_pos];
		ordered_messages->estado[index] = unordered_messages->estado[old_pos];
	}

#endif

/** load_next_pedestrian_state_message
 * Used to load the next message data to shared memory
 * Idea is check the current cell index to see if we can simply get a message from the current cell
 * If we are at the end of the current cell then loop till we find the next cell with messages (this way we ignore cells with no messages)
 * @param messages the message list
 * @param partition_matrix the PBM
 * @param relative_cell the relative partition cell position from the agent position
 * @param cell_index_max the maximum index of the current partition cell
 * @param agent_grid_cell the agents partition cell position
 * @param cell_index the current cell index in agent_grid_cell+relative_cell
 * @return true if a message has been loaded into sm false otherwise
 */
__device__ bool load_next_pedestrian_state_message(xmachine_message_pedestrian_state_list* messages, xmachine_message_pedestrian_state_PBM* partition_matrix, glm::ivec3 relative_cell, int cell_index_max, glm::ivec3 agent_grid_cell, int cell_index)
{
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	int move_cell = true;
	cell_index ++;

	//see if we need to move to a new partition cell
	if(cell_index < cell_index_max)
		move_cell = false;

	while(move_cell)
	{
		//get the next relative grid position 
        if (next_cell2D(&relative_cell))
		{
			//calculate the next cells grid position and hash
			glm::ivec3 next_cell_position = agent_grid_cell + relative_cell;
			int next_cell_hash = message_pedestrian_state_hash(next_cell_position);
			//use the hash to calculate the start index
			int cell_index_min = tex1Dfetch(tex_xmachine_message_pedestrian_state_pbm_start, next_cell_hash + d_tex_xmachine_message_pedestrian_state_pbm_start_offset);
			cell_index_max = tex1Dfetch(tex_xmachine_message_pedestrian_state_pbm_end_or_count, next_cell_hash + d_tex_xmachine_message_pedestrian_state_pbm_end_or_count_offset);
			//check for messages in the cell (cell index max is the count for atomic sorting)
#ifdef FAST_ATOMIC_SORTING
			if (cell_index_max > 0)
			{
				//when using fast atomics value represents bin count not last index!
				cell_index_max += cell_index_min; //when using fast atomics value represents bin count not last index!
#else
			if (cell_index_min != 0xffffffff)
			{
#endif
				//start from the cell index min
				cell_index = cell_index_min;
				//exit the loop as we have found a valid cell with message data
				move_cell = false;
			}
		}
		else
		{
			//we have exhausted all the neighbouring cells so there are no more messages
			return false;
		}
	}
	
	//get message data using texture fetch
	xmachine_message_pedestrian_state temp_message;
	temp_message._relative_cell = relative_cell;
	temp_message._cell_index_max = cell_index_max;
	temp_message._cell_index = cell_index;
	temp_message._agent_grid_cell = agent_grid_cell;

	//Using texture cache
  temp_message.x = tex1Dfetch(tex_xmachine_message_pedestrian_state_x, cell_index + d_tex_xmachine_message_pedestrian_state_x_offset); temp_message.y = tex1Dfetch(tex_xmachine_message_pedestrian_state_y, cell_index + d_tex_xmachine_message_pedestrian_state_y_offset); temp_message.z = tex1Dfetch(tex_xmachine_message_pedestrian_state_z, cell_index + d_tex_xmachine_message_pedestrian_state_z_offset); temp_message.estado = tex1Dfetch(tex_xmachine_message_pedestrian_state_estado, cell_index + d_tex_xmachine_message_pedestrian_state_estado_offset); 

	//load it into shared memory (no sync as no sharing between threads)
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_pedestrian_state));
	xmachine_message_pedestrian_state* sm_message = ((xmachine_message_pedestrian_state*)&message_share[message_index]);
	sm_message[0] = temp_message;

	return true;
}


/*
 * get first spatial partitioned pedestrian_state message (first batch load into shared memory)
 */
__device__ xmachine_message_pedestrian_state* get_first_pedestrian_state_message(xmachine_message_pedestrian_state_list* messages, xmachine_message_pedestrian_state_PBM* partition_matrix, float x, float y, float z){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];

	// If there are no messages, do not load any messages
	if(d_message_pedestrian_state_count == 0){
		return nullptr;
	}

	glm::ivec3 relative_cell = glm::ivec3(-2, -1, -1);
	int cell_index_max = 0;
	int cell_index = 0;
	glm::vec3 position = glm::vec3(x, y, z);
	glm::ivec3 agent_grid_cell = message_pedestrian_state_grid_position(position);
	
	if (load_next_pedestrian_state_message(messages, partition_matrix, relative_cell, cell_index_max, agent_grid_cell, cell_index))
	{
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_pedestrian_state));
		return ((xmachine_message_pedestrian_state*)&message_share[message_index]);
	}
	else
	{
		return nullptr;
	}
}

/*
 * get next spatial partitioned pedestrian_state message (either from SM or next batch load)
 */
__device__ xmachine_message_pedestrian_state* get_next_pedestrian_state_message(xmachine_message_pedestrian_state* message, xmachine_message_pedestrian_state_list* messages, xmachine_message_pedestrian_state_PBM* partition_matrix){
	
	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	// If there are no messages, do not load any messages
	if(d_message_pedestrian_state_count == 0){
		return nullptr;
	}
	
	if (load_next_pedestrian_state_message(messages, partition_matrix, message->_relative_cell, message->_cell_index_max, message->_agent_grid_cell, message->_cell_index))
	{
		//get conflict free address of 
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_pedestrian_state));
		return ((xmachine_message_pedestrian_state*)&message_share[message_index]);
	}
	else
		return nullptr;
	
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created navmap_cell message functions */


/* Message functions */

template <int AGENT_TYPE>
__device__ void add_navmap_cell_message(xmachine_message_navmap_cell_list* messages, int x, int y, int exit_no, float height, float collision_x, float collision_y){
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x * gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;

		int index = global_position.x + (global_position.y * width);

		
		messages->x[index] = x;			
		messages->y[index] = y;			
		messages->exit_no[index] = exit_no;			
		messages->height[index] = height;			
		messages->collision_x[index] = collision_x;			
		messages->collision_y[index] = collision_y;			
	}
	//else CONTINUOUS agents can not write to discrete space
}

//Used by continuous agents this accesses messages with texture cache. agent_x and agent_y are discrete positions in the message space
__device__ xmachine_message_navmap_cell* get_first_navmap_cell_message_continuous(xmachine_message_navmap_cell_list* messages,  int agent_x, int agent_y){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	xmachine_message_navmap_cell* message_share = (xmachine_message_navmap_cell*)&sm_data[0];
	
	int range = d_message_navmap_cell_range;
	int width = d_message_navmap_cell_width;
	
	glm::ivec2 global_position;
	global_position.x = sWRAP(agent_x-range , width);
	global_position.y = sWRAP(agent_y-range , width);
	

	int index = ((global_position.y)* width) + global_position.x;
	
	xmachine_message_navmap_cell temp_message;
	temp_message._position = glm::ivec2(agent_x, agent_y);
	temp_message._relative = glm::ivec2(-range, -range);

	temp_message.x = tex1Dfetch(tex_xmachine_message_navmap_cell_x, index + d_tex_xmachine_message_navmap_cell_x_offset);temp_message.y = tex1Dfetch(tex_xmachine_message_navmap_cell_y, index + d_tex_xmachine_message_navmap_cell_y_offset);temp_message.exit_no = tex1Dfetch(tex_xmachine_message_navmap_cell_exit_no, index + d_tex_xmachine_message_navmap_cell_exit_no_offset);temp_message.height = tex1Dfetch(tex_xmachine_message_navmap_cell_height, index + d_tex_xmachine_message_navmap_cell_height_offset);temp_message.collision_x = tex1Dfetch(tex_xmachine_message_navmap_cell_collision_x, index + d_tex_xmachine_message_navmap_cell_collision_x_offset);temp_message.collision_y = tex1Dfetch(tex_xmachine_message_navmap_cell_collision_y, index + d_tex_xmachine_message_navmap_cell_collision_y_offset);
	
	message_share[threadIdx.x] = temp_message;

	//return top left of messages
	return &message_share[threadIdx.x];
}

//Get next navmap_cell message  continuous
//Used by continuous agents this accesses messages with texture cache (agent position in discrete space was set when accessing first message)
__device__ xmachine_message_navmap_cell* get_next_navmap_cell_message_continuous(xmachine_message_navmap_cell* message, xmachine_message_navmap_cell_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	xmachine_message_navmap_cell* message_share = (xmachine_message_navmap_cell*)&sm_data[0];
	
	int range = d_message_navmap_cell_range;
	int width = d_message_navmap_cell_width;

	//Get previous position
	glm::ivec2 previous_relative = message->_relative;

	//exit if at (range, range)
	if (previous_relative.x == (range))
        if (previous_relative.y == (range))
		    return nullptr;

	//calculate next message relative position
	glm::ivec2 next_relative = previous_relative;
	next_relative.x += 1;
	if ((next_relative.x)>range){
		next_relative.x = -range;
		next_relative.y = previous_relative.y + 1;
	}

	//skip own message
	if (next_relative.x == 0)
        if (next_relative.y == 0)
		    next_relative.x += 1;

	glm::ivec2 global_position;
	global_position.x =	sWRAP(message->_position.x + next_relative.x, width);
	global_position.y = sWRAP(message->_position.y + next_relative.y, width);

	int index = ((global_position.y)* width) + (global_position.x);
	
	xmachine_message_navmap_cell temp_message;
	temp_message._position = message->_position;
	temp_message._relative = next_relative;

	temp_message.x = tex1Dfetch(tex_xmachine_message_navmap_cell_x, index + d_tex_xmachine_message_navmap_cell_x_offset);	temp_message.y = tex1Dfetch(tex_xmachine_message_navmap_cell_y, index + d_tex_xmachine_message_navmap_cell_y_offset);	temp_message.exit_no = tex1Dfetch(tex_xmachine_message_navmap_cell_exit_no, index + d_tex_xmachine_message_navmap_cell_exit_no_offset);	temp_message.height = tex1Dfetch(tex_xmachine_message_navmap_cell_height, index + d_tex_xmachine_message_navmap_cell_height_offset);	temp_message.collision_x = tex1Dfetch(tex_xmachine_message_navmap_cell_collision_x, index + d_tex_xmachine_message_navmap_cell_collision_x_offset);	temp_message.collision_y = tex1Dfetch(tex_xmachine_message_navmap_cell_collision_y, index + d_tex_xmachine_message_navmap_cell_collision_y_offset);	

	message_share[threadIdx.x] = temp_message;

	return &message_share[threadIdx.x];
}

//method used by discrete agents accessing discrete messages to load messages into shared memory
__device__ void navmap_cell_message_to_sm(xmachine_message_navmap_cell_list* messages, char* message_share, int sm_index, int global_index){
		xmachine_message_navmap_cell temp_message;
		
		temp_message.x = messages->x[global_index];		
		temp_message.y = messages->y[global_index];		
		temp_message.exit_no = messages->exit_no[global_index];		
		temp_message.height = messages->height[global_index];		
		temp_message.collision_x = messages->collision_x[global_index];		
		temp_message.collision_y = messages->collision_y[global_index];		

	  int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_navmap_cell));
	  xmachine_message_navmap_cell* sm_message = ((xmachine_message_navmap_cell*)&message_share[message_index]);
	  sm_message[0] = temp_message;
}

//Get first navmap_cell message 
//Used by discrete agents this accesses messages with texture cache. Agent position is determined by position in the grid/block
//Possibility of upto 8 thread divergences
__device__ xmachine_message_navmap_cell* get_first_navmap_cell_message_discrete(xmachine_message_navmap_cell_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	char* message_share = (char*)&sm_data[0];
  
	__syncthreads();

	int range = d_message_navmap_cell_range;
	int width = d_message_navmap_cell_width;
	int sm_grid_width = blockDim.x + (range* 2);
	
	
	glm::ivec2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//calculate the position in shared memory of first load
	glm::ivec2 sm_pos;
	sm_pos.x = threadIdx.x + range;
	sm_pos.y = threadIdx.y + range;
	int sm_index = (sm_pos.y * sm_grid_width) + sm_pos.x;

	//each thread loads to shared memory (coalesced read)
	navmap_cell_message_to_sm(messages, message_share, sm_index, index);

	//check for edge conditions
	int left_border = (threadIdx.x < range);
	int right_border = (threadIdx.x >= (blockDim.x-range));
	int top_border = (threadIdx.y < range);
	int bottom_border = (threadIdx.y >= (blockDim.y-range));

	
	int  border_index;
	int  sm_border_index;

	//left
	if (left_border){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (sm_pos.y * sm_grid_width) + threadIdx.x;
		
		navmap_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//right
	if (right_border){
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (sm_pos.y * sm_grid_width) + (sm_pos.x + range);

		navmap_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top
	if (top_border){
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + sm_pos.x;

		navmap_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom
	if (bottom_border){
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + sm_pos.x;

		navmap_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top left
	if ((top_border)&&(left_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + threadIdx.x;
		
		navmap_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//top right
	if ((top_border)&&(right_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index_2d.y = sWRAP(border_index_2d.y - range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = (threadIdx.y * sm_grid_width) + (sm_pos.x + range);
		
		navmap_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom right
	if ((bottom_border)&&(right_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x + range, width);
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + (sm_pos.x + range);
		
		navmap_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	//bottom left
	if ((bottom_border)&&(left_border)){	
		glm::ivec2 border_index_2d = global_position;
		border_index_2d.x = sWRAP(border_index_2d.x - range, width);
		border_index_2d.y = sWRAP(border_index_2d.y + range, width);
		border_index = (border_index_2d.y * width) + border_index_2d.x;
		sm_border_index = ((sm_pos.y + range) * sm_grid_width) + threadIdx.x;
		
		navmap_cell_message_to_sm(messages, message_share, sm_border_index, border_index);
	}

	__syncthreads();
	
  
	//top left of block position sm index
	sm_index = (threadIdx.y * sm_grid_width) + threadIdx.x;
	
	int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_navmap_cell));
	xmachine_message_navmap_cell* temp = ((xmachine_message_navmap_cell*)&message_share[message_index]);
	temp->_relative = glm::ivec2(-range, -range); //this is the relative position
	return temp;
}

//Get next navmap_cell message 
//Used by discrete agents this accesses messages through shared memory which were all loaded on first message retrieval call.
__device__ xmachine_message_navmap_cell* get_next_navmap_cell_message_discrete(xmachine_message_navmap_cell* message, xmachine_message_navmap_cell_list* messages){

	//shared memory get from offset dependant on sm usage in function
	extern __shared__ int sm_data [];

	char* message_share = (char*)&sm_data[0];
  
	__syncthreads();
	
	int range = d_message_navmap_cell_range;
	int sm_grid_width = blockDim.x+(range*2);


	//Get previous position
	glm::ivec2 previous_relative = message->_relative;

	//exit if at (range, range)
	if (previous_relative.x == range)
        if (previous_relative.y == range)
		    return nullptr;

	//calculate next message relative position
	glm::ivec2 next_relative = previous_relative;
	next_relative.x += 1;
	if ((next_relative.x)>range){
		next_relative.x = -range;
		next_relative.y = previous_relative.y + 1;
	}

	//skip own message
	if (next_relative.x == 0)
        if (next_relative.y == 0)
		    next_relative.x += 1;


	//calculate the next message position
	glm::ivec2 next_position;// = block_position+next_relative;
	//offset next position by the sm border size
	next_position.x = threadIdx.x + next_relative.x + range;
	next_position.y = threadIdx.y + next_relative.y + range;

	int sm_index = next_position.x + (next_position.y * sm_grid_width);
	
	__syncthreads();
  
	int message_index = SHARE_INDEX(sm_index, sizeof(xmachine_message_navmap_cell));
	xmachine_message_navmap_cell* temp = ((xmachine_message_navmap_cell*)&message_share[message_index]);
	temp->_relative = next_relative; //this is the relative position
	return temp;
}

//Get first navmap_cell message
template <int AGENT_TYPE>
__device__ xmachine_message_navmap_cell* get_first_navmap_cell_message(xmachine_message_navmap_cell_list* messages, int agent_x, int agent_y){

	if (AGENT_TYPE == DISCRETE_2D)	//use shared memory method
		return get_first_navmap_cell_message_discrete(messages);
	else	//use texture fetching method
		return get_first_navmap_cell_message_continuous(messages, agent_x, agent_y);

}

//Get next navmap_cell message
template <int AGENT_TYPE>
__device__ xmachine_message_navmap_cell* get_next_navmap_cell_message(xmachine_message_navmap_cell* message, xmachine_message_navmap_cell_list* messages){

	if (AGENT_TYPE == DISCRETE_2D)	//use shared memory method
		return get_next_navmap_cell_message_discrete(message, messages);
	else	//use texture fetching method
		return get_next_navmap_cell_message_continuous(message, messages);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created check_in message functions */


/** add_check_in_message
 * Add non partitioned or spatially partitioned check_in message
 * @param messages xmachine_message_check_in_list message list to add too
 * @param id agent variable of type unsigned int
 */
__device__ void add_check_in_message(xmachine_message_check_in_list* messages, unsigned int id){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_check_in_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_check_in_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_check_in_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_check_in Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;

}

/**
 * Scatter non partitioned or spatially partitioned check_in message (for optional messages)
 * @param messages scatter_optional_check_in_messages Sparse xmachine_message_check_in_list message list
 * @param message_swap temp xmachine_message_check_in_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_check_in_messages(xmachine_message_check_in_list* messages, xmachine_message_check_in_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_check_in_count;

		//AoS - xmachine_message_check_in Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];				
	}
}

/** reset_check_in_swaps
 * Reset non partitioned or spatially partitioned check_in message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_check_in_swaps(xmachine_message_check_in_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_check_in* get_first_check_in_message(xmachine_message_check_in_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_check_in_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_check_in Coalesced memory read
	xmachine_message_check_in temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_check_in));
	xmachine_message_check_in* sm_message = ((xmachine_message_check_in*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_check_in*)&message_share[d_SM_START]);
}

__device__ xmachine_message_check_in* get_next_check_in_message(xmachine_message_check_in* message, xmachine_message_check_in_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_check_in_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_check_in_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_check_in Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_check_in temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_check_in));
		xmachine_message_check_in* sm_message = ((xmachine_message_check_in*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_check_in));
	return ((xmachine_message_check_in*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created check_in_response message functions */


/** add_check_in_response_message
 * Add non partitioned or spatially partitioned check_in_response message
 * @param messages xmachine_message_check_in_response_list message list to add too
 * @param id agent variable of type unsigned int
 */
__device__ void add_check_in_response_message(xmachine_message_check_in_response_list* messages, unsigned int id){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_check_in_response_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_check_in_response_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_check_in_response_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_check_in_response Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;

}

/**
 * Scatter non partitioned or spatially partitioned check_in_response message (for optional messages)
 * @param messages scatter_optional_check_in_response_messages Sparse xmachine_message_check_in_response_list message list
 * @param message_swap temp xmachine_message_check_in_response_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_check_in_response_messages(xmachine_message_check_in_response_list* messages, xmachine_message_check_in_response_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_check_in_response_count;

		//AoS - xmachine_message_check_in_response Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];				
	}
}

/** reset_check_in_response_swaps
 * Reset non partitioned or spatially partitioned check_in_response message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_check_in_response_swaps(xmachine_message_check_in_response_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_check_in_response* get_first_check_in_response_message(xmachine_message_check_in_response_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_check_in_response_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_check_in_response Coalesced memory read
	xmachine_message_check_in_response temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_check_in_response));
	xmachine_message_check_in_response* sm_message = ((xmachine_message_check_in_response*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_check_in_response*)&message_share[d_SM_START]);
}

__device__ xmachine_message_check_in_response* get_next_check_in_response_message(xmachine_message_check_in_response* message, xmachine_message_check_in_response_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_check_in_response_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_check_in_response_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_check_in_response Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_check_in_response temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_check_in_response));
		xmachine_message_check_in_response* sm_message = ((xmachine_message_check_in_response*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_check_in_response));
	return ((xmachine_message_check_in_response*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created chair_petition message functions */


/** add_chair_petition_message
 * Add non partitioned or spatially partitioned chair_petition message
 * @param messages xmachine_message_chair_petition_list message list to add too
 * @param id agent variable of type unsigned int
 */
__device__ void add_chair_petition_message(xmachine_message_chair_petition_list* messages, unsigned int id){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_chair_petition_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_chair_petition_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_chair_petition_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_chair_petition Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;

}

/**
 * Scatter non partitioned or spatially partitioned chair_petition message (for optional messages)
 * @param messages scatter_optional_chair_petition_messages Sparse xmachine_message_chair_petition_list message list
 * @param message_swap temp xmachine_message_chair_petition_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_chair_petition_messages(xmachine_message_chair_petition_list* messages, xmachine_message_chair_petition_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_chair_petition_count;

		//AoS - xmachine_message_chair_petition Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];				
	}
}

/** reset_chair_petition_swaps
 * Reset non partitioned or spatially partitioned chair_petition message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_chair_petition_swaps(xmachine_message_chair_petition_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_chair_petition* get_first_chair_petition_message(xmachine_message_chair_petition_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_chair_petition_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_chair_petition Coalesced memory read
	xmachine_message_chair_petition temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_chair_petition));
	xmachine_message_chair_petition* sm_message = ((xmachine_message_chair_petition*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_chair_petition*)&message_share[d_SM_START]);
}

__device__ xmachine_message_chair_petition* get_next_chair_petition_message(xmachine_message_chair_petition* message, xmachine_message_chair_petition_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_chair_petition_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_chair_petition_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_chair_petition Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_chair_petition temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_chair_petition));
		xmachine_message_chair_petition* sm_message = ((xmachine_message_chair_petition*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_chair_petition));
	return ((xmachine_message_chair_petition*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created chair_response message functions */


/** add_chair_response_message
 * Add non partitioned or spatially partitioned chair_response message
 * @param messages xmachine_message_chair_response_list message list to add too
 * @param id agent variable of type unsigned int
 * @param chair_no agent variable of type int
 */
__device__ void add_chair_response_message(xmachine_message_chair_response_list* messages, unsigned int id, int chair_no){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_chair_response_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_chair_response_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_chair_response_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_chair_response Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->chair_no[index] = chair_no;

}

/**
 * Scatter non partitioned or spatially partitioned chair_response message (for optional messages)
 * @param messages scatter_optional_chair_response_messages Sparse xmachine_message_chair_response_list message list
 * @param message_swap temp xmachine_message_chair_response_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_chair_response_messages(xmachine_message_chair_response_list* messages, xmachine_message_chair_response_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_chair_response_count;

		//AoS - xmachine_message_chair_response Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->chair_no[output_index] = messages_swap->chair_no[index];				
	}
}

/** reset_chair_response_swaps
 * Reset non partitioned or spatially partitioned chair_response message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_chair_response_swaps(xmachine_message_chair_response_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_chair_response* get_first_chair_response_message(xmachine_message_chair_response_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_chair_response_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_chair_response Coalesced memory read
	xmachine_message_chair_response temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.chair_no = messages->chair_no[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_chair_response));
	xmachine_message_chair_response* sm_message = ((xmachine_message_chair_response*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_chair_response*)&message_share[d_SM_START]);
}

__device__ xmachine_message_chair_response* get_next_chair_response_message(xmachine_message_chair_response* message, xmachine_message_chair_response_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_chair_response_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_chair_response_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_chair_response Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_chair_response temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.chair_no = messages->chair_no[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_chair_response));
		xmachine_message_chair_response* sm_message = ((xmachine_message_chair_response*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_chair_response));
	return ((xmachine_message_chair_response*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created free_chair message functions */


/** add_free_chair_message
 * Add non partitioned or spatially partitioned free_chair message
 * @param messages xmachine_message_free_chair_list message list to add too
 * @param chair_no agent variable of type unsigned int
 */
__device__ void add_free_chair_message(xmachine_message_free_chair_list* messages, unsigned int chair_no){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_free_chair_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_free_chair_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_free_chair_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_free_chair Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->chair_no[index] = chair_no;

}

/**
 * Scatter non partitioned or spatially partitioned free_chair message (for optional messages)
 * @param messages scatter_optional_free_chair_messages Sparse xmachine_message_free_chair_list message list
 * @param message_swap temp xmachine_message_free_chair_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_free_chair_messages(xmachine_message_free_chair_list* messages, xmachine_message_free_chair_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_free_chair_count;

		//AoS - xmachine_message_free_chair Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->chair_no[output_index] = messages_swap->chair_no[index];				
	}
}

/** reset_free_chair_swaps
 * Reset non partitioned or spatially partitioned free_chair message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_free_chair_swaps(xmachine_message_free_chair_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_free_chair* get_first_free_chair_message(xmachine_message_free_chair_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_free_chair_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_free_chair Coalesced memory read
	xmachine_message_free_chair temp_message;
	temp_message._position = messages->_position[index];
	temp_message.chair_no = messages->chair_no[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_free_chair));
	xmachine_message_free_chair* sm_message = ((xmachine_message_free_chair*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_free_chair*)&message_share[d_SM_START]);
}

__device__ xmachine_message_free_chair* get_next_free_chair_message(xmachine_message_free_chair* message, xmachine_message_free_chair_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_free_chair_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_free_chair_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_free_chair Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_free_chair temp_message;
		temp_message._position = messages->_position[index];
		temp_message.chair_no = messages->chair_no[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_free_chair));
		xmachine_message_free_chair* sm_message = ((xmachine_message_free_chair*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_free_chair));
	return ((xmachine_message_free_chair*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created chair_state message functions */


/** add_chair_state_message
 * Add non partitioned or spatially partitioned chair_state message
 * @param messages xmachine_message_chair_state_list message list to add too
 * @param id agent variable of type unsigned int
 * @param state agent variable of type int
 */
__device__ void add_chair_state_message(xmachine_message_chair_state_list* messages, unsigned int id, int state){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_chair_state_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_chair_state_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_chair_state_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_chair_state Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->state[index] = state;

}

/**
 * Scatter non partitioned or spatially partitioned chair_state message (for optional messages)
 * @param messages scatter_optional_chair_state_messages Sparse xmachine_message_chair_state_list message list
 * @param message_swap temp xmachine_message_chair_state_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_chair_state_messages(xmachine_message_chair_state_list* messages, xmachine_message_chair_state_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_chair_state_count;

		//AoS - xmachine_message_chair_state Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->state[output_index] = messages_swap->state[index];				
	}
}

/** reset_chair_state_swaps
 * Reset non partitioned or spatially partitioned chair_state message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_chair_state_swaps(xmachine_message_chair_state_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_chair_state* get_first_chair_state_message(xmachine_message_chair_state_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_chair_state_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_chair_state Coalesced memory read
	xmachine_message_chair_state temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.state = messages->state[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_chair_state));
	xmachine_message_chair_state* sm_message = ((xmachine_message_chair_state*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_chair_state*)&message_share[d_SM_START]);
}

__device__ xmachine_message_chair_state* get_next_chair_state_message(xmachine_message_chair_state* message, xmachine_message_chair_state_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_chair_state_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_chair_state_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_chair_state Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_chair_state temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.state = messages->state[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_chair_state));
		xmachine_message_chair_state* sm_message = ((xmachine_message_chair_state*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_chair_state));
	return ((xmachine_message_chair_state*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created chair_contact message functions */


/** add_chair_contact_message
 * Add non partitioned or spatially partitioned chair_contact message
 * @param messages xmachine_message_chair_contact_list message list to add too
 * @param id agent variable of type unsigned int
 * @param chair_no agent variable of type unsigned int
 * @param state agent variable of type int
 */
__device__ void add_chair_contact_message(xmachine_message_chair_contact_list* messages, unsigned int id, unsigned int chair_no, int state){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_chair_contact_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_chair_contact_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_chair_contact_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_chair_contact Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->chair_no[index] = chair_no;
	messages->state[index] = state;

}

/**
 * Scatter non partitioned or spatially partitioned chair_contact message (for optional messages)
 * @param messages scatter_optional_chair_contact_messages Sparse xmachine_message_chair_contact_list message list
 * @param message_swap temp xmachine_message_chair_contact_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_chair_contact_messages(xmachine_message_chair_contact_list* messages, xmachine_message_chair_contact_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_chair_contact_count;

		//AoS - xmachine_message_chair_contact Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->chair_no[output_index] = messages_swap->chair_no[index];
		messages->state[output_index] = messages_swap->state[index];				
	}
}

/** reset_chair_contact_swaps
 * Reset non partitioned or spatially partitioned chair_contact message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_chair_contact_swaps(xmachine_message_chair_contact_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_chair_contact* get_first_chair_contact_message(xmachine_message_chair_contact_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_chair_contact_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_chair_contact Coalesced memory read
	xmachine_message_chair_contact temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.chair_no = messages->chair_no[index];
	temp_message.state = messages->state[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_chair_contact));
	xmachine_message_chair_contact* sm_message = ((xmachine_message_chair_contact*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_chair_contact*)&message_share[d_SM_START]);
}

__device__ xmachine_message_chair_contact* get_next_chair_contact_message(xmachine_message_chair_contact* message, xmachine_message_chair_contact_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_chair_contact_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_chair_contact_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_chair_contact Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_chair_contact temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.chair_no = messages->chair_no[index];
		temp_message.state = messages->state[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_chair_contact));
		xmachine_message_chair_contact* sm_message = ((xmachine_message_chair_contact*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_chair_contact));
	return ((xmachine_message_chair_contact*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created bed_state message functions */


/** add_bed_state_message
 * Add non partitioned or spatially partitioned bed_state message
 * @param messages xmachine_message_bed_state_list message list to add too
 * @param id agent variable of type unsigned int
 * @param state agent variable of type int
 */
__device__ void add_bed_state_message(xmachine_message_bed_state_list* messages, unsigned int id, int state){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_bed_state_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_bed_state_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_bed_state_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_bed_state Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->state[index] = state;

}

/**
 * Scatter non partitioned or spatially partitioned bed_state message (for optional messages)
 * @param messages scatter_optional_bed_state_messages Sparse xmachine_message_bed_state_list message list
 * @param message_swap temp xmachine_message_bed_state_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_bed_state_messages(xmachine_message_bed_state_list* messages, xmachine_message_bed_state_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_bed_state_count;

		//AoS - xmachine_message_bed_state Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->state[output_index] = messages_swap->state[index];				
	}
}

/** reset_bed_state_swaps
 * Reset non partitioned or spatially partitioned bed_state message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_bed_state_swaps(xmachine_message_bed_state_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_bed_state* get_first_bed_state_message(xmachine_message_bed_state_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_bed_state_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_bed_state Coalesced memory read
	xmachine_message_bed_state temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.state = messages->state[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_bed_state));
	xmachine_message_bed_state* sm_message = ((xmachine_message_bed_state*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_bed_state*)&message_share[d_SM_START]);
}

__device__ xmachine_message_bed_state* get_next_bed_state_message(xmachine_message_bed_state* message, xmachine_message_bed_state_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_bed_state_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_bed_state_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_bed_state Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_bed_state temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.state = messages->state[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_bed_state));
		xmachine_message_bed_state* sm_message = ((xmachine_message_bed_state*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_bed_state));
	return ((xmachine_message_bed_state*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created bed_contact message functions */


/** add_bed_contact_message
 * Add non partitioned or spatially partitioned bed_contact message
 * @param messages xmachine_message_bed_contact_list message list to add too
 * @param id agent variable of type unsigned int
 * @param bed_no agent variable of type unsigned int
 * @param state agent variable of type int
 */
__device__ void add_bed_contact_message(xmachine_message_bed_contact_list* messages, unsigned int id, unsigned int bed_no, int state){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_bed_contact_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_bed_contact_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_bed_contact_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_bed_contact Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->bed_no[index] = bed_no;
	messages->state[index] = state;

}

/**
 * Scatter non partitioned or spatially partitioned bed_contact message (for optional messages)
 * @param messages scatter_optional_bed_contact_messages Sparse xmachine_message_bed_contact_list message list
 * @param message_swap temp xmachine_message_bed_contact_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_bed_contact_messages(xmachine_message_bed_contact_list* messages, xmachine_message_bed_contact_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_bed_contact_count;

		//AoS - xmachine_message_bed_contact Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->bed_no[output_index] = messages_swap->bed_no[index];
		messages->state[output_index] = messages_swap->state[index];				
	}
}

/** reset_bed_contact_swaps
 * Reset non partitioned or spatially partitioned bed_contact message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_bed_contact_swaps(xmachine_message_bed_contact_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_bed_contact* get_first_bed_contact_message(xmachine_message_bed_contact_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_bed_contact_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_bed_contact Coalesced memory read
	xmachine_message_bed_contact temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.bed_no = messages->bed_no[index];
	temp_message.state = messages->state[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_bed_contact));
	xmachine_message_bed_contact* sm_message = ((xmachine_message_bed_contact*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_bed_contact*)&message_share[d_SM_START]);
}

__device__ xmachine_message_bed_contact* get_next_bed_contact_message(xmachine_message_bed_contact* message, xmachine_message_bed_contact_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_bed_contact_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_bed_contact_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_bed_contact Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_bed_contact temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.bed_no = messages->bed_no[index];
		temp_message.state = messages->state[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_bed_contact));
		xmachine_message_bed_contact* sm_message = ((xmachine_message_bed_contact*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_bed_contact));
	return ((xmachine_message_bed_contact*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created box_petition message functions */


/** add_box_petition_message
 * Add non partitioned or spatially partitioned box_petition message
 * @param messages xmachine_message_box_petition_list message list to add too
 * @param id agent variable of type unsigned int
 * @param box_no agent variable of type unsigned int
 */
__device__ void add_box_petition_message(xmachine_message_box_petition_list* messages, unsigned int id, unsigned int box_no){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_box_petition_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_box_petition_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_box_petition_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_box_petition Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->box_no[index] = box_no;

}

/**
 * Scatter non partitioned or spatially partitioned box_petition message (for optional messages)
 * @param messages scatter_optional_box_petition_messages Sparse xmachine_message_box_petition_list message list
 * @param message_swap temp xmachine_message_box_petition_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_box_petition_messages(xmachine_message_box_petition_list* messages, xmachine_message_box_petition_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_box_petition_count;

		//AoS - xmachine_message_box_petition Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->box_no[output_index] = messages_swap->box_no[index];				
	}
}

/** reset_box_petition_swaps
 * Reset non partitioned or spatially partitioned box_petition message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_box_petition_swaps(xmachine_message_box_petition_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_box_petition* get_first_box_petition_message(xmachine_message_box_petition_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_box_petition_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_box_petition Coalesced memory read
	xmachine_message_box_petition temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.box_no = messages->box_no[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_box_petition));
	xmachine_message_box_petition* sm_message = ((xmachine_message_box_petition*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_box_petition*)&message_share[d_SM_START]);
}

__device__ xmachine_message_box_petition* get_next_box_petition_message(xmachine_message_box_petition* message, xmachine_message_box_petition_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_box_petition_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_box_petition_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_box_petition Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_box_petition temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.box_no = messages->box_no[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_box_petition));
		xmachine_message_box_petition* sm_message = ((xmachine_message_box_petition*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_box_petition));
	return ((xmachine_message_box_petition*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created box_response message functions */


/** add_box_response_message
 * Add non partitioned or spatially partitioned box_response message
 * @param messages xmachine_message_box_response_list message list to add too
 * @param id agent variable of type unsigned int
 * @param doctor_no agent variable of type unsigned int
 * @param priority agent variable of type unsigned int
 */
__device__ void add_box_response_message(xmachine_message_box_response_list* messages, unsigned int id, unsigned int doctor_no, unsigned int priority){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_box_response_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_box_response_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_box_response_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_box_response Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->doctor_no[index] = doctor_no;
	messages->priority[index] = priority;

}

/**
 * Scatter non partitioned or spatially partitioned box_response message (for optional messages)
 * @param messages scatter_optional_box_response_messages Sparse xmachine_message_box_response_list message list
 * @param message_swap temp xmachine_message_box_response_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_box_response_messages(xmachine_message_box_response_list* messages, xmachine_message_box_response_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_box_response_count;

		//AoS - xmachine_message_box_response Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->doctor_no[output_index] = messages_swap->doctor_no[index];
		messages->priority[output_index] = messages_swap->priority[index];				
	}
}

/** reset_box_response_swaps
 * Reset non partitioned or spatially partitioned box_response message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_box_response_swaps(xmachine_message_box_response_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_box_response* get_first_box_response_message(xmachine_message_box_response_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_box_response_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_box_response Coalesced memory read
	xmachine_message_box_response temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.doctor_no = messages->doctor_no[index];
	temp_message.priority = messages->priority[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_box_response));
	xmachine_message_box_response* sm_message = ((xmachine_message_box_response*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_box_response*)&message_share[d_SM_START]);
}

__device__ xmachine_message_box_response* get_next_box_response_message(xmachine_message_box_response* message, xmachine_message_box_response_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_box_response_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_box_response_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_box_response Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_box_response temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.doctor_no = messages->doctor_no[index];
		temp_message.priority = messages->priority[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_box_response));
		xmachine_message_box_response* sm_message = ((xmachine_message_box_response*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_box_response));
	return ((xmachine_message_box_response*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created specialist_reached message functions */


/** add_specialist_reached_message
 * Add non partitioned or spatially partitioned specialist_reached message
 * @param messages xmachine_message_specialist_reached_list message list to add too
 * @param id agent variable of type unsigned int
 * @param specialist_no agent variable of type unsigned int
 */
__device__ void add_specialist_reached_message(xmachine_message_specialist_reached_list* messages, unsigned int id, unsigned int specialist_no){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_specialist_reached_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_specialist_reached_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_specialist_reached_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_specialist_reached Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->specialist_no[index] = specialist_no;

}

/**
 * Scatter non partitioned or spatially partitioned specialist_reached message (for optional messages)
 * @param messages scatter_optional_specialist_reached_messages Sparse xmachine_message_specialist_reached_list message list
 * @param message_swap temp xmachine_message_specialist_reached_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_specialist_reached_messages(xmachine_message_specialist_reached_list* messages, xmachine_message_specialist_reached_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_specialist_reached_count;

		//AoS - xmachine_message_specialist_reached Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->specialist_no[output_index] = messages_swap->specialist_no[index];				
	}
}

/** reset_specialist_reached_swaps
 * Reset non partitioned or spatially partitioned specialist_reached message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_specialist_reached_swaps(xmachine_message_specialist_reached_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_specialist_reached* get_first_specialist_reached_message(xmachine_message_specialist_reached_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_specialist_reached_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_specialist_reached Coalesced memory read
	xmachine_message_specialist_reached temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.specialist_no = messages->specialist_no[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_specialist_reached));
	xmachine_message_specialist_reached* sm_message = ((xmachine_message_specialist_reached*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_specialist_reached*)&message_share[d_SM_START]);
}

__device__ xmachine_message_specialist_reached* get_next_specialist_reached_message(xmachine_message_specialist_reached* message, xmachine_message_specialist_reached_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_specialist_reached_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_specialist_reached_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_specialist_reached Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_specialist_reached temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.specialist_no = messages->specialist_no[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_specialist_reached));
		xmachine_message_specialist_reached* sm_message = ((xmachine_message_specialist_reached*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_specialist_reached));
	return ((xmachine_message_specialist_reached*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created specialist_terminated message functions */


/** add_specialist_terminated_message
 * Add non partitioned or spatially partitioned specialist_terminated message
 * @param messages xmachine_message_specialist_terminated_list message list to add too
 * @param id agent variable of type unsigned int
 */
__device__ void add_specialist_terminated_message(xmachine_message_specialist_terminated_list* messages, unsigned int id){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_specialist_terminated_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_specialist_terminated_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_specialist_terminated_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_specialist_terminated Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;

}

/**
 * Scatter non partitioned or spatially partitioned specialist_terminated message (for optional messages)
 * @param messages scatter_optional_specialist_terminated_messages Sparse xmachine_message_specialist_terminated_list message list
 * @param message_swap temp xmachine_message_specialist_terminated_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_specialist_terminated_messages(xmachine_message_specialist_terminated_list* messages, xmachine_message_specialist_terminated_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_specialist_terminated_count;

		//AoS - xmachine_message_specialist_terminated Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];				
	}
}

/** reset_specialist_terminated_swaps
 * Reset non partitioned or spatially partitioned specialist_terminated message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_specialist_terminated_swaps(xmachine_message_specialist_terminated_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_specialist_terminated* get_first_specialist_terminated_message(xmachine_message_specialist_terminated_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_specialist_terminated_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_specialist_terminated Coalesced memory read
	xmachine_message_specialist_terminated temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_specialist_terminated));
	xmachine_message_specialist_terminated* sm_message = ((xmachine_message_specialist_terminated*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_specialist_terminated*)&message_share[d_SM_START]);
}

__device__ xmachine_message_specialist_terminated* get_next_specialist_terminated_message(xmachine_message_specialist_terminated* message, xmachine_message_specialist_terminated_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_specialist_terminated_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_specialist_terminated_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_specialist_terminated Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_specialist_terminated temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_specialist_terminated));
		xmachine_message_specialist_terminated* sm_message = ((xmachine_message_specialist_terminated*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_specialist_terminated));
	return ((xmachine_message_specialist_terminated*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created free_specialist message functions */


/** add_free_specialist_message
 * Add non partitioned or spatially partitioned free_specialist message
 * @param messages xmachine_message_free_specialist_list message list to add too
 * @param specialist_no agent variable of type unsigned int
 */
__device__ void add_free_specialist_message(xmachine_message_free_specialist_list* messages, unsigned int specialist_no){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_free_specialist_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_free_specialist_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_free_specialist_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_free_specialist Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->specialist_no[index] = specialist_no;

}

/**
 * Scatter non partitioned or spatially partitioned free_specialist message (for optional messages)
 * @param messages scatter_optional_free_specialist_messages Sparse xmachine_message_free_specialist_list message list
 * @param message_swap temp xmachine_message_free_specialist_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_free_specialist_messages(xmachine_message_free_specialist_list* messages, xmachine_message_free_specialist_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_free_specialist_count;

		//AoS - xmachine_message_free_specialist Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->specialist_no[output_index] = messages_swap->specialist_no[index];				
	}
}

/** reset_free_specialist_swaps
 * Reset non partitioned or spatially partitioned free_specialist message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_free_specialist_swaps(xmachine_message_free_specialist_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_free_specialist* get_first_free_specialist_message(xmachine_message_free_specialist_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_free_specialist_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_free_specialist Coalesced memory read
	xmachine_message_free_specialist temp_message;
	temp_message._position = messages->_position[index];
	temp_message.specialist_no = messages->specialist_no[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_free_specialist));
	xmachine_message_free_specialist* sm_message = ((xmachine_message_free_specialist*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_free_specialist*)&message_share[d_SM_START]);
}

__device__ xmachine_message_free_specialist* get_next_free_specialist_message(xmachine_message_free_specialist* message, xmachine_message_free_specialist_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_free_specialist_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_free_specialist_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_free_specialist Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_free_specialist temp_message;
		temp_message._position = messages->_position[index];
		temp_message.specialist_no = messages->specialist_no[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_free_specialist));
		xmachine_message_free_specialist* sm_message = ((xmachine_message_free_specialist*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_free_specialist));
	return ((xmachine_message_free_specialist*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created specialist_petition message functions */


/** add_specialist_petition_message
 * Add non partitioned or spatially partitioned specialist_petition message
 * @param messages xmachine_message_specialist_petition_list message list to add too
 * @param id agent variable of type unsigned int
 * @param priority agent variable of type unsigned int
 * @param specialist_no agent variable of type unsigned int
 */
__device__ void add_specialist_petition_message(xmachine_message_specialist_petition_list* messages, unsigned int id, unsigned int priority, unsigned int specialist_no){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_specialist_petition_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_specialist_petition_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_specialist_petition_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_specialist_petition Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->priority[index] = priority;
	messages->specialist_no[index] = specialist_no;

}

/**
 * Scatter non partitioned or spatially partitioned specialist_petition message (for optional messages)
 * @param messages scatter_optional_specialist_petition_messages Sparse xmachine_message_specialist_petition_list message list
 * @param message_swap temp xmachine_message_specialist_petition_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_specialist_petition_messages(xmachine_message_specialist_petition_list* messages, xmachine_message_specialist_petition_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_specialist_petition_count;

		//AoS - xmachine_message_specialist_petition Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->priority[output_index] = messages_swap->priority[index];
		messages->specialist_no[output_index] = messages_swap->specialist_no[index];				
	}
}

/** reset_specialist_petition_swaps
 * Reset non partitioned or spatially partitioned specialist_petition message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_specialist_petition_swaps(xmachine_message_specialist_petition_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_specialist_petition* get_first_specialist_petition_message(xmachine_message_specialist_petition_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_specialist_petition_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_specialist_petition Coalesced memory read
	xmachine_message_specialist_petition temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.priority = messages->priority[index];
	temp_message.specialist_no = messages->specialist_no[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_specialist_petition));
	xmachine_message_specialist_petition* sm_message = ((xmachine_message_specialist_petition*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_specialist_petition*)&message_share[d_SM_START]);
}

__device__ xmachine_message_specialist_petition* get_next_specialist_petition_message(xmachine_message_specialist_petition* message, xmachine_message_specialist_petition_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_specialist_petition_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_specialist_petition_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_specialist_petition Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_specialist_petition temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.priority = messages->priority[index];
		temp_message.specialist_no = messages->specialist_no[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_specialist_petition));
		xmachine_message_specialist_petition* sm_message = ((xmachine_message_specialist_petition*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_specialist_petition));
	return ((xmachine_message_specialist_petition*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created specialist_response message functions */


/** add_specialist_response_message
 * Add non partitioned or spatially partitioned specialist_response message
 * @param messages xmachine_message_specialist_response_list message list to add too
 * @param id agent variable of type unsigned int
 * @param specialist_ready agent variable of type int
 */
__device__ void add_specialist_response_message(xmachine_message_specialist_response_list* messages, unsigned int id, int specialist_ready){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_specialist_response_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_specialist_response_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_specialist_response_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_specialist_response Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->specialist_ready[index] = specialist_ready;

}

/**
 * Scatter non partitioned or spatially partitioned specialist_response message (for optional messages)
 * @param messages scatter_optional_specialist_response_messages Sparse xmachine_message_specialist_response_list message list
 * @param message_swap temp xmachine_message_specialist_response_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_specialist_response_messages(xmachine_message_specialist_response_list* messages, xmachine_message_specialist_response_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_specialist_response_count;

		//AoS - xmachine_message_specialist_response Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->specialist_ready[output_index] = messages_swap->specialist_ready[index];				
	}
}

/** reset_specialist_response_swaps
 * Reset non partitioned or spatially partitioned specialist_response message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_specialist_response_swaps(xmachine_message_specialist_response_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_specialist_response* get_first_specialist_response_message(xmachine_message_specialist_response_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_specialist_response_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_specialist_response Coalesced memory read
	xmachine_message_specialist_response temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.specialist_ready = messages->specialist_ready[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_specialist_response));
	xmachine_message_specialist_response* sm_message = ((xmachine_message_specialist_response*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_specialist_response*)&message_share[d_SM_START]);
}

__device__ xmachine_message_specialist_response* get_next_specialist_response_message(xmachine_message_specialist_response* message, xmachine_message_specialist_response_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_specialist_response_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_specialist_response_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_specialist_response Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_specialist_response temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.specialist_ready = messages->specialist_ready[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_specialist_response));
		xmachine_message_specialist_response* sm_message = ((xmachine_message_specialist_response*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_specialist_response));
	return ((xmachine_message_specialist_response*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created doctor_reached message functions */


/** add_doctor_reached_message
 * Add non partitioned or spatially partitioned doctor_reached message
 * @param messages xmachine_message_doctor_reached_list message list to add too
 * @param id agent variable of type unsigned int
 * @param doctor_no agent variable of type unsigned int
 */
__device__ void add_doctor_reached_message(xmachine_message_doctor_reached_list* messages, unsigned int id, unsigned int doctor_no){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_doctor_reached_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_doctor_reached_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_doctor_reached_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_doctor_reached Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->doctor_no[index] = doctor_no;

}

/**
 * Scatter non partitioned or spatially partitioned doctor_reached message (for optional messages)
 * @param messages scatter_optional_doctor_reached_messages Sparse xmachine_message_doctor_reached_list message list
 * @param message_swap temp xmachine_message_doctor_reached_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_doctor_reached_messages(xmachine_message_doctor_reached_list* messages, xmachine_message_doctor_reached_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_doctor_reached_count;

		//AoS - xmachine_message_doctor_reached Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->doctor_no[output_index] = messages_swap->doctor_no[index];				
	}
}

/** reset_doctor_reached_swaps
 * Reset non partitioned or spatially partitioned doctor_reached message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_doctor_reached_swaps(xmachine_message_doctor_reached_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_doctor_reached* get_first_doctor_reached_message(xmachine_message_doctor_reached_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_doctor_reached_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_doctor_reached Coalesced memory read
	xmachine_message_doctor_reached temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.doctor_no = messages->doctor_no[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_doctor_reached));
	xmachine_message_doctor_reached* sm_message = ((xmachine_message_doctor_reached*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_doctor_reached*)&message_share[d_SM_START]);
}

__device__ xmachine_message_doctor_reached* get_next_doctor_reached_message(xmachine_message_doctor_reached* message, xmachine_message_doctor_reached_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_doctor_reached_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_doctor_reached_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_doctor_reached Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_doctor_reached temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.doctor_no = messages->doctor_no[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_doctor_reached));
		xmachine_message_doctor_reached* sm_message = ((xmachine_message_doctor_reached*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_doctor_reached));
	return ((xmachine_message_doctor_reached*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created free_doctor message functions */


/** add_free_doctor_message
 * Add non partitioned or spatially partitioned free_doctor message
 * @param messages xmachine_message_free_doctor_list message list to add too
 * @param doctor_no agent variable of type unsigned int
 */
__device__ void add_free_doctor_message(xmachine_message_free_doctor_list* messages, unsigned int doctor_no){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_free_doctor_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_free_doctor_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_free_doctor_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_free_doctor Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->doctor_no[index] = doctor_no;

}

/**
 * Scatter non partitioned or spatially partitioned free_doctor message (for optional messages)
 * @param messages scatter_optional_free_doctor_messages Sparse xmachine_message_free_doctor_list message list
 * @param message_swap temp xmachine_message_free_doctor_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_free_doctor_messages(xmachine_message_free_doctor_list* messages, xmachine_message_free_doctor_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_free_doctor_count;

		//AoS - xmachine_message_free_doctor Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->doctor_no[output_index] = messages_swap->doctor_no[index];				
	}
}

/** reset_free_doctor_swaps
 * Reset non partitioned or spatially partitioned free_doctor message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_free_doctor_swaps(xmachine_message_free_doctor_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_free_doctor* get_first_free_doctor_message(xmachine_message_free_doctor_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_free_doctor_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_free_doctor Coalesced memory read
	xmachine_message_free_doctor temp_message;
	temp_message._position = messages->_position[index];
	temp_message.doctor_no = messages->doctor_no[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_free_doctor));
	xmachine_message_free_doctor* sm_message = ((xmachine_message_free_doctor*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_free_doctor*)&message_share[d_SM_START]);
}

__device__ xmachine_message_free_doctor* get_next_free_doctor_message(xmachine_message_free_doctor* message, xmachine_message_free_doctor_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_free_doctor_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_free_doctor_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_free_doctor Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_free_doctor temp_message;
		temp_message._position = messages->_position[index];
		temp_message.doctor_no = messages->doctor_no[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_free_doctor));
		xmachine_message_free_doctor* sm_message = ((xmachine_message_free_doctor*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_free_doctor));
	return ((xmachine_message_free_doctor*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created attention_terminated message functions */


/** add_attention_terminated_message
 * Add non partitioned or spatially partitioned attention_terminated message
 * @param messages xmachine_message_attention_terminated_list message list to add too
 * @param id agent variable of type unsigned int
 */
__device__ void add_attention_terminated_message(xmachine_message_attention_terminated_list* messages, unsigned int id){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_attention_terminated_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_attention_terminated_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_attention_terminated_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_attention_terminated Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;

}

/**
 * Scatter non partitioned or spatially partitioned attention_terminated message (for optional messages)
 * @param messages scatter_optional_attention_terminated_messages Sparse xmachine_message_attention_terminated_list message list
 * @param message_swap temp xmachine_message_attention_terminated_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_attention_terminated_messages(xmachine_message_attention_terminated_list* messages, xmachine_message_attention_terminated_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_attention_terminated_count;

		//AoS - xmachine_message_attention_terminated Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];				
	}
}

/** reset_attention_terminated_swaps
 * Reset non partitioned or spatially partitioned attention_terminated message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_attention_terminated_swaps(xmachine_message_attention_terminated_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_attention_terminated* get_first_attention_terminated_message(xmachine_message_attention_terminated_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_attention_terminated_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_attention_terminated Coalesced memory read
	xmachine_message_attention_terminated temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_attention_terminated));
	xmachine_message_attention_terminated* sm_message = ((xmachine_message_attention_terminated*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_attention_terminated*)&message_share[d_SM_START]);
}

__device__ xmachine_message_attention_terminated* get_next_attention_terminated_message(xmachine_message_attention_terminated* message, xmachine_message_attention_terminated_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_attention_terminated_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_attention_terminated_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_attention_terminated Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_attention_terminated temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_attention_terminated));
		xmachine_message_attention_terminated* sm_message = ((xmachine_message_attention_terminated*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_attention_terminated));
	return ((xmachine_message_attention_terminated*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created doctor_petition message functions */


/** add_doctor_petition_message
 * Add non partitioned or spatially partitioned doctor_petition message
 * @param messages xmachine_message_doctor_petition_list message list to add too
 * @param id agent variable of type unsigned int
 * @param priority agent variable of type unsigned int
 */
__device__ void add_doctor_petition_message(xmachine_message_doctor_petition_list* messages, unsigned int id, unsigned int priority){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_doctor_petition_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_doctor_petition_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_doctor_petition_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_doctor_petition Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->priority[index] = priority;

}

/**
 * Scatter non partitioned or spatially partitioned doctor_petition message (for optional messages)
 * @param messages scatter_optional_doctor_petition_messages Sparse xmachine_message_doctor_petition_list message list
 * @param message_swap temp xmachine_message_doctor_petition_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_doctor_petition_messages(xmachine_message_doctor_petition_list* messages, xmachine_message_doctor_petition_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_doctor_petition_count;

		//AoS - xmachine_message_doctor_petition Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->priority[output_index] = messages_swap->priority[index];				
	}
}

/** reset_doctor_petition_swaps
 * Reset non partitioned or spatially partitioned doctor_petition message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_doctor_petition_swaps(xmachine_message_doctor_petition_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_doctor_petition* get_first_doctor_petition_message(xmachine_message_doctor_petition_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_doctor_petition_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_doctor_petition Coalesced memory read
	xmachine_message_doctor_petition temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.priority = messages->priority[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_doctor_petition));
	xmachine_message_doctor_petition* sm_message = ((xmachine_message_doctor_petition*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_doctor_petition*)&message_share[d_SM_START]);
}

__device__ xmachine_message_doctor_petition* get_next_doctor_petition_message(xmachine_message_doctor_petition* message, xmachine_message_doctor_petition_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_doctor_petition_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_doctor_petition_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_doctor_petition Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_doctor_petition temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.priority = messages->priority[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_doctor_petition));
		xmachine_message_doctor_petition* sm_message = ((xmachine_message_doctor_petition*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_doctor_petition));
	return ((xmachine_message_doctor_petition*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created doctor_response message functions */


/** add_doctor_response_message
 * Add non partitioned or spatially partitioned doctor_response message
 * @param messages xmachine_message_doctor_response_list message list to add too
 * @param id agent variable of type unsigned int
 * @param doctor_no agent variable of type int
 */
__device__ void add_doctor_response_message(xmachine_message_doctor_response_list* messages, unsigned int id, int doctor_no){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_doctor_response_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_doctor_response_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_doctor_response_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_doctor_response Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->doctor_no[index] = doctor_no;

}

/**
 * Scatter non partitioned or spatially partitioned doctor_response message (for optional messages)
 * @param messages scatter_optional_doctor_response_messages Sparse xmachine_message_doctor_response_list message list
 * @param message_swap temp xmachine_message_doctor_response_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_doctor_response_messages(xmachine_message_doctor_response_list* messages, xmachine_message_doctor_response_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_doctor_response_count;

		//AoS - xmachine_message_doctor_response Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->doctor_no[output_index] = messages_swap->doctor_no[index];				
	}
}

/** reset_doctor_response_swaps
 * Reset non partitioned or spatially partitioned doctor_response message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_doctor_response_swaps(xmachine_message_doctor_response_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_doctor_response* get_first_doctor_response_message(xmachine_message_doctor_response_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_doctor_response_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_doctor_response Coalesced memory read
	xmachine_message_doctor_response temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.doctor_no = messages->doctor_no[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_doctor_response));
	xmachine_message_doctor_response* sm_message = ((xmachine_message_doctor_response*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_doctor_response*)&message_share[d_SM_START]);
}

__device__ xmachine_message_doctor_response* get_next_doctor_response_message(xmachine_message_doctor_response* message, xmachine_message_doctor_response_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_doctor_response_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_doctor_response_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_doctor_response Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_doctor_response temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.doctor_no = messages->doctor_no[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_doctor_response));
		xmachine_message_doctor_response* sm_message = ((xmachine_message_doctor_response*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_doctor_response));
	return ((xmachine_message_doctor_response*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created bed_petition message functions */


/** add_bed_petition_message
 * Add non partitioned or spatially partitioned bed_petition message
 * @param messages xmachine_message_bed_petition_list message list to add too
 * @param id agent variable of type unsigned int
 */
__device__ void add_bed_petition_message(xmachine_message_bed_petition_list* messages, unsigned int id){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_bed_petition_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_bed_petition_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_bed_petition_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_bed_petition Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;

}

/**
 * Scatter non partitioned or spatially partitioned bed_petition message (for optional messages)
 * @param messages scatter_optional_bed_petition_messages Sparse xmachine_message_bed_petition_list message list
 * @param message_swap temp xmachine_message_bed_petition_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_bed_petition_messages(xmachine_message_bed_petition_list* messages, xmachine_message_bed_petition_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_bed_petition_count;

		//AoS - xmachine_message_bed_petition Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];				
	}
}

/** reset_bed_petition_swaps
 * Reset non partitioned or spatially partitioned bed_petition message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_bed_petition_swaps(xmachine_message_bed_petition_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_bed_petition* get_first_bed_petition_message(xmachine_message_bed_petition_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_bed_petition_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_bed_petition Coalesced memory read
	xmachine_message_bed_petition temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_bed_petition));
	xmachine_message_bed_petition* sm_message = ((xmachine_message_bed_petition*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_bed_petition*)&message_share[d_SM_START]);
}

__device__ xmachine_message_bed_petition* get_next_bed_petition_message(xmachine_message_bed_petition* message, xmachine_message_bed_petition_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_bed_petition_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_bed_petition_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_bed_petition Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_bed_petition temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_bed_petition));
		xmachine_message_bed_petition* sm_message = ((xmachine_message_bed_petition*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_bed_petition));
	return ((xmachine_message_bed_petition*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created bed_response message functions */


/** add_bed_response_message
 * Add non partitioned or spatially partitioned bed_response message
 * @param messages xmachine_message_bed_response_list message list to add too
 * @param id agent variable of type unsigned int
 * @param bed_no agent variable of type int
 */
__device__ void add_bed_response_message(xmachine_message_bed_response_list* messages, unsigned int id, int bed_no){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_bed_response_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_bed_response_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_bed_response_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_bed_response Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->bed_no[index] = bed_no;

}

/**
 * Scatter non partitioned or spatially partitioned bed_response message (for optional messages)
 * @param messages scatter_optional_bed_response_messages Sparse xmachine_message_bed_response_list message list
 * @param message_swap temp xmachine_message_bed_response_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_bed_response_messages(xmachine_message_bed_response_list* messages, xmachine_message_bed_response_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_bed_response_count;

		//AoS - xmachine_message_bed_response Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->bed_no[output_index] = messages_swap->bed_no[index];				
	}
}

/** reset_bed_response_swaps
 * Reset non partitioned or spatially partitioned bed_response message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_bed_response_swaps(xmachine_message_bed_response_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_bed_response* get_first_bed_response_message(xmachine_message_bed_response_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_bed_response_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_bed_response Coalesced memory read
	xmachine_message_bed_response temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.bed_no = messages->bed_no[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_bed_response));
	xmachine_message_bed_response* sm_message = ((xmachine_message_bed_response*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_bed_response*)&message_share[d_SM_START]);
}

__device__ xmachine_message_bed_response* get_next_bed_response_message(xmachine_message_bed_response* message, xmachine_message_bed_response_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_bed_response_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_bed_response_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_bed_response Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_bed_response temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.bed_no = messages->bed_no[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_bed_response));
		xmachine_message_bed_response* sm_message = ((xmachine_message_bed_response*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_bed_response));
	return ((xmachine_message_bed_response*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created triage_petition message functions */


/** add_triage_petition_message
 * Add non partitioned or spatially partitioned triage_petition message
 * @param messages xmachine_message_triage_petition_list message list to add too
 * @param id agent variable of type unsigned int
 */
__device__ void add_triage_petition_message(xmachine_message_triage_petition_list* messages, unsigned int id){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_triage_petition_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_triage_petition_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_triage_petition_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_triage_petition Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;

}

/**
 * Scatter non partitioned or spatially partitioned triage_petition message (for optional messages)
 * @param messages scatter_optional_triage_petition_messages Sparse xmachine_message_triage_petition_list message list
 * @param message_swap temp xmachine_message_triage_petition_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_triage_petition_messages(xmachine_message_triage_petition_list* messages, xmachine_message_triage_petition_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_triage_petition_count;

		//AoS - xmachine_message_triage_petition Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];				
	}
}

/** reset_triage_petition_swaps
 * Reset non partitioned or spatially partitioned triage_petition message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_triage_petition_swaps(xmachine_message_triage_petition_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_triage_petition* get_first_triage_petition_message(xmachine_message_triage_petition_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_triage_petition_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_triage_petition Coalesced memory read
	xmachine_message_triage_petition temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_triage_petition));
	xmachine_message_triage_petition* sm_message = ((xmachine_message_triage_petition*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_triage_petition*)&message_share[d_SM_START]);
}

__device__ xmachine_message_triage_petition* get_next_triage_petition_message(xmachine_message_triage_petition* message, xmachine_message_triage_petition_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_triage_petition_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_triage_petition_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_triage_petition Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_triage_petition temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_triage_petition));
		xmachine_message_triage_petition* sm_message = ((xmachine_message_triage_petition*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_triage_petition));
	return ((xmachine_message_triage_petition*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created triage_response message functions */


/** add_triage_response_message
 * Add non partitioned or spatially partitioned triage_response message
 * @param messages xmachine_message_triage_response_list message list to add too
 * @param id agent variable of type unsigned int
 * @param box_no agent variable of type int
 */
__device__ void add_triage_response_message(xmachine_message_triage_response_list* messages, unsigned int id, int box_no){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_triage_response_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_triage_response_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_triage_response_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_triage_response Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->id[index] = id;
	messages->box_no[index] = box_no;

}

/**
 * Scatter non partitioned or spatially partitioned triage_response message (for optional messages)
 * @param messages scatter_optional_triage_response_messages Sparse xmachine_message_triage_response_list message list
 * @param message_swap temp xmachine_message_triage_response_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_triage_response_messages(xmachine_message_triage_response_list* messages, xmachine_message_triage_response_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_triage_response_count;

		//AoS - xmachine_message_triage_response Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->id[output_index] = messages_swap->id[index];
		messages->box_no[output_index] = messages_swap->box_no[index];				
	}
}

/** reset_triage_response_swaps
 * Reset non partitioned or spatially partitioned triage_response message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_triage_response_swaps(xmachine_message_triage_response_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_triage_response* get_first_triage_response_message(xmachine_message_triage_response_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_triage_response_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_triage_response Coalesced memory read
	xmachine_message_triage_response temp_message;
	temp_message._position = messages->_position[index];
	temp_message.id = messages->id[index];
	temp_message.box_no = messages->box_no[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_triage_response));
	xmachine_message_triage_response* sm_message = ((xmachine_message_triage_response*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_triage_response*)&message_share[d_SM_START]);
}

__device__ xmachine_message_triage_response* get_next_triage_response_message(xmachine_message_triage_response* message, xmachine_message_triage_response_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_triage_response_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_triage_response_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_triage_response Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_triage_response temp_message;
		temp_message._position = messages->_position[index];
		temp_message.id = messages->id[index];
		temp_message.box_no = messages->box_no[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_triage_response));
		xmachine_message_triage_response* sm_message = ((xmachine_message_triage_response*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_triage_response));
	return ((xmachine_message_triage_response*)&message_share[message_index]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created free_box message functions */


/** add_free_box_message
 * Add non partitioned or spatially partitioned free_box message
 * @param messages xmachine_message_free_box_list message list to add too
 * @param box_no agent variable of type int
 */
__device__ void add_free_box_message(xmachine_message_free_box_list* messages, int box_no){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_free_box_count;

	int _position;
	int _scan_input;

	//decide output position
	if(d_message_free_box_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}else if (d_message_free_box_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_free_box Coalesced memory write
	messages->_scan_input[index] = _scan_input;	
	messages->_position[index] = _position;
	messages->box_no[index] = box_no;

}

/**
 * Scatter non partitioned or spatially partitioned free_box message (for optional messages)
 * @param messages scatter_optional_free_box_messages Sparse xmachine_message_free_box_list message list
 * @param message_swap temp xmachine_message_free_box_list message list to scatter sparse messages to
 */
__global__ void scatter_optional_free_box_messages(xmachine_message_free_box_list* messages, xmachine_message_free_box_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_free_box_count;

		//AoS - xmachine_message_free_box Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->box_no[output_index] = messages_swap->box_no[index];				
	}
}

/** reset_free_box_swaps
 * Reset non partitioned or spatially partitioned free_box message swaps (for scattering optional messages)
 * @param message_swap message list to reset _position and _scan_input values back to 0
 */
__global__ void reset_free_box_swaps(xmachine_message_free_box_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

__device__ xmachine_message_free_box* get_first_free_box_message(xmachine_message_free_box_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = (ceil((float)d_message_free_box_count/ blockDim.x)* blockDim.x);

	//if no messages then return a null pointer (false)
	if (wrap_size == 0)
		return nullptr;

	//global thread index
	int global_index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//global thread index
	int index = WRAP(global_index, wrap_size);

	//SoA to AoS - xmachine_message_free_box Coalesced memory read
	xmachine_message_free_box temp_message;
	temp_message._position = messages->_position[index];
	temp_message.box_no = messages->box_no[index];

	//AoS to shared memory
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_free_box));
	xmachine_message_free_box* sm_message = ((xmachine_message_free_box*)&message_share[message_index]);
	sm_message[0] = temp_message;

	__syncthreads();

  //HACK FOR 64 bit addressing issue in sm
	return ((xmachine_message_free_box*)&message_share[d_SM_START]);
}

__device__ xmachine_message_free_box* get_next_free_box_message(xmachine_message_free_box* message, xmachine_message_free_box_list* messages){

	extern __shared__ int sm_data [];
	char* message_share = (char*)&sm_data[0];
	
	//wrap size is the number of tiles required to load all messages
	int wrap_size = ceil((float)d_message_free_box_count/ blockDim.x)*blockDim.x;

	int i = WRAP((message->_position + 1),wrap_size);

	//If end of messages (last message not multiple of gridsize) go to 0 index
	if (i >= d_message_free_box_count)
		i = 0;

	//Check if back to start position of first message
	if (i == WRAP((blockDim.x* blockIdx.x), wrap_size))
		return nullptr;

	int tile = floor((float)i/(blockDim.x)); //tile is round down position over blockDim
	i = i % blockDim.x;						 //mod i for shared memory index

	//if count == Block Size load next tile int shared memory values
	if (i == 0){
		__syncthreads();					//make sure we don't change shared memory until all threads are here (important for emu-debug mode)
		
		//SoA to AoS - xmachine_message_free_box Coalesced memory read
		int index = (tile* blockDim.x) + threadIdx.x;
		xmachine_message_free_box temp_message;
		temp_message._position = messages->_position[index];
		temp_message.box_no = messages->box_no[index];

		//AoS to shared memory
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x+threadIdx.x, sizeof(xmachine_message_free_box));
		xmachine_message_free_box* sm_message = ((xmachine_message_free_box*)&message_share[message_index]);
		sm_message[0] = temp_message;

		__syncthreads();					//make sure we don't start returning messages until all threads have updated shared memory
	}

	int message_index = SHARE_INDEX(i, sizeof(xmachine_message_free_box));
	return ((xmachine_message_free_box*)&message_share[message_index]);
}

	
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created GPU kernels  */



/**
 *
 */
__global__ void GPUFLAME_output_pedestrian_location(xmachine_memory_agent_list* agents, xmachine_message_pedestrian_location_list* pedestrian_location_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_pedestrian_location Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !output_pedestrian_location(&agent, pedestrian_location_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_pedestrian_location Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_avoid_pedestrians(xmachine_memory_agent_list* agents, xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_avoid_pedestrians Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !avoid_pedestrians(&agent, pedestrian_location_messages, partition_matrix, rand48);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_avoid_pedestrians Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_output_pedestrian_state(xmachine_memory_agent_list* agents, xmachine_message_pedestrian_state_list* pedestrian_state_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_pedestrian_state Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !output_pedestrian_state(&agent, pedestrian_state_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_pedestrian_state Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_infect_patients(xmachine_memory_agent_list* agents, xmachine_message_pedestrian_state_list* pedestrian_state_messages, xmachine_message_pedestrian_state_PBM* partition_matrix, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_infect_patients Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !infect_patients(&agent, pedestrian_state_messages, partition_matrix, rand48);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_infect_patients Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_infect_patients_UCI(xmachine_memory_agent_list* agents, xmachine_message_pedestrian_state_list* pedestrian_state_messages, xmachine_message_pedestrian_state_PBM* partition_matrix, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_infect_patients_UCI Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !infect_patients_UCI(&agent, pedestrian_state_messages, partition_matrix, rand48);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_infect_patients_UCI Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_move(xmachine_memory_agent_list* agents, xmachine_message_check_in_list* check_in_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_move Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !move(&agent, check_in_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_move Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_receive_chair_state(xmachine_memory_agent_list* agents, xmachine_message_chair_state_list* chair_state_messages, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_chair_state Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];
	} else {
	
	agent.id = 0;
	agent.x = 0;
	agent.y = 0;
	agent.velx = 0;
	agent.vely = 0;
	agent.steer_x = 0;
	agent.steer_y = 0;
	agent.height = 0;
	agent.exit_no = 0;
	agent.speed = 0;
	agent.lod = 0;
	agent.animate = 0;
	agent.animate_dir = 0;
	agent.estado = 0;
	agent.tick = 0;
	agent.estado_movimiento = 0;
	agent.go_to_x = 0;
	agent.go_to_y = 0;
	agent.checkpoint = 0;
	agent.chair_no = 0;
	agent.box_no = 0;
	agent.doctor_no = 0;
	agent.specialist_no = 0;
	agent.bed_no = 0;
	agent.priority = 0;
	agent.vaccine = 0;
	}

	//FLAME function call
	int dead = !receive_chair_state(&agent, chair_state_messages, rand48);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_chair_state Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
	}
}

/**
 *
 */
__global__ void GPUFLAME_output_chair_contact(xmachine_memory_agent_list* agents, xmachine_message_chair_contact_list* chair_contact_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_chair_contact Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !output_chair_contact(&agent, chair_contact_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_chair_contact Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_output_free_chair(xmachine_memory_agent_list* agents, xmachine_message_free_chair_list* free_chair_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_free_chair Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !output_free_chair(&agent, free_chair_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_free_chair Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_output_chair_petition(xmachine_memory_agent_list* agents, xmachine_message_chair_petition_list* chair_petition_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_chair_petition Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !output_chair_petition(&agent, chair_petition_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_chair_petition Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_receive_chair_response(xmachine_memory_agent_list* agents, xmachine_message_chair_response_list* chair_response_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_chair_response Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];
	} else {
	
	agent.id = 0;
	agent.x = 0;
	agent.y = 0;
	agent.velx = 0;
	agent.vely = 0;
	agent.steer_x = 0;
	agent.steer_y = 0;
	agent.height = 0;
	agent.exit_no = 0;
	agent.speed = 0;
	agent.lod = 0;
	agent.animate = 0;
	agent.animate_dir = 0;
	agent.estado = 0;
	agent.tick = 0;
	agent.estado_movimiento = 0;
	agent.go_to_x = 0;
	agent.go_to_y = 0;
	agent.checkpoint = 0;
	agent.chair_no = 0;
	agent.box_no = 0;
	agent.doctor_no = 0;
	agent.specialist_no = 0;
	agent.bed_no = 0;
	agent.priority = 0;
	agent.vaccine = 0;
	}

	//FLAME function call
	int dead = !receive_chair_response(&agent, chair_response_messages);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_chair_response Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
	}
}

/**
 *
 */
__global__ void GPUFLAME_receive_check_in_response(xmachine_memory_agent_list* agents, xmachine_message_check_in_response_list* check_in_response_messages, xmachine_message_free_chair_list* free_chair_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_check_in_response Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];
	} else {
	
	agent.id = 0;
	agent.x = 0;
	agent.y = 0;
	agent.velx = 0;
	agent.vely = 0;
	agent.steer_x = 0;
	agent.steer_y = 0;
	agent.height = 0;
	agent.exit_no = 0;
	agent.speed = 0;
	agent.lod = 0;
	agent.animate = 0;
	agent.animate_dir = 0;
	agent.estado = 0;
	agent.tick = 0;
	agent.estado_movimiento = 0;
	agent.go_to_x = 0;
	agent.go_to_y = 0;
	agent.checkpoint = 0;
	agent.chair_no = 0;
	agent.box_no = 0;
	agent.doctor_no = 0;
	agent.specialist_no = 0;
	agent.bed_no = 0;
	agent.priority = 0;
	agent.vaccine = 0;
	}

	//FLAME function call
	int dead = !receive_check_in_response(&agent, check_in_response_messages, free_chair_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_check_in_response Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
	}
}

/**
 *
 */
__global__ void GPUFLAME_output_box_petition(xmachine_memory_agent_list* agents, xmachine_message_box_petition_list* box_petition_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_box_petition Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !output_box_petition(&agent, box_petition_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_box_petition Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_receive_box_response(xmachine_memory_agent_list* agents, xmachine_message_box_response_list* box_response_messages, xmachine_message_free_box_list* free_box_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_box_response Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];
	} else {
	
	agent.id = 0;
	agent.x = 0;
	agent.y = 0;
	agent.velx = 0;
	agent.vely = 0;
	agent.steer_x = 0;
	agent.steer_y = 0;
	agent.height = 0;
	agent.exit_no = 0;
	agent.speed = 0;
	agent.lod = 0;
	agent.animate = 0;
	agent.animate_dir = 0;
	agent.estado = 0;
	agent.tick = 0;
	agent.estado_movimiento = 0;
	agent.go_to_x = 0;
	agent.go_to_y = 0;
	agent.checkpoint = 0;
	agent.chair_no = 0;
	agent.box_no = 0;
	agent.doctor_no = 0;
	agent.specialist_no = 0;
	agent.bed_no = 0;
	agent.priority = 0;
	agent.vaccine = 0;
	}

	//FLAME function call
	int dead = !receive_box_response(&agent, box_response_messages, free_box_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_box_response Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
	}
}

/**
 *
 */
__global__ void GPUFLAME_output_doctor_petition(xmachine_memory_agent_list* agents, xmachine_message_doctor_petition_list* doctor_petition_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_doctor_petition Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !output_doctor_petition(&agent, doctor_petition_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_doctor_petition Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_receive_doctor_response(xmachine_memory_agent_list* agents, xmachine_message_doctor_response_list* doctor_response_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_doctor_response Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];
	} else {
	
	agent.id = 0;
	agent.x = 0;
	agent.y = 0;
	agent.velx = 0;
	agent.vely = 0;
	agent.steer_x = 0;
	agent.steer_y = 0;
	agent.height = 0;
	agent.exit_no = 0;
	agent.speed = 0;
	agent.lod = 0;
	agent.animate = 0;
	agent.animate_dir = 0;
	agent.estado = 0;
	agent.tick = 0;
	agent.estado_movimiento = 0;
	agent.go_to_x = 0;
	agent.go_to_y = 0;
	agent.checkpoint = 0;
	agent.chair_no = 0;
	agent.box_no = 0;
	agent.doctor_no = 0;
	agent.specialist_no = 0;
	agent.bed_no = 0;
	agent.priority = 0;
	agent.vaccine = 0;
	}

	//FLAME function call
	int dead = !receive_doctor_response(&agent, doctor_response_messages);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_doctor_response Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
	}
}

/**
 *
 */
__global__ void GPUFLAME_receive_attention_terminated(xmachine_memory_agent_list* agents, xmachine_message_attention_terminated_list* attention_terminated_messages, xmachine_message_free_doctor_list* free_doctor_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_attention_terminated Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];
	} else {
	
	agent.id = 0;
	agent.x = 0;
	agent.y = 0;
	agent.velx = 0;
	agent.vely = 0;
	agent.steer_x = 0;
	agent.steer_y = 0;
	agent.height = 0;
	agent.exit_no = 0;
	agent.speed = 0;
	agent.lod = 0;
	agent.animate = 0;
	agent.animate_dir = 0;
	agent.estado = 0;
	agent.tick = 0;
	agent.estado_movimiento = 0;
	agent.go_to_x = 0;
	agent.go_to_y = 0;
	agent.checkpoint = 0;
	agent.chair_no = 0;
	agent.box_no = 0;
	agent.doctor_no = 0;
	agent.specialist_no = 0;
	agent.bed_no = 0;
	agent.priority = 0;
	agent.vaccine = 0;
	}

	//FLAME function call
	int dead = !receive_attention_terminated(&agent, attention_terminated_messages, free_doctor_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_attention_terminated Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
	}
}

/**
 *
 */
__global__ void GPUFLAME_receive_specialist_terminated(xmachine_memory_agent_list* agents, xmachine_message_specialist_terminated_list* specialist_terminated_messages, xmachine_message_free_specialist_list* free_specialist_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_specialist_terminated Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];
	} else {
	
	agent.id = 0;
	agent.x = 0;
	agent.y = 0;
	agent.velx = 0;
	agent.vely = 0;
	agent.steer_x = 0;
	agent.steer_y = 0;
	agent.height = 0;
	agent.exit_no = 0;
	agent.speed = 0;
	agent.lod = 0;
	agent.animate = 0;
	agent.animate_dir = 0;
	agent.estado = 0;
	agent.tick = 0;
	agent.estado_movimiento = 0;
	agent.go_to_x = 0;
	agent.go_to_y = 0;
	agent.checkpoint = 0;
	agent.chair_no = 0;
	agent.box_no = 0;
	agent.doctor_no = 0;
	agent.specialist_no = 0;
	agent.bed_no = 0;
	agent.priority = 0;
	agent.vaccine = 0;
	}

	//FLAME function call
	int dead = !receive_specialist_terminated(&agent, specialist_terminated_messages, free_specialist_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_specialist_terminated Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
	}
}

/**
 *
 */
__global__ void GPUFLAME_output_doctor_reached(xmachine_memory_agent_list* agents, xmachine_message_doctor_reached_list* doctor_reached_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_doctor_reached Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !output_doctor_reached(&agent, doctor_reached_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_doctor_reached Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_receive_specialist_response(xmachine_memory_agent_list* agents, xmachine_message_specialist_response_list* specialist_response_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_specialist_response Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];
	} else {
	
	agent.id = 0;
	agent.x = 0;
	agent.y = 0;
	agent.velx = 0;
	agent.vely = 0;
	agent.steer_x = 0;
	agent.steer_y = 0;
	agent.height = 0;
	agent.exit_no = 0;
	agent.speed = 0;
	agent.lod = 0;
	agent.animate = 0;
	agent.animate_dir = 0;
	agent.estado = 0;
	agent.tick = 0;
	agent.estado_movimiento = 0;
	agent.go_to_x = 0;
	agent.go_to_y = 0;
	agent.checkpoint = 0;
	agent.chair_no = 0;
	agent.box_no = 0;
	agent.doctor_no = 0;
	agent.specialist_no = 0;
	agent.bed_no = 0;
	agent.priority = 0;
	agent.vaccine = 0;
	}

	//FLAME function call
	int dead = !receive_specialist_response(&agent, specialist_response_messages);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_specialist_response Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
	}
}

/**
 *
 */
__global__ void GPUFLAME_output_specialist_petition(xmachine_memory_agent_list* agents, xmachine_message_specialist_petition_list* specialist_petition_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_specialist_petition Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !output_specialist_petition(&agent, specialist_petition_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_specialist_petition Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_output_specialist_reached(xmachine_memory_agent_list* agents, xmachine_message_specialist_reached_list* specialist_reached_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_specialist_reached Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !output_specialist_reached(&agent, specialist_reached_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_specialist_reached Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_output_bed_petition(xmachine_memory_agent_list* agents, xmachine_message_bed_petition_list* bed_petition_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_bed_petition Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !output_bed_petition(&agent, bed_petition_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_bed_petition Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_receive_bed_state(xmachine_memory_agent_list* agents, xmachine_message_bed_state_list* bed_state_messages, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_bed_state Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];
	} else {
	
	agent.id = 0;
	agent.x = 0;
	agent.y = 0;
	agent.velx = 0;
	agent.vely = 0;
	agent.steer_x = 0;
	agent.steer_y = 0;
	agent.height = 0;
	agent.exit_no = 0;
	agent.speed = 0;
	agent.lod = 0;
	agent.animate = 0;
	agent.animate_dir = 0;
	agent.estado = 0;
	agent.tick = 0;
	agent.estado_movimiento = 0;
	agent.go_to_x = 0;
	agent.go_to_y = 0;
	agent.checkpoint = 0;
	agent.chair_no = 0;
	agent.box_no = 0;
	agent.doctor_no = 0;
	agent.specialist_no = 0;
	agent.bed_no = 0;
	agent.priority = 0;
	agent.vaccine = 0;
	}

	//FLAME function call
	int dead = !receive_bed_state(&agent, bed_state_messages, rand48);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_bed_state Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
	}
}

/**
 *
 */
__global__ void GPUFLAME_receive_bed_response(xmachine_memory_agent_list* agents, xmachine_message_bed_response_list* bed_response_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_bed_response Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];
	} else {
	
	agent.id = 0;
	agent.x = 0;
	agent.y = 0;
	agent.velx = 0;
	agent.vely = 0;
	agent.steer_x = 0;
	agent.steer_y = 0;
	agent.height = 0;
	agent.exit_no = 0;
	agent.speed = 0;
	agent.lod = 0;
	agent.animate = 0;
	agent.animate_dir = 0;
	agent.estado = 0;
	agent.tick = 0;
	agent.estado_movimiento = 0;
	agent.go_to_x = 0;
	agent.go_to_y = 0;
	agent.checkpoint = 0;
	agent.chair_no = 0;
	agent.box_no = 0;
	agent.doctor_no = 0;
	agent.specialist_no = 0;
	agent.bed_no = 0;
	agent.priority = 0;
	agent.vaccine = 0;
	}

	//FLAME function call
	int dead = !receive_bed_response(&agent, bed_response_messages);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_bed_response Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
	}
}

/**
 *
 */
__global__ void GPUFLAME_output_bed_contact(xmachine_memory_agent_list* agents, xmachine_message_bed_contact_list* bed_contact_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_bed_contact Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !output_bed_contact(&agent, bed_contact_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_bed_contact Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_output_triage_petition(xmachine_memory_agent_list* agents, xmachine_message_triage_petition_list* triage_petition_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_count)
        return;
    

	//SoA to AoS - xmachine_memory_output_triage_petition Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];

	//FLAME function call
	int dead = !output_triage_petition(&agent, triage_petition_messages	);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_triage_petition Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
}

/**
 *
 */
__global__ void GPUFLAME_receive_triage_response(xmachine_memory_agent_list* agents, xmachine_message_triage_response_list* triage_response_messages, xmachine_message_free_chair_list* free_chair_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_triage_response Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    
	agent.id = agents->id[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.velx = agents->velx[index];
	agent.vely = agents->vely[index];
	agent.steer_x = agents->steer_x[index];
	agent.steer_y = agents->steer_y[index];
	agent.height = agents->height[index];
	agent.exit_no = agents->exit_no[index];
	agent.speed = agents->speed[index];
	agent.lod = agents->lod[index];
	agent.animate = agents->animate[index];
	agent.animate_dir = agents->animate_dir[index];
	agent.estado = agents->estado[index];
	agent.tick = agents->tick[index];
	agent.estado_movimiento = agents->estado_movimiento[index];
	agent.go_to_x = agents->go_to_x[index];
	agent.go_to_y = agents->go_to_y[index];
	agent.checkpoint = agents->checkpoint[index];
	agent.chair_no = agents->chair_no[index];
	agent.box_no = agents->box_no[index];
	agent.doctor_no = agents->doctor_no[index];
	agent.specialist_no = agents->specialist_no[index];
	agent.bed_no = agents->bed_no[index];
	agent.priority = agents->priority[index];
	agent.vaccine = agents->vaccine[index];
	} else {
	
	agent.id = 0;
	agent.x = 0;
	agent.y = 0;
	agent.velx = 0;
	agent.vely = 0;
	agent.steer_x = 0;
	agent.steer_y = 0;
	agent.height = 0;
	agent.exit_no = 0;
	agent.speed = 0;
	agent.lod = 0;
	agent.animate = 0;
	agent.animate_dir = 0;
	agent.estado = 0;
	agent.tick = 0;
	agent.estado_movimiento = 0;
	agent.go_to_x = 0;
	agent.go_to_y = 0;
	agent.checkpoint = 0;
	agent.chair_no = 0;
	agent.box_no = 0;
	agent.doctor_no = 0;
	agent.specialist_no = 0;
	agent.bed_no = 0;
	agent.priority = 0;
	agent.vaccine = 0;
	}

	//FLAME function call
	int dead = !receive_triage_response(&agent, triage_response_messages, free_chair_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_agent_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_triage_response Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->velx[index] = agent.velx;
	agents->vely[index] = agent.vely;
	agents->steer_x[index] = agent.steer_x;
	agents->steer_y[index] = agent.steer_y;
	agents->height[index] = agent.height;
	agents->exit_no[index] = agent.exit_no;
	agents->speed[index] = agent.speed;
	agents->lod[index] = agent.lod;
	agents->animate[index] = agent.animate;
	agents->animate_dir[index] = agent.animate_dir;
	agents->estado[index] = agent.estado;
	agents->tick[index] = agent.tick;
	agents->estado_movimiento[index] = agent.estado_movimiento;
	agents->go_to_x[index] = agent.go_to_x;
	agents->go_to_y[index] = agent.go_to_y;
	agents->checkpoint[index] = agent.checkpoint;
	agents->chair_no[index] = agent.chair_no;
	agents->box_no[index] = agent.box_no;
	agents->doctor_no[index] = agent.doctor_no;
	agents->specialist_no[index] = agent.specialist_no;
	agents->bed_no[index] = agent.bed_no;
	agents->priority[index] = agent.priority;
	agents->vaccine[index] = agent.vaccine;
	}
}

/**
 *
 */
__global__ void GPUFLAME_output_navmap_cells(xmachine_memory_navmap_list* agents, xmachine_message_navmap_cell_list* navmap_cell_messages){
	
	
	//discrete agent: index is position in 2D agent grid
	int width = (blockDim.x * gridDim.x);
	glm::ivec2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//SoA to AoS - xmachine_memory_output_navmap_cells Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_navmap agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.exit_no = agents->exit_no[index];
	agent.height = agents->height[index];
	agent.collision_x = agents->collision_x[index];
	agent.collision_y = agents->collision_y[index];
	agent.exit0_x = agents->exit0_x[index];
	agent.exit0_y = agents->exit0_y[index];
	agent.exit1_x = agents->exit1_x[index];
	agent.exit1_y = agents->exit1_y[index];
	agent.exit2_x = agents->exit2_x[index];
	agent.exit2_y = agents->exit2_y[index];
	agent.exit3_x = agents->exit3_x[index];
	agent.exit3_y = agents->exit3_y[index];
	agent.exit4_x = agents->exit4_x[index];
	agent.exit4_y = agents->exit4_y[index];
	agent.exit5_x = agents->exit5_x[index];
	agent.exit5_y = agents->exit5_y[index];
	agent.exit6_x = agents->exit6_x[index];
	agent.exit6_y = agents->exit6_y[index];
	agent.cant_generados = agents->cant_generados[index];

	//FLAME function call
	output_navmap_cells(&agent, navmap_cell_messages	);
	

	

	//AoS to SoA - xmachine_memory_output_navmap_cells Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->exit_no[index] = agent.exit_no;
	agents->height[index] = agent.height;
	agents->collision_x[index] = agent.collision_x;
	agents->collision_y[index] = agent.collision_y;
	agents->exit0_x[index] = agent.exit0_x;
	agents->exit0_y[index] = agent.exit0_y;
	agents->exit1_x[index] = agent.exit1_x;
	agents->exit1_y[index] = agent.exit1_y;
	agents->exit2_x[index] = agent.exit2_x;
	agents->exit2_y[index] = agent.exit2_y;
	agents->exit3_x[index] = agent.exit3_x;
	agents->exit3_y[index] = agent.exit3_y;
	agents->exit4_x[index] = agent.exit4_x;
	agents->exit4_y[index] = agent.exit4_y;
	agents->exit5_x[index] = agent.exit5_x;
	agents->exit5_y[index] = agent.exit5_y;
	agents->exit6_x[index] = agent.exit6_x;
	agents->exit6_y[index] = agent.exit6_y;
	agents->cant_generados[index] = agent.cant_generados;
}

/**
 *
 */
__global__ void GPUFLAME_generate_pedestrians(xmachine_memory_navmap_list* agents, xmachine_memory_agent_list* agent_agents, RNG_rand48* rand48){
	
	
	//discrete agent: index is position in 2D agent grid
	int width = (blockDim.x * gridDim.x);
	glm::ivec2 global_position;
	global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
	global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = global_position.x + (global_position.y * width);
	

	//SoA to AoS - xmachine_memory_generate_pedestrians Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_navmap agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.exit_no = agents->exit_no[index];
	agent.height = agents->height[index];
	agent.collision_x = agents->collision_x[index];
	agent.collision_y = agents->collision_y[index];
	agent.exit0_x = agents->exit0_x[index];
	agent.exit0_y = agents->exit0_y[index];
	agent.exit1_x = agents->exit1_x[index];
	agent.exit1_y = agents->exit1_y[index];
	agent.exit2_x = agents->exit2_x[index];
	agent.exit2_y = agents->exit2_y[index];
	agent.exit3_x = agents->exit3_x[index];
	agent.exit3_y = agents->exit3_y[index];
	agent.exit4_x = agents->exit4_x[index];
	agent.exit4_y = agents->exit4_y[index];
	agent.exit5_x = agents->exit5_x[index];
	agent.exit5_y = agents->exit5_y[index];
	agent.exit6_x = agents->exit6_x[index];
	agent.exit6_y = agents->exit6_y[index];
	agent.cant_generados = agents->cant_generados[index];

	//FLAME function call
	generate_pedestrians(&agent, agent_agents, rand48);
	

	

	//AoS to SoA - xmachine_memory_generate_pedestrians Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->exit_no[index] = agent.exit_no;
	agents->height[index] = agent.height;
	agents->collision_x[index] = agent.collision_x;
	agents->collision_y[index] = agent.collision_y;
	agents->exit0_x[index] = agent.exit0_x;
	agents->exit0_y[index] = agent.exit0_y;
	agents->exit1_x[index] = agent.exit1_x;
	agents->exit1_y[index] = agent.exit1_y;
	agents->exit2_x[index] = agent.exit2_x;
	agents->exit2_y[index] = agent.exit2_y;
	agents->exit3_x[index] = agent.exit3_x;
	agents->exit3_y[index] = agent.exit3_y;
	agents->exit4_x[index] = agent.exit4_x;
	agents->exit4_y[index] = agent.exit4_y;
	agents->exit5_x[index] = agent.exit5_x;
	agents->exit5_y[index] = agent.exit5_y;
	agents->exit6_x[index] = agent.exit6_x;
	agents->exit6_y[index] = agent.exit6_y;
	agents->cant_generados[index] = agent.cant_generados;
}

/**
 *
 */
__global__ void GPUFLAME_output_chair_state(xmachine_memory_chair_list* agents, xmachine_message_chair_contact_list* chair_contact_messages, xmachine_message_chair_state_list* chair_state_messages, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_output_chair_state Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_chair agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_chair_count){
    
	agent.id = agents->id[index];
	agent.tick = agents->tick[index];
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.state = agents->state[index];
	} else {
	
	agent.id = 0;
	agent.tick = 0;
	agent.x = 0;
	agent.y = 0;
	agent.state = 0;
	}

	//FLAME function call
	int dead = !output_chair_state(&agent, chair_contact_messages, chair_state_messages	, rand48);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_chair_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_chair_state Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->tick[index] = agent.tick;
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->state[index] = agent.state;
	}
}

/**
 *
 */
__global__ void GPUFLAME_output_bed_state(xmachine_memory_bed_list* agents, xmachine_message_bed_contact_list* bed_contact_messages, xmachine_message_bed_state_list* bed_state_messages, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_output_bed_state Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_bed agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_bed_count){
    
	agent.id = agents->id[index];
	agent.tick = agents->tick[index];
	agent.state = agents->state[index];
	} else {
	
	agent.id = 0;
	agent.tick = 0;
	agent.state = 0;
	}

	//FLAME function call
	int dead = !output_bed_state(&agent, bed_contact_messages, bed_state_messages	, rand48);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_bed_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_output_bed_state Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->tick[index] = agent.tick;
	agents->state[index] = agent.state;
	}
}

/**
 *
 */
__global__ void GPUFLAME_receive_doctor_petitions(xmachine_memory_doctor_manager_list* agents, xmachine_message_doctor_petition_list* doctor_petition_messages, xmachine_message_doctor_response_list* doctor_response_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_doctor_petitions Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_doctor_manager agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_doctor_manager_count){
    
	agent.tick = agents->tick[index];
	agent.rear = agents->rear[index];
	agent.size = agents->size[index];
    agent.doctors_occupied = &(agents->doctors_occupied[index]);
	agent.free_doctors = agents->free_doctors[index];
    agent.patientQueue = &(agents->patientQueue[index]);
	} else {
	
	agent.tick = 0;
	agent.rear = 0;
	agent.size = 0;
    agent.doctors_occupied = nullptr;
	agent.free_doctors = 4;
    agent.patientQueue = nullptr;
	}

	//FLAME function call
	int dead = !receive_doctor_petitions(&agent, doctor_petition_messages, doctor_response_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_doctor_manager_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_doctor_petitions Coalesced memory write (ignore arrays)
	agents->tick[index] = agent.tick;
	agents->rear[index] = agent.rear;
	agents->size[index] = agent.size;
	agents->free_doctors[index] = agent.free_doctors;
	}
}

/**
 *
 */
__global__ void GPUFLAME_receive_free_doctors(xmachine_memory_doctor_manager_list* agents, xmachine_message_free_doctor_list* free_doctor_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_free_doctors Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_doctor_manager agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_doctor_manager_count){
    
	agent.tick = agents->tick[index];
	agent.rear = agents->rear[index];
	agent.size = agents->size[index];
    agent.doctors_occupied = &(agents->doctors_occupied[index]);
	agent.free_doctors = agents->free_doctors[index];
    agent.patientQueue = &(agents->patientQueue[index]);
	} else {
	
	agent.tick = 0;
	agent.rear = 0;
	agent.size = 0;
    agent.doctors_occupied = nullptr;
	agent.free_doctors = 4;
    agent.patientQueue = nullptr;
	}

	//FLAME function call
	int dead = !receive_free_doctors(&agent, free_doctor_messages);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_doctor_manager_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_free_doctors Coalesced memory write (ignore arrays)
	agents->tick[index] = agent.tick;
	agents->rear[index] = agent.rear;
	agents->size[index] = agent.size;
	agents->free_doctors[index] = agent.free_doctors;
	}
}

/**
 *
 */
__global__ void GPUFLAME_receive_specialist_petitions(xmachine_memory_specialist_manager_list* agents, xmachine_message_specialist_petition_list* specialist_petition_messages, xmachine_message_specialist_response_list* specialist_response_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_specialist_petitions Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_specialist_manager agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_specialist_manager_count){
    
	agent.id = agents->id[index];
    agent.tick = &(agents->tick[index]);
    agent.free_specialist = &(agents->free_specialist[index]);
    agent.rear = &(agents->rear[index]);
    agent.size = &(agents->size[index]);
    agent.surgicalQueue = &(agents->surgicalQueue[index]);
    agent.pediatricsQueue = &(agents->pediatricsQueue[index]);
    agent.gynecologistQueue = &(agents->gynecologistQueue[index]);
    agent.geriatricsQueue = &(agents->geriatricsQueue[index]);
    agent.psychiatristQueue = &(agents->psychiatristQueue[index]);
	} else {
	
	agent.id = 0;
    agent.tick = nullptr;
    agent.free_specialist = nullptr;
    agent.rear = nullptr;
    agent.size = nullptr;
    agent.surgicalQueue = nullptr;
    agent.pediatricsQueue = nullptr;
    agent.gynecologistQueue = nullptr;
    agent.geriatricsQueue = nullptr;
    agent.psychiatristQueue = nullptr;
	}

	//FLAME function call
	int dead = !receive_specialist_petitions(&agent, specialist_petition_messages, specialist_response_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_specialist_manager_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_specialist_petitions Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	}
}

/**
 *
 */
__global__ void GPUFLAME_receive_free_specialist(xmachine_memory_specialist_manager_list* agents, xmachine_message_free_specialist_list* free_specialist_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_free_specialist Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_specialist_manager agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_specialist_manager_count){
    
	agent.id = agents->id[index];
    agent.tick = &(agents->tick[index]);
    agent.free_specialist = &(agents->free_specialist[index]);
    agent.rear = &(agents->rear[index]);
    agent.size = &(agents->size[index]);
    agent.surgicalQueue = &(agents->surgicalQueue[index]);
    agent.pediatricsQueue = &(agents->pediatricsQueue[index]);
    agent.gynecologistQueue = &(agents->gynecologistQueue[index]);
    agent.geriatricsQueue = &(agents->geriatricsQueue[index]);
    agent.psychiatristQueue = &(agents->psychiatristQueue[index]);
	} else {
	
	agent.id = 0;
    agent.tick = nullptr;
    agent.free_specialist = nullptr;
    agent.rear = nullptr;
    agent.size = nullptr;
    agent.surgicalQueue = nullptr;
    agent.pediatricsQueue = nullptr;
    agent.gynecologistQueue = nullptr;
    agent.geriatricsQueue = nullptr;
    agent.psychiatristQueue = nullptr;
	}

	//FLAME function call
	int dead = !receive_free_specialist(&agent, free_specialist_messages);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_specialist_manager_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_free_specialist Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	}
}

/**
 *
 */
__global__ void GPUFLAME_receive_specialist_reached(xmachine_memory_specialist_list* agents, xmachine_message_specialist_reached_list* specialist_reached_messages, xmachine_message_specialist_terminated_list* specialist_terminated_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_specialist_reached Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_specialist agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_specialist_count){
    
	agent.id = agents->id[index];
	agent.current_patient = agents->current_patient[index];
	agent.tick = agents->tick[index];
	} else {
	
	agent.id = 0;
	agent.current_patient = 0;
	agent.tick = 0;
	}

	//FLAME function call
	int dead = !receive_specialist_reached(&agent, specialist_reached_messages, specialist_terminated_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_specialist_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_specialist_reached Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->current_patient[index] = agent.current_patient;
	agents->tick[index] = agent.tick;
	}
}

/**
 *
 */
__global__ void GPUFLAME_reception_server(xmachine_memory_receptionist_list* agents, xmachine_message_check_in_list* check_in_messages, xmachine_message_check_in_response_list* check_in_response_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_reception_server Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_receptionist agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_receptionist_count){
    
    agent.patientQueue = &(agents->patientQueue[index]);
	agent.front = agents->front[index];
	agent.rear = agents->rear[index];
	agent.size = agents->size[index];
	agent.tick = agents->tick[index];
	agent.current_patient = agents->current_patient[index];
	agent.attend_patient = agents->attend_patient[index];
	} else {
	
    agent.patientQueue = nullptr;
	agent.front = 0;
	agent.rear = 0;
	agent.size = 0;
	agent.tick = 0;
	agent.current_patient = -1;
	agent.attend_patient = 0;
	}

	//FLAME function call
	int dead = !reception_server(&agent, check_in_messages, check_in_response_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_receptionist_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_reception_server Coalesced memory write (ignore arrays)
	agents->front[index] = agent.front;
	agents->rear[index] = agent.rear;
	agents->size[index] = agent.size;
	agents->tick[index] = agent.tick;
	agents->current_patient[index] = agent.current_patient;
	agents->attend_patient[index] = agent.attend_patient;
	}
}

/**
 *
 */
__global__ void GPUFLAME_generate_chairs(xmachine_memory_agent_generator_list* agents, xmachine_memory_chair_list* chair_agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_generator_count)
        return;
    

	//SoA to AoS - xmachine_memory_generate_chairs Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent_generator agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.chairs_generated = agents->chairs_generated[index];
	agent.beds_generated = agents->beds_generated[index];
	agent.boxes_generated = agents->boxes_generated[index];
	agent.doctors_generated = agents->doctors_generated[index];
	agent.specialists_generated = agents->specialists_generated[index];
	agent.personal_generated = agents->personal_generated[index];

	//FLAME function call
	int dead = !generate_chairs(&agent, chair_agents);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_generate_chairs Coalesced memory write (ignore arrays)
	agents->chairs_generated[index] = agent.chairs_generated;
	agents->beds_generated[index] = agent.beds_generated;
	agents->boxes_generated[index] = agent.boxes_generated;
	agents->doctors_generated[index] = agent.doctors_generated;
	agents->specialists_generated[index] = agent.specialists_generated;
	agents->personal_generated[index] = agent.personal_generated;
}

/**
 *
 */
__global__ void GPUFLAME_generate_beds(xmachine_memory_agent_generator_list* agents, xmachine_memory_bed_list* bed_agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_generator_count)
        return;
    

	//SoA to AoS - xmachine_memory_generate_beds Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent_generator agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.chairs_generated = agents->chairs_generated[index];
	agent.beds_generated = agents->beds_generated[index];
	agent.boxes_generated = agents->boxes_generated[index];
	agent.doctors_generated = agents->doctors_generated[index];
	agent.specialists_generated = agents->specialists_generated[index];
	agent.personal_generated = agents->personal_generated[index];

	//FLAME function call
	int dead = !generate_beds(&agent, bed_agents);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_generate_beds Coalesced memory write (ignore arrays)
	agents->chairs_generated[index] = agent.chairs_generated;
	agents->beds_generated[index] = agent.beds_generated;
	agents->boxes_generated[index] = agent.boxes_generated;
	agents->doctors_generated[index] = agent.doctors_generated;
	agents->specialists_generated[index] = agent.specialists_generated;
	agents->personal_generated[index] = agent.personal_generated;
}

/**
 *
 */
__global__ void GPUFLAME_generate_boxes(xmachine_memory_agent_generator_list* agents, xmachine_memory_box_list* box_agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_generator_count)
        return;
    

	//SoA to AoS - xmachine_memory_generate_boxes Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent_generator agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.chairs_generated = agents->chairs_generated[index];
	agent.beds_generated = agents->beds_generated[index];
	agent.boxes_generated = agents->boxes_generated[index];
	agent.doctors_generated = agents->doctors_generated[index];
	agent.specialists_generated = agents->specialists_generated[index];
	agent.personal_generated = agents->personal_generated[index];

	//FLAME function call
	int dead = !generate_boxes(&agent, box_agents);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_generate_boxes Coalesced memory write (ignore arrays)
	agents->chairs_generated[index] = agent.chairs_generated;
	agents->beds_generated[index] = agent.beds_generated;
	agents->boxes_generated[index] = agent.boxes_generated;
	agents->doctors_generated[index] = agent.doctors_generated;
	agents->specialists_generated[index] = agent.specialists_generated;
	agents->personal_generated[index] = agent.personal_generated;
}

/**
 *
 */
__global__ void GPUFLAME_generate_doctors(xmachine_memory_agent_generator_list* agents, xmachine_memory_doctor_list* doctor_agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_generator_count)
        return;
    

	//SoA to AoS - xmachine_memory_generate_doctors Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent_generator agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.chairs_generated = agents->chairs_generated[index];
	agent.beds_generated = agents->beds_generated[index];
	agent.boxes_generated = agents->boxes_generated[index];
	agent.doctors_generated = agents->doctors_generated[index];
	agent.specialists_generated = agents->specialists_generated[index];
	agent.personal_generated = agents->personal_generated[index];

	//FLAME function call
	int dead = !generate_doctors(&agent, doctor_agents);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_generate_doctors Coalesced memory write (ignore arrays)
	agents->chairs_generated[index] = agent.chairs_generated;
	agents->beds_generated[index] = agent.beds_generated;
	agents->boxes_generated[index] = agent.boxes_generated;
	agents->doctors_generated[index] = agent.doctors_generated;
	agents->specialists_generated[index] = agent.specialists_generated;
	agents->personal_generated[index] = agent.personal_generated;
}

/**
 *
 */
__global__ void GPUFLAME_generate_specialists(xmachine_memory_agent_generator_list* agents, xmachine_memory_specialist_list* specialist_agents){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_generator_count)
        return;
    

	//SoA to AoS - xmachine_memory_generate_specialists Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent_generator agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.chairs_generated = agents->chairs_generated[index];
	agent.beds_generated = agents->beds_generated[index];
	agent.boxes_generated = agents->boxes_generated[index];
	agent.doctors_generated = agents->doctors_generated[index];
	agent.specialists_generated = agents->specialists_generated[index];
	agent.personal_generated = agents->personal_generated[index];

	//FLAME function call
	int dead = !generate_specialists(&agent, specialist_agents);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_generate_specialists Coalesced memory write (ignore arrays)
	agents->chairs_generated[index] = agent.chairs_generated;
	agents->beds_generated[index] = agent.beds_generated;
	agents->boxes_generated[index] = agent.boxes_generated;
	agents->doctors_generated[index] = agent.doctors_generated;
	agents->specialists_generated[index] = agent.specialists_generated;
	agents->personal_generated[index] = agent.personal_generated;
}

/**
 *
 */
__global__ void GPUFLAME_generate_personal(xmachine_memory_agent_generator_list* agents, xmachine_memory_agent_list* agent_agents, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_agent_generator_count)
        return;
    

	//SoA to AoS - xmachine_memory_generate_personal Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_agent_generator agent;
    
    // Thread bounds already checked, but the agent function will still execute. load default values?
	
	agent.chairs_generated = agents->chairs_generated[index];
	agent.beds_generated = agents->beds_generated[index];
	agent.boxes_generated = agents->boxes_generated[index];
	agent.doctors_generated = agents->doctors_generated[index];
	agent.specialists_generated = agents->specialists_generated[index];
	agent.personal_generated = agents->personal_generated[index];

	//FLAME function call
	int dead = !generate_personal(&agent, agent_agents, rand48);
	

	//continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_generate_personal Coalesced memory write (ignore arrays)
	agents->chairs_generated[index] = agent.chairs_generated;
	agents->beds_generated[index] = agent.beds_generated;
	agents->boxes_generated[index] = agent.boxes_generated;
	agents->doctors_generated[index] = agent.doctors_generated;
	agents->specialists_generated[index] = agent.specialists_generated;
	agents->personal_generated[index] = agent.personal_generated;
}

/**
 *
 */
__global__ void GPUFLAME_attend_chair_petitions(xmachine_memory_chair_admin_list* agents, xmachine_message_chair_petition_list* chair_petition_messages, xmachine_message_chair_response_list* chair_response_messages, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_attend_chair_petitions Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_chair_admin agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_chair_admin_count){
    
	agent.id = agents->id[index];
    agent.chairArray = &(agents->chairArray[index]);
	} else {
	
	agent.id = 0;
    agent.chairArray = nullptr;
	}

	//FLAME function call
	int dead = !attend_chair_petitions(&agent, chair_petition_messages, chair_response_messages	, rand48);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_chair_admin_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_attend_chair_petitions Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	}
}

/**
 *
 */
__global__ void GPUFLAME_receive_free_chair(xmachine_memory_chair_admin_list* agents, xmachine_message_free_chair_list* free_chair_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_free_chair Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_chair_admin agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_chair_admin_count){
    
	agent.id = agents->id[index];
    agent.chairArray = &(agents->chairArray[index]);
	} else {
	
	agent.id = 0;
    agent.chairArray = nullptr;
	}

	//FLAME function call
	int dead = !receive_free_chair(&agent, free_chair_messages);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_chair_admin_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_free_chair Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	}
}

/**
 *
 */
__global__ void GPUFLAME_attend_bed_petitions(xmachine_memory_uci_list* agents, xmachine_message_bed_petition_list* bed_petition_messages, xmachine_message_bed_response_list* bed_response_messages, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_attend_bed_petitions Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_uci agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_uci_count){
    
	agent.tick = agents->tick[index];
    agent.bedArray = &(agents->bedArray[index]);
	} else {
	
	agent.tick = 0;
    agent.bedArray = nullptr;
	}

	//FLAME function call
	int dead = !attend_bed_petitions(&agent, bed_petition_messages, bed_response_messages	, rand48);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_uci_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_attend_bed_petitions Coalesced memory write (ignore arrays)
	agents->tick[index] = agent.tick;
	}
}

/**
 *
 */
__global__ void GPUFLAME_box_server(xmachine_memory_box_list* agents, xmachine_message_box_petition_list* box_petition_messages, xmachine_message_box_response_list* box_response_messages, RNG_rand48* rand48){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_box_server Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_box agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_box_count){
    
	agent.id = agents->id[index];
	agent.current_patient = agents->current_patient[index];
	agent.tick = agents->tick[index];
	} else {
	
	agent.id = 0;
	agent.current_patient = 0;
	agent.tick = 0;
	}

	//FLAME function call
	int dead = !box_server(&agent, box_petition_messages, box_response_messages	, rand48);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_box_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_box_server Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->current_patient[index] = agent.current_patient;
	agents->tick[index] = agent.tick;
	}
}

/**
 *
 */
__global__ void GPUFLAME_doctor_server(xmachine_memory_doctor_list* agents, xmachine_message_doctor_reached_list* doctor_reached_messages, xmachine_message_attention_terminated_list* attention_terminated_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_doctor_server Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_doctor agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_doctor_count){
    
	agent.id = agents->id[index];
	agent.current_patient = agents->current_patient[index];
	agent.tick = agents->tick[index];
	} else {
	
	agent.id = 0;
	agent.current_patient = 0;
	agent.tick = 0;
	}

	//FLAME function call
	int dead = !doctor_server(&agent, doctor_reached_messages, attention_terminated_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_doctor_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_doctor_server Coalesced memory write (ignore arrays)
	agents->id[index] = agent.id;
	agents->current_patient[index] = agent.current_patient;
	agents->tick[index] = agent.tick;
	}
}

/**
 *
 */
__global__ void GPUFLAME_receive_triage_petitions(xmachine_memory_triage_list* agents, xmachine_message_triage_petition_list* triage_petition_messages, xmachine_message_triage_response_list* triage_response_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_triage_petitions Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_triage agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_triage_count){
    
	agent.front = agents->front[index];
	agent.rear = agents->rear[index];
	agent.size = agents->size[index];
	agent.tick = agents->tick[index];
    agent.free_boxes = &(agents->free_boxes[index]);
    agent.patientQueue = &(agents->patientQueue[index]);
	} else {
	
	agent.front = 0;
	agent.rear = 0;
	agent.size = 0;
	agent.tick = 0;
    agent.free_boxes = nullptr;
    agent.patientQueue = nullptr;
	}

	//FLAME function call
	int dead = !receive_triage_petitions(&agent, triage_petition_messages, triage_response_messages	);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_triage_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_triage_petitions Coalesced memory write (ignore arrays)
	agents->front[index] = agent.front;
	agents->rear[index] = agent.rear;
	agents->size[index] = agent.size;
	agents->tick[index] = agent.tick;
	}
}

/**
 *
 */
__global__ void GPUFLAME_receive_free_box(xmachine_memory_triage_list* agents, xmachine_message_free_box_list* free_box_messages){
	
	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  
    
    //No partitioned input requires threads to be launched beyond the agent count to ensure full block sizes
    

	//SoA to AoS - xmachine_memory_receive_free_box Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_triage agent;
    //No partitioned input may launch more threads than required - only load agent data within bounds. 
    if (index < d_xmachine_memory_triage_count){
    
	agent.front = agents->front[index];
	agent.rear = agents->rear[index];
	agent.size = agents->size[index];
	agent.tick = agents->tick[index];
    agent.free_boxes = &(agents->free_boxes[index]);
    agent.patientQueue = &(agents->patientQueue[index]);
	} else {
	
	agent.front = 0;
	agent.rear = 0;
	agent.size = 0;
	agent.tick = 0;
    agent.free_boxes = nullptr;
    agent.patientQueue = nullptr;
	}

	//FLAME function call
	int dead = !receive_free_box(&agent, free_box_messages);
	

	
    //No partitioned input may launch more threads than required - only write agent data within bounds. 
    if (index < d_xmachine_memory_triage_count){
    //continuous agent: set reallocation flag
	agents->_scan_input[index]  = dead; 

	//AoS to SoA - xmachine_memory_receive_free_box Coalesced memory write (ignore arrays)
	agents->front[index] = agent.front;
	agents->rear[index] = agent.rear;
	agents->size[index] = agent.size;
	agents->tick[index] = agent.tick;
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
