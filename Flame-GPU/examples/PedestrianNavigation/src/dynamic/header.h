
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



#ifndef __HEADER
#define __HEADER

#if defined __NVCC__
   // Disable annotation on defaulted function warnings (glm 0.9.9 and CUDA 9.0 introduced this warning)
   #pragma diag_suppress esa_on_defaulted_function_ignored 
#endif

#define GLM_FORCE_NO_CTOR_INIT
#include <glm/glm.hpp>

/* General standard definitions */
//Threads per block (agents per block)
#define THREADS_PER_TILE 64
//Definition for any agent function or helper function
#define __FLAME_GPU_FUNC__ __device__
//Definition for a function used to initialise environment variables
#define __FLAME_GPU_INIT_FUNC__
#define __FLAME_GPU_STEP_FUNC__
#define __FLAME_GPU_EXIT_FUNC__
#define __FLAME_GPU_HOST_FUNC__ __host__

#define USE_CUDA_STREAMS
#define FAST_ATOMIC_SORTING

// FLAME GPU Version Macros.
#define FLAME_GPU_MAJOR_VERSION 1
#define FLAME_GPU_MINOR_VERSION 5
#define FLAME_GPU_PATCH_VERSION 0

typedef unsigned int uint;

//FLAME GPU vector types float, (i)nteger, (u)nsigned integer, (d)ouble
typedef glm::vec2 fvec2;
typedef glm::vec3 fvec3;
typedef glm::vec4 fvec4;
typedef glm::ivec2 ivec2;
typedef glm::ivec3 ivec3;
typedef glm::ivec4 ivec4;
typedef glm::uvec2 uvec2;
typedef glm::uvec3 uvec3;
typedef glm::uvec4 uvec4;
typedef glm::dvec2 dvec2;
typedef glm::dvec3 dvec3;
typedef glm::dvec4 dvec4;

	

/* Agent population size definitions must be a multiple of THREADS_PER_TILE (default 64) */
//Maximum buffer size (largest agent buffer size)
#define buffer_size_MAX 65536

//Maximum population size of xmachine_memory_agent
#define xmachine_memory_agent_MAX 65536

//Maximum population size of xmachine_memory_navmap
#define xmachine_memory_navmap_MAX 65536

//Maximum population size of xmachine_memory_chair
#define xmachine_memory_chair_MAX 65536

//Maximum population size of xmachine_memory_doctor_manager
#define xmachine_memory_doctor_manager_MAX 65536

//Maximum population size of xmachine_memory_specialist_manager
#define xmachine_memory_specialist_manager_MAX 65536

//Maximum population size of xmachine_memory_specialist
#define xmachine_memory_specialist_MAX 65536

//Maximum population size of xmachine_memory_receptionist
#define xmachine_memory_receptionist_MAX 65536

//Maximum population size of xmachine_memory_agent_generator
#define xmachine_memory_agent_generator_MAX 65536

//Maximum population size of xmachine_memory_chair_admin
#define xmachine_memory_chair_admin_MAX 16

//Maximum population size of xmachine_memory_box
#define xmachine_memory_box_MAX 65536

//Maximum population size of xmachine_memory_doctor
#define xmachine_memory_doctor_MAX 65536

//Maximum population size of xmachine_memory_triage
#define xmachine_memory_triage_MAX 65536 
//Agent variable array length for xmachine_memory_doctor_manager->doctors_occupied
#define xmachine_memory_doctor_manager_doctors_occupied_LENGTH 4 
//Agent variable array length for xmachine_memory_doctor_manager->patientQueue
#define xmachine_memory_doctor_manager_patientQueue_LENGTH 35 
//Agent variable array length for xmachine_memory_specialist_manager->tick
#define xmachine_memory_specialist_manager_tick_LENGTH 5 
//Agent variable array length for xmachine_memory_specialist_manager->free_specialist
#define xmachine_memory_specialist_manager_free_specialist_LENGTH 5 
//Agent variable array length for xmachine_memory_specialist_manager->rear
#define xmachine_memory_specialist_manager_rear_LENGTH 5 
//Agent variable array length for xmachine_memory_specialist_manager->size
#define xmachine_memory_specialist_manager_size_LENGTH 5 
//Agent variable array length for xmachine_memory_specialist_manager->surgicalQueue
#define xmachine_memory_specialist_manager_surgicalQueue_LENGTH 35 
//Agent variable array length for xmachine_memory_specialist_manager->pediatricsQueue
#define xmachine_memory_specialist_manager_pediatricsQueue_LENGTH 35 
//Agent variable array length for xmachine_memory_specialist_manager->gynecologistQueue
#define xmachine_memory_specialist_manager_gynecologistQueue_LENGTH 35 
//Agent variable array length for xmachine_memory_specialist_manager->geriatricsQueue
#define xmachine_memory_specialist_manager_geriatricsQueue_LENGTH 35 
//Agent variable array length for xmachine_memory_specialist_manager->psychiatristQueue
#define xmachine_memory_specialist_manager_psychiatristQueue_LENGTH 35 
//Agent variable array length for xmachine_memory_receptionist->patientQueue
#define xmachine_memory_receptionist_patientQueue_LENGTH 100 
//Agent variable array length for xmachine_memory_chair_admin->chairArray
#define xmachine_memory_chair_admin_chairArray_LENGTH 35 
//Agent variable array length for xmachine_memory_triage->boxArray
#define xmachine_memory_triage_boxArray_LENGTH 3 
//Agent variable array length for xmachine_memory_triage->patientQueue
#define xmachine_memory_triage_patientQueue_LENGTH 100


  
  
/* Message population size definitions */
//Maximum population size of xmachine_mmessage_pedestrian_location
#define xmachine_message_pedestrian_location_MAX 65536

//Maximum population size of xmachine_mmessage_pedestrian_state
#define xmachine_message_pedestrian_state_MAX 65536

//Maximum population size of xmachine_mmessage_navmap_cell
#define xmachine_message_navmap_cell_MAX 65536

//Maximum population size of xmachine_mmessage_check_in
#define xmachine_message_check_in_MAX 65536

//Maximum population size of xmachine_mmessage_check_in_response
#define xmachine_message_check_in_response_MAX 65536

//Maximum population size of xmachine_mmessage_chair_petition
#define xmachine_message_chair_petition_MAX 65536

//Maximum population size of xmachine_mmessage_chair_response
#define xmachine_message_chair_response_MAX 65536

//Maximum population size of xmachine_mmessage_chair_state
#define xmachine_message_chair_state_MAX 65536

//Maximum population size of xmachine_mmessage_free_chair
#define xmachine_message_free_chair_MAX 65536

//Maximum population size of xmachine_mmessage_chair_contact
#define xmachine_message_chair_contact_MAX 65536

//Maximum population size of xmachine_mmessage_box_petition
#define xmachine_message_box_petition_MAX 65536

//Maximum population size of xmachine_mmessage_box_response
#define xmachine_message_box_response_MAX 65536

//Maximum population size of xmachine_mmessage_specialist_reached
#define xmachine_message_specialist_reached_MAX 65536

//Maximum population size of xmachine_mmessage_specialist_petition
#define xmachine_message_specialist_petition_MAX 65536

//Maximum population size of xmachine_mmessage_doctor_reached
#define xmachine_message_doctor_reached_MAX 65536

//Maximum population size of xmachine_mmessage_free_doctor
#define xmachine_message_free_doctor_MAX 65536

//Maximum population size of xmachine_mmessage_attention_terminated
#define xmachine_message_attention_terminated_MAX 65536

//Maximum population size of xmachine_mmessage_doctor_petition
#define xmachine_message_doctor_petition_MAX 65536

//Maximum population size of xmachine_mmessage_doctor_response
#define xmachine_message_doctor_response_MAX 65536

//Maximum population size of xmachine_mmessage_specialist_response
#define xmachine_message_specialist_response_MAX 65536

//Maximum population size of xmachine_mmessage_triage_petition
#define xmachine_message_triage_petition_MAX 65536

//Maximum population size of xmachine_mmessage_triage_response
#define xmachine_message_triage_response_MAX 65536


/* Define preprocessor symbols for each message to specify the type, to simplify / improve portability */

#define xmachine_message_pedestrian_location_partitioningSpatial
#define xmachine_message_pedestrian_state_partitioningSpatial
#define xmachine_message_navmap_cell_partitioningDiscrete
#define xmachine_message_check_in_partitioningNone
#define xmachine_message_check_in_response_partitioningNone
#define xmachine_message_chair_petition_partitioningNone
#define xmachine_message_chair_response_partitioningNone
#define xmachine_message_chair_state_partitioningNone
#define xmachine_message_free_chair_partitioningNone
#define xmachine_message_chair_contact_partitioningNone
#define xmachine_message_box_petition_partitioningNone
#define xmachine_message_box_response_partitioningNone
#define xmachine_message_specialist_reached_partitioningNone
#define xmachine_message_specialist_petition_partitioningNone
#define xmachine_message_doctor_reached_partitioningNone
#define xmachine_message_free_doctor_partitioningNone
#define xmachine_message_attention_terminated_partitioningNone
#define xmachine_message_doctor_petition_partitioningNone
#define xmachine_message_doctor_response_partitioningNone
#define xmachine_message_specialist_response_partitioningNone
#define xmachine_message_triage_petition_partitioningNone
#define xmachine_message_triage_response_partitioningNone

/* Spatial partitioning grid size definitions */
//xmachine_message_pedestrian_location partition grid size (gridDim.X*gridDim.Y*gridDim.Z)
#define xmachine_message_pedestrian_location_grid_size 6400
//xmachine_message_pedestrian_state partition grid size (gridDim.X*gridDim.Y*gridDim.Z)
#define xmachine_message_pedestrian_state_grid_size 16384

/* Static Graph size definitions*/
  

/* Default visualisation Colour indices */
 
#define FLAME_GPU_VISUALISATION_COLOUR_BLACK 0
#define FLAME_GPU_VISUALISATION_COLOUR_RED 1
#define FLAME_GPU_VISUALISATION_COLOUR_GREEN 2
#define FLAME_GPU_VISUALISATION_COLOUR_BLUE 3
#define FLAME_GPU_VISUALISATION_COLOUR_YELLOW 4
#define FLAME_GPU_VISUALISATION_COLOUR_CYAN 5
#define FLAME_GPU_VISUALISATION_COLOUR_MAGENTA 6
#define FLAME_GPU_VISUALISATION_COLOUR_WHITE 7
#define FLAME_GPU_VISUALISATION_COLOUR_BROWN 8

/* enum types */

/**
 * MESSAGE_OUTPUT used for all continuous messaging
 */
enum MESSAGE_OUTPUT{
	single_message,
	optional_message,
};

/**
 * AGENT_TYPE used for templates device message functions
 */
enum AGENT_TYPE{
	CONTINUOUS,
	DISCRETE_2D
};


/* Agent structures */

/** struct xmachine_memory_agent
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_agent
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    float velx;    /**< X-machine memory variable velx of type float.*/
    float vely;    /**< X-machine memory variable vely of type float.*/
    float steer_x;    /**< X-machine memory variable steer_x of type float.*/
    float steer_y;    /**< X-machine memory variable steer_y of type float.*/
    float height;    /**< X-machine memory variable height of type float.*/
    int exit_no;    /**< X-machine memory variable exit_no of type int.*/
    float speed;    /**< X-machine memory variable speed of type float.*/
    int lod;    /**< X-machine memory variable lod of type int.*/
    float animate;    /**< X-machine memory variable animate of type float.*/
    int animate_dir;    /**< X-machine memory variable animate_dir of type int.*/
    int estado;    /**< X-machine memory variable estado of type int.*/
    int tick;    /**< X-machine memory variable tick of type int.*/
    unsigned int estado_movimiento;    /**< X-machine memory variable estado_movimiento of type unsigned int.*/
    unsigned int go_to_x;    /**< X-machine memory variable go_to_x of type unsigned int.*/
    unsigned int go_to_y;    /**< X-machine memory variable go_to_y of type unsigned int.*/
    unsigned int checkpoint;    /**< X-machine memory variable checkpoint of type unsigned int.*/
    int chair_no;    /**< X-machine memory variable chair_no of type int.*/
    unsigned int box_no;    /**< X-machine memory variable box_no of type unsigned int.*/
    unsigned int doctor_no;    /**< X-machine memory variable doctor_no of type unsigned int.*/
    unsigned int specialist_no;    /**< X-machine memory variable specialist_no of type unsigned int.*/
    unsigned int priority;    /**< X-machine memory variable priority of type unsigned int.*/
};

/** struct xmachine_memory_navmap
 * discrete valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_navmap
{
    int x;    /**< X-machine memory variable x of type int.*/
    int y;    /**< X-machine memory variable y of type int.*/
    int exit_no;    /**< X-machine memory variable exit_no of type int.*/
    float height;    /**< X-machine memory variable height of type float.*/
    float collision_x;    /**< X-machine memory variable collision_x of type float.*/
    float collision_y;    /**< X-machine memory variable collision_y of type float.*/
    float exit0_x;    /**< X-machine memory variable exit0_x of type float.*/
    float exit0_y;    /**< X-machine memory variable exit0_y of type float.*/
    float exit1_x;    /**< X-machine memory variable exit1_x of type float.*/
    float exit1_y;    /**< X-machine memory variable exit1_y of type float.*/
    float exit2_x;    /**< X-machine memory variable exit2_x of type float.*/
    float exit2_y;    /**< X-machine memory variable exit2_y of type float.*/
    float exit3_x;    /**< X-machine memory variable exit3_x of type float.*/
    float exit3_y;    /**< X-machine memory variable exit3_y of type float.*/
    float exit4_x;    /**< X-machine memory variable exit4_x of type float.*/
    float exit4_y;    /**< X-machine memory variable exit4_y of type float.*/
    float exit5_x;    /**< X-machine memory variable exit5_x of type float.*/
    float exit5_y;    /**< X-machine memory variable exit5_y of type float.*/
    float exit6_x;    /**< X-machine memory variable exit6_x of type float.*/
    float exit6_y;    /**< X-machine memory variable exit6_y of type float.*/
    unsigned int cant_generados;    /**< X-machine memory variable cant_generados of type unsigned int.*/
};

/** struct xmachine_memory_chair
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_chair
{
    int id;    /**< X-machine memory variable id of type int.*/
    int x;    /**< X-machine memory variable x of type int.*/
    int y;    /**< X-machine memory variable y of type int.*/
    int state;    /**< X-machine memory variable state of type int.*/
};

/** struct xmachine_memory_doctor_manager
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_doctor_manager
{
    unsigned int tick;    /**< X-machine memory variable tick of type unsigned int.*/
    unsigned int rear;    /**< X-machine memory variable rear of type unsigned int.*/
    unsigned int size;    /**< X-machine memory variable size of type unsigned int.*/
    int *doctors_occupied;    /**< X-machine memory variable doctors_occupied of type int.*/
    unsigned int free_doctors;    /**< X-machine memory variable free_doctors of type unsigned int.*/
    ivec2 *patientQueue;    /**< X-machine memory variable patientQueue of type ivec2.*/
};

/** struct xmachine_memory_specialist_manager
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_specialist_manager
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    unsigned int *tick;    /**< X-machine memory variable tick of type unsigned int.*/
    unsigned int *free_specialist;    /**< X-machine memory variable free_specialist of type unsigned int.*/
    unsigned int *rear;    /**< X-machine memory variable rear of type unsigned int.*/
    unsigned int *size;    /**< X-machine memory variable size of type unsigned int.*/
    ivec2 *surgicalQueue;    /**< X-machine memory variable surgicalQueue of type ivec2.*/
    ivec2 *pediatricsQueue;    /**< X-machine memory variable pediatricsQueue of type ivec2.*/
    ivec2 *gynecologistQueue;    /**< X-machine memory variable gynecologistQueue of type ivec2.*/
    ivec2 *geriatricsQueue;    /**< X-machine memory variable geriatricsQueue of type ivec2.*/
    ivec2 *psychiatristQueue;    /**< X-machine memory variable psychiatristQueue of type ivec2.*/
};

/** struct xmachine_memory_specialist
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_specialist
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    unsigned int current_patient;    /**< X-machine memory variable current_patient of type unsigned int.*/
    unsigned int tick;    /**< X-machine memory variable tick of type unsigned int.*/
};

/** struct xmachine_memory_receptionist
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_receptionist
{
    int x;    /**< X-machine memory variable x of type int.*/
    int y;    /**< X-machine memory variable y of type int.*/
    unsigned int *patientQueue;    /**< X-machine memory variable patientQueue of type unsigned int.*/
    unsigned int front;    /**< X-machine memory variable front of type unsigned int.*/
    unsigned int rear;    /**< X-machine memory variable rear of type unsigned int.*/
    unsigned int size;    /**< X-machine memory variable size of type unsigned int.*/
    unsigned int tick;    /**< X-machine memory variable tick of type unsigned int.*/
    int current_patient;    /**< X-machine memory variable current_patient of type int.*/
    int attend_patient;    /**< X-machine memory variable attend_patient of type int.*/
    int estado;    /**< X-machine memory variable estado of type int.*/
};

/** struct xmachine_memory_agent_generator
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_agent_generator
{
    int chairs_generated;    /**< X-machine memory variable chairs_generated of type int.*/
    int boxes_generated;    /**< X-machine memory variable boxes_generated of type int.*/
    int doctors_generated;    /**< X-machine memory variable doctors_generated of type int.*/
    int specialists_generated;    /**< X-machine memory variable specialists_generated of type int.*/
};

/** struct xmachine_memory_chair_admin
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_chair_admin
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    unsigned int *chairArray;    /**< X-machine memory variable chairArray of type unsigned int.*/
};

/** struct xmachine_memory_box
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_box
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    unsigned int attending;    /**< X-machine memory variable attending of type unsigned int.*/
    unsigned int tick;    /**< X-machine memory variable tick of type unsigned int.*/
};

/** struct xmachine_memory_doctor
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_doctor
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    int current_patient;    /**< X-machine memory variable current_patient of type int.*/
    unsigned int tick;    /**< X-machine memory variable tick of type unsigned int.*/
};

/** struct xmachine_memory_triage
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_triage
{
    unsigned int front;    /**< X-machine memory variable front of type unsigned int.*/
    unsigned int rear;    /**< X-machine memory variable rear of type unsigned int.*/
    unsigned int size;    /**< X-machine memory variable size of type unsigned int.*/
    unsigned int tick;    /**< X-machine memory variable tick of type unsigned int.*/
    unsigned int *boxArray;    /**< X-machine memory variable boxArray of type unsigned int.*/
    unsigned int *patientQueue;    /**< X-machine memory variable patientQueue of type unsigned int.*/
};



/* Message structures */

/** struct xmachine_message_pedestrian_location
 * Spatial Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_pedestrian_location
{	
    /* Spatial Partitioning Variables */
    glm::ivec3 _relative_cell;    /**< Relative cell position from agent grid cell position range -1 to 1 */
    int _cell_index_max;    /**< Max boundary value of current cell */
    glm::ivec3 _agent_grid_cell;  /**< Agents partition cell position */
    int _cell_index;        /**< Index of position in current cell */  
      
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/  
    float z;        /**< Message variable z of type float.*/  
    int estado;        /**< Message variable estado of type int.*/
};

/** struct xmachine_message_pedestrian_state
 * Spatial Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_pedestrian_state
{	
    /* Spatial Partitioning Variables */
    glm::ivec3 _relative_cell;    /**< Relative cell position from agent grid cell position range -1 to 1 */
    int _cell_index_max;    /**< Max boundary value of current cell */
    glm::ivec3 _agent_grid_cell;  /**< Agents partition cell position */
    int _cell_index;        /**< Index of position in current cell */  
      
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/  
    float z;        /**< Message variable z of type float.*/  
    int estado;        /**< Message variable estado of type int.*/
};

/** struct xmachine_message_navmap_cell
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_navmap_cell
{	
    /* Discrete Partitioning Variables */
    glm::ivec2 _position;         /**< 2D position of message*/
    glm::ivec2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int x;        /**< Message variable x of type int.*/  
    int y;        /**< Message variable y of type int.*/  
    int exit_no;        /**< Message variable exit_no of type int.*/  
    float height;        /**< Message variable height of type float.*/  
    float collision_x;        /**< Message variable collision_x of type float.*/  
    float collision_y;        /**< Message variable collision_y of type float.*/
};

/** struct xmachine_message_check_in
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_check_in
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/
};

/** struct xmachine_message_check_in_response
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_check_in_response
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/
};

/** struct xmachine_message_chair_petition
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_chair_petition
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/
};

/** struct xmachine_message_chair_response
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_chair_response
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/  
    int chair_no;        /**< Message variable chair_no of type int.*/
};

/** struct xmachine_message_chair_state
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_chair_state
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/  
    int state;        /**< Message variable state of type int.*/
};

/** struct xmachine_message_free_chair
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_free_chair
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int chair_no;        /**< Message variable chair_no of type unsigned int.*/
};

/** struct xmachine_message_chair_contact
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_chair_contact
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/  
    unsigned int chair_no;        /**< Message variable chair_no of type unsigned int.*/  
    int state;        /**< Message variable state of type int.*/
};

/** struct xmachine_message_box_petition
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_box_petition
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/  
    unsigned int box_no;        /**< Message variable box_no of type unsigned int.*/
};

/** struct xmachine_message_box_response
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_box_response
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/  
    unsigned int doctor_no;        /**< Message variable doctor_no of type unsigned int.*/  
    unsigned int priority;        /**< Message variable priority of type unsigned int.*/
};

/** struct xmachine_message_specialist_reached
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_specialist_reached
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/  
    unsigned int specialist_no;        /**< Message variable specialist_no of type unsigned int.*/
};

/** struct xmachine_message_specialist_petition
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_specialist_petition
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/  
    unsigned int priority;        /**< Message variable priority of type unsigned int.*/  
    unsigned int specialist_no;        /**< Message variable specialist_no of type unsigned int.*/
};

/** struct xmachine_message_doctor_reached
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_doctor_reached
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/  
    unsigned int doctor_no;        /**< Message variable doctor_no of type unsigned int.*/
};

/** struct xmachine_message_free_doctor
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_free_doctor
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int doctor_no;        /**< Message variable doctor_no of type unsigned int.*/
};

/** struct xmachine_message_attention_terminated
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_attention_terminated
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/
};

/** struct xmachine_message_doctor_petition
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_doctor_petition
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/  
    unsigned int doctor_no;        /**< Message variable doctor_no of type unsigned int.*/  
    unsigned int priority;        /**< Message variable priority of type unsigned int.*/
};

/** struct xmachine_message_doctor_response
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_doctor_response
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/  
    int doctor_no;        /**< Message variable doctor_no of type int.*/
};

/** struct xmachine_message_specialist_response
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_specialist_response
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/  
    int specialist_ready;        /**< Message variable specialist_ready of type int.*/
};

/** struct xmachine_message_triage_petition
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_triage_petition
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/
};

/** struct xmachine_message_triage_response
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_triage_response
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/  
    int box_no;        /**< Message variable box_no of type int.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_agent_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_agent_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_agent_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_agent_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_agent_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
    float x [xmachine_memory_agent_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_agent_MAX];    /**< X-machine memory variable list y of type float.*/
    float velx [xmachine_memory_agent_MAX];    /**< X-machine memory variable list velx of type float.*/
    float vely [xmachine_memory_agent_MAX];    /**< X-machine memory variable list vely of type float.*/
    float steer_x [xmachine_memory_agent_MAX];    /**< X-machine memory variable list steer_x of type float.*/
    float steer_y [xmachine_memory_agent_MAX];    /**< X-machine memory variable list steer_y of type float.*/
    float height [xmachine_memory_agent_MAX];    /**< X-machine memory variable list height of type float.*/
    int exit_no [xmachine_memory_agent_MAX];    /**< X-machine memory variable list exit_no of type int.*/
    float speed [xmachine_memory_agent_MAX];    /**< X-machine memory variable list speed of type float.*/
    int lod [xmachine_memory_agent_MAX];    /**< X-machine memory variable list lod of type int.*/
    float animate [xmachine_memory_agent_MAX];    /**< X-machine memory variable list animate of type float.*/
    int animate_dir [xmachine_memory_agent_MAX];    /**< X-machine memory variable list animate_dir of type int.*/
    int estado [xmachine_memory_agent_MAX];    /**< X-machine memory variable list estado of type int.*/
    int tick [xmachine_memory_agent_MAX];    /**< X-machine memory variable list tick of type int.*/
    unsigned int estado_movimiento [xmachine_memory_agent_MAX];    /**< X-machine memory variable list estado_movimiento of type unsigned int.*/
    unsigned int go_to_x [xmachine_memory_agent_MAX];    /**< X-machine memory variable list go_to_x of type unsigned int.*/
    unsigned int go_to_y [xmachine_memory_agent_MAX];    /**< X-machine memory variable list go_to_y of type unsigned int.*/
    unsigned int checkpoint [xmachine_memory_agent_MAX];    /**< X-machine memory variable list checkpoint of type unsigned int.*/
    int chair_no [xmachine_memory_agent_MAX];    /**< X-machine memory variable list chair_no of type int.*/
    unsigned int box_no [xmachine_memory_agent_MAX];    /**< X-machine memory variable list box_no of type unsigned int.*/
    unsigned int doctor_no [xmachine_memory_agent_MAX];    /**< X-machine memory variable list doctor_no of type unsigned int.*/
    unsigned int specialist_no [xmachine_memory_agent_MAX];    /**< X-machine memory variable list specialist_no of type unsigned int.*/
    unsigned int priority [xmachine_memory_agent_MAX];    /**< X-machine memory variable list priority of type unsigned int.*/
};

/** struct xmachine_memory_navmap_list
 * discrete valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_navmap_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_navmap_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_navmap_MAX];  /**< Used during parallel prefix sum */
    
    int x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list x of type int.*/
    int y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list y of type int.*/
    int exit_no [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit_no of type int.*/
    float height [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list height of type float.*/
    float collision_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list collision_x of type float.*/
    float collision_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list collision_y of type float.*/
    float exit0_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit0_x of type float.*/
    float exit0_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit0_y of type float.*/
    float exit1_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit1_x of type float.*/
    float exit1_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit1_y of type float.*/
    float exit2_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit2_x of type float.*/
    float exit2_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit2_y of type float.*/
    float exit3_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit3_x of type float.*/
    float exit3_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit3_y of type float.*/
    float exit4_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit4_x of type float.*/
    float exit4_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit4_y of type float.*/
    float exit5_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit5_x of type float.*/
    float exit5_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit5_y of type float.*/
    float exit6_x [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit6_x of type float.*/
    float exit6_y [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list exit6_y of type float.*/
    unsigned int cant_generados [xmachine_memory_navmap_MAX];    /**< X-machine memory variable list cant_generados of type unsigned int.*/
};

/** struct xmachine_memory_chair_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_chair_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_chair_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_chair_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_memory_chair_MAX];    /**< X-machine memory variable list id of type int.*/
    int x [xmachine_memory_chair_MAX];    /**< X-machine memory variable list x of type int.*/
    int y [xmachine_memory_chair_MAX];    /**< X-machine memory variable list y of type int.*/
    int state [xmachine_memory_chair_MAX];    /**< X-machine memory variable list state of type int.*/
};

/** struct xmachine_memory_doctor_manager_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_doctor_manager_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_doctor_manager_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_doctor_manager_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int tick [xmachine_memory_doctor_manager_MAX];    /**< X-machine memory variable list tick of type unsigned int.*/
    unsigned int rear [xmachine_memory_doctor_manager_MAX];    /**< X-machine memory variable list rear of type unsigned int.*/
    unsigned int size [xmachine_memory_doctor_manager_MAX];    /**< X-machine memory variable list size of type unsigned int.*/
    int doctors_occupied [xmachine_memory_doctor_manager_MAX*4];    /**< X-machine memory variable list doctors_occupied of type int.*/
    unsigned int free_doctors [xmachine_memory_doctor_manager_MAX];    /**< X-machine memory variable list free_doctors of type unsigned int.*/
    ivec2 patientQueue [xmachine_memory_doctor_manager_MAX*35];    /**< X-machine memory variable list patientQueue of type ivec2.*/
};

/** struct xmachine_memory_specialist_manager_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_specialist_manager_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_specialist_manager_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_specialist_manager_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_specialist_manager_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
    unsigned int tick [xmachine_memory_specialist_manager_MAX*5];    /**< X-machine memory variable list tick of type unsigned int.*/
    unsigned int free_specialist [xmachine_memory_specialist_manager_MAX*5];    /**< X-machine memory variable list free_specialist of type unsigned int.*/
    unsigned int rear [xmachine_memory_specialist_manager_MAX*5];    /**< X-machine memory variable list rear of type unsigned int.*/
    unsigned int size [xmachine_memory_specialist_manager_MAX*5];    /**< X-machine memory variable list size of type unsigned int.*/
    ivec2 surgicalQueue [xmachine_memory_specialist_manager_MAX*35];    /**< X-machine memory variable list surgicalQueue of type ivec2.*/
    ivec2 pediatricsQueue [xmachine_memory_specialist_manager_MAX*35];    /**< X-machine memory variable list pediatricsQueue of type ivec2.*/
    ivec2 gynecologistQueue [xmachine_memory_specialist_manager_MAX*35];    /**< X-machine memory variable list gynecologistQueue of type ivec2.*/
    ivec2 geriatricsQueue [xmachine_memory_specialist_manager_MAX*35];    /**< X-machine memory variable list geriatricsQueue of type ivec2.*/
    ivec2 psychiatristQueue [xmachine_memory_specialist_manager_MAX*35];    /**< X-machine memory variable list psychiatristQueue of type ivec2.*/
};

/** struct xmachine_memory_specialist_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_specialist_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_specialist_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_specialist_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_specialist_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
    unsigned int current_patient [xmachine_memory_specialist_MAX];    /**< X-machine memory variable list current_patient of type unsigned int.*/
    unsigned int tick [xmachine_memory_specialist_MAX];    /**< X-machine memory variable list tick of type unsigned int.*/
};

/** struct xmachine_memory_receptionist_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_receptionist_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_receptionist_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_receptionist_MAX];  /**< Used during parallel prefix sum */
    
    int x [xmachine_memory_receptionist_MAX];    /**< X-machine memory variable list x of type int.*/
    int y [xmachine_memory_receptionist_MAX];    /**< X-machine memory variable list y of type int.*/
    unsigned int patientQueue [xmachine_memory_receptionist_MAX*100];    /**< X-machine memory variable list patientQueue of type unsigned int.*/
    unsigned int front [xmachine_memory_receptionist_MAX];    /**< X-machine memory variable list front of type unsigned int.*/
    unsigned int rear [xmachine_memory_receptionist_MAX];    /**< X-machine memory variable list rear of type unsigned int.*/
    unsigned int size [xmachine_memory_receptionist_MAX];    /**< X-machine memory variable list size of type unsigned int.*/
    unsigned int tick [xmachine_memory_receptionist_MAX];    /**< X-machine memory variable list tick of type unsigned int.*/
    int current_patient [xmachine_memory_receptionist_MAX];    /**< X-machine memory variable list current_patient of type int.*/
    int attend_patient [xmachine_memory_receptionist_MAX];    /**< X-machine memory variable list attend_patient of type int.*/
    int estado [xmachine_memory_receptionist_MAX];    /**< X-machine memory variable list estado of type int.*/
};

/** struct xmachine_memory_agent_generator_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_agent_generator_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_agent_generator_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_agent_generator_MAX];  /**< Used during parallel prefix sum */
    
    int chairs_generated [xmachine_memory_agent_generator_MAX];    /**< X-machine memory variable list chairs_generated of type int.*/
    int boxes_generated [xmachine_memory_agent_generator_MAX];    /**< X-machine memory variable list boxes_generated of type int.*/
    int doctors_generated [xmachine_memory_agent_generator_MAX];    /**< X-machine memory variable list doctors_generated of type int.*/
    int specialists_generated [xmachine_memory_agent_generator_MAX];    /**< X-machine memory variable list specialists_generated of type int.*/
};

/** struct xmachine_memory_chair_admin_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_chair_admin_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_chair_admin_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_chair_admin_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_chair_admin_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
    unsigned int chairArray [xmachine_memory_chair_admin_MAX*35];    /**< X-machine memory variable list chairArray of type unsigned int.*/
};

/** struct xmachine_memory_box_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_box_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_box_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_box_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_box_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
    unsigned int attending [xmachine_memory_box_MAX];    /**< X-machine memory variable list attending of type unsigned int.*/
    unsigned int tick [xmachine_memory_box_MAX];    /**< X-machine memory variable list tick of type unsigned int.*/
};

/** struct xmachine_memory_doctor_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_doctor_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_doctor_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_doctor_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_doctor_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
    int current_patient [xmachine_memory_doctor_MAX];    /**< X-machine memory variable list current_patient of type int.*/
    unsigned int tick [xmachine_memory_doctor_MAX];    /**< X-machine memory variable list tick of type unsigned int.*/
};

/** struct xmachine_memory_triage_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_triage_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_triage_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_triage_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int front [xmachine_memory_triage_MAX];    /**< X-machine memory variable list front of type unsigned int.*/
    unsigned int rear [xmachine_memory_triage_MAX];    /**< X-machine memory variable list rear of type unsigned int.*/
    unsigned int size [xmachine_memory_triage_MAX];    /**< X-machine memory variable list size of type unsigned int.*/
    unsigned int tick [xmachine_memory_triage_MAX];    /**< X-machine memory variable list tick of type unsigned int.*/
    unsigned int boxArray [xmachine_memory_triage_MAX*3];    /**< X-machine memory variable list boxArray of type unsigned int.*/
    unsigned int patientQueue [xmachine_memory_triage_MAX*100];    /**< X-machine memory variable list patientQueue of type unsigned int.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_pedestrian_location_list
 * Spatial Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_pedestrian_location_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_pedestrian_location_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_pedestrian_location_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_message_pedestrian_location_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_pedestrian_location_MAX];    /**< Message memory variable list y of type float.*/
    float z [xmachine_message_pedestrian_location_MAX];    /**< Message memory variable list z of type float.*/
    int estado [xmachine_message_pedestrian_location_MAX];    /**< Message memory variable list estado of type int.*/
    
};

/** struct xmachine_message_pedestrian_state_list
 * Spatial Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_pedestrian_state_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_pedestrian_state_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_pedestrian_state_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_message_pedestrian_state_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_pedestrian_state_MAX];    /**< Message memory variable list y of type float.*/
    float z [xmachine_message_pedestrian_state_MAX];    /**< Message memory variable list z of type float.*/
    int estado [xmachine_message_pedestrian_state_MAX];    /**< Message memory variable list estado of type int.*/
    
};

/** struct xmachine_message_navmap_cell_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_navmap_cell_list
{
    int x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list x of type int.*/
    int y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list y of type int.*/
    int exit_no [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list exit_no of type int.*/
    float height [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list height of type float.*/
    float collision_x [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list collision_x of type float.*/
    float collision_y [xmachine_message_navmap_cell_MAX];    /**< Message memory variable list collision_y of type float.*/
    
};

/** struct xmachine_message_check_in_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_check_in_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_check_in_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_check_in_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_check_in_MAX];    /**< Message memory variable list id of type unsigned int.*/
    
};

/** struct xmachine_message_check_in_response_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_check_in_response_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_check_in_response_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_check_in_response_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_check_in_response_MAX];    /**< Message memory variable list id of type unsigned int.*/
    
};

/** struct xmachine_message_chair_petition_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_chair_petition_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_chair_petition_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_chair_petition_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_chair_petition_MAX];    /**< Message memory variable list id of type unsigned int.*/
    
};

/** struct xmachine_message_chair_response_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_chair_response_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_chair_response_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_chair_response_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_chair_response_MAX];    /**< Message memory variable list id of type unsigned int.*/
    int chair_no [xmachine_message_chair_response_MAX];    /**< Message memory variable list chair_no of type int.*/
    
};

/** struct xmachine_message_chair_state_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_chair_state_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_chair_state_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_chair_state_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_chair_state_MAX];    /**< Message memory variable list id of type unsigned int.*/
    int state [xmachine_message_chair_state_MAX];    /**< Message memory variable list state of type int.*/
    
};

/** struct xmachine_message_free_chair_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_free_chair_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_free_chair_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_free_chair_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int chair_no [xmachine_message_free_chair_MAX];    /**< Message memory variable list chair_no of type unsigned int.*/
    
};

/** struct xmachine_message_chair_contact_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_chair_contact_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_chair_contact_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_chair_contact_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_chair_contact_MAX];    /**< Message memory variable list id of type unsigned int.*/
    unsigned int chair_no [xmachine_message_chair_contact_MAX];    /**< Message memory variable list chair_no of type unsigned int.*/
    int state [xmachine_message_chair_contact_MAX];    /**< Message memory variable list state of type int.*/
    
};

/** struct xmachine_message_box_petition_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_box_petition_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_box_petition_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_box_petition_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_box_petition_MAX];    /**< Message memory variable list id of type unsigned int.*/
    unsigned int box_no [xmachine_message_box_petition_MAX];    /**< Message memory variable list box_no of type unsigned int.*/
    
};

/** struct xmachine_message_box_response_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_box_response_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_box_response_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_box_response_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_box_response_MAX];    /**< Message memory variable list id of type unsigned int.*/
    unsigned int doctor_no [xmachine_message_box_response_MAX];    /**< Message memory variable list doctor_no of type unsigned int.*/
    unsigned int priority [xmachine_message_box_response_MAX];    /**< Message memory variable list priority of type unsigned int.*/
    
};

/** struct xmachine_message_specialist_reached_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_specialist_reached_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_specialist_reached_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_specialist_reached_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_specialist_reached_MAX];    /**< Message memory variable list id of type unsigned int.*/
    unsigned int specialist_no [xmachine_message_specialist_reached_MAX];    /**< Message memory variable list specialist_no of type unsigned int.*/
    
};

/** struct xmachine_message_specialist_petition_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_specialist_petition_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_specialist_petition_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_specialist_petition_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_specialist_petition_MAX];    /**< Message memory variable list id of type unsigned int.*/
    unsigned int priority [xmachine_message_specialist_petition_MAX];    /**< Message memory variable list priority of type unsigned int.*/
    unsigned int specialist_no [xmachine_message_specialist_petition_MAX];    /**< Message memory variable list specialist_no of type unsigned int.*/
    
};

/** struct xmachine_message_doctor_reached_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_doctor_reached_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_doctor_reached_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_doctor_reached_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_doctor_reached_MAX];    /**< Message memory variable list id of type unsigned int.*/
    unsigned int doctor_no [xmachine_message_doctor_reached_MAX];    /**< Message memory variable list doctor_no of type unsigned int.*/
    
};

/** struct xmachine_message_free_doctor_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_free_doctor_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_free_doctor_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_free_doctor_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int doctor_no [xmachine_message_free_doctor_MAX];    /**< Message memory variable list doctor_no of type unsigned int.*/
    
};

/** struct xmachine_message_attention_terminated_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_attention_terminated_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_attention_terminated_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_attention_terminated_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_attention_terminated_MAX];    /**< Message memory variable list id of type unsigned int.*/
    
};

/** struct xmachine_message_doctor_petition_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_doctor_petition_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_doctor_petition_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_doctor_petition_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_doctor_petition_MAX];    /**< Message memory variable list id of type unsigned int.*/
    unsigned int doctor_no [xmachine_message_doctor_petition_MAX];    /**< Message memory variable list doctor_no of type unsigned int.*/
    unsigned int priority [xmachine_message_doctor_petition_MAX];    /**< Message memory variable list priority of type unsigned int.*/
    
};

/** struct xmachine_message_doctor_response_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_doctor_response_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_doctor_response_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_doctor_response_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_doctor_response_MAX];    /**< Message memory variable list id of type unsigned int.*/
    int doctor_no [xmachine_message_doctor_response_MAX];    /**< Message memory variable list doctor_no of type int.*/
    
};

/** struct xmachine_message_specialist_response_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_specialist_response_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_specialist_response_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_specialist_response_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_specialist_response_MAX];    /**< Message memory variable list id of type unsigned int.*/
    int specialist_ready [xmachine_message_specialist_response_MAX];    /**< Message memory variable list specialist_ready of type int.*/
    
};

/** struct xmachine_message_triage_petition_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_triage_petition_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_triage_petition_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_triage_petition_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_triage_petition_MAX];    /**< Message memory variable list id of type unsigned int.*/
    
};

/** struct xmachine_message_triage_response_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_triage_response_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_triage_response_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_triage_response_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_triage_response_MAX];    /**< Message memory variable list id of type unsigned int.*/
    int box_no [xmachine_message_triage_response_MAX];    /**< Message memory variable list box_no of type int.*/
    
};



/* Spatially Partitioned Message boundary Matrices */

/** struct xmachine_message_pedestrian_location_PBM
 * Partition Boundary Matrix (PBM) for xmachine_message_pedestrian_location 
 */
struct xmachine_message_pedestrian_location_PBM
{
	int start[xmachine_message_pedestrian_location_grid_size];
	int end_or_count[xmachine_message_pedestrian_location_grid_size];
};

/** struct xmachine_message_pedestrian_state_PBM
 * Partition Boundary Matrix (PBM) for xmachine_message_pedestrian_state 
 */
struct xmachine_message_pedestrian_state_PBM
{
	int start[xmachine_message_pedestrian_state_grid_size];
	int end_or_count[xmachine_message_pedestrian_state_grid_size];
};



/* Graph structures */


/* Graph Edge Partitioned message boundary structures */


/* Graph utility functions, usable in agent functions and implemented in FLAMEGPU_Kernels */


  /* Random */
  /** struct RNG_rand48
  *	structure used to hold list seeds
  */
  struct RNG_rand48
  {
  glm::uvec2 A, C;
  glm::uvec2 seeds[buffer_size_MAX];
  };


/** getOutputDir
* Gets the output directory of the simulation. This is the same as the 0.xml input directory.
* @return a const char pointer to string denoting the output directory
*/
const char* getOutputDir();

  /* Random Functions (usable in agent functions) implemented in FLAMEGPU_Kernels */

  /**
  * Templated random function using a DISCRETE_2D template calculates the agent index using a 2D block
  * which requires extra processing but will work for CONTINUOUS agents. Using a CONTINUOUS template will
  * not work for DISCRETE_2D agent.
  * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
  * @return			returns a random float value
  */
  template <int AGENT_TYPE> __FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);
/**
 * Non templated random function calls the templated version with DISCRETE_2D which will work in either case
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
__FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);

/* Agent function prototypes */

/**
 * output_pedestrian_location FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param pedestrian_location_messages Pointer to output message list of type xmachine_message_pedestrian_location_list. Must be passed as an argument to the add_pedestrian_location_message function ??.
 */
__FLAME_GPU_FUNC__ int output_pedestrian_location(xmachine_memory_agent* agent, xmachine_message_pedestrian_location_list* pedestrian_location_messages);

/**
 * avoid_pedestrians FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param pedestrian_location_messages  pedestrian_location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_pedestrian_location_message and get_next_pedestrian_location_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_pedestrian_location_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int avoid_pedestrians(xmachine_memory_agent* agent, xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix, RNG_rand48* rand48);

/**
 * output_pedestrian_state FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param pedestrian_state_messages Pointer to output message list of type xmachine_message_pedestrian_state_list. Must be passed as an argument to the add_pedestrian_state_message function ??.
 */
__FLAME_GPU_FUNC__ int output_pedestrian_state(xmachine_memory_agent* agent, xmachine_message_pedestrian_state_list* pedestrian_state_messages);

/**
 * infect_pedestrians FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param pedestrian_state_messages  pedestrian_state_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_pedestrian_state_message and get_next_pedestrian_state_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_pedestrian_state_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int infect_pedestrians(xmachine_memory_agent* agent, xmachine_message_pedestrian_state_list* pedestrian_state_messages, xmachine_message_pedestrian_state_PBM* partition_matrix, RNG_rand48* rand48);

/**
 * move FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param check_in_messages Pointer to output message list of type xmachine_message_check_in_list. Must be passed as an argument to the add_check_in_message function ??.
 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_agent* agent, xmachine_message_check_in_list* check_in_messages);

/**
 * receive_chair_state FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param chair_state_messages  chair_state_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_chair_state_message and get_next_chair_state_message functions.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int receive_chair_state(xmachine_memory_agent* agent, xmachine_message_chair_state_list* chair_state_messages, RNG_rand48* rand48);

/**
 * output_chair_contact FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param chair_contact_messages Pointer to output message list of type xmachine_message_chair_contact_list. Must be passed as an argument to the add_chair_contact_message function ??.
 */
__FLAME_GPU_FUNC__ int output_chair_contact(xmachine_memory_agent* agent, xmachine_message_chair_contact_list* chair_contact_messages);

/**
 * output_free_chair FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param free_chair_messages Pointer to output message list of type xmachine_message_free_chair_list. Must be passed as an argument to the add_free_chair_message function ??.
 */
__FLAME_GPU_FUNC__ int output_free_chair(xmachine_memory_agent* agent, xmachine_message_free_chair_list* free_chair_messages);

/**
 * output_chair_petition FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param chair_petition_messages Pointer to output message list of type xmachine_message_chair_petition_list. Must be passed as an argument to the add_chair_petition_message function ??.
 */
__FLAME_GPU_FUNC__ int output_chair_petition(xmachine_memory_agent* agent, xmachine_message_chair_petition_list* chair_petition_messages);

/**
 * receive_chair_response FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param chair_response_messages  chair_response_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_chair_response_message and get_next_chair_response_message functions.
 */
__FLAME_GPU_FUNC__ int receive_chair_response(xmachine_memory_agent* agent, xmachine_message_chair_response_list* chair_response_messages);

/**
 * receive_check_in_response FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param check_in_response_messages  check_in_response_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_check_in_response_message and get_next_check_in_response_message functions.* @param chair_petition_messages Pointer to output message list of type xmachine_message_chair_petition_list. Must be passed as an argument to the add_chair_petition_message function ??.
 */
__FLAME_GPU_FUNC__ int receive_check_in_response(xmachine_memory_agent* agent, xmachine_message_check_in_response_list* check_in_response_messages, xmachine_message_chair_petition_list* chair_petition_messages);

/**
 * output_box_petition FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param box_petition_messages Pointer to output message list of type xmachine_message_box_petition_list. Must be passed as an argument to the add_box_petition_message function ??.
 */
__FLAME_GPU_FUNC__ int output_box_petition(xmachine_memory_agent* agent, xmachine_message_box_petition_list* box_petition_messages);

/**
 * receive_box_response FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param box_response_messages  box_response_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_box_response_message and get_next_box_response_message functions.
 */
__FLAME_GPU_FUNC__ int receive_box_response(xmachine_memory_agent* agent, xmachine_message_box_response_list* box_response_messages);

/**
 * output_doctor_petition FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param doctor_petition_messages Pointer to output message list of type xmachine_message_doctor_petition_list. Must be passed as an argument to the add_doctor_petition_message function ??.
 */
__FLAME_GPU_FUNC__ int output_doctor_petition(xmachine_memory_agent* agent, xmachine_message_doctor_petition_list* doctor_petition_messages);

/**
 * receive_doctor_response FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param doctor_response_messages  doctor_response_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_doctor_response_message and get_next_doctor_response_message functions.
 */
__FLAME_GPU_FUNC__ int receive_doctor_response(xmachine_memory_agent* agent, xmachine_message_doctor_response_list* doctor_response_messages);

/**
 * receive_attention_terminated FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param attention_terminated_messages  attention_terminated_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_attention_terminated_message and get_next_attention_terminated_message functions.* @param free_doctor_messages Pointer to output message list of type xmachine_message_free_doctor_list. Must be passed as an argument to the add_free_doctor_message function ??.
 */
__FLAME_GPU_FUNC__ int receive_attention_terminated(xmachine_memory_agent* agent, xmachine_message_attention_terminated_list* attention_terminated_messages, xmachine_message_free_doctor_list* free_doctor_messages);

/**
 * output_doctor_reached FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param doctor_reached_messages Pointer to output message list of type xmachine_message_doctor_reached_list. Must be passed as an argument to the add_doctor_reached_message function ??.
 */
__FLAME_GPU_FUNC__ int output_doctor_reached(xmachine_memory_agent* agent, xmachine_message_doctor_reached_list* doctor_reached_messages);

/**
 * receive_specialist_response FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param specialist_response_messages  specialist_response_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_specialist_response_message and get_next_specialist_response_message functions.
 */
__FLAME_GPU_FUNC__ int receive_specialist_response(xmachine_memory_agent* agent, xmachine_message_specialist_response_list* specialist_response_messages);

/**
 * output_specialist_petition FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param specialist_petition_messages Pointer to output message list of type xmachine_message_specialist_petition_list. Must be passed as an argument to the add_specialist_petition_message function ??.
 */
__FLAME_GPU_FUNC__ int output_specialist_petition(xmachine_memory_agent* agent, xmachine_message_specialist_petition_list* specialist_petition_messages);

/**
 * output_specialist_reached FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param specialist_reached_messages Pointer to output message list of type xmachine_message_specialist_reached_list. Must be passed as an argument to the add_specialist_reached_message function ??.
 */
__FLAME_GPU_FUNC__ int output_specialist_reached(xmachine_memory_agent* agent, xmachine_message_specialist_reached_list* specialist_reached_messages);

/**
 * output_triage_petition FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param triage_petition_messages Pointer to output message list of type xmachine_message_triage_petition_list. Must be passed as an argument to the add_triage_petition_message function ??.
 */
__FLAME_GPU_FUNC__ int output_triage_petition(xmachine_memory_agent* agent, xmachine_message_triage_petition_list* triage_petition_messages);

/**
 * receive_triage_response FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param triage_response_messages  triage_response_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_triage_response_message and get_next_triage_response_message functions.* @param chair_petition_messages Pointer to output message list of type xmachine_message_chair_petition_list. Must be passed as an argument to the add_chair_petition_message function ??.
 */
__FLAME_GPU_FUNC__ int receive_triage_response(xmachine_memory_agent* agent, xmachine_message_triage_response_list* triage_response_messages, xmachine_message_chair_petition_list* chair_petition_messages);

/**
 * output_navmap_cells FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_navmap. This represents a single agent instance and can be modified directly.
 * @param navmap_cell_messages Pointer to output message list of type xmachine_message_navmap_cell_list. Must be passed as an argument to the add_navmap_cell_message function ??.
 */
__FLAME_GPU_FUNC__ int output_navmap_cells(xmachine_memory_navmap* agent, xmachine_message_navmap_cell_list* navmap_cell_messages);

/**
 * generate_pedestrians FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_navmap. This represents a single agent instance and can be modified directly.
 * @param agent_agents Pointer to agent list of type xmachine_memory_agent_list. This must be passed as an argument to the add_agent_agent function to add a new agent.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int generate_pedestrians(xmachine_memory_navmap* agent, xmachine_memory_agent_list* agent_agents, RNG_rand48* rand48);

/**
 * output_chair_state FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_chair. This represents a single agent instance and can be modified directly.
 * @param chair_contact_messages  chair_contact_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_chair_contact_message and get_next_chair_contact_message functions.* @param chair_state_messages Pointer to output message list of type xmachine_message_chair_state_list. Must be passed as an argument to the add_chair_state_message function ??.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int output_chair_state(xmachine_memory_chair* agent, xmachine_message_chair_contact_list* chair_contact_messages, xmachine_message_chair_state_list* chair_state_messages, RNG_rand48* rand48);

/**
 * receive_doctor_petitions FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_doctor_manager. This represents a single agent instance and can be modified directly.
 * @param doctor_petition_messages  doctor_petition_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_doctor_petition_message and get_next_doctor_petition_message functions.* @param doctor_response_messages Pointer to output message list of type xmachine_message_doctor_response_list. Must be passed as an argument to the add_doctor_response_message function ??.
 */
__FLAME_GPU_FUNC__ int receive_doctor_petitions(xmachine_memory_doctor_manager* agent, xmachine_message_doctor_petition_list* doctor_petition_messages, xmachine_message_doctor_response_list* doctor_response_messages);

/**
 * receive_free_doctors FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_doctor_manager. This represents a single agent instance and can be modified directly.
 * @param free_doctor_messages  free_doctor_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_free_doctor_message and get_next_free_doctor_message functions.
 */
__FLAME_GPU_FUNC__ int receive_free_doctors(xmachine_memory_doctor_manager* agent, xmachine_message_free_doctor_list* free_doctor_messages);

/**
 * receive_specialist_petitions FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_specialist_manager. This represents a single agent instance and can be modified directly.
 * @param specialist_petition_messages  specialist_petition_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_specialist_petition_message and get_next_specialist_petition_message functions.* @param specialist_response_messages Pointer to output message list of type xmachine_message_specialist_response_list. Must be passed as an argument to the add_specialist_response_message function ??.
 */
__FLAME_GPU_FUNC__ int receive_specialist_petitions(xmachine_memory_specialist_manager* agent, xmachine_message_specialist_petition_list* specialist_petition_messages, xmachine_message_specialist_response_list* specialist_response_messages);

/**
 * receive_specialist_reached FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_specialist. This represents a single agent instance and can be modified directly.
 * @param specialist_reached_messages  specialist_reached_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_specialist_reached_message and get_next_specialist_reached_message functions.* @param attention_terminated_messages Pointer to output message list of type xmachine_message_attention_terminated_list. Must be passed as an argument to the add_attention_terminated_message function ??.
 */
__FLAME_GPU_FUNC__ int receive_specialist_reached(xmachine_memory_specialist* agent, xmachine_message_specialist_reached_list* specialist_reached_messages, xmachine_message_attention_terminated_list* attention_terminated_messages);

/**
 * receptionServer FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_receptionist. This represents a single agent instance and can be modified directly.
 * @param check_in_messages  check_in_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_check_in_message and get_next_check_in_message functions.* @param check_in_response_messages Pointer to output message list of type xmachine_message_check_in_response_list. Must be passed as an argument to the add_check_in_response_message function ??.
 */
__FLAME_GPU_FUNC__ int receptionServer(xmachine_memory_receptionist* agent, xmachine_message_check_in_list* check_in_messages, xmachine_message_check_in_response_list* check_in_response_messages);

/**
 * infect_receptionist FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_receptionist. This represents a single agent instance and can be modified directly.
 * @param pedestrian_state_messages  pedestrian_state_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_pedestrian_state_message and get_next_pedestrian_state_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_pedestrian_state_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int infect_receptionist(xmachine_memory_receptionist* agent, xmachine_message_pedestrian_state_list* pedestrian_state_messages, xmachine_message_pedestrian_state_PBM* partition_matrix, RNG_rand48* rand48);

/**
 * generate_chairs FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent_generator. This represents a single agent instance and can be modified directly.
 * @param chair_agents Pointer to agent list of type xmachine_memory_chair_list. This must be passed as an argument to the add_chair_agent function to add a new agent.
 */
__FLAME_GPU_FUNC__ int generate_chairs(xmachine_memory_agent_generator* agent, xmachine_memory_chair_list* chair_agents);

/**
 * generate_boxes FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent_generator. This represents a single agent instance and can be modified directly.
 * @param box_agents Pointer to agent list of type xmachine_memory_box_list. This must be passed as an argument to the add_box_agent function to add a new agent.
 */
__FLAME_GPU_FUNC__ int generate_boxes(xmachine_memory_agent_generator* agent, xmachine_memory_box_list* box_agents);

/**
 * generate_doctors FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent_generator. This represents a single agent instance and can be modified directly.
 * @param doctor_agents Pointer to agent list of type xmachine_memory_doctor_list. This must be passed as an argument to the add_doctor_agent function to add a new agent.
 */
__FLAME_GPU_FUNC__ int generate_doctors(xmachine_memory_agent_generator* agent, xmachine_memory_doctor_list* doctor_agents);

/**
 * generate_specialists FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent_generator. This represents a single agent instance and can be modified directly.
 * @param specialist_agents Pointer to agent list of type xmachine_memory_specialist_list. This must be passed as an argument to the add_specialist_agent function to add a new agent.
 */
__FLAME_GPU_FUNC__ int generate_specialists(xmachine_memory_agent_generator* agent, xmachine_memory_specialist_list* specialist_agents);

/**
 * attend_chair_petitions FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_chair_admin. This represents a single agent instance and can be modified directly.
 * @param chair_petition_messages  chair_petition_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_chair_petition_message and get_next_chair_petition_message functions.* @param chair_response_messages Pointer to output message list of type xmachine_message_chair_response_list. Must be passed as an argument to the add_chair_response_message function ??.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int attend_chair_petitions(xmachine_memory_chair_admin* agent, xmachine_message_chair_petition_list* chair_petition_messages, xmachine_message_chair_response_list* chair_response_messages, RNG_rand48* rand48);

/**
 * receive_free_chair FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_chair_admin. This represents a single agent instance and can be modified directly.
 * @param free_chair_messages  free_chair_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_free_chair_message and get_next_free_chair_message functions.
 */
__FLAME_GPU_FUNC__ int receive_free_chair(xmachine_memory_chair_admin* agent, xmachine_message_free_chair_list* free_chair_messages);

/**
 * box_server FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_box. This represents a single agent instance and can be modified directly.
 * @param box_petition_messages  box_petition_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_box_petition_message and get_next_box_petition_message functions.
 */
__FLAME_GPU_FUNC__ int box_server(xmachine_memory_box* agent, xmachine_message_box_petition_list* box_petition_messages);

/**
 * attend_box_patient FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_box. This represents a single agent instance and can be modified directly.
 * @param box_response_messages Pointer to output message list of type xmachine_message_box_response_list. Must be passed as an argument to the add_box_response_message function ??.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int attend_box_patient(xmachine_memory_box* agent, xmachine_message_box_response_list* box_response_messages, RNG_rand48* rand48);

/**
 * doctor_server FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_doctor. This represents a single agent instance and can be modified directly.
 * @param doctor_reached_messages  doctor_reached_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_doctor_reached_message and get_next_doctor_reached_message functions.* @param attention_terminated_messages Pointer to output message list of type xmachine_message_attention_terminated_list. Must be passed as an argument to the add_attention_terminated_message function ??.
 */
__FLAME_GPU_FUNC__ int doctor_server(xmachine_memory_doctor* agent, xmachine_message_doctor_reached_list* doctor_reached_messages, xmachine_message_attention_terminated_list* attention_terminated_messages);

/**
 * receive_triage_petitions FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_triage. This represents a single agent instance and can be modified directly.
 * @param triage_petition_messages  triage_petition_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_triage_petition_message and get_next_triage_petition_message functions.* @param triage_response_messages Pointer to output message list of type xmachine_message_triage_response_list. Must be passed as an argument to the add_triage_response_message function ??.
 */
__FLAME_GPU_FUNC__ int receive_triage_petitions(xmachine_memory_triage* agent, xmachine_message_triage_petition_list* triage_petition_messages, xmachine_message_triage_response_list* triage_response_messages);

  
/* Message Function Prototypes for Spatially Partitioned pedestrian_location message implemented in FLAMEGPU_Kernels */

/** add_pedestrian_location_message
 * Function for all types of message partitioning
 * Adds a new pedestrian_location agent to the xmachine_memory_pedestrian_location_list list using a linear mapping
 * @param agents	xmachine_memory_pedestrian_location_list agent list
 * @param x	message variable of type float
 * @param y	message variable of type float
 * @param z	message variable of type float
 * @param estado	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_pedestrian_location_message(xmachine_message_pedestrian_location_list* pedestrian_location_messages, float x, float y, float z, int estado);
 
/** get_first_pedestrian_location_message
 * Get first message function for spatially partitioned messages
 * @param pedestrian_location_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @param agentz z position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_pedestrian_location * get_first_pedestrian_location_message(xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix, float x, float y, float z);

/** get_next_pedestrian_location_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param pedestrian_location_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_pedestrian_location * get_next_pedestrian_location_message(xmachine_message_pedestrian_location* current, xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix);

  
/* Message Function Prototypes for Spatially Partitioned pedestrian_state message implemented in FLAMEGPU_Kernels */

/** add_pedestrian_state_message
 * Function for all types of message partitioning
 * Adds a new pedestrian_state agent to the xmachine_memory_pedestrian_state_list list using a linear mapping
 * @param agents	xmachine_memory_pedestrian_state_list agent list
 * @param x	message variable of type float
 * @param y	message variable of type float
 * @param z	message variable of type float
 * @param estado	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_pedestrian_state_message(xmachine_message_pedestrian_state_list* pedestrian_state_messages, float x, float y, float z, int estado);
 
/** get_first_pedestrian_state_message
 * Get first message function for spatially partitioned messages
 * @param pedestrian_state_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @param agentz z position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_pedestrian_state * get_first_pedestrian_state_message(xmachine_message_pedestrian_state_list* pedestrian_state_messages, xmachine_message_pedestrian_state_PBM* partition_matrix, float x, float y, float z);

/** get_next_pedestrian_state_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param pedestrian_state_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_pedestrian_state * get_next_pedestrian_state_message(xmachine_message_pedestrian_state* current, xmachine_message_pedestrian_state_list* pedestrian_state_messages, xmachine_message_pedestrian_state_PBM* partition_matrix);

  
/* Message Function Prototypes for Discrete Partitioned navmap_cell message implemented in FLAMEGPU_Kernels */

/** add_navmap_cell_message
 * Function for all types of message partitioning
 * Adds a new navmap_cell agent to the xmachine_memory_navmap_cell_list list using a linear mapping
 * @param agents	xmachine_memory_navmap_cell_list agent list
 * @param x	message variable of type int
 * @param y	message variable of type int
 * @param exit_no	message variable of type int
 * @param height	message variable of type float
 * @param collision_x	message variable of type float
 * @param collision_y	message variable of type float
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_navmap_cell_message(xmachine_message_navmap_cell_list* navmap_cell_messages, int x, int y, int exit_no, float height, float collision_x, float collision_y);
 
/** get_first_navmap_cell_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param navmap_cell_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_navmap_cell * get_first_navmap_cell_message(xmachine_message_navmap_cell_list* navmap_cell_messages, int agentx, int agent_y);

/** get_next_navmap_cell_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param navmap_cell_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_navmap_cell * get_next_navmap_cell_message(xmachine_message_navmap_cell* current, xmachine_message_navmap_cell_list* navmap_cell_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) check_in message implemented in FLAMEGPU_Kernels */

/** add_check_in_message
 * Function for all types of message partitioning
 * Adds a new check_in agent to the xmachine_memory_check_in_list list using a linear mapping
 * @param agents	xmachine_memory_check_in_list agent list
 * @param id	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_check_in_message(xmachine_message_check_in_list* check_in_messages, unsigned int id);
 
/** get_first_check_in_message
 * Get first message function for non partitioned (brute force) messages
 * @param check_in_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_check_in * get_first_check_in_message(xmachine_message_check_in_list* check_in_messages);

/** get_next_check_in_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param check_in_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_check_in * get_next_check_in_message(xmachine_message_check_in* current, xmachine_message_check_in_list* check_in_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) check_in_response message implemented in FLAMEGPU_Kernels */

/** add_check_in_response_message
 * Function for all types of message partitioning
 * Adds a new check_in_response agent to the xmachine_memory_check_in_response_list list using a linear mapping
 * @param agents	xmachine_memory_check_in_response_list agent list
 * @param id	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_check_in_response_message(xmachine_message_check_in_response_list* check_in_response_messages, unsigned int id);
 
/** get_first_check_in_response_message
 * Get first message function for non partitioned (brute force) messages
 * @param check_in_response_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_check_in_response * get_first_check_in_response_message(xmachine_message_check_in_response_list* check_in_response_messages);

/** get_next_check_in_response_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param check_in_response_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_check_in_response * get_next_check_in_response_message(xmachine_message_check_in_response* current, xmachine_message_check_in_response_list* check_in_response_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) chair_petition message implemented in FLAMEGPU_Kernels */

/** add_chair_petition_message
 * Function for all types of message partitioning
 * Adds a new chair_petition agent to the xmachine_memory_chair_petition_list list using a linear mapping
 * @param agents	xmachine_memory_chair_petition_list agent list
 * @param id	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_chair_petition_message(xmachine_message_chair_petition_list* chair_petition_messages, unsigned int id);
 
/** get_first_chair_petition_message
 * Get first message function for non partitioned (brute force) messages
 * @param chair_petition_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_chair_petition * get_first_chair_petition_message(xmachine_message_chair_petition_list* chair_petition_messages);

/** get_next_chair_petition_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param chair_petition_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_chair_petition * get_next_chair_petition_message(xmachine_message_chair_petition* current, xmachine_message_chair_petition_list* chair_petition_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) chair_response message implemented in FLAMEGPU_Kernels */

/** add_chair_response_message
 * Function for all types of message partitioning
 * Adds a new chair_response agent to the xmachine_memory_chair_response_list list using a linear mapping
 * @param agents	xmachine_memory_chair_response_list agent list
 * @param id	message variable of type unsigned int
 * @param chair_no	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_chair_response_message(xmachine_message_chair_response_list* chair_response_messages, unsigned int id, int chair_no);
 
/** get_first_chair_response_message
 * Get first message function for non partitioned (brute force) messages
 * @param chair_response_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_chair_response * get_first_chair_response_message(xmachine_message_chair_response_list* chair_response_messages);

/** get_next_chair_response_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param chair_response_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_chair_response * get_next_chair_response_message(xmachine_message_chair_response* current, xmachine_message_chair_response_list* chair_response_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) chair_state message implemented in FLAMEGPU_Kernels */

/** add_chair_state_message
 * Function for all types of message partitioning
 * Adds a new chair_state agent to the xmachine_memory_chair_state_list list using a linear mapping
 * @param agents	xmachine_memory_chair_state_list agent list
 * @param id	message variable of type unsigned int
 * @param state	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_chair_state_message(xmachine_message_chair_state_list* chair_state_messages, unsigned int id, int state);
 
/** get_first_chair_state_message
 * Get first message function for non partitioned (brute force) messages
 * @param chair_state_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_chair_state * get_first_chair_state_message(xmachine_message_chair_state_list* chair_state_messages);

/** get_next_chair_state_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param chair_state_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_chair_state * get_next_chair_state_message(xmachine_message_chair_state* current, xmachine_message_chair_state_list* chair_state_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) free_chair message implemented in FLAMEGPU_Kernels */

/** add_free_chair_message
 * Function for all types of message partitioning
 * Adds a new free_chair agent to the xmachine_memory_free_chair_list list using a linear mapping
 * @param agents	xmachine_memory_free_chair_list agent list
 * @param chair_no	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_free_chair_message(xmachine_message_free_chair_list* free_chair_messages, unsigned int chair_no);
 
/** get_first_free_chair_message
 * Get first message function for non partitioned (brute force) messages
 * @param free_chair_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_free_chair * get_first_free_chair_message(xmachine_message_free_chair_list* free_chair_messages);

/** get_next_free_chair_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param free_chair_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_free_chair * get_next_free_chair_message(xmachine_message_free_chair* current, xmachine_message_free_chair_list* free_chair_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) chair_contact message implemented in FLAMEGPU_Kernels */

/** add_chair_contact_message
 * Function for all types of message partitioning
 * Adds a new chair_contact agent to the xmachine_memory_chair_contact_list list using a linear mapping
 * @param agents	xmachine_memory_chair_contact_list agent list
 * @param id	message variable of type unsigned int
 * @param chair_no	message variable of type unsigned int
 * @param state	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_chair_contact_message(xmachine_message_chair_contact_list* chair_contact_messages, unsigned int id, unsigned int chair_no, int state);
 
/** get_first_chair_contact_message
 * Get first message function for non partitioned (brute force) messages
 * @param chair_contact_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_chair_contact * get_first_chair_contact_message(xmachine_message_chair_contact_list* chair_contact_messages);

/** get_next_chair_contact_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param chair_contact_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_chair_contact * get_next_chair_contact_message(xmachine_message_chair_contact* current, xmachine_message_chair_contact_list* chair_contact_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) box_petition message implemented in FLAMEGPU_Kernels */

/** add_box_petition_message
 * Function for all types of message partitioning
 * Adds a new box_petition agent to the xmachine_memory_box_petition_list list using a linear mapping
 * @param agents	xmachine_memory_box_petition_list agent list
 * @param id	message variable of type unsigned int
 * @param box_no	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_box_petition_message(xmachine_message_box_petition_list* box_petition_messages, unsigned int id, unsigned int box_no);
 
/** get_first_box_petition_message
 * Get first message function for non partitioned (brute force) messages
 * @param box_petition_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_box_petition * get_first_box_petition_message(xmachine_message_box_petition_list* box_petition_messages);

/** get_next_box_petition_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param box_petition_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_box_petition * get_next_box_petition_message(xmachine_message_box_petition* current, xmachine_message_box_petition_list* box_petition_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) box_response message implemented in FLAMEGPU_Kernels */

/** add_box_response_message
 * Function for all types of message partitioning
 * Adds a new box_response agent to the xmachine_memory_box_response_list list using a linear mapping
 * @param agents	xmachine_memory_box_response_list agent list
 * @param id	message variable of type unsigned int
 * @param doctor_no	message variable of type unsigned int
 * @param priority	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_box_response_message(xmachine_message_box_response_list* box_response_messages, unsigned int id, unsigned int doctor_no, unsigned int priority);
 
/** get_first_box_response_message
 * Get first message function for non partitioned (brute force) messages
 * @param box_response_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_box_response * get_first_box_response_message(xmachine_message_box_response_list* box_response_messages);

/** get_next_box_response_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param box_response_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_box_response * get_next_box_response_message(xmachine_message_box_response* current, xmachine_message_box_response_list* box_response_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) specialist_reached message implemented in FLAMEGPU_Kernels */

/** add_specialist_reached_message
 * Function for all types of message partitioning
 * Adds a new specialist_reached agent to the xmachine_memory_specialist_reached_list list using a linear mapping
 * @param agents	xmachine_memory_specialist_reached_list agent list
 * @param id	message variable of type unsigned int
 * @param specialist_no	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_specialist_reached_message(xmachine_message_specialist_reached_list* specialist_reached_messages, unsigned int id, unsigned int specialist_no);
 
/** get_first_specialist_reached_message
 * Get first message function for non partitioned (brute force) messages
 * @param specialist_reached_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_specialist_reached * get_first_specialist_reached_message(xmachine_message_specialist_reached_list* specialist_reached_messages);

/** get_next_specialist_reached_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param specialist_reached_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_specialist_reached * get_next_specialist_reached_message(xmachine_message_specialist_reached* current, xmachine_message_specialist_reached_list* specialist_reached_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) specialist_petition message implemented in FLAMEGPU_Kernels */

/** add_specialist_petition_message
 * Function for all types of message partitioning
 * Adds a new specialist_petition agent to the xmachine_memory_specialist_petition_list list using a linear mapping
 * @param agents	xmachine_memory_specialist_petition_list agent list
 * @param id	message variable of type unsigned int
 * @param priority	message variable of type unsigned int
 * @param specialist_no	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_specialist_petition_message(xmachine_message_specialist_petition_list* specialist_petition_messages, unsigned int id, unsigned int priority, unsigned int specialist_no);
 
/** get_first_specialist_petition_message
 * Get first message function for non partitioned (brute force) messages
 * @param specialist_petition_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_specialist_petition * get_first_specialist_petition_message(xmachine_message_specialist_petition_list* specialist_petition_messages);

/** get_next_specialist_petition_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param specialist_petition_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_specialist_petition * get_next_specialist_petition_message(xmachine_message_specialist_petition* current, xmachine_message_specialist_petition_list* specialist_petition_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) doctor_reached message implemented in FLAMEGPU_Kernels */

/** add_doctor_reached_message
 * Function for all types of message partitioning
 * Adds a new doctor_reached agent to the xmachine_memory_doctor_reached_list list using a linear mapping
 * @param agents	xmachine_memory_doctor_reached_list agent list
 * @param id	message variable of type unsigned int
 * @param doctor_no	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_doctor_reached_message(xmachine_message_doctor_reached_list* doctor_reached_messages, unsigned int id, unsigned int doctor_no);
 
/** get_first_doctor_reached_message
 * Get first message function for non partitioned (brute force) messages
 * @param doctor_reached_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_doctor_reached * get_first_doctor_reached_message(xmachine_message_doctor_reached_list* doctor_reached_messages);

/** get_next_doctor_reached_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param doctor_reached_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_doctor_reached * get_next_doctor_reached_message(xmachine_message_doctor_reached* current, xmachine_message_doctor_reached_list* doctor_reached_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) free_doctor message implemented in FLAMEGPU_Kernels */

/** add_free_doctor_message
 * Function for all types of message partitioning
 * Adds a new free_doctor agent to the xmachine_memory_free_doctor_list list using a linear mapping
 * @param agents	xmachine_memory_free_doctor_list agent list
 * @param doctor_no	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_free_doctor_message(xmachine_message_free_doctor_list* free_doctor_messages, unsigned int doctor_no);
 
/** get_first_free_doctor_message
 * Get first message function for non partitioned (brute force) messages
 * @param free_doctor_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_free_doctor * get_first_free_doctor_message(xmachine_message_free_doctor_list* free_doctor_messages);

/** get_next_free_doctor_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param free_doctor_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_free_doctor * get_next_free_doctor_message(xmachine_message_free_doctor* current, xmachine_message_free_doctor_list* free_doctor_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) attention_terminated message implemented in FLAMEGPU_Kernels */

/** add_attention_terminated_message
 * Function for all types of message partitioning
 * Adds a new attention_terminated agent to the xmachine_memory_attention_terminated_list list using a linear mapping
 * @param agents	xmachine_memory_attention_terminated_list agent list
 * @param id	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_attention_terminated_message(xmachine_message_attention_terminated_list* attention_terminated_messages, unsigned int id);
 
/** get_first_attention_terminated_message
 * Get first message function for non partitioned (brute force) messages
 * @param attention_terminated_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_attention_terminated * get_first_attention_terminated_message(xmachine_message_attention_terminated_list* attention_terminated_messages);

/** get_next_attention_terminated_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param attention_terminated_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_attention_terminated * get_next_attention_terminated_message(xmachine_message_attention_terminated* current, xmachine_message_attention_terminated_list* attention_terminated_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) doctor_petition message implemented in FLAMEGPU_Kernels */

/** add_doctor_petition_message
 * Function for all types of message partitioning
 * Adds a new doctor_petition agent to the xmachine_memory_doctor_petition_list list using a linear mapping
 * @param agents	xmachine_memory_doctor_petition_list agent list
 * @param id	message variable of type unsigned int
 * @param doctor_no	message variable of type unsigned int
 * @param priority	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_doctor_petition_message(xmachine_message_doctor_petition_list* doctor_petition_messages, unsigned int id, unsigned int doctor_no, unsigned int priority);
 
/** get_first_doctor_petition_message
 * Get first message function for non partitioned (brute force) messages
 * @param doctor_petition_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_doctor_petition * get_first_doctor_petition_message(xmachine_message_doctor_petition_list* doctor_petition_messages);

/** get_next_doctor_petition_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param doctor_petition_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_doctor_petition * get_next_doctor_petition_message(xmachine_message_doctor_petition* current, xmachine_message_doctor_petition_list* doctor_petition_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) doctor_response message implemented in FLAMEGPU_Kernels */

/** add_doctor_response_message
 * Function for all types of message partitioning
 * Adds a new doctor_response agent to the xmachine_memory_doctor_response_list list using a linear mapping
 * @param agents	xmachine_memory_doctor_response_list agent list
 * @param id	message variable of type unsigned int
 * @param doctor_no	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_doctor_response_message(xmachine_message_doctor_response_list* doctor_response_messages, unsigned int id, int doctor_no);
 
/** get_first_doctor_response_message
 * Get first message function for non partitioned (brute force) messages
 * @param doctor_response_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_doctor_response * get_first_doctor_response_message(xmachine_message_doctor_response_list* doctor_response_messages);

/** get_next_doctor_response_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param doctor_response_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_doctor_response * get_next_doctor_response_message(xmachine_message_doctor_response* current, xmachine_message_doctor_response_list* doctor_response_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) specialist_response message implemented in FLAMEGPU_Kernels */

/** add_specialist_response_message
 * Function for all types of message partitioning
 * Adds a new specialist_response agent to the xmachine_memory_specialist_response_list list using a linear mapping
 * @param agents	xmachine_memory_specialist_response_list agent list
 * @param id	message variable of type unsigned int
 * @param specialist_ready	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_specialist_response_message(xmachine_message_specialist_response_list* specialist_response_messages, unsigned int id, int specialist_ready);
 
/** get_first_specialist_response_message
 * Get first message function for non partitioned (brute force) messages
 * @param specialist_response_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_specialist_response * get_first_specialist_response_message(xmachine_message_specialist_response_list* specialist_response_messages);

/** get_next_specialist_response_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param specialist_response_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_specialist_response * get_next_specialist_response_message(xmachine_message_specialist_response* current, xmachine_message_specialist_response_list* specialist_response_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) triage_petition message implemented in FLAMEGPU_Kernels */

/** add_triage_petition_message
 * Function for all types of message partitioning
 * Adds a new triage_petition agent to the xmachine_memory_triage_petition_list list using a linear mapping
 * @param agents	xmachine_memory_triage_petition_list agent list
 * @param id	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_triage_petition_message(xmachine_message_triage_petition_list* triage_petition_messages, unsigned int id);
 
/** get_first_triage_petition_message
 * Get first message function for non partitioned (brute force) messages
 * @param triage_petition_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_triage_petition * get_first_triage_petition_message(xmachine_message_triage_petition_list* triage_petition_messages);

/** get_next_triage_petition_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param triage_petition_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_triage_petition * get_next_triage_petition_message(xmachine_message_triage_petition* current, xmachine_message_triage_petition_list* triage_petition_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) triage_response message implemented in FLAMEGPU_Kernels */

/** add_triage_response_message
 * Function for all types of message partitioning
 * Adds a new triage_response agent to the xmachine_memory_triage_response_list list using a linear mapping
 * @param agents	xmachine_memory_triage_response_list agent list
 * @param id	message variable of type unsigned int
 * @param box_no	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_triage_response_message(xmachine_message_triage_response_list* triage_response_messages, unsigned int id, int box_no);
 
/** get_first_triage_response_message
 * Get first message function for non partitioned (brute force) messages
 * @param triage_response_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_triage_response * get_first_triage_response_message(xmachine_message_triage_response_list* triage_response_messages);

/** get_next_triage_response_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param triage_response_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_triage_response * get_next_triage_response_message(xmachine_message_triage_response* current, xmachine_message_triage_response_list* triage_response_messages);

  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_agent_agent
 * Adds a new continuous valued agent agent to the xmachine_memory_agent_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_agent_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param velx	agent agent variable of type float
 * @param vely	agent agent variable of type float
 * @param steer_x	agent agent variable of type float
 * @param steer_y	agent agent variable of type float
 * @param height	agent agent variable of type float
 * @param exit_no	agent agent variable of type int
 * @param speed	agent agent variable of type float
 * @param lod	agent agent variable of type int
 * @param animate	agent agent variable of type float
 * @param animate_dir	agent agent variable of type int
 * @param estado	agent agent variable of type int
 * @param tick	agent agent variable of type int
 * @param estado_movimiento	agent agent variable of type unsigned int
 * @param go_to_x	agent agent variable of type unsigned int
 * @param go_to_y	agent agent variable of type unsigned int
 * @param checkpoint	agent agent variable of type unsigned int
 * @param chair_no	agent agent variable of type int
 * @param box_no	agent agent variable of type unsigned int
 * @param doctor_no	agent agent variable of type unsigned int
 * @param specialist_no	agent agent variable of type unsigned int
 * @param priority	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_agent_agent(xmachine_memory_agent_list* agents, unsigned int id, float x, float y, float velx, float vely, float steer_x, float steer_y, float height, int exit_no, float speed, int lod, float animate, int animate_dir, int estado, int tick, unsigned int estado_movimiento, unsigned int go_to_x, unsigned int go_to_y, unsigned int checkpoint, int chair_no, unsigned int box_no, unsigned int doctor_no, unsigned int specialist_no, unsigned int priority);

/** add_chair_agent
 * Adds a new continuous valued chair agent to the xmachine_memory_chair_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_chair_list agent list
 * @param id	agent agent variable of type int
 * @param x	agent agent variable of type int
 * @param y	agent agent variable of type int
 * @param state	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_chair_agent(xmachine_memory_chair_list* agents, int id, int x, int y, int state);

/** add_doctor_manager_agent
 * Adds a new continuous valued doctor_manager agent to the xmachine_memory_doctor_manager_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_doctor_manager_list agent list
 * @param tick	agent agent variable of type unsigned int
 * @param rear	agent agent variable of type unsigned int
 * @param size	agent agent variable of type unsigned int
 * @param free_doctors	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_doctor_manager_agent(xmachine_memory_doctor_manager_list* agents, unsigned int tick, unsigned int rear, unsigned int size, unsigned int free_doctors);

/** get_doctor_manager_agent_array_value
 *  Template function for accessing doctor_manager agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_doctor_manager_agent_array_value(T *array, unsigned int index);

/** set_doctor_manager_agent_array_value
 *  Template function for setting doctor_manager agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_doctor_manager_agent_array_value(T *array, unsigned int index, T value);


  

/** add_specialist_manager_agent
 * Adds a new continuous valued specialist_manager agent to the xmachine_memory_specialist_manager_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_specialist_manager_list agent list
 * @param id	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_specialist_manager_agent(xmachine_memory_specialist_manager_list* agents, unsigned int id);

/** get_specialist_manager_agent_array_value
 *  Template function for accessing specialist_manager agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_specialist_manager_agent_array_value(T *array, unsigned int index);

/** set_specialist_manager_agent_array_value
 *  Template function for setting specialist_manager agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_specialist_manager_agent_array_value(T *array, unsigned int index, T value);


  

/** add_specialist_agent
 * Adds a new continuous valued specialist agent to the xmachine_memory_specialist_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_specialist_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param current_patient	agent agent variable of type unsigned int
 * @param tick	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_specialist_agent(xmachine_memory_specialist_list* agents, unsigned int id, unsigned int current_patient, unsigned int tick);

/** add_receptionist_agent
 * Adds a new continuous valued receptionist agent to the xmachine_memory_receptionist_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_receptionist_list agent list
 * @param x	agent agent variable of type int
 * @param y	agent agent variable of type int
 * @param front	agent agent variable of type unsigned int
 * @param rear	agent agent variable of type unsigned int
 * @param size	agent agent variable of type unsigned int
 * @param tick	agent agent variable of type unsigned int
 * @param current_patient	agent agent variable of type int
 * @param attend_patient	agent agent variable of type int
 * @param estado	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_receptionist_agent(xmachine_memory_receptionist_list* agents, int x, int y, unsigned int front, unsigned int rear, unsigned int size, unsigned int tick, int current_patient, int attend_patient, int estado);

/** get_receptionist_agent_array_value
 *  Template function for accessing receptionist agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_receptionist_agent_array_value(T *array, unsigned int index);

/** set_receptionist_agent_array_value
 *  Template function for setting receptionist agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_receptionist_agent_array_value(T *array, unsigned int index, T value);


  

/** add_agent_generator_agent
 * Adds a new continuous valued agent_generator agent to the xmachine_memory_agent_generator_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_agent_generator_list agent list
 * @param chairs_generated	agent agent variable of type int
 * @param boxes_generated	agent agent variable of type int
 * @param doctors_generated	agent agent variable of type int
 * @param specialists_generated	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_agent_generator_agent(xmachine_memory_agent_generator_list* agents, int chairs_generated, int boxes_generated, int doctors_generated, int specialists_generated);

/** add_chair_admin_agent
 * Adds a new continuous valued chair_admin agent to the xmachine_memory_chair_admin_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_chair_admin_list agent list
 * @param id	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_chair_admin_agent(xmachine_memory_chair_admin_list* agents, unsigned int id);

/** get_chair_admin_agent_array_value
 *  Template function for accessing chair_admin agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_chair_admin_agent_array_value(T *array, unsigned int index);

/** set_chair_admin_agent_array_value
 *  Template function for setting chair_admin agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_chair_admin_agent_array_value(T *array, unsigned int index, T value);


  

/** add_box_agent
 * Adds a new continuous valued box agent to the xmachine_memory_box_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_box_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param attending	agent agent variable of type unsigned int
 * @param tick	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_box_agent(xmachine_memory_box_list* agents, unsigned int id, unsigned int attending, unsigned int tick);

/** add_doctor_agent
 * Adds a new continuous valued doctor agent to the xmachine_memory_doctor_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_doctor_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param current_patient	agent agent variable of type int
 * @param tick	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_doctor_agent(xmachine_memory_doctor_list* agents, unsigned int id, int current_patient, unsigned int tick);

/** add_triage_agent
 * Adds a new continuous valued triage agent to the xmachine_memory_triage_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_triage_list agent list
 * @param front	agent agent variable of type unsigned int
 * @param rear	agent agent variable of type unsigned int
 * @param size	agent agent variable of type unsigned int
 * @param tick	agent agent variable of type unsigned int
 */
__FLAME_GPU_FUNC__ void add_triage_agent(xmachine_memory_triage_list* agents, unsigned int front, unsigned int rear, unsigned int size, unsigned int tick);

/** get_triage_agent_array_value
 *  Template function for accessing triage agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_triage_agent_array_value(T *array, unsigned int index);

/** set_triage_agent_array_value
 *  Template function for setting triage agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_triage_agent_array_value(T *array, unsigned int index, T value);


  


/* Graph loading function prototypes implemented in io.cu */


  
/* Simulation function prototypes implemented in simulation.cu */
/** getIterationNumber
 *  Get the iteration number (host)
 */
extern unsigned int getIterationNumber();

/** initialise
 * Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
 * @param input        XML file path for agent initial configuration
 */
extern void initialise(char * input);

/** cleanup
 * Function cleans up any memory allocations on the host and device
 */
extern void cleanup();

/** singleIteration
 *	Performs a single iteration of the simulation. I.e. performs each agent function on each function layer in the correct order.
 */
extern void singleIteration();

/** saveIterationData
 * Reads the current agent data fromt he device and saves it to XML
 * @param	outputpath	file path to XML file used for output of agent data
 * @param	iteration_number
 * @param h_agents Pointer to agent list on the host
 * @param d_agents Pointer to agent list on the GPU device
 * @param h_xmachine_memory_agent_count Pointer to agent counter
 * @param h_navmaps Pointer to agent list on the host
 * @param d_navmaps Pointer to agent list on the GPU device
 * @param h_xmachine_memory_navmap_count Pointer to agent counter
 * @param h_chairs Pointer to agent list on the host
 * @param d_chairs Pointer to agent list on the GPU device
 * @param h_xmachine_memory_chair_count Pointer to agent counter
 * @param h_doctor_managers Pointer to agent list on the host
 * @param d_doctor_managers Pointer to agent list on the GPU device
 * @param h_xmachine_memory_doctor_manager_count Pointer to agent counter
 * @param h_specialist_managers Pointer to agent list on the host
 * @param d_specialist_managers Pointer to agent list on the GPU device
 * @param h_xmachine_memory_specialist_manager_count Pointer to agent counter
 * @param h_specialists Pointer to agent list on the host
 * @param d_specialists Pointer to agent list on the GPU device
 * @param h_xmachine_memory_specialist_count Pointer to agent counter
 * @param h_receptionists Pointer to agent list on the host
 * @param d_receptionists Pointer to agent list on the GPU device
 * @param h_xmachine_memory_receptionist_count Pointer to agent counter
 * @param h_agent_generators Pointer to agent list on the host
 * @param d_agent_generators Pointer to agent list on the GPU device
 * @param h_xmachine_memory_agent_generator_count Pointer to agent counter
 * @param h_chair_admins Pointer to agent list on the host
 * @param d_chair_admins Pointer to agent list on the GPU device
 * @param h_xmachine_memory_chair_admin_count Pointer to agent counter
 * @param h_boxs Pointer to agent list on the host
 * @param d_boxs Pointer to agent list on the GPU device
 * @param h_xmachine_memory_box_count Pointer to agent counter
 * @param h_doctors Pointer to agent list on the host
 * @param d_doctors Pointer to agent list on the GPU device
 * @param h_xmachine_memory_doctor_count Pointer to agent counter
 * @param h_triages Pointer to agent list on the host
 * @param d_triages Pointer to agent list on the GPU device
 * @param h_xmachine_memory_triage_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_agent_list* h_agents_default, xmachine_memory_agent_list* d_agents_default, int h_xmachine_memory_agent_default_count,xmachine_memory_navmap_list* h_navmaps_static, xmachine_memory_navmap_list* d_navmaps_static, int h_xmachine_memory_navmap_static_count,xmachine_memory_chair_list* h_chairs_defaultChair, xmachine_memory_chair_list* d_chairs_defaultChair, int h_xmachine_memory_chair_defaultChair_count,xmachine_memory_doctor_manager_list* h_doctor_managers_defaultDoctorManager, xmachine_memory_doctor_manager_list* d_doctor_managers_defaultDoctorManager, int h_xmachine_memory_doctor_manager_defaultDoctorManager_count,xmachine_memory_specialist_manager_list* h_specialist_managers_defaultSpecialistManager, xmachine_memory_specialist_manager_list* d_specialist_managers_defaultSpecialistManager, int h_xmachine_memory_specialist_manager_defaultSpecialistManager_count,xmachine_memory_specialist_list* h_specialists_defaultSpecialist, xmachine_memory_specialist_list* d_specialists_defaultSpecialist, int h_xmachine_memory_specialist_defaultSpecialist_count,xmachine_memory_receptionist_list* h_receptionists_defaultReceptionist, xmachine_memory_receptionist_list* d_receptionists_defaultReceptionist, int h_xmachine_memory_receptionist_defaultReceptionist_count,xmachine_memory_agent_generator_list* h_agent_generators_defaultGenerator, xmachine_memory_agent_generator_list* d_agent_generators_defaultGenerator, int h_xmachine_memory_agent_generator_defaultGenerator_count,xmachine_memory_chair_admin_list* h_chair_admins_defaultAdmin, xmachine_memory_chair_admin_list* d_chair_admins_defaultAdmin, int h_xmachine_memory_chair_admin_defaultAdmin_count,xmachine_memory_box_list* h_boxs_defaultBox, xmachine_memory_box_list* d_boxs_defaultBox, int h_xmachine_memory_box_defaultBox_count,xmachine_memory_doctor_list* h_doctors_defaultDoctor, xmachine_memory_doctor_list* d_doctors_defaultDoctor, int h_xmachine_memory_doctor_defaultDoctor_count,xmachine_memory_triage_list* h_triages_defaultTriage, xmachine_memory_triage_list* d_triages_defaultTriage, int h_xmachine_memory_triage_defaultTriage_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_agents Pointer to agent list on the host
 * @param h_xmachine_memory_agent_count Pointer to agent counter
 * @param h_navmaps Pointer to agent list on the host
 * @param h_xmachine_memory_navmap_count Pointer to agent counter
 * @param h_chairs Pointer to agent list on the host
 * @param h_xmachine_memory_chair_count Pointer to agent counter
 * @param h_doctor_managers Pointer to agent list on the host
 * @param h_xmachine_memory_doctor_manager_count Pointer to agent counter
 * @param h_specialist_managers Pointer to agent list on the host
 * @param h_xmachine_memory_specialist_manager_count Pointer to agent counter
 * @param h_specialists Pointer to agent list on the host
 * @param h_xmachine_memory_specialist_count Pointer to agent counter
 * @param h_receptionists Pointer to agent list on the host
 * @param h_xmachine_memory_receptionist_count Pointer to agent counter
 * @param h_agent_generators Pointer to agent list on the host
 * @param h_xmachine_memory_agent_generator_count Pointer to agent counter
 * @param h_chair_admins Pointer to agent list on the host
 * @param h_xmachine_memory_chair_admin_count Pointer to agent counter
 * @param h_boxs Pointer to agent list on the host
 * @param h_xmachine_memory_box_count Pointer to agent counter
 * @param h_doctors Pointer to agent list on the host
 * @param h_xmachine_memory_doctor_count Pointer to agent counter
 * @param h_triages Pointer to agent list on the host
 * @param h_xmachine_memory_triage_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_agent_list* h_agents, int* h_xmachine_memory_agent_count,xmachine_memory_navmap_list* h_navmaps, int* h_xmachine_memory_navmap_count,xmachine_memory_chair_list* h_chairs, int* h_xmachine_memory_chair_count,xmachine_memory_doctor_manager_list* h_doctor_managers, int* h_xmachine_memory_doctor_manager_count,xmachine_memory_specialist_manager_list* h_specialist_managers, int* h_xmachine_memory_specialist_manager_count,xmachine_memory_specialist_list* h_specialists, int* h_xmachine_memory_specialist_count,xmachine_memory_receptionist_list* h_receptionists, int* h_xmachine_memory_receptionist_count,xmachine_memory_agent_generator_list* h_agent_generators, int* h_xmachine_memory_agent_generator_count,xmachine_memory_chair_admin_list* h_chair_admins, int* h_xmachine_memory_chair_admin_count,xmachine_memory_box_list* h_boxs, int* h_xmachine_memory_box_count,xmachine_memory_doctor_list* h_doctors, int* h_xmachine_memory_doctor_count,xmachine_memory_triage_list* h_triages, int* h_xmachine_memory_triage_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_agent_MAX_count
 * Gets the max agent count for the agent agent type 
 * @return		the maximum agent agent count
 */
extern int get_agent_agent_MAX_count();



/** get_agent_agent_default_count
 * Gets the agent count for the agent agent type in state default
 * @return		the current agent agent count in state default
 */
extern int get_agent_agent_default_count();

/** reset_default_count
 * Resets the agent count of the agent in state default to 0. This is useful for interacting with some visualisations.
 */
extern void reset_agent_default_count();

/** get_device_agent_default_agents
 * Gets a pointer to xmachine_memory_agent_list on the GPU device
 * @return		a xmachine_memory_agent_list on the GPU device
 */
extern xmachine_memory_agent_list* get_device_agent_default_agents();

/** get_host_agent_default_agents
 * Gets a pointer to xmachine_memory_agent_list on the CPU host
 * @return		a xmachine_memory_agent_list on the CPU host
 */
extern xmachine_memory_agent_list* get_host_agent_default_agents();


/** sort_agents_default
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_agents_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_agent_list* agents));


    
/** get_agent_navmap_MAX_count
 * Gets the max agent count for the navmap agent type 
 * @return		the maximum navmap agent count
 */
extern int get_agent_navmap_MAX_count();



/** get_agent_navmap_static_count
 * Gets the agent count for the navmap agent type in state static
 * @return		the current navmap agent count in state static
 */
extern int get_agent_navmap_static_count();

/** reset_static_count
 * Resets the agent count of the navmap in state static to 0. This is useful for interacting with some visualisations.
 */
extern void reset_navmap_static_count();

/** get_device_navmap_static_agents
 * Gets a pointer to xmachine_memory_navmap_list on the GPU device
 * @return		a xmachine_memory_navmap_list on the GPU device
 */
extern xmachine_memory_navmap_list* get_device_navmap_static_agents();

/** get_host_navmap_static_agents
 * Gets a pointer to xmachine_memory_navmap_list on the CPU host
 * @return		a xmachine_memory_navmap_list on the CPU host
 */
extern xmachine_memory_navmap_list* get_host_navmap_static_agents();


/** get_navmap_population_width
 * Gets an int value representing the xmachine_memory_navmap population width.
 * @return		xmachine_memory_navmap population width
 */
extern int get_navmap_population_width();

    
/** get_agent_chair_MAX_count
 * Gets the max agent count for the chair agent type 
 * @return		the maximum chair agent count
 */
extern int get_agent_chair_MAX_count();



/** get_agent_chair_defaultChair_count
 * Gets the agent count for the chair agent type in state defaultChair
 * @return		the current chair agent count in state defaultChair
 */
extern int get_agent_chair_defaultChair_count();

/** reset_defaultChair_count
 * Resets the agent count of the chair in state defaultChair to 0. This is useful for interacting with some visualisations.
 */
extern void reset_chair_defaultChair_count();

/** get_device_chair_defaultChair_agents
 * Gets a pointer to xmachine_memory_chair_list on the GPU device
 * @return		a xmachine_memory_chair_list on the GPU device
 */
extern xmachine_memory_chair_list* get_device_chair_defaultChair_agents();

/** get_host_chair_defaultChair_agents
 * Gets a pointer to xmachine_memory_chair_list on the CPU host
 * @return		a xmachine_memory_chair_list on the CPU host
 */
extern xmachine_memory_chair_list* get_host_chair_defaultChair_agents();


/** sort_chairs_defaultChair
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_chairs_defaultChair(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_chair_list* agents));


    
/** get_agent_doctor_manager_MAX_count
 * Gets the max agent count for the doctor_manager agent type 
 * @return		the maximum doctor_manager agent count
 */
extern int get_agent_doctor_manager_MAX_count();



/** get_agent_doctor_manager_defaultDoctorManager_count
 * Gets the agent count for the doctor_manager agent type in state defaultDoctorManager
 * @return		the current doctor_manager agent count in state defaultDoctorManager
 */
extern int get_agent_doctor_manager_defaultDoctorManager_count();

/** reset_defaultDoctorManager_count
 * Resets the agent count of the doctor_manager in state defaultDoctorManager to 0. This is useful for interacting with some visualisations.
 */
extern void reset_doctor_manager_defaultDoctorManager_count();

/** get_device_doctor_manager_defaultDoctorManager_agents
 * Gets a pointer to xmachine_memory_doctor_manager_list on the GPU device
 * @return		a xmachine_memory_doctor_manager_list on the GPU device
 */
extern xmachine_memory_doctor_manager_list* get_device_doctor_manager_defaultDoctorManager_agents();

/** get_host_doctor_manager_defaultDoctorManager_agents
 * Gets a pointer to xmachine_memory_doctor_manager_list on the CPU host
 * @return		a xmachine_memory_doctor_manager_list on the CPU host
 */
extern xmachine_memory_doctor_manager_list* get_host_doctor_manager_defaultDoctorManager_agents();


/** sort_doctor_managers_defaultDoctorManager
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_doctor_managers_defaultDoctorManager(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_doctor_manager_list* agents));


    
/** get_agent_specialist_manager_MAX_count
 * Gets the max agent count for the specialist_manager agent type 
 * @return		the maximum specialist_manager agent count
 */
extern int get_agent_specialist_manager_MAX_count();



/** get_agent_specialist_manager_defaultSpecialistManager_count
 * Gets the agent count for the specialist_manager agent type in state defaultSpecialistManager
 * @return		the current specialist_manager agent count in state defaultSpecialistManager
 */
extern int get_agent_specialist_manager_defaultSpecialistManager_count();

/** reset_defaultSpecialistManager_count
 * Resets the agent count of the specialist_manager in state defaultSpecialistManager to 0. This is useful for interacting with some visualisations.
 */
extern void reset_specialist_manager_defaultSpecialistManager_count();

/** get_device_specialist_manager_defaultSpecialistManager_agents
 * Gets a pointer to xmachine_memory_specialist_manager_list on the GPU device
 * @return		a xmachine_memory_specialist_manager_list on the GPU device
 */
extern xmachine_memory_specialist_manager_list* get_device_specialist_manager_defaultSpecialistManager_agents();

/** get_host_specialist_manager_defaultSpecialistManager_agents
 * Gets a pointer to xmachine_memory_specialist_manager_list on the CPU host
 * @return		a xmachine_memory_specialist_manager_list on the CPU host
 */
extern xmachine_memory_specialist_manager_list* get_host_specialist_manager_defaultSpecialistManager_agents();


/** sort_specialist_managers_defaultSpecialistManager
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_specialist_managers_defaultSpecialistManager(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_specialist_manager_list* agents));


    
/** get_agent_specialist_MAX_count
 * Gets the max agent count for the specialist agent type 
 * @return		the maximum specialist agent count
 */
extern int get_agent_specialist_MAX_count();



/** get_agent_specialist_defaultSpecialist_count
 * Gets the agent count for the specialist agent type in state defaultSpecialist
 * @return		the current specialist agent count in state defaultSpecialist
 */
extern int get_agent_specialist_defaultSpecialist_count();

/** reset_defaultSpecialist_count
 * Resets the agent count of the specialist in state defaultSpecialist to 0. This is useful for interacting with some visualisations.
 */
extern void reset_specialist_defaultSpecialist_count();

/** get_device_specialist_defaultSpecialist_agents
 * Gets a pointer to xmachine_memory_specialist_list on the GPU device
 * @return		a xmachine_memory_specialist_list on the GPU device
 */
extern xmachine_memory_specialist_list* get_device_specialist_defaultSpecialist_agents();

/** get_host_specialist_defaultSpecialist_agents
 * Gets a pointer to xmachine_memory_specialist_list on the CPU host
 * @return		a xmachine_memory_specialist_list on the CPU host
 */
extern xmachine_memory_specialist_list* get_host_specialist_defaultSpecialist_agents();


/** sort_specialists_defaultSpecialist
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_specialists_defaultSpecialist(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_specialist_list* agents));


    
/** get_agent_receptionist_MAX_count
 * Gets the max agent count for the receptionist agent type 
 * @return		the maximum receptionist agent count
 */
extern int get_agent_receptionist_MAX_count();



/** get_agent_receptionist_defaultReceptionist_count
 * Gets the agent count for the receptionist agent type in state defaultReceptionist
 * @return		the current receptionist agent count in state defaultReceptionist
 */
extern int get_agent_receptionist_defaultReceptionist_count();

/** reset_defaultReceptionist_count
 * Resets the agent count of the receptionist in state defaultReceptionist to 0. This is useful for interacting with some visualisations.
 */
extern void reset_receptionist_defaultReceptionist_count();

/** get_device_receptionist_defaultReceptionist_agents
 * Gets a pointer to xmachine_memory_receptionist_list on the GPU device
 * @return		a xmachine_memory_receptionist_list on the GPU device
 */
extern xmachine_memory_receptionist_list* get_device_receptionist_defaultReceptionist_agents();

/** get_host_receptionist_defaultReceptionist_agents
 * Gets a pointer to xmachine_memory_receptionist_list on the CPU host
 * @return		a xmachine_memory_receptionist_list on the CPU host
 */
extern xmachine_memory_receptionist_list* get_host_receptionist_defaultReceptionist_agents();


/** sort_receptionists_defaultReceptionist
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_receptionists_defaultReceptionist(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_receptionist_list* agents));


    
/** get_agent_agent_generator_MAX_count
 * Gets the max agent count for the agent_generator agent type 
 * @return		the maximum agent_generator agent count
 */
extern int get_agent_agent_generator_MAX_count();



/** get_agent_agent_generator_defaultGenerator_count
 * Gets the agent count for the agent_generator agent type in state defaultGenerator
 * @return		the current agent_generator agent count in state defaultGenerator
 */
extern int get_agent_agent_generator_defaultGenerator_count();

/** reset_defaultGenerator_count
 * Resets the agent count of the agent_generator in state defaultGenerator to 0. This is useful for interacting with some visualisations.
 */
extern void reset_agent_generator_defaultGenerator_count();

/** get_device_agent_generator_defaultGenerator_agents
 * Gets a pointer to xmachine_memory_agent_generator_list on the GPU device
 * @return		a xmachine_memory_agent_generator_list on the GPU device
 */
extern xmachine_memory_agent_generator_list* get_device_agent_generator_defaultGenerator_agents();

/** get_host_agent_generator_defaultGenerator_agents
 * Gets a pointer to xmachine_memory_agent_generator_list on the CPU host
 * @return		a xmachine_memory_agent_generator_list on the CPU host
 */
extern xmachine_memory_agent_generator_list* get_host_agent_generator_defaultGenerator_agents();


/** sort_agent_generators_defaultGenerator
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_agent_generators_defaultGenerator(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_agent_generator_list* agents));


    
/** get_agent_chair_admin_MAX_count
 * Gets the max agent count for the chair_admin agent type 
 * @return		the maximum chair_admin agent count
 */
extern int get_agent_chair_admin_MAX_count();



/** get_agent_chair_admin_defaultAdmin_count
 * Gets the agent count for the chair_admin agent type in state defaultAdmin
 * @return		the current chair_admin agent count in state defaultAdmin
 */
extern int get_agent_chair_admin_defaultAdmin_count();

/** reset_defaultAdmin_count
 * Resets the agent count of the chair_admin in state defaultAdmin to 0. This is useful for interacting with some visualisations.
 */
extern void reset_chair_admin_defaultAdmin_count();

/** get_device_chair_admin_defaultAdmin_agents
 * Gets a pointer to xmachine_memory_chair_admin_list on the GPU device
 * @return		a xmachine_memory_chair_admin_list on the GPU device
 */
extern xmachine_memory_chair_admin_list* get_device_chair_admin_defaultAdmin_agents();

/** get_host_chair_admin_defaultAdmin_agents
 * Gets a pointer to xmachine_memory_chair_admin_list on the CPU host
 * @return		a xmachine_memory_chair_admin_list on the CPU host
 */
extern xmachine_memory_chair_admin_list* get_host_chair_admin_defaultAdmin_agents();


/** sort_chair_admins_defaultAdmin
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_chair_admins_defaultAdmin(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_chair_admin_list* agents));


    
/** get_agent_box_MAX_count
 * Gets the max agent count for the box agent type 
 * @return		the maximum box agent count
 */
extern int get_agent_box_MAX_count();



/** get_agent_box_defaultBox_count
 * Gets the agent count for the box agent type in state defaultBox
 * @return		the current box agent count in state defaultBox
 */
extern int get_agent_box_defaultBox_count();

/** reset_defaultBox_count
 * Resets the agent count of the box in state defaultBox to 0. This is useful for interacting with some visualisations.
 */
extern void reset_box_defaultBox_count();

/** get_device_box_defaultBox_agents
 * Gets a pointer to xmachine_memory_box_list on the GPU device
 * @return		a xmachine_memory_box_list on the GPU device
 */
extern xmachine_memory_box_list* get_device_box_defaultBox_agents();

/** get_host_box_defaultBox_agents
 * Gets a pointer to xmachine_memory_box_list on the CPU host
 * @return		a xmachine_memory_box_list on the CPU host
 */
extern xmachine_memory_box_list* get_host_box_defaultBox_agents();


/** sort_boxs_defaultBox
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_boxs_defaultBox(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_box_list* agents));


    
/** get_agent_doctor_MAX_count
 * Gets the max agent count for the doctor agent type 
 * @return		the maximum doctor agent count
 */
extern int get_agent_doctor_MAX_count();



/** get_agent_doctor_defaultDoctor_count
 * Gets the agent count for the doctor agent type in state defaultDoctor
 * @return		the current doctor agent count in state defaultDoctor
 */
extern int get_agent_doctor_defaultDoctor_count();

/** reset_defaultDoctor_count
 * Resets the agent count of the doctor in state defaultDoctor to 0. This is useful for interacting with some visualisations.
 */
extern void reset_doctor_defaultDoctor_count();

/** get_device_doctor_defaultDoctor_agents
 * Gets a pointer to xmachine_memory_doctor_list on the GPU device
 * @return		a xmachine_memory_doctor_list on the GPU device
 */
extern xmachine_memory_doctor_list* get_device_doctor_defaultDoctor_agents();

/** get_host_doctor_defaultDoctor_agents
 * Gets a pointer to xmachine_memory_doctor_list on the CPU host
 * @return		a xmachine_memory_doctor_list on the CPU host
 */
extern xmachine_memory_doctor_list* get_host_doctor_defaultDoctor_agents();


/** sort_doctors_defaultDoctor
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_doctors_defaultDoctor(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_doctor_list* agents));


    
/** get_agent_triage_MAX_count
 * Gets the max agent count for the triage agent type 
 * @return		the maximum triage agent count
 */
extern int get_agent_triage_MAX_count();



/** get_agent_triage_defaultTriage_count
 * Gets the agent count for the triage agent type in state defaultTriage
 * @return		the current triage agent count in state defaultTriage
 */
extern int get_agent_triage_defaultTriage_count();

/** reset_defaultTriage_count
 * Resets the agent count of the triage in state defaultTriage to 0. This is useful for interacting with some visualisations.
 */
extern void reset_triage_defaultTriage_count();

/** get_device_triage_defaultTriage_agents
 * Gets a pointer to xmachine_memory_triage_list on the GPU device
 * @return		a xmachine_memory_triage_list on the GPU device
 */
extern xmachine_memory_triage_list* get_device_triage_defaultTriage_agents();

/** get_host_triage_defaultTriage_agents
 * Gets a pointer to xmachine_memory_triage_list on the CPU host
 * @return		a xmachine_memory_triage_list on the CPU host
 */
extern xmachine_memory_triage_list* get_host_triage_defaultTriage_agents();


/** sort_triages_defaultTriage
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_triages_defaultTriage(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_triage_list* agents));



/* Host based access of agent variables*/

/** unsigned int get_agent_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_agent_default_variable_id(unsigned int index);

/** float get_agent_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_agent_default_variable_x(unsigned int index);

/** float get_agent_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_agent_default_variable_y(unsigned int index);

/** float get_agent_default_variable_velx(unsigned int index)
 * Gets the value of the velx variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable velx
 */
__host__ float get_agent_default_variable_velx(unsigned int index);

/** float get_agent_default_variable_vely(unsigned int index)
 * Gets the value of the vely variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable vely
 */
__host__ float get_agent_default_variable_vely(unsigned int index);

/** float get_agent_default_variable_steer_x(unsigned int index)
 * Gets the value of the steer_x variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable steer_x
 */
__host__ float get_agent_default_variable_steer_x(unsigned int index);

/** float get_agent_default_variable_steer_y(unsigned int index)
 * Gets the value of the steer_y variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable steer_y
 */
__host__ float get_agent_default_variable_steer_y(unsigned int index);

/** float get_agent_default_variable_height(unsigned int index)
 * Gets the value of the height variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable height
 */
__host__ float get_agent_default_variable_height(unsigned int index);

/** int get_agent_default_variable_exit_no(unsigned int index)
 * Gets the value of the exit_no variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit_no
 */
__host__ int get_agent_default_variable_exit_no(unsigned int index);

/** float get_agent_default_variable_speed(unsigned int index)
 * Gets the value of the speed variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable speed
 */
__host__ float get_agent_default_variable_speed(unsigned int index);

/** int get_agent_default_variable_lod(unsigned int index)
 * Gets the value of the lod variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable lod
 */
__host__ int get_agent_default_variable_lod(unsigned int index);

/** float get_agent_default_variable_animate(unsigned int index)
 * Gets the value of the animate variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable animate
 */
__host__ float get_agent_default_variable_animate(unsigned int index);

/** int get_agent_default_variable_animate_dir(unsigned int index)
 * Gets the value of the animate_dir variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable animate_dir
 */
__host__ int get_agent_default_variable_animate_dir(unsigned int index);

/** int get_agent_default_variable_estado(unsigned int index)
 * Gets the value of the estado variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable estado
 */
__host__ int get_agent_default_variable_estado(unsigned int index);

/** int get_agent_default_variable_tick(unsigned int index)
 * Gets the value of the tick variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable tick
 */
__host__ int get_agent_default_variable_tick(unsigned int index);

/** unsigned int get_agent_default_variable_estado_movimiento(unsigned int index)
 * Gets the value of the estado_movimiento variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable estado_movimiento
 */
__host__ unsigned int get_agent_default_variable_estado_movimiento(unsigned int index);

/** unsigned int get_agent_default_variable_go_to_x(unsigned int index)
 * Gets the value of the go_to_x variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable go_to_x
 */
__host__ unsigned int get_agent_default_variable_go_to_x(unsigned int index);

/** unsigned int get_agent_default_variable_go_to_y(unsigned int index)
 * Gets the value of the go_to_y variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable go_to_y
 */
__host__ unsigned int get_agent_default_variable_go_to_y(unsigned int index);

/** unsigned int get_agent_default_variable_checkpoint(unsigned int index)
 * Gets the value of the checkpoint variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable checkpoint
 */
__host__ unsigned int get_agent_default_variable_checkpoint(unsigned int index);

/** int get_agent_default_variable_chair_no(unsigned int index)
 * Gets the value of the chair_no variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable chair_no
 */
__host__ int get_agent_default_variable_chair_no(unsigned int index);

/** unsigned int get_agent_default_variable_box_no(unsigned int index)
 * Gets the value of the box_no variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable box_no
 */
__host__ unsigned int get_agent_default_variable_box_no(unsigned int index);

/** unsigned int get_agent_default_variable_doctor_no(unsigned int index)
 * Gets the value of the doctor_no variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable doctor_no
 */
__host__ unsigned int get_agent_default_variable_doctor_no(unsigned int index);

/** unsigned int get_agent_default_variable_specialist_no(unsigned int index)
 * Gets the value of the specialist_no variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable specialist_no
 */
__host__ unsigned int get_agent_default_variable_specialist_no(unsigned int index);

/** unsigned int get_agent_default_variable_priority(unsigned int index)
 * Gets the value of the priority variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable priority
 */
__host__ unsigned int get_agent_default_variable_priority(unsigned int index);

/** int get_navmap_static_variable_x(unsigned int index)
 * Gets the value of the x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_navmap_static_variable_x(unsigned int index);

/** int get_navmap_static_variable_y(unsigned int index)
 * Gets the value of the y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_navmap_static_variable_y(unsigned int index);

/** int get_navmap_static_variable_exit_no(unsigned int index)
 * Gets the value of the exit_no variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit_no
 */
__host__ int get_navmap_static_variable_exit_no(unsigned int index);

/** float get_navmap_static_variable_height(unsigned int index)
 * Gets the value of the height variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable height
 */
__host__ float get_navmap_static_variable_height(unsigned int index);

/** float get_navmap_static_variable_collision_x(unsigned int index)
 * Gets the value of the collision_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable collision_x
 */
__host__ float get_navmap_static_variable_collision_x(unsigned int index);

/** float get_navmap_static_variable_collision_y(unsigned int index)
 * Gets the value of the collision_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable collision_y
 */
__host__ float get_navmap_static_variable_collision_y(unsigned int index);

/** float get_navmap_static_variable_exit0_x(unsigned int index)
 * Gets the value of the exit0_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit0_x
 */
__host__ float get_navmap_static_variable_exit0_x(unsigned int index);

/** float get_navmap_static_variable_exit0_y(unsigned int index)
 * Gets the value of the exit0_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit0_y
 */
__host__ float get_navmap_static_variable_exit0_y(unsigned int index);

/** float get_navmap_static_variable_exit1_x(unsigned int index)
 * Gets the value of the exit1_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit1_x
 */
__host__ float get_navmap_static_variable_exit1_x(unsigned int index);

/** float get_navmap_static_variable_exit1_y(unsigned int index)
 * Gets the value of the exit1_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit1_y
 */
__host__ float get_navmap_static_variable_exit1_y(unsigned int index);

/** float get_navmap_static_variable_exit2_x(unsigned int index)
 * Gets the value of the exit2_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit2_x
 */
__host__ float get_navmap_static_variable_exit2_x(unsigned int index);

/** float get_navmap_static_variable_exit2_y(unsigned int index)
 * Gets the value of the exit2_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit2_y
 */
__host__ float get_navmap_static_variable_exit2_y(unsigned int index);

/** float get_navmap_static_variable_exit3_x(unsigned int index)
 * Gets the value of the exit3_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit3_x
 */
__host__ float get_navmap_static_variable_exit3_x(unsigned int index);

/** float get_navmap_static_variable_exit3_y(unsigned int index)
 * Gets the value of the exit3_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit3_y
 */
__host__ float get_navmap_static_variable_exit3_y(unsigned int index);

/** float get_navmap_static_variable_exit4_x(unsigned int index)
 * Gets the value of the exit4_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit4_x
 */
__host__ float get_navmap_static_variable_exit4_x(unsigned int index);

/** float get_navmap_static_variable_exit4_y(unsigned int index)
 * Gets the value of the exit4_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit4_y
 */
__host__ float get_navmap_static_variable_exit4_y(unsigned int index);

/** float get_navmap_static_variable_exit5_x(unsigned int index)
 * Gets the value of the exit5_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit5_x
 */
__host__ float get_navmap_static_variable_exit5_x(unsigned int index);

/** float get_navmap_static_variable_exit5_y(unsigned int index)
 * Gets the value of the exit5_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit5_y
 */
__host__ float get_navmap_static_variable_exit5_y(unsigned int index);

/** float get_navmap_static_variable_exit6_x(unsigned int index)
 * Gets the value of the exit6_x variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit6_x
 */
__host__ float get_navmap_static_variable_exit6_x(unsigned int index);

/** float get_navmap_static_variable_exit6_y(unsigned int index)
 * Gets the value of the exit6_y variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable exit6_y
 */
__host__ float get_navmap_static_variable_exit6_y(unsigned int index);

/** unsigned int get_navmap_static_variable_cant_generados(unsigned int index)
 * Gets the value of the cant_generados variable of an navmap agent in the static state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable cant_generados
 */
__host__ unsigned int get_navmap_static_variable_cant_generados(unsigned int index);

/** int get_chair_defaultChair_variable_id(unsigned int index)
 * Gets the value of the id variable of an chair agent in the defaultChair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_chair_defaultChair_variable_id(unsigned int index);

/** int get_chair_defaultChair_variable_x(unsigned int index)
 * Gets the value of the x variable of an chair agent in the defaultChair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_chair_defaultChair_variable_x(unsigned int index);

/** int get_chair_defaultChair_variable_y(unsigned int index)
 * Gets the value of the y variable of an chair agent in the defaultChair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_chair_defaultChair_variable_y(unsigned int index);

/** int get_chair_defaultChair_variable_state(unsigned int index)
 * Gets the value of the state variable of an chair agent in the defaultChair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable state
 */
__host__ int get_chair_defaultChair_variable_state(unsigned int index);

/** unsigned int get_doctor_manager_defaultDoctorManager_variable_tick(unsigned int index)
 * Gets the value of the tick variable of an doctor_manager agent in the defaultDoctorManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable tick
 */
__host__ unsigned int get_doctor_manager_defaultDoctorManager_variable_tick(unsigned int index);

/** unsigned int get_doctor_manager_defaultDoctorManager_variable_rear(unsigned int index)
 * Gets the value of the rear variable of an doctor_manager agent in the defaultDoctorManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable rear
 */
__host__ unsigned int get_doctor_manager_defaultDoctorManager_variable_rear(unsigned int index);

/** unsigned int get_doctor_manager_defaultDoctorManager_variable_size(unsigned int index)
 * Gets the value of the size variable of an doctor_manager agent in the defaultDoctorManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable size
 */
__host__ unsigned int get_doctor_manager_defaultDoctorManager_variable_size(unsigned int index);

/** int get_doctor_manager_defaultDoctorManager_variable_doctors_occupied(unsigned int index, unsigned int element)
 * Gets the element-th value of the doctors_occupied variable array of an doctor_manager agent in the defaultDoctorManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable doctors_occupied
 */
__host__ int get_doctor_manager_defaultDoctorManager_variable_doctors_occupied(unsigned int index, unsigned int element);

/** unsigned int get_doctor_manager_defaultDoctorManager_variable_free_doctors(unsigned int index)
 * Gets the value of the free_doctors variable of an doctor_manager agent in the defaultDoctorManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable free_doctors
 */
__host__ unsigned int get_doctor_manager_defaultDoctorManager_variable_free_doctors(unsigned int index);

/** ivec2 get_doctor_manager_defaultDoctorManager_variable_patientQueue(unsigned int index, unsigned int element)
 * Gets the element-th value of the patientQueue variable array of an doctor_manager agent in the defaultDoctorManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable patientQueue
 */
__host__ ivec2 get_doctor_manager_defaultDoctorManager_variable_patientQueue(unsigned int index, unsigned int element);

/** unsigned int get_specialist_manager_defaultSpecialistManager_variable_id(unsigned int index)
 * Gets the value of the id variable of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_specialist_manager_defaultSpecialistManager_variable_id(unsigned int index);

/** unsigned int get_specialist_manager_defaultSpecialistManager_variable_tick(unsigned int index, unsigned int element)
 * Gets the element-th value of the tick variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable tick
 */
__host__ unsigned int get_specialist_manager_defaultSpecialistManager_variable_tick(unsigned int index, unsigned int element);

/** unsigned int get_specialist_manager_defaultSpecialistManager_variable_free_specialist(unsigned int index, unsigned int element)
 * Gets the element-th value of the free_specialist variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable free_specialist
 */
__host__ unsigned int get_specialist_manager_defaultSpecialistManager_variable_free_specialist(unsigned int index, unsigned int element);

/** unsigned int get_specialist_manager_defaultSpecialistManager_variable_rear(unsigned int index, unsigned int element)
 * Gets the element-th value of the rear variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable rear
 */
__host__ unsigned int get_specialist_manager_defaultSpecialistManager_variable_rear(unsigned int index, unsigned int element);

/** unsigned int get_specialist_manager_defaultSpecialistManager_variable_size(unsigned int index, unsigned int element)
 * Gets the element-th value of the size variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable size
 */
__host__ unsigned int get_specialist_manager_defaultSpecialistManager_variable_size(unsigned int index, unsigned int element);

/** ivec2 get_specialist_manager_defaultSpecialistManager_variable_surgicalQueue(unsigned int index, unsigned int element)
 * Gets the element-th value of the surgicalQueue variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable surgicalQueue
 */
__host__ ivec2 get_specialist_manager_defaultSpecialistManager_variable_surgicalQueue(unsigned int index, unsigned int element);

/** ivec2 get_specialist_manager_defaultSpecialistManager_variable_pediatricsQueue(unsigned int index, unsigned int element)
 * Gets the element-th value of the pediatricsQueue variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable pediatricsQueue
 */
__host__ ivec2 get_specialist_manager_defaultSpecialistManager_variable_pediatricsQueue(unsigned int index, unsigned int element);

/** ivec2 get_specialist_manager_defaultSpecialistManager_variable_gynecologistQueue(unsigned int index, unsigned int element)
 * Gets the element-th value of the gynecologistQueue variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable gynecologistQueue
 */
__host__ ivec2 get_specialist_manager_defaultSpecialistManager_variable_gynecologistQueue(unsigned int index, unsigned int element);

/** ivec2 get_specialist_manager_defaultSpecialistManager_variable_geriatricsQueue(unsigned int index, unsigned int element)
 * Gets the element-th value of the geriatricsQueue variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable geriatricsQueue
 */
__host__ ivec2 get_specialist_manager_defaultSpecialistManager_variable_geriatricsQueue(unsigned int index, unsigned int element);

/** ivec2 get_specialist_manager_defaultSpecialistManager_variable_psychiatristQueue(unsigned int index, unsigned int element)
 * Gets the element-th value of the psychiatristQueue variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable psychiatristQueue
 */
__host__ ivec2 get_specialist_manager_defaultSpecialistManager_variable_psychiatristQueue(unsigned int index, unsigned int element);

/** unsigned int get_specialist_defaultSpecialist_variable_id(unsigned int index)
 * Gets the value of the id variable of an specialist agent in the defaultSpecialist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_specialist_defaultSpecialist_variable_id(unsigned int index);

/** unsigned int get_specialist_defaultSpecialist_variable_current_patient(unsigned int index)
 * Gets the value of the current_patient variable of an specialist agent in the defaultSpecialist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_patient
 */
__host__ unsigned int get_specialist_defaultSpecialist_variable_current_patient(unsigned int index);

/** unsigned int get_specialist_defaultSpecialist_variable_tick(unsigned int index)
 * Gets the value of the tick variable of an specialist agent in the defaultSpecialist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable tick
 */
__host__ unsigned int get_specialist_defaultSpecialist_variable_tick(unsigned int index);

/** int get_receptionist_defaultReceptionist_variable_x(unsigned int index)
 * Gets the value of the x variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_receptionist_defaultReceptionist_variable_x(unsigned int index);

/** int get_receptionist_defaultReceptionist_variable_y(unsigned int index)
 * Gets the value of the y variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_receptionist_defaultReceptionist_variable_y(unsigned int index);

/** unsigned int get_receptionist_defaultReceptionist_variable_patientQueue(unsigned int index, unsigned int element)
 * Gets the element-th value of the patientQueue variable array of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable patientQueue
 */
__host__ unsigned int get_receptionist_defaultReceptionist_variable_patientQueue(unsigned int index, unsigned int element);

/** unsigned int get_receptionist_defaultReceptionist_variable_front(unsigned int index)
 * Gets the value of the front variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable front
 */
__host__ unsigned int get_receptionist_defaultReceptionist_variable_front(unsigned int index);

/** unsigned int get_receptionist_defaultReceptionist_variable_rear(unsigned int index)
 * Gets the value of the rear variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable rear
 */
__host__ unsigned int get_receptionist_defaultReceptionist_variable_rear(unsigned int index);

/** unsigned int get_receptionist_defaultReceptionist_variable_size(unsigned int index)
 * Gets the value of the size variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable size
 */
__host__ unsigned int get_receptionist_defaultReceptionist_variable_size(unsigned int index);

/** unsigned int get_receptionist_defaultReceptionist_variable_tick(unsigned int index)
 * Gets the value of the tick variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable tick
 */
__host__ unsigned int get_receptionist_defaultReceptionist_variable_tick(unsigned int index);

/** int get_receptionist_defaultReceptionist_variable_current_patient(unsigned int index)
 * Gets the value of the current_patient variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_patient
 */
__host__ int get_receptionist_defaultReceptionist_variable_current_patient(unsigned int index);

/** int get_receptionist_defaultReceptionist_variable_attend_patient(unsigned int index)
 * Gets the value of the attend_patient variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable attend_patient
 */
__host__ int get_receptionist_defaultReceptionist_variable_attend_patient(unsigned int index);

/** int get_receptionist_defaultReceptionist_variable_estado(unsigned int index)
 * Gets the value of the estado variable of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable estado
 */
__host__ int get_receptionist_defaultReceptionist_variable_estado(unsigned int index);

/** int get_agent_generator_defaultGenerator_variable_chairs_generated(unsigned int index)
 * Gets the value of the chairs_generated variable of an agent_generator agent in the defaultGenerator state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable chairs_generated
 */
__host__ int get_agent_generator_defaultGenerator_variable_chairs_generated(unsigned int index);

/** int get_agent_generator_defaultGenerator_variable_boxes_generated(unsigned int index)
 * Gets the value of the boxes_generated variable of an agent_generator agent in the defaultGenerator state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable boxes_generated
 */
__host__ int get_agent_generator_defaultGenerator_variable_boxes_generated(unsigned int index);

/** int get_agent_generator_defaultGenerator_variable_doctors_generated(unsigned int index)
 * Gets the value of the doctors_generated variable of an agent_generator agent in the defaultGenerator state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable doctors_generated
 */
__host__ int get_agent_generator_defaultGenerator_variable_doctors_generated(unsigned int index);

/** int get_agent_generator_defaultGenerator_variable_specialists_generated(unsigned int index)
 * Gets the value of the specialists_generated variable of an agent_generator agent in the defaultGenerator state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable specialists_generated
 */
__host__ int get_agent_generator_defaultGenerator_variable_specialists_generated(unsigned int index);

/** unsigned int get_chair_admin_defaultAdmin_variable_id(unsigned int index)
 * Gets the value of the id variable of an chair_admin agent in the defaultAdmin state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_chair_admin_defaultAdmin_variable_id(unsigned int index);

/** unsigned int get_chair_admin_defaultAdmin_variable_chairArray(unsigned int index, unsigned int element)
 * Gets the element-th value of the chairArray variable array of an chair_admin agent in the defaultAdmin state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable chairArray
 */
__host__ unsigned int get_chair_admin_defaultAdmin_variable_chairArray(unsigned int index, unsigned int element);

/** unsigned int get_box_defaultBox_variable_id(unsigned int index)
 * Gets the value of the id variable of an box agent in the defaultBox state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_box_defaultBox_variable_id(unsigned int index);

/** unsigned int get_box_defaultBox_variable_attending(unsigned int index)
 * Gets the value of the attending variable of an box agent in the defaultBox state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable attending
 */
__host__ unsigned int get_box_defaultBox_variable_attending(unsigned int index);

/** unsigned int get_box_defaultBox_variable_tick(unsigned int index)
 * Gets the value of the tick variable of an box agent in the defaultBox state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable tick
 */
__host__ unsigned int get_box_defaultBox_variable_tick(unsigned int index);

/** unsigned int get_doctor_defaultDoctor_variable_id(unsigned int index)
 * Gets the value of the id variable of an doctor agent in the defaultDoctor state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_doctor_defaultDoctor_variable_id(unsigned int index);

/** int get_doctor_defaultDoctor_variable_current_patient(unsigned int index)
 * Gets the value of the current_patient variable of an doctor agent in the defaultDoctor state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_patient
 */
__host__ int get_doctor_defaultDoctor_variable_current_patient(unsigned int index);

/** unsigned int get_doctor_defaultDoctor_variable_tick(unsigned int index)
 * Gets the value of the tick variable of an doctor agent in the defaultDoctor state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable tick
 */
__host__ unsigned int get_doctor_defaultDoctor_variable_tick(unsigned int index);

/** unsigned int get_triage_defaultTriage_variable_front(unsigned int index)
 * Gets the value of the front variable of an triage agent in the defaultTriage state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable front
 */
__host__ unsigned int get_triage_defaultTriage_variable_front(unsigned int index);

/** unsigned int get_triage_defaultTriage_variable_rear(unsigned int index)
 * Gets the value of the rear variable of an triage agent in the defaultTriage state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable rear
 */
__host__ unsigned int get_triage_defaultTriage_variable_rear(unsigned int index);

/** unsigned int get_triage_defaultTriage_variable_size(unsigned int index)
 * Gets the value of the size variable of an triage agent in the defaultTriage state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable size
 */
__host__ unsigned int get_triage_defaultTriage_variable_size(unsigned int index);

/** unsigned int get_triage_defaultTriage_variable_tick(unsigned int index)
 * Gets the value of the tick variable of an triage agent in the defaultTriage state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable tick
 */
__host__ unsigned int get_triage_defaultTriage_variable_tick(unsigned int index);

/** unsigned int get_triage_defaultTriage_variable_boxArray(unsigned int index, unsigned int element)
 * Gets the element-th value of the boxArray variable array of an triage agent in the defaultTriage state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable boxArray
 */
__host__ unsigned int get_triage_defaultTriage_variable_boxArray(unsigned int index, unsigned int element);

/** unsigned int get_triage_defaultTriage_variable_patientQueue(unsigned int index, unsigned int element)
 * Gets the element-th value of the patientQueue variable array of an triage agent in the defaultTriage state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable patientQueue
 */
__host__ unsigned int get_triage_defaultTriage_variable_patientQueue(unsigned int index, unsigned int element);




/* Host based agent creation functions */

/** h_allocate_agent_agent
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated agent struct.
 */
xmachine_memory_agent* h_allocate_agent_agent();
/** h_free_agent_agent
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_agent(xmachine_memory_agent** agent);
/** h_allocate_agent_agent_array
 * Utility function to allocate an array of structs for  agent agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_agent** h_allocate_agent_agent_array(unsigned int count);
/** h_free_agent_agent_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_agent_array(xmachine_memory_agent*** agents, unsigned int count);


/** h_add_agent_agent_default
 * Host function to add a single agent of type agent to the default state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_agent_default instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_agent_default(xmachine_memory_agent* agent);

/** h_add_agents_agent_default(
 * Host function to add multiple agents of type agent to the default state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of agent agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_agent_default(xmachine_memory_agent** agents, unsigned int count);

/** h_allocate_agent_navmap
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated navmap struct.
 */
xmachine_memory_navmap* h_allocate_agent_navmap();
/** h_free_agent_navmap
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_navmap(xmachine_memory_navmap** agent);
/** h_allocate_agent_navmap_array
 * Utility function to allocate an array of structs for  navmap agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_navmap** h_allocate_agent_navmap_array(unsigned int count);
/** h_free_agent_navmap_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_navmap_array(xmachine_memory_navmap*** agents, unsigned int count);


/** h_add_agent_navmap_static
 * Host function to add a single agent of type navmap to the static state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_navmap_static instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_navmap_static(xmachine_memory_navmap* agent);

/** h_add_agents_navmap_static(
 * Host function to add multiple agents of type navmap to the static state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of navmap agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_navmap_static(xmachine_memory_navmap** agents, unsigned int count);

/** h_allocate_agent_chair
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated chair struct.
 */
xmachine_memory_chair* h_allocate_agent_chair();
/** h_free_agent_chair
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_chair(xmachine_memory_chair** agent);
/** h_allocate_agent_chair_array
 * Utility function to allocate an array of structs for  chair agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_chair** h_allocate_agent_chair_array(unsigned int count);
/** h_free_agent_chair_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_chair_array(xmachine_memory_chair*** agents, unsigned int count);


/** h_add_agent_chair_defaultChair
 * Host function to add a single agent of type chair to the defaultChair state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_chair_defaultChair instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_chair_defaultChair(xmachine_memory_chair* agent);

/** h_add_agents_chair_defaultChair(
 * Host function to add multiple agents of type chair to the defaultChair state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of chair agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_chair_defaultChair(xmachine_memory_chair** agents, unsigned int count);

/** h_allocate_agent_doctor_manager
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated doctor_manager struct.
 */
xmachine_memory_doctor_manager* h_allocate_agent_doctor_manager();
/** h_free_agent_doctor_manager
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_doctor_manager(xmachine_memory_doctor_manager** agent);
/** h_allocate_agent_doctor_manager_array
 * Utility function to allocate an array of structs for  doctor_manager agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_doctor_manager** h_allocate_agent_doctor_manager_array(unsigned int count);
/** h_free_agent_doctor_manager_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_doctor_manager_array(xmachine_memory_doctor_manager*** agents, unsigned int count);


/** h_add_agent_doctor_manager_defaultDoctorManager
 * Host function to add a single agent of type doctor_manager to the defaultDoctorManager state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_doctor_manager_defaultDoctorManager instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_doctor_manager_defaultDoctorManager(xmachine_memory_doctor_manager* agent);

/** h_add_agents_doctor_manager_defaultDoctorManager(
 * Host function to add multiple agents of type doctor_manager to the defaultDoctorManager state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of doctor_manager agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_doctor_manager_defaultDoctorManager(xmachine_memory_doctor_manager** agents, unsigned int count);

/** h_allocate_agent_specialist_manager
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated specialist_manager struct.
 */
xmachine_memory_specialist_manager* h_allocate_agent_specialist_manager();
/** h_free_agent_specialist_manager
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_specialist_manager(xmachine_memory_specialist_manager** agent);
/** h_allocate_agent_specialist_manager_array
 * Utility function to allocate an array of structs for  specialist_manager agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_specialist_manager** h_allocate_agent_specialist_manager_array(unsigned int count);
/** h_free_agent_specialist_manager_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_specialist_manager_array(xmachine_memory_specialist_manager*** agents, unsigned int count);


/** h_add_agent_specialist_manager_defaultSpecialistManager
 * Host function to add a single agent of type specialist_manager to the defaultSpecialistManager state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_specialist_manager_defaultSpecialistManager instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_specialist_manager_defaultSpecialistManager(xmachine_memory_specialist_manager* agent);

/** h_add_agents_specialist_manager_defaultSpecialistManager(
 * Host function to add multiple agents of type specialist_manager to the defaultSpecialistManager state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of specialist_manager agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_specialist_manager_defaultSpecialistManager(xmachine_memory_specialist_manager** agents, unsigned int count);

/** h_allocate_agent_specialist
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated specialist struct.
 */
xmachine_memory_specialist* h_allocate_agent_specialist();
/** h_free_agent_specialist
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_specialist(xmachine_memory_specialist** agent);
/** h_allocate_agent_specialist_array
 * Utility function to allocate an array of structs for  specialist agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_specialist** h_allocate_agent_specialist_array(unsigned int count);
/** h_free_agent_specialist_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_specialist_array(xmachine_memory_specialist*** agents, unsigned int count);


/** h_add_agent_specialist_defaultSpecialist
 * Host function to add a single agent of type specialist to the defaultSpecialist state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_specialist_defaultSpecialist instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_specialist_defaultSpecialist(xmachine_memory_specialist* agent);

/** h_add_agents_specialist_defaultSpecialist(
 * Host function to add multiple agents of type specialist to the defaultSpecialist state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of specialist agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_specialist_defaultSpecialist(xmachine_memory_specialist** agents, unsigned int count);

/** h_allocate_agent_receptionist
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated receptionist struct.
 */
xmachine_memory_receptionist* h_allocate_agent_receptionist();
/** h_free_agent_receptionist
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_receptionist(xmachine_memory_receptionist** agent);
/** h_allocate_agent_receptionist_array
 * Utility function to allocate an array of structs for  receptionist agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_receptionist** h_allocate_agent_receptionist_array(unsigned int count);
/** h_free_agent_receptionist_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_receptionist_array(xmachine_memory_receptionist*** agents, unsigned int count);


/** h_add_agent_receptionist_defaultReceptionist
 * Host function to add a single agent of type receptionist to the defaultReceptionist state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_receptionist_defaultReceptionist instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_receptionist_defaultReceptionist(xmachine_memory_receptionist* agent);

/** h_add_agents_receptionist_defaultReceptionist(
 * Host function to add multiple agents of type receptionist to the defaultReceptionist state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of receptionist agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_receptionist_defaultReceptionist(xmachine_memory_receptionist** agents, unsigned int count);

/** h_allocate_agent_agent_generator
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated agent_generator struct.
 */
xmachine_memory_agent_generator* h_allocate_agent_agent_generator();
/** h_free_agent_agent_generator
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_agent_generator(xmachine_memory_agent_generator** agent);
/** h_allocate_agent_agent_generator_array
 * Utility function to allocate an array of structs for  agent_generator agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_agent_generator** h_allocate_agent_agent_generator_array(unsigned int count);
/** h_free_agent_agent_generator_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_agent_generator_array(xmachine_memory_agent_generator*** agents, unsigned int count);


/** h_add_agent_agent_generator_defaultGenerator
 * Host function to add a single agent of type agent_generator to the defaultGenerator state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_agent_generator_defaultGenerator instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_agent_generator_defaultGenerator(xmachine_memory_agent_generator* agent);

/** h_add_agents_agent_generator_defaultGenerator(
 * Host function to add multiple agents of type agent_generator to the defaultGenerator state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of agent_generator agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_agent_generator_defaultGenerator(xmachine_memory_agent_generator** agents, unsigned int count);

/** h_allocate_agent_chair_admin
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated chair_admin struct.
 */
xmachine_memory_chair_admin* h_allocate_agent_chair_admin();
/** h_free_agent_chair_admin
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_chair_admin(xmachine_memory_chair_admin** agent);
/** h_allocate_agent_chair_admin_array
 * Utility function to allocate an array of structs for  chair_admin agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_chair_admin** h_allocate_agent_chair_admin_array(unsigned int count);
/** h_free_agent_chair_admin_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_chair_admin_array(xmachine_memory_chair_admin*** agents, unsigned int count);


/** h_add_agent_chair_admin_defaultAdmin
 * Host function to add a single agent of type chair_admin to the defaultAdmin state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_chair_admin_defaultAdmin instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_chair_admin_defaultAdmin(xmachine_memory_chair_admin* agent);

/** h_add_agents_chair_admin_defaultAdmin(
 * Host function to add multiple agents of type chair_admin to the defaultAdmin state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of chair_admin agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_chair_admin_defaultAdmin(xmachine_memory_chair_admin** agents, unsigned int count);

/** h_allocate_agent_box
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated box struct.
 */
xmachine_memory_box* h_allocate_agent_box();
/** h_free_agent_box
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_box(xmachine_memory_box** agent);
/** h_allocate_agent_box_array
 * Utility function to allocate an array of structs for  box agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_box** h_allocate_agent_box_array(unsigned int count);
/** h_free_agent_box_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_box_array(xmachine_memory_box*** agents, unsigned int count);


/** h_add_agent_box_defaultBox
 * Host function to add a single agent of type box to the defaultBox state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_box_defaultBox instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_box_defaultBox(xmachine_memory_box* agent);

/** h_add_agents_box_defaultBox(
 * Host function to add multiple agents of type box to the defaultBox state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of box agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_box_defaultBox(xmachine_memory_box** agents, unsigned int count);

/** h_allocate_agent_doctor
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated doctor struct.
 */
xmachine_memory_doctor* h_allocate_agent_doctor();
/** h_free_agent_doctor
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_doctor(xmachine_memory_doctor** agent);
/** h_allocate_agent_doctor_array
 * Utility function to allocate an array of structs for  doctor agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_doctor** h_allocate_agent_doctor_array(unsigned int count);
/** h_free_agent_doctor_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_doctor_array(xmachine_memory_doctor*** agents, unsigned int count);


/** h_add_agent_doctor_defaultDoctor
 * Host function to add a single agent of type doctor to the defaultDoctor state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_doctor_defaultDoctor instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_doctor_defaultDoctor(xmachine_memory_doctor* agent);

/** h_add_agents_doctor_defaultDoctor(
 * Host function to add multiple agents of type doctor to the defaultDoctor state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of doctor agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_doctor_defaultDoctor(xmachine_memory_doctor** agents, unsigned int count);

/** h_allocate_agent_triage
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated triage struct.
 */
xmachine_memory_triage* h_allocate_agent_triage();
/** h_free_agent_triage
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_triage(xmachine_memory_triage** agent);
/** h_allocate_agent_triage_array
 * Utility function to allocate an array of structs for  triage agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_triage** h_allocate_agent_triage_array(unsigned int count);
/** h_free_agent_triage_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_triage_array(xmachine_memory_triage*** agents, unsigned int count);


/** h_add_agent_triage_defaultTriage
 * Host function to add a single agent of type triage to the defaultTriage state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_triage_defaultTriage instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_triage_defaultTriage(xmachine_memory_triage* agent);

/** h_add_agents_triage_defaultTriage(
 * Host function to add multiple agents of type triage to the defaultTriage state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of triage agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_triage_defaultTriage(xmachine_memory_triage** agents, unsigned int count);

  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** unsigned int reduce_agent_default_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_agent_default_id_variable();



/** unsigned int count_agent_default_id_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_agent_default_id_variable(unsigned int count_value);

/** unsigned int min_agent_default_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_agent_default_id_variable();
/** unsigned int max_agent_default_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_agent_default_id_variable();

/** float reduce_agent_default_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_x_variable();



/** float min_agent_default_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_x_variable();
/** float max_agent_default_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_x_variable();

/** float reduce_agent_default_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_y_variable();



/** float min_agent_default_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_y_variable();
/** float max_agent_default_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_y_variable();

/** float reduce_agent_default_velx_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_velx_variable();



/** float min_agent_default_velx_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_velx_variable();
/** float max_agent_default_velx_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_velx_variable();

/** float reduce_agent_default_vely_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_vely_variable();



/** float min_agent_default_vely_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_vely_variable();
/** float max_agent_default_vely_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_vely_variable();

/** float reduce_agent_default_steer_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_steer_x_variable();



/** float min_agent_default_steer_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_steer_x_variable();
/** float max_agent_default_steer_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_steer_x_variable();

/** float reduce_agent_default_steer_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_steer_y_variable();



/** float min_agent_default_steer_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_steer_y_variable();
/** float max_agent_default_steer_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_steer_y_variable();

/** float reduce_agent_default_height_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_height_variable();



/** float min_agent_default_height_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_height_variable();
/** float max_agent_default_height_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_height_variable();

/** int reduce_agent_default_exit_no_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_exit_no_variable();



/** int count_agent_default_exit_no_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_exit_no_variable(int count_value);

/** int min_agent_default_exit_no_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_exit_no_variable();
/** int max_agent_default_exit_no_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_exit_no_variable();

/** float reduce_agent_default_speed_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_speed_variable();



/** float min_agent_default_speed_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_speed_variable();
/** float max_agent_default_speed_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_speed_variable();

/** int reduce_agent_default_lod_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_lod_variable();



/** int count_agent_default_lod_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_lod_variable(int count_value);

/** int min_agent_default_lod_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_lod_variable();
/** int max_agent_default_lod_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_lod_variable();

/** float reduce_agent_default_animate_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_agent_default_animate_variable();



/** float min_agent_default_animate_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_agent_default_animate_variable();
/** float max_agent_default_animate_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_agent_default_animate_variable();

/** int reduce_agent_default_animate_dir_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_animate_dir_variable();



/** int count_agent_default_animate_dir_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_animate_dir_variable(int count_value);

/** int min_agent_default_animate_dir_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_animate_dir_variable();
/** int max_agent_default_animate_dir_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_animate_dir_variable();

/** int reduce_agent_default_estado_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_estado_variable();



/** int count_agent_default_estado_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_estado_variable(int count_value);

/** int min_agent_default_estado_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_estado_variable();
/** int max_agent_default_estado_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_estado_variable();

/** int reduce_agent_default_tick_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_tick_variable();



/** int count_agent_default_tick_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_tick_variable(int count_value);

/** int min_agent_default_tick_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_tick_variable();
/** int max_agent_default_tick_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_tick_variable();

/** unsigned int reduce_agent_default_estado_movimiento_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_agent_default_estado_movimiento_variable();



/** unsigned int count_agent_default_estado_movimiento_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_agent_default_estado_movimiento_variable(unsigned int count_value);

/** unsigned int min_agent_default_estado_movimiento_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_agent_default_estado_movimiento_variable();
/** unsigned int max_agent_default_estado_movimiento_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_agent_default_estado_movimiento_variable();

/** unsigned int reduce_agent_default_go_to_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_agent_default_go_to_x_variable();



/** unsigned int count_agent_default_go_to_x_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_agent_default_go_to_x_variable(unsigned int count_value);

/** unsigned int min_agent_default_go_to_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_agent_default_go_to_x_variable();
/** unsigned int max_agent_default_go_to_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_agent_default_go_to_x_variable();

/** unsigned int reduce_agent_default_go_to_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_agent_default_go_to_y_variable();



/** unsigned int count_agent_default_go_to_y_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_agent_default_go_to_y_variable(unsigned int count_value);

/** unsigned int min_agent_default_go_to_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_agent_default_go_to_y_variable();
/** unsigned int max_agent_default_go_to_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_agent_default_go_to_y_variable();

/** unsigned int reduce_agent_default_checkpoint_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_agent_default_checkpoint_variable();



/** unsigned int count_agent_default_checkpoint_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_agent_default_checkpoint_variable(unsigned int count_value);

/** unsigned int min_agent_default_checkpoint_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_agent_default_checkpoint_variable();
/** unsigned int max_agent_default_checkpoint_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_agent_default_checkpoint_variable();

/** int reduce_agent_default_chair_no_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_chair_no_variable();



/** int count_agent_default_chair_no_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_chair_no_variable(int count_value);

/** int min_agent_default_chair_no_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_chair_no_variable();
/** int max_agent_default_chair_no_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_chair_no_variable();

/** unsigned int reduce_agent_default_box_no_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_agent_default_box_no_variable();



/** unsigned int count_agent_default_box_no_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_agent_default_box_no_variable(unsigned int count_value);

/** unsigned int min_agent_default_box_no_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_agent_default_box_no_variable();
/** unsigned int max_agent_default_box_no_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_agent_default_box_no_variable();

/** unsigned int reduce_agent_default_doctor_no_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_agent_default_doctor_no_variable();



/** unsigned int count_agent_default_doctor_no_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_agent_default_doctor_no_variable(unsigned int count_value);

/** unsigned int min_agent_default_doctor_no_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_agent_default_doctor_no_variable();
/** unsigned int max_agent_default_doctor_no_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_agent_default_doctor_no_variable();

/** unsigned int reduce_agent_default_specialist_no_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_agent_default_specialist_no_variable();



/** unsigned int count_agent_default_specialist_no_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_agent_default_specialist_no_variable(unsigned int count_value);

/** unsigned int min_agent_default_specialist_no_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_agent_default_specialist_no_variable();
/** unsigned int max_agent_default_specialist_no_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_agent_default_specialist_no_variable();

/** unsigned int reduce_agent_default_priority_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_agent_default_priority_variable();



/** unsigned int count_agent_default_priority_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_agent_default_priority_variable(unsigned int count_value);

/** unsigned int min_agent_default_priority_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_agent_default_priority_variable();
/** unsigned int max_agent_default_priority_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_agent_default_priority_variable();

/** int reduce_navmap_static_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_navmap_static_x_variable();



/** int count_navmap_static_x_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_navmap_static_x_variable(int count_value);

/** int min_navmap_static_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_navmap_static_x_variable();
/** int max_navmap_static_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_navmap_static_x_variable();

/** int reduce_navmap_static_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_navmap_static_y_variable();



/** int count_navmap_static_y_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_navmap_static_y_variable(int count_value);

/** int min_navmap_static_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_navmap_static_y_variable();
/** int max_navmap_static_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_navmap_static_y_variable();

/** int reduce_navmap_static_exit_no_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_navmap_static_exit_no_variable();



/** int count_navmap_static_exit_no_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_navmap_static_exit_no_variable(int count_value);

/** int min_navmap_static_exit_no_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_navmap_static_exit_no_variable();
/** int max_navmap_static_exit_no_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_navmap_static_exit_no_variable();

/** float reduce_navmap_static_height_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_height_variable();



/** float min_navmap_static_height_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_height_variable();
/** float max_navmap_static_height_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_height_variable();

/** float reduce_navmap_static_collision_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_collision_x_variable();



/** float min_navmap_static_collision_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_collision_x_variable();
/** float max_navmap_static_collision_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_collision_x_variable();

/** float reduce_navmap_static_collision_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_collision_y_variable();



/** float min_navmap_static_collision_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_collision_y_variable();
/** float max_navmap_static_collision_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_collision_y_variable();

/** float reduce_navmap_static_exit0_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit0_x_variable();



/** float min_navmap_static_exit0_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit0_x_variable();
/** float max_navmap_static_exit0_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit0_x_variable();

/** float reduce_navmap_static_exit0_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit0_y_variable();



/** float min_navmap_static_exit0_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit0_y_variable();
/** float max_navmap_static_exit0_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit0_y_variable();

/** float reduce_navmap_static_exit1_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit1_x_variable();



/** float min_navmap_static_exit1_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit1_x_variable();
/** float max_navmap_static_exit1_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit1_x_variable();

/** float reduce_navmap_static_exit1_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit1_y_variable();



/** float min_navmap_static_exit1_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit1_y_variable();
/** float max_navmap_static_exit1_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit1_y_variable();

/** float reduce_navmap_static_exit2_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit2_x_variable();



/** float min_navmap_static_exit2_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit2_x_variable();
/** float max_navmap_static_exit2_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit2_x_variable();

/** float reduce_navmap_static_exit2_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit2_y_variable();



/** float min_navmap_static_exit2_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit2_y_variable();
/** float max_navmap_static_exit2_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit2_y_variable();

/** float reduce_navmap_static_exit3_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit3_x_variable();



/** float min_navmap_static_exit3_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit3_x_variable();
/** float max_navmap_static_exit3_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit3_x_variable();

/** float reduce_navmap_static_exit3_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit3_y_variable();



/** float min_navmap_static_exit3_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit3_y_variable();
/** float max_navmap_static_exit3_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit3_y_variable();

/** float reduce_navmap_static_exit4_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit4_x_variable();



/** float min_navmap_static_exit4_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit4_x_variable();
/** float max_navmap_static_exit4_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit4_x_variable();

/** float reduce_navmap_static_exit4_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit4_y_variable();



/** float min_navmap_static_exit4_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit4_y_variable();
/** float max_navmap_static_exit4_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit4_y_variable();

/** float reduce_navmap_static_exit5_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit5_x_variable();



/** float min_navmap_static_exit5_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit5_x_variable();
/** float max_navmap_static_exit5_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit5_x_variable();

/** float reduce_navmap_static_exit5_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit5_y_variable();



/** float min_navmap_static_exit5_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit5_y_variable();
/** float max_navmap_static_exit5_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit5_y_variable();

/** float reduce_navmap_static_exit6_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit6_x_variable();



/** float min_navmap_static_exit6_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit6_x_variable();
/** float max_navmap_static_exit6_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit6_x_variable();

/** float reduce_navmap_static_exit6_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_navmap_static_exit6_y_variable();



/** float min_navmap_static_exit6_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_navmap_static_exit6_y_variable();
/** float max_navmap_static_exit6_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_navmap_static_exit6_y_variable();

/** unsigned int reduce_navmap_static_cant_generados_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_navmap_static_cant_generados_variable();



/** unsigned int count_navmap_static_cant_generados_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_navmap_static_cant_generados_variable(unsigned int count_value);

/** unsigned int min_navmap_static_cant_generados_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_navmap_static_cant_generados_variable();
/** unsigned int max_navmap_static_cant_generados_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_navmap_static_cant_generados_variable();

/** int reduce_chair_defaultChair_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_chair_defaultChair_id_variable();



/** int count_chair_defaultChair_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_chair_defaultChair_id_variable(int count_value);

/** int min_chair_defaultChair_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_chair_defaultChair_id_variable();
/** int max_chair_defaultChair_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_chair_defaultChair_id_variable();

/** int reduce_chair_defaultChair_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_chair_defaultChair_x_variable();



/** int count_chair_defaultChair_x_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_chair_defaultChair_x_variable(int count_value);

/** int min_chair_defaultChair_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_chair_defaultChair_x_variable();
/** int max_chair_defaultChair_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_chair_defaultChair_x_variable();

/** int reduce_chair_defaultChair_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_chair_defaultChair_y_variable();



/** int count_chair_defaultChair_y_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_chair_defaultChair_y_variable(int count_value);

/** int min_chair_defaultChair_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_chair_defaultChair_y_variable();
/** int max_chair_defaultChair_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_chair_defaultChair_y_variable();

/** int reduce_chair_defaultChair_state_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_chair_defaultChair_state_variable();



/** int count_chair_defaultChair_state_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_chair_defaultChair_state_variable(int count_value);

/** int min_chair_defaultChair_state_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_chair_defaultChair_state_variable();
/** int max_chair_defaultChair_state_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_chair_defaultChair_state_variable();

/** unsigned int reduce_doctor_manager_defaultDoctorManager_tick_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_doctor_manager_defaultDoctorManager_tick_variable();



/** unsigned int count_doctor_manager_defaultDoctorManager_tick_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_doctor_manager_defaultDoctorManager_tick_variable(unsigned int count_value);

/** unsigned int min_doctor_manager_defaultDoctorManager_tick_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_doctor_manager_defaultDoctorManager_tick_variable();
/** unsigned int max_doctor_manager_defaultDoctorManager_tick_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_doctor_manager_defaultDoctorManager_tick_variable();

/** unsigned int reduce_doctor_manager_defaultDoctorManager_rear_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_doctor_manager_defaultDoctorManager_rear_variable();



/** unsigned int count_doctor_manager_defaultDoctorManager_rear_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_doctor_manager_defaultDoctorManager_rear_variable(unsigned int count_value);

/** unsigned int min_doctor_manager_defaultDoctorManager_rear_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_doctor_manager_defaultDoctorManager_rear_variable();
/** unsigned int max_doctor_manager_defaultDoctorManager_rear_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_doctor_manager_defaultDoctorManager_rear_variable();

/** unsigned int reduce_doctor_manager_defaultDoctorManager_size_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_doctor_manager_defaultDoctorManager_size_variable();



/** unsigned int count_doctor_manager_defaultDoctorManager_size_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_doctor_manager_defaultDoctorManager_size_variable(unsigned int count_value);

/** unsigned int min_doctor_manager_defaultDoctorManager_size_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_doctor_manager_defaultDoctorManager_size_variable();
/** unsigned int max_doctor_manager_defaultDoctorManager_size_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_doctor_manager_defaultDoctorManager_size_variable();

/** unsigned int reduce_doctor_manager_defaultDoctorManager_free_doctors_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_doctor_manager_defaultDoctorManager_free_doctors_variable();



/** unsigned int count_doctor_manager_defaultDoctorManager_free_doctors_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_doctor_manager_defaultDoctorManager_free_doctors_variable(unsigned int count_value);

/** unsigned int min_doctor_manager_defaultDoctorManager_free_doctors_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_doctor_manager_defaultDoctorManager_free_doctors_variable();
/** unsigned int max_doctor_manager_defaultDoctorManager_free_doctors_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_doctor_manager_defaultDoctorManager_free_doctors_variable();

/** unsigned int reduce_specialist_manager_defaultSpecialistManager_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_specialist_manager_defaultSpecialistManager_id_variable();



/** unsigned int count_specialist_manager_defaultSpecialistManager_id_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_specialist_manager_defaultSpecialistManager_id_variable(unsigned int count_value);

/** unsigned int min_specialist_manager_defaultSpecialistManager_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_specialist_manager_defaultSpecialistManager_id_variable();
/** unsigned int max_specialist_manager_defaultSpecialistManager_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_specialist_manager_defaultSpecialistManager_id_variable();

/** unsigned int reduce_specialist_defaultSpecialist_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_specialist_defaultSpecialist_id_variable();



/** unsigned int count_specialist_defaultSpecialist_id_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_specialist_defaultSpecialist_id_variable(unsigned int count_value);

/** unsigned int min_specialist_defaultSpecialist_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_specialist_defaultSpecialist_id_variable();
/** unsigned int max_specialist_defaultSpecialist_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_specialist_defaultSpecialist_id_variable();

/** unsigned int reduce_specialist_defaultSpecialist_current_patient_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_specialist_defaultSpecialist_current_patient_variable();



/** unsigned int count_specialist_defaultSpecialist_current_patient_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_specialist_defaultSpecialist_current_patient_variable(unsigned int count_value);

/** unsigned int min_specialist_defaultSpecialist_current_patient_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_specialist_defaultSpecialist_current_patient_variable();
/** unsigned int max_specialist_defaultSpecialist_current_patient_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_specialist_defaultSpecialist_current_patient_variable();

/** unsigned int reduce_specialist_defaultSpecialist_tick_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_specialist_defaultSpecialist_tick_variable();



/** unsigned int count_specialist_defaultSpecialist_tick_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_specialist_defaultSpecialist_tick_variable(unsigned int count_value);

/** unsigned int min_specialist_defaultSpecialist_tick_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_specialist_defaultSpecialist_tick_variable();
/** unsigned int max_specialist_defaultSpecialist_tick_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_specialist_defaultSpecialist_tick_variable();

/** int reduce_receptionist_defaultReceptionist_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_receptionist_defaultReceptionist_x_variable();



/** int count_receptionist_defaultReceptionist_x_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_receptionist_defaultReceptionist_x_variable(int count_value);

/** int min_receptionist_defaultReceptionist_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_receptionist_defaultReceptionist_x_variable();
/** int max_receptionist_defaultReceptionist_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_receptionist_defaultReceptionist_x_variable();

/** int reduce_receptionist_defaultReceptionist_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_receptionist_defaultReceptionist_y_variable();



/** int count_receptionist_defaultReceptionist_y_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_receptionist_defaultReceptionist_y_variable(int count_value);

/** int min_receptionist_defaultReceptionist_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_receptionist_defaultReceptionist_y_variable();
/** int max_receptionist_defaultReceptionist_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_receptionist_defaultReceptionist_y_variable();

/** unsigned int reduce_receptionist_defaultReceptionist_front_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_receptionist_defaultReceptionist_front_variable();



/** unsigned int count_receptionist_defaultReceptionist_front_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_receptionist_defaultReceptionist_front_variable(unsigned int count_value);

/** unsigned int min_receptionist_defaultReceptionist_front_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_receptionist_defaultReceptionist_front_variable();
/** unsigned int max_receptionist_defaultReceptionist_front_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_receptionist_defaultReceptionist_front_variable();

/** unsigned int reduce_receptionist_defaultReceptionist_rear_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_receptionist_defaultReceptionist_rear_variable();



/** unsigned int count_receptionist_defaultReceptionist_rear_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_receptionist_defaultReceptionist_rear_variable(unsigned int count_value);

/** unsigned int min_receptionist_defaultReceptionist_rear_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_receptionist_defaultReceptionist_rear_variable();
/** unsigned int max_receptionist_defaultReceptionist_rear_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_receptionist_defaultReceptionist_rear_variable();

/** unsigned int reduce_receptionist_defaultReceptionist_size_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_receptionist_defaultReceptionist_size_variable();



/** unsigned int count_receptionist_defaultReceptionist_size_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_receptionist_defaultReceptionist_size_variable(unsigned int count_value);

/** unsigned int min_receptionist_defaultReceptionist_size_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_receptionist_defaultReceptionist_size_variable();
/** unsigned int max_receptionist_defaultReceptionist_size_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_receptionist_defaultReceptionist_size_variable();

/** unsigned int reduce_receptionist_defaultReceptionist_tick_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_receptionist_defaultReceptionist_tick_variable();



/** unsigned int count_receptionist_defaultReceptionist_tick_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_receptionist_defaultReceptionist_tick_variable(unsigned int count_value);

/** unsigned int min_receptionist_defaultReceptionist_tick_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_receptionist_defaultReceptionist_tick_variable();
/** unsigned int max_receptionist_defaultReceptionist_tick_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_receptionist_defaultReceptionist_tick_variable();

/** int reduce_receptionist_defaultReceptionist_current_patient_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_receptionist_defaultReceptionist_current_patient_variable();



/** int count_receptionist_defaultReceptionist_current_patient_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_receptionist_defaultReceptionist_current_patient_variable(int count_value);

/** int min_receptionist_defaultReceptionist_current_patient_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_receptionist_defaultReceptionist_current_patient_variable();
/** int max_receptionist_defaultReceptionist_current_patient_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_receptionist_defaultReceptionist_current_patient_variable();

/** int reduce_receptionist_defaultReceptionist_attend_patient_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_receptionist_defaultReceptionist_attend_patient_variable();



/** int count_receptionist_defaultReceptionist_attend_patient_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_receptionist_defaultReceptionist_attend_patient_variable(int count_value);

/** int min_receptionist_defaultReceptionist_attend_patient_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_receptionist_defaultReceptionist_attend_patient_variable();
/** int max_receptionist_defaultReceptionist_attend_patient_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_receptionist_defaultReceptionist_attend_patient_variable();

/** int reduce_receptionist_defaultReceptionist_estado_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_receptionist_defaultReceptionist_estado_variable();



/** int count_receptionist_defaultReceptionist_estado_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_receptionist_defaultReceptionist_estado_variable(int count_value);

/** int min_receptionist_defaultReceptionist_estado_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_receptionist_defaultReceptionist_estado_variable();
/** int max_receptionist_defaultReceptionist_estado_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_receptionist_defaultReceptionist_estado_variable();

/** int reduce_agent_generator_defaultGenerator_chairs_generated_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_generator_defaultGenerator_chairs_generated_variable();



/** int count_agent_generator_defaultGenerator_chairs_generated_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_generator_defaultGenerator_chairs_generated_variable(int count_value);

/** int min_agent_generator_defaultGenerator_chairs_generated_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_generator_defaultGenerator_chairs_generated_variable();
/** int max_agent_generator_defaultGenerator_chairs_generated_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_generator_defaultGenerator_chairs_generated_variable();

/** int reduce_agent_generator_defaultGenerator_boxes_generated_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_generator_defaultGenerator_boxes_generated_variable();



/** int count_agent_generator_defaultGenerator_boxes_generated_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_generator_defaultGenerator_boxes_generated_variable(int count_value);

/** int min_agent_generator_defaultGenerator_boxes_generated_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_generator_defaultGenerator_boxes_generated_variable();
/** int max_agent_generator_defaultGenerator_boxes_generated_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_generator_defaultGenerator_boxes_generated_variable();

/** int reduce_agent_generator_defaultGenerator_doctors_generated_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_generator_defaultGenerator_doctors_generated_variable();



/** int count_agent_generator_defaultGenerator_doctors_generated_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_generator_defaultGenerator_doctors_generated_variable(int count_value);

/** int min_agent_generator_defaultGenerator_doctors_generated_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_generator_defaultGenerator_doctors_generated_variable();
/** int max_agent_generator_defaultGenerator_doctors_generated_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_generator_defaultGenerator_doctors_generated_variable();

/** int reduce_agent_generator_defaultGenerator_specialists_generated_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_generator_defaultGenerator_specialists_generated_variable();



/** int count_agent_generator_defaultGenerator_specialists_generated_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_generator_defaultGenerator_specialists_generated_variable(int count_value);

/** int min_agent_generator_defaultGenerator_specialists_generated_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_generator_defaultGenerator_specialists_generated_variable();
/** int max_agent_generator_defaultGenerator_specialists_generated_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_generator_defaultGenerator_specialists_generated_variable();

/** unsigned int reduce_chair_admin_defaultAdmin_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_chair_admin_defaultAdmin_id_variable();



/** unsigned int count_chair_admin_defaultAdmin_id_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_chair_admin_defaultAdmin_id_variable(unsigned int count_value);

/** unsigned int min_chair_admin_defaultAdmin_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_chair_admin_defaultAdmin_id_variable();
/** unsigned int max_chair_admin_defaultAdmin_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_chair_admin_defaultAdmin_id_variable();

/** unsigned int reduce_box_defaultBox_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_box_defaultBox_id_variable();



/** unsigned int count_box_defaultBox_id_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_box_defaultBox_id_variable(unsigned int count_value);

/** unsigned int min_box_defaultBox_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_box_defaultBox_id_variable();
/** unsigned int max_box_defaultBox_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_box_defaultBox_id_variable();

/** unsigned int reduce_box_defaultBox_attending_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_box_defaultBox_attending_variable();



/** unsigned int count_box_defaultBox_attending_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_box_defaultBox_attending_variable(unsigned int count_value);

/** unsigned int min_box_defaultBox_attending_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_box_defaultBox_attending_variable();
/** unsigned int max_box_defaultBox_attending_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_box_defaultBox_attending_variable();

/** unsigned int reduce_box_defaultBox_tick_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_box_defaultBox_tick_variable();



/** unsigned int count_box_defaultBox_tick_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_box_defaultBox_tick_variable(unsigned int count_value);

/** unsigned int min_box_defaultBox_tick_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_box_defaultBox_tick_variable();
/** unsigned int max_box_defaultBox_tick_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_box_defaultBox_tick_variable();

/** unsigned int reduce_doctor_defaultDoctor_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_doctor_defaultDoctor_id_variable();



/** unsigned int count_doctor_defaultDoctor_id_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_doctor_defaultDoctor_id_variable(unsigned int count_value);

/** unsigned int min_doctor_defaultDoctor_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_doctor_defaultDoctor_id_variable();
/** unsigned int max_doctor_defaultDoctor_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_doctor_defaultDoctor_id_variable();

/** int reduce_doctor_defaultDoctor_current_patient_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_doctor_defaultDoctor_current_patient_variable();



/** int count_doctor_defaultDoctor_current_patient_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_doctor_defaultDoctor_current_patient_variable(int count_value);

/** int min_doctor_defaultDoctor_current_patient_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_doctor_defaultDoctor_current_patient_variable();
/** int max_doctor_defaultDoctor_current_patient_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_doctor_defaultDoctor_current_patient_variable();

/** unsigned int reduce_doctor_defaultDoctor_tick_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_doctor_defaultDoctor_tick_variable();



/** unsigned int count_doctor_defaultDoctor_tick_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_doctor_defaultDoctor_tick_variable(unsigned int count_value);

/** unsigned int min_doctor_defaultDoctor_tick_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_doctor_defaultDoctor_tick_variable();
/** unsigned int max_doctor_defaultDoctor_tick_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_doctor_defaultDoctor_tick_variable();

/** unsigned int reduce_triage_defaultTriage_front_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_triage_defaultTriage_front_variable();



/** unsigned int count_triage_defaultTriage_front_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_triage_defaultTriage_front_variable(unsigned int count_value);

/** unsigned int min_triage_defaultTriage_front_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_triage_defaultTriage_front_variable();
/** unsigned int max_triage_defaultTriage_front_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_triage_defaultTriage_front_variable();

/** unsigned int reduce_triage_defaultTriage_rear_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_triage_defaultTriage_rear_variable();



/** unsigned int count_triage_defaultTriage_rear_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_triage_defaultTriage_rear_variable(unsigned int count_value);

/** unsigned int min_triage_defaultTriage_rear_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_triage_defaultTriage_rear_variable();
/** unsigned int max_triage_defaultTriage_rear_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_triage_defaultTriage_rear_variable();

/** unsigned int reduce_triage_defaultTriage_size_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_triage_defaultTriage_size_variable();



/** unsigned int count_triage_defaultTriage_size_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_triage_defaultTriage_size_variable(unsigned int count_value);

/** unsigned int min_triage_defaultTriage_size_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_triage_defaultTriage_size_variable();
/** unsigned int max_triage_defaultTriage_size_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_triage_defaultTriage_size_variable();

/** unsigned int reduce_triage_defaultTriage_tick_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_triage_defaultTriage_tick_variable();



/** unsigned int count_triage_defaultTriage_tick_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_triage_defaultTriage_tick_variable(unsigned int count_value);

/** unsigned int min_triage_defaultTriage_tick_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_triage_defaultTriage_tick_variable();
/** unsigned int max_triage_defaultTriage_tick_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_triage_defaultTriage_tick_variable();


  
/* global constant variables */

__constant__ float EMMISION_RATE_EXIT1;

__constant__ float EMMISION_RATE_EXIT2;

__constant__ float EMMISION_RATE_EXIT3;

__constant__ float EMMISION_RATE_EXIT4;

__constant__ float EMMISION_RATE_EXIT5;

__constant__ float EMMISION_RATE_EXIT6;

__constant__ float EMMISION_RATE_EXIT7;

__constant__ int EXIT1_PROBABILITY;

__constant__ int EXIT2_PROBABILITY;

__constant__ int EXIT3_PROBABILITY;

__constant__ int EXIT4_PROBABILITY;

__constant__ int EXIT5_PROBABILITY;

__constant__ int EXIT6_PROBABILITY;

__constant__ int EXIT7_PROBABILITY;

__constant__ int EXIT1_STATE;

__constant__ int EXIT2_STATE;

__constant__ int EXIT3_STATE;

__constant__ int EXIT4_STATE;

__constant__ int EXIT5_STATE;

__constant__ int EXIT6_STATE;

__constant__ int EXIT7_STATE;

__constant__ int EXIT1_CELL_COUNT;

__constant__ int EXIT2_CELL_COUNT;

__constant__ int EXIT3_CELL_COUNT;

__constant__ int EXIT4_CELL_COUNT;

__constant__ int EXIT5_CELL_COUNT;

__constant__ int EXIT6_CELL_COUNT;

__constant__ int EXIT7_CELL_COUNT;

__constant__ float TIME_SCALER;

__constant__ float STEER_WEIGHT;

__constant__ float AVOID_WEIGHT;

__constant__ float COLLISION_WEIGHT;

__constant__ float GOAL_WEIGHT;

/** set_EMMISION_RATE_EXIT1
 * Sets the constant variable EMMISION_RATE_EXIT1 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT1 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT1(float* h_EMMISION_RATE_EXIT1);

extern const float* get_EMMISION_RATE_EXIT1();


extern float h_env_EMMISION_RATE_EXIT1;

/** set_EMMISION_RATE_EXIT2
 * Sets the constant variable EMMISION_RATE_EXIT2 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT2 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT2(float* h_EMMISION_RATE_EXIT2);

extern const float* get_EMMISION_RATE_EXIT2();


extern float h_env_EMMISION_RATE_EXIT2;

/** set_EMMISION_RATE_EXIT3
 * Sets the constant variable EMMISION_RATE_EXIT3 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT3 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT3(float* h_EMMISION_RATE_EXIT3);

extern const float* get_EMMISION_RATE_EXIT3();


extern float h_env_EMMISION_RATE_EXIT3;

/** set_EMMISION_RATE_EXIT4
 * Sets the constant variable EMMISION_RATE_EXIT4 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT4 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT4(float* h_EMMISION_RATE_EXIT4);

extern const float* get_EMMISION_RATE_EXIT4();


extern float h_env_EMMISION_RATE_EXIT4;

/** set_EMMISION_RATE_EXIT5
 * Sets the constant variable EMMISION_RATE_EXIT5 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT5 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT5(float* h_EMMISION_RATE_EXIT5);

extern const float* get_EMMISION_RATE_EXIT5();


extern float h_env_EMMISION_RATE_EXIT5;

/** set_EMMISION_RATE_EXIT6
 * Sets the constant variable EMMISION_RATE_EXIT6 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT6 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT6(float* h_EMMISION_RATE_EXIT6);

extern const float* get_EMMISION_RATE_EXIT6();


extern float h_env_EMMISION_RATE_EXIT6;

/** set_EMMISION_RATE_EXIT7
 * Sets the constant variable EMMISION_RATE_EXIT7 on the device which can then be used in the agent functions.
 * @param h_EMMISION_RATE_EXIT7 value to set the variable
 */
extern void set_EMMISION_RATE_EXIT7(float* h_EMMISION_RATE_EXIT7);

extern const float* get_EMMISION_RATE_EXIT7();


extern float h_env_EMMISION_RATE_EXIT7;

/** set_EXIT1_PROBABILITY
 * Sets the constant variable EXIT1_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT1_PROBABILITY value to set the variable
 */
extern void set_EXIT1_PROBABILITY(int* h_EXIT1_PROBABILITY);

extern const int* get_EXIT1_PROBABILITY();


extern int h_env_EXIT1_PROBABILITY;

/** set_EXIT2_PROBABILITY
 * Sets the constant variable EXIT2_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT2_PROBABILITY value to set the variable
 */
extern void set_EXIT2_PROBABILITY(int* h_EXIT2_PROBABILITY);

extern const int* get_EXIT2_PROBABILITY();


extern int h_env_EXIT2_PROBABILITY;

/** set_EXIT3_PROBABILITY
 * Sets the constant variable EXIT3_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT3_PROBABILITY value to set the variable
 */
extern void set_EXIT3_PROBABILITY(int* h_EXIT3_PROBABILITY);

extern const int* get_EXIT3_PROBABILITY();


extern int h_env_EXIT3_PROBABILITY;

/** set_EXIT4_PROBABILITY
 * Sets the constant variable EXIT4_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT4_PROBABILITY value to set the variable
 */
extern void set_EXIT4_PROBABILITY(int* h_EXIT4_PROBABILITY);

extern const int* get_EXIT4_PROBABILITY();


extern int h_env_EXIT4_PROBABILITY;

/** set_EXIT5_PROBABILITY
 * Sets the constant variable EXIT5_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT5_PROBABILITY value to set the variable
 */
extern void set_EXIT5_PROBABILITY(int* h_EXIT5_PROBABILITY);

extern const int* get_EXIT5_PROBABILITY();


extern int h_env_EXIT5_PROBABILITY;

/** set_EXIT6_PROBABILITY
 * Sets the constant variable EXIT6_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT6_PROBABILITY value to set the variable
 */
extern void set_EXIT6_PROBABILITY(int* h_EXIT6_PROBABILITY);

extern const int* get_EXIT6_PROBABILITY();


extern int h_env_EXIT6_PROBABILITY;

/** set_EXIT7_PROBABILITY
 * Sets the constant variable EXIT7_PROBABILITY on the device which can then be used in the agent functions.
 * @param h_EXIT7_PROBABILITY value to set the variable
 */
extern void set_EXIT7_PROBABILITY(int* h_EXIT7_PROBABILITY);

extern const int* get_EXIT7_PROBABILITY();


extern int h_env_EXIT7_PROBABILITY;

/** set_EXIT1_STATE
 * Sets the constant variable EXIT1_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT1_STATE value to set the variable
 */
extern void set_EXIT1_STATE(int* h_EXIT1_STATE);

extern const int* get_EXIT1_STATE();


extern int h_env_EXIT1_STATE;

/** set_EXIT2_STATE
 * Sets the constant variable EXIT2_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT2_STATE value to set the variable
 */
extern void set_EXIT2_STATE(int* h_EXIT2_STATE);

extern const int* get_EXIT2_STATE();


extern int h_env_EXIT2_STATE;

/** set_EXIT3_STATE
 * Sets the constant variable EXIT3_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT3_STATE value to set the variable
 */
extern void set_EXIT3_STATE(int* h_EXIT3_STATE);

extern const int* get_EXIT3_STATE();


extern int h_env_EXIT3_STATE;

/** set_EXIT4_STATE
 * Sets the constant variable EXIT4_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT4_STATE value to set the variable
 */
extern void set_EXIT4_STATE(int* h_EXIT4_STATE);

extern const int* get_EXIT4_STATE();


extern int h_env_EXIT4_STATE;

/** set_EXIT5_STATE
 * Sets the constant variable EXIT5_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT5_STATE value to set the variable
 */
extern void set_EXIT5_STATE(int* h_EXIT5_STATE);

extern const int* get_EXIT5_STATE();


extern int h_env_EXIT5_STATE;

/** set_EXIT6_STATE
 * Sets the constant variable EXIT6_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT6_STATE value to set the variable
 */
extern void set_EXIT6_STATE(int* h_EXIT6_STATE);

extern const int* get_EXIT6_STATE();


extern int h_env_EXIT6_STATE;

/** set_EXIT7_STATE
 * Sets the constant variable EXIT7_STATE on the device which can then be used in the agent functions.
 * @param h_EXIT7_STATE value to set the variable
 */
extern void set_EXIT7_STATE(int* h_EXIT7_STATE);

extern const int* get_EXIT7_STATE();


extern int h_env_EXIT7_STATE;

/** set_EXIT1_CELL_COUNT
 * Sets the constant variable EXIT1_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT1_CELL_COUNT value to set the variable
 */
extern void set_EXIT1_CELL_COUNT(int* h_EXIT1_CELL_COUNT);

extern const int* get_EXIT1_CELL_COUNT();


extern int h_env_EXIT1_CELL_COUNT;

/** set_EXIT2_CELL_COUNT
 * Sets the constant variable EXIT2_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT2_CELL_COUNT value to set the variable
 */
extern void set_EXIT2_CELL_COUNT(int* h_EXIT2_CELL_COUNT);

extern const int* get_EXIT2_CELL_COUNT();


extern int h_env_EXIT2_CELL_COUNT;

/** set_EXIT3_CELL_COUNT
 * Sets the constant variable EXIT3_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT3_CELL_COUNT value to set the variable
 */
extern void set_EXIT3_CELL_COUNT(int* h_EXIT3_CELL_COUNT);

extern const int* get_EXIT3_CELL_COUNT();


extern int h_env_EXIT3_CELL_COUNT;

/** set_EXIT4_CELL_COUNT
 * Sets the constant variable EXIT4_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT4_CELL_COUNT value to set the variable
 */
extern void set_EXIT4_CELL_COUNT(int* h_EXIT4_CELL_COUNT);

extern const int* get_EXIT4_CELL_COUNT();


extern int h_env_EXIT4_CELL_COUNT;

/** set_EXIT5_CELL_COUNT
 * Sets the constant variable EXIT5_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT5_CELL_COUNT value to set the variable
 */
extern void set_EXIT5_CELL_COUNT(int* h_EXIT5_CELL_COUNT);

extern const int* get_EXIT5_CELL_COUNT();


extern int h_env_EXIT5_CELL_COUNT;

/** set_EXIT6_CELL_COUNT
 * Sets the constant variable EXIT6_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT6_CELL_COUNT value to set the variable
 */
extern void set_EXIT6_CELL_COUNT(int* h_EXIT6_CELL_COUNT);

extern const int* get_EXIT6_CELL_COUNT();


extern int h_env_EXIT6_CELL_COUNT;

/** set_EXIT7_CELL_COUNT
 * Sets the constant variable EXIT7_CELL_COUNT on the device which can then be used in the agent functions.
 * @param h_EXIT7_CELL_COUNT value to set the variable
 */
extern void set_EXIT7_CELL_COUNT(int* h_EXIT7_CELL_COUNT);

extern const int* get_EXIT7_CELL_COUNT();


extern int h_env_EXIT7_CELL_COUNT;

/** set_TIME_SCALER
 * Sets the constant variable TIME_SCALER on the device which can then be used in the agent functions.
 * @param h_TIME_SCALER value to set the variable
 */
extern void set_TIME_SCALER(float* h_TIME_SCALER);

extern const float* get_TIME_SCALER();


extern float h_env_TIME_SCALER;

/** set_STEER_WEIGHT
 * Sets the constant variable STEER_WEIGHT on the device which can then be used in the agent functions.
 * @param h_STEER_WEIGHT value to set the variable
 */
extern void set_STEER_WEIGHT(float* h_STEER_WEIGHT);

extern const float* get_STEER_WEIGHT();


extern float h_env_STEER_WEIGHT;

/** set_AVOID_WEIGHT
 * Sets the constant variable AVOID_WEIGHT on the device which can then be used in the agent functions.
 * @param h_AVOID_WEIGHT value to set the variable
 */
extern void set_AVOID_WEIGHT(float* h_AVOID_WEIGHT);

extern const float* get_AVOID_WEIGHT();


extern float h_env_AVOID_WEIGHT;

/** set_COLLISION_WEIGHT
 * Sets the constant variable COLLISION_WEIGHT on the device which can then be used in the agent functions.
 * @param h_COLLISION_WEIGHT value to set the variable
 */
extern void set_COLLISION_WEIGHT(float* h_COLLISION_WEIGHT);

extern const float* get_COLLISION_WEIGHT();


extern float h_env_COLLISION_WEIGHT;

/** set_GOAL_WEIGHT
 * Sets the constant variable GOAL_WEIGHT on the device which can then be used in the agent functions.
 * @param h_GOAL_WEIGHT value to set the variable
 */
extern void set_GOAL_WEIGHT(float* h_GOAL_WEIGHT);

extern const float* get_GOAL_WEIGHT();


extern float h_env_GOAL_WEIGHT;


/** getMaximumBound
 * Returns the maximum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the maximum x, y and z positions of all agents
 */
glm::vec3 getMaximumBounds();

/** getMinimumBounds
 * Returns the minimum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the minimum x, y and z positions of all agents
 */
glm::vec3 getMinimumBounds();
    
    
#ifdef VISUALISATION
/** initVisualisation
 * Prototype for method which initialises the visualisation. Must be implemented in separate file
 * @param argc	the argument count from the main function used with GLUT
 * @param argv	the argument values from the main function used with GLUT
 */
extern void initVisualisation();

extern void runVisualisation();


#endif

#if defined(PROFILE)
#include "nvToolsExt.h"

#define PROFILE_WHITE   0x00eeeeee
#define PROFILE_GREEN   0x0000ff00
#define PROFILE_BLUE    0x000000ff
#define PROFILE_YELLOW  0x00ffff00
#define PROFILE_MAGENTA 0x00ff00ff
#define PROFILE_CYAN    0x0000ffff
#define PROFILE_RED     0x00ff0000
#define PROFILE_GREY    0x00999999
#define PROFILE_LILAC   0xC8A2C8

const uint32_t profile_colors[] = {
  PROFILE_WHITE,
  PROFILE_GREEN,
  PROFILE_BLUE,
  PROFILE_YELLOW,
  PROFILE_MAGENTA,
  PROFILE_CYAN,
  PROFILE_RED,
  PROFILE_GREY,
  PROFILE_LILAC
};
const int num_profile_colors = sizeof(profile_colors) / sizeof(uint32_t);

// Externed value containing colour information.
extern unsigned int g_profile_colour_id;

#define PROFILE_PUSH_RANGE(name) { \
    unsigned int color_id = g_profile_colour_id % num_profile_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = profile_colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
    g_profile_colour_id++; \
}
#define PROFILE_POP_RANGE() nvtxRangePop();

// Class for simple fire-and-forget profile ranges (ie. functions with multiple return conditions.)
class ProfileScopedRange {
public:
    ProfileScopedRange(const char * name){
      PROFILE_PUSH_RANGE(name);
    }
    ~ProfileScopedRange(){
      PROFILE_POP_RANGE();
    }
};
#define PROFILE_SCOPED_RANGE(name) ProfileScopedRange uniq_name_using_macros(name);
#else
#define PROFILE_PUSH_RANGE(name)
#define PROFILE_POP_RANGE()
#define PROFILE_SCOPED_RANGE(name)
#endif


#endif //__HEADER

