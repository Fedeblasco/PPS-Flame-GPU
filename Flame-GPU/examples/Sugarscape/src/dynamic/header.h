
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


  
  
/* Message population size definitions */
//Maximum population size of xmachine_mmessage_cell_state
#define xmachine_message_cell_state_MAX 65536

//Maximum population size of xmachine_mmessage_movement_request
#define xmachine_message_movement_request_MAX 65536

//Maximum population size of xmachine_mmessage_movement_response
#define xmachine_message_movement_response_MAX 65536


/* Define preprocessor symbols for each message to specify the type, to simplify / improve portability */

#define xmachine_message_cell_state_partitioningDiscrete
#define xmachine_message_movement_request_partitioningDiscrete
#define xmachine_message_movement_response_partitioningDiscrete

/* Spatial partitioning grid size definitions */

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
 * discrete valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_agent
{
    int location_id;    /**< X-machine memory variable location_id of type int.*/
    int agent_id;    /**< X-machine memory variable agent_id of type int.*/
    int state;    /**< X-machine memory variable state of type int.*/
    int sugar_level;    /**< X-machine memory variable sugar_level of type int.*/
    int metabolism;    /**< X-machine memory variable metabolism of type int.*/
    int env_sugar_level;    /**< X-machine memory variable env_sugar_level of type int.*/
};



/* Message structures */

/** struct xmachine_message_cell_state
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_cell_state
{	
    /* Discrete Partitioning Variables */
    glm::ivec2 _position;         /**< 2D position of message*/
    glm::ivec2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int location_id;        /**< Message variable location_id of type int.*/  
    int state;        /**< Message variable state of type int.*/  
    int env_sugar_level;        /**< Message variable env_sugar_level of type int.*/
};

/** struct xmachine_message_movement_request
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_movement_request
{	
    /* Discrete Partitioning Variables */
    glm::ivec2 _position;         /**< 2D position of message*/
    glm::ivec2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int agent_id;        /**< Message variable agent_id of type int.*/  
    int location_id;        /**< Message variable location_id of type int.*/  
    int sugar_level;        /**< Message variable sugar_level of type int.*/  
    int metabolism;        /**< Message variable metabolism of type int.*/
};

/** struct xmachine_message_movement_response
 * Discrete Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_movement_response
{	
    /* Discrete Partitioning Variables */
    glm::ivec2 _position;         /**< 2D position of message*/
    glm::ivec2 _relative;         /**< 2D position of message relative to the agent (range +- radius) */  
      
    int location_id;        /**< Message variable location_id of type int.*/  
    int agent_id;        /**< Message variable agent_id of type int.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_agent_list
 * discrete valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_agent_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_agent_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_agent_MAX];  /**< Used during parallel prefix sum */
    
    int location_id [xmachine_memory_agent_MAX];    /**< X-machine memory variable list location_id of type int.*/
    int agent_id [xmachine_memory_agent_MAX];    /**< X-machine memory variable list agent_id of type int.*/
    int state [xmachine_memory_agent_MAX];    /**< X-machine memory variable list state of type int.*/
    int sugar_level [xmachine_memory_agent_MAX];    /**< X-machine memory variable list sugar_level of type int.*/
    int metabolism [xmachine_memory_agent_MAX];    /**< X-machine memory variable list metabolism of type int.*/
    int env_sugar_level [xmachine_memory_agent_MAX];    /**< X-machine memory variable list env_sugar_level of type int.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_cell_state_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_cell_state_list
{
    int location_id [xmachine_message_cell_state_MAX];    /**< Message memory variable list location_id of type int.*/
    int state [xmachine_message_cell_state_MAX];    /**< Message memory variable list state of type int.*/
    int env_sugar_level [xmachine_message_cell_state_MAX];    /**< Message memory variable list env_sugar_level of type int.*/
    
};

/** struct xmachine_message_movement_request_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_movement_request_list
{
    int agent_id [xmachine_message_movement_request_MAX];    /**< Message memory variable list agent_id of type int.*/
    int location_id [xmachine_message_movement_request_MAX];    /**< Message memory variable list location_id of type int.*/
    int sugar_level [xmachine_message_movement_request_MAX];    /**< Message memory variable list sugar_level of type int.*/
    int metabolism [xmachine_message_movement_request_MAX];    /**< Message memory variable list metabolism of type int.*/
    
};

/** struct xmachine_message_movement_response_list
 * Discrete Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_movement_response_list
{
    int location_id [xmachine_message_movement_response_MAX];    /**< Message memory variable list location_id of type int.*/
    int agent_id [xmachine_message_movement_response_MAX];    /**< Message memory variable list agent_id of type int.*/
    
};



/* Spatially Partitioned Message boundary Matrices */



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
 * metabolise_and_growback FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int metabolise_and_growback(xmachine_memory_agent* agent);

/**
 * output_cell_state FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param cell_state_messages Pointer to output message list of type xmachine_message_cell_state_list. Must be passed as an argument to the add_cell_state_message function ??.
 */
__FLAME_GPU_FUNC__ int output_cell_state(xmachine_memory_agent* agent, xmachine_message_cell_state_list* cell_state_messages);

/**
 * movement_request FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param cell_state_messages  cell_state_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_cell_state_message and get_next_cell_state_message functions.* @param movement_request_messages Pointer to output message list of type xmachine_message_movement_request_list. Must be passed as an argument to the add_movement_request_message function ??.
 */
__FLAME_GPU_FUNC__ int movement_request(xmachine_memory_agent* agent, xmachine_message_cell_state_list* cell_state_messages, xmachine_message_movement_request_list* movement_request_messages);

/**
 * movement_response FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param movement_request_messages  movement_request_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_movement_request_message and get_next_movement_request_message functions.* @param movement_response_messages Pointer to output message list of type xmachine_message_movement_response_list. Must be passed as an argument to the add_movement_response_message function ??.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int movement_response(xmachine_memory_agent* agent, xmachine_message_movement_request_list* movement_request_messages, xmachine_message_movement_response_list* movement_response_messages, RNG_rand48* rand48);

/**
 * movement_transaction FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param movement_response_messages  movement_response_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_movement_response_message and get_next_movement_response_message functions.
 */
__FLAME_GPU_FUNC__ int movement_transaction(xmachine_memory_agent* agent, xmachine_message_movement_response_list* movement_response_messages);

  
/* Message Function Prototypes for Discrete Partitioned cell_state message implemented in FLAMEGPU_Kernels */

/** add_cell_state_message
 * Function for all types of message partitioning
 * Adds a new cell_state agent to the xmachine_memory_cell_state_list list using a linear mapping
 * @param agents	xmachine_memory_cell_state_list agent list
 * @param location_id	message variable of type int
 * @param state	message variable of type int
 * @param env_sugar_level	message variable of type int
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_cell_state_message(xmachine_message_cell_state_list* cell_state_messages, int location_id, int state, int env_sugar_level);
 
/** get_first_cell_state_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param cell_state_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_cell_state * get_first_cell_state_message(xmachine_message_cell_state_list* cell_state_messages, int agentx, int agent_y);

/** get_next_cell_state_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param cell_state_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_cell_state * get_next_cell_state_message(xmachine_message_cell_state* current, xmachine_message_cell_state_list* cell_state_messages);

  
/* Message Function Prototypes for Discrete Partitioned movement_request message implemented in FLAMEGPU_Kernels */

/** add_movement_request_message
 * Function for all types of message partitioning
 * Adds a new movement_request agent to the xmachine_memory_movement_request_list list using a linear mapping
 * @param agents	xmachine_memory_movement_request_list agent list
 * @param agent_id	message variable of type int
 * @param location_id	message variable of type int
 * @param sugar_level	message variable of type int
 * @param metabolism	message variable of type int
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_movement_request_message(xmachine_message_movement_request_list* movement_request_messages, int agent_id, int location_id, int sugar_level, int metabolism);
 
/** get_first_movement_request_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param movement_request_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_movement_request * get_first_movement_request_message(xmachine_message_movement_request_list* movement_request_messages, int agentx, int agent_y);

/** get_next_movement_request_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param movement_request_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_movement_request * get_next_movement_request_message(xmachine_message_movement_request* current, xmachine_message_movement_request_list* movement_request_messages);

  
/* Message Function Prototypes for Discrete Partitioned movement_response message implemented in FLAMEGPU_Kernels */

/** add_movement_response_message
 * Function for all types of message partitioning
 * Adds a new movement_response agent to the xmachine_memory_movement_response_list list using a linear mapping
 * @param agents	xmachine_memory_movement_response_list agent list
 * @param location_id	message variable of type int
 * @param agent_id	message variable of type int
 */
 template <int AGENT_TYPE>
 __FLAME_GPU_FUNC__ void add_movement_response_message(xmachine_message_movement_response_list* movement_response_messages, int location_id, int agent_id);
 
/** get_first_movement_response_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param movement_response_messages message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_movement_response * get_first_movement_response_message(xmachine_message_movement_response_list* movement_response_messages, int agentx, int agent_y);

/** get_next_movement_response_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param movement_response_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
template <int AGENT_TYPE> __FLAME_GPU_FUNC__ xmachine_message_movement_response * get_next_movement_response_message(xmachine_message_movement_response* current, xmachine_message_movement_response_list* movement_response_messages);

  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */


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
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_agent_list* h_agents_default, xmachine_memory_agent_list* d_agents_default, int h_xmachine_memory_agent_default_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_agents Pointer to agent list on the host
 * @param h_xmachine_memory_agent_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_agent_list* h_agents, int* h_xmachine_memory_agent_count);


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


/** get_agent_population_width
 * Gets an int value representing the xmachine_memory_agent population width.
 * @return		xmachine_memory_agent population width
 */
extern int get_agent_population_width();


/* Host based access of agent variables*/

/** int get_agent_default_variable_location_id(unsigned int index)
 * Gets the value of the location_id variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable location_id
 */
__host__ int get_agent_default_variable_location_id(unsigned int index);

/** int get_agent_default_variable_agent_id(unsigned int index)
 * Gets the value of the agent_id variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable agent_id
 */
__host__ int get_agent_default_variable_agent_id(unsigned int index);

/** int get_agent_default_variable_state(unsigned int index)
 * Gets the value of the state variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable state
 */
__host__ int get_agent_default_variable_state(unsigned int index);

/** int get_agent_default_variable_sugar_level(unsigned int index)
 * Gets the value of the sugar_level variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable sugar_level
 */
__host__ int get_agent_default_variable_sugar_level(unsigned int index);

/** int get_agent_default_variable_metabolism(unsigned int index)
 * Gets the value of the metabolism variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable metabolism
 */
__host__ int get_agent_default_variable_metabolism(unsigned int index);

/** int get_agent_default_variable_env_sugar_level(unsigned int index)
 * Gets the value of the env_sugar_level variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable env_sugar_level
 */
__host__ int get_agent_default_variable_env_sugar_level(unsigned int index);




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

  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** int reduce_agent_default_location_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_location_id_variable();



/** int count_agent_default_location_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_location_id_variable(int count_value);

/** int min_agent_default_location_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_location_id_variable();
/** int max_agent_default_location_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_location_id_variable();

/** int reduce_agent_default_agent_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_agent_id_variable();



/** int count_agent_default_agent_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_agent_id_variable(int count_value);

/** int min_agent_default_agent_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_agent_id_variable();
/** int max_agent_default_agent_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_agent_id_variable();

/** int reduce_agent_default_state_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_state_variable();



/** int count_agent_default_state_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_state_variable(int count_value);

/** int min_agent_default_state_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_state_variable();
/** int max_agent_default_state_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_state_variable();

/** int reduce_agent_default_sugar_level_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_sugar_level_variable();



/** int count_agent_default_sugar_level_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_sugar_level_variable(int count_value);

/** int min_agent_default_sugar_level_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_sugar_level_variable();
/** int max_agent_default_sugar_level_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_sugar_level_variable();

/** int reduce_agent_default_metabolism_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_metabolism_variable();



/** int count_agent_default_metabolism_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_metabolism_variable(int count_value);

/** int min_agent_default_metabolism_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_metabolism_variable();
/** int max_agent_default_metabolism_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_metabolism_variable();

/** int reduce_agent_default_env_sugar_level_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_agent_default_env_sugar_level_variable();



/** int count_agent_default_env_sugar_level_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_agent_default_env_sugar_level_variable(int count_value);

/** int min_agent_default_env_sugar_level_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_agent_default_env_sugar_level_variable();
/** int max_agent_default_env_sugar_level_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_agent_default_env_sugar_level_variable();


  
/* global constant variables */


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

