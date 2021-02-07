
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
#define buffer_size_MAX 1024

//Maximum population size of xmachine_memory_Man
#define xmachine_memory_Man_MAX 1024

//Maximum population size of xmachine_memory_Woman
#define xmachine_memory_Woman_MAX 1024 
//Agent variable array length for xmachine_memory_Man->preferred_woman
#define xmachine_memory_Man_preferred_woman_LENGTH 1024 
//Agent variable array length for xmachine_memory_Woman->preferred_man
#define xmachine_memory_Woman_preferred_man_LENGTH 1024


  
  
/* Message population size definitions */
//Maximum population size of xmachine_mmessage_proposal
#define xmachine_message_proposal_MAX 1024

//Maximum population size of xmachine_mmessage_notification
#define xmachine_message_notification_MAX 1024


/* Define preprocessor symbols for each message to specify the type, to simplify / improve portability */

#define xmachine_message_proposal_partitioningNone
#define xmachine_message_notification_partitioningNone

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

/** struct xmachine_memory_Man
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Man
{
    int id;    /**< X-machine memory variable id of type int.*/
    int round;    /**< X-machine memory variable round of type int.*/
    int engaged_to;    /**< X-machine memory variable engaged_to of type int.*/
    int *preferred_woman;    /**< X-machine memory variable preferred_woman of type int.*/
};

/** struct xmachine_memory_Woman
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Woman
{
    int id;    /**< X-machine memory variable id of type int.*/
    int current_suitor;    /**< X-machine memory variable current_suitor of type int.*/
    int current_suitor_rank;    /**< X-machine memory variable current_suitor_rank of type int.*/
    int *preferred_man;    /**< X-machine memory variable preferred_man of type int.*/
};



/* Message structures */

/** struct xmachine_message_proposal
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_proposal
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    int id;        /**< Message variable id of type int.*/  
    int woman;        /**< Message variable woman of type int.*/
};

/** struct xmachine_message_notification
 * Brute force: No Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_notification
{	
    /* Brute force Partitioning Variables */
    int _position;          /**< 1D position of message in linear message list */   
      
    int id;        /**< Message variable id of type int.*/  
    int suitor;        /**< Message variable suitor of type int.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_Man_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_Man_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_Man_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_Man_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_memory_Man_MAX];    /**< X-machine memory variable list id of type int.*/
    int round [xmachine_memory_Man_MAX];    /**< X-machine memory variable list round of type int.*/
    int engaged_to [xmachine_memory_Man_MAX];    /**< X-machine memory variable list engaged_to of type int.*/
    int preferred_woman [xmachine_memory_Man_MAX*1024];    /**< X-machine memory variable list preferred_woman of type int.*/
};

/** struct xmachine_memory_Woman_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_Woman_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_Woman_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_Woman_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_memory_Woman_MAX];    /**< X-machine memory variable list id of type int.*/
    int current_suitor [xmachine_memory_Woman_MAX];    /**< X-machine memory variable list current_suitor of type int.*/
    int current_suitor_rank [xmachine_memory_Woman_MAX];    /**< X-machine memory variable list current_suitor_rank of type int.*/
    int preferred_man [xmachine_memory_Woman_MAX*1024];    /**< X-machine memory variable list preferred_man of type int.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_proposal_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_proposal_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_proposal_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_proposal_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_message_proposal_MAX];    /**< Message memory variable list id of type int.*/
    int woman [xmachine_message_proposal_MAX];    /**< Message memory variable list woman of type int.*/
    
};

/** struct xmachine_message_notification_list
 * Brute force: No Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_notification_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_notification_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_notification_MAX];  /**< Used during parallel prefix sum */
    
    int id [xmachine_message_notification_MAX];    /**< Message memory variable list id of type int.*/
    int suitor [xmachine_message_notification_MAX];    /**< Message memory variable list suitor of type int.*/
    
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
 * make_proposals FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Man. This represents a single agent instance and can be modified directly.
 * @param proposal_messages Pointer to output message list of type xmachine_message_proposal_list. Must be passed as an argument to the add_proposal_message function ??.
 */
__FLAME_GPU_FUNC__ int make_proposals(xmachine_memory_Man* agent, xmachine_message_proposal_list* proposal_messages);

/**
 * check_notifications FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Man. This represents a single agent instance and can be modified directly.
 * @param notification_messages  notification_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_notification_message and get_next_notification_message functions.
 */
__FLAME_GPU_FUNC__ int check_notifications(xmachine_memory_Man* agent, xmachine_message_notification_list* notification_messages);

/**
 * check_resolved FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Man. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int check_resolved(xmachine_memory_Man* agent);

/**
 * check_proposals FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Woman. This represents a single agent instance and can be modified directly.
 * @param proposal_messages  proposal_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_proposal_message and get_next_proposal_message functions.
 */
__FLAME_GPU_FUNC__ int check_proposals(xmachine_memory_Woman* agent, xmachine_message_proposal_list* proposal_messages);

/**
 * notify_suitors FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Woman. This represents a single agent instance and can be modified directly.
 * @param notification_messages Pointer to output message list of type xmachine_message_notification_list. Must be passed as an argument to the add_notification_message function ??.
 */
__FLAME_GPU_FUNC__ int notify_suitors(xmachine_memory_Woman* agent, xmachine_message_notification_list* notification_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) proposal message implemented in FLAMEGPU_Kernels */

/** add_proposal_message
 * Function for all types of message partitioning
 * Adds a new proposal agent to the xmachine_memory_proposal_list list using a linear mapping
 * @param agents	xmachine_memory_proposal_list agent list
 * @param id	message variable of type int
 * @param woman	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_proposal_message(xmachine_message_proposal_list* proposal_messages, int id, int woman);
 
/** get_first_proposal_message
 * Get first message function for non partitioned (brute force) messages
 * @param proposal_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_proposal * get_first_proposal_message(xmachine_message_proposal_list* proposal_messages);

/** get_next_proposal_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param proposal_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_proposal * get_next_proposal_message(xmachine_message_proposal* current, xmachine_message_proposal_list* proposal_messages);

  
/* Message Function Prototypes for Brute force (No Partitioning) notification message implemented in FLAMEGPU_Kernels */

/** add_notification_message
 * Function for all types of message partitioning
 * Adds a new notification agent to the xmachine_memory_notification_list list using a linear mapping
 * @param agents	xmachine_memory_notification_list agent list
 * @param id	message variable of type int
 * @param suitor	message variable of type int
 */
 
 __FLAME_GPU_FUNC__ void add_notification_message(xmachine_message_notification_list* notification_messages, int id, int suitor);
 
/** get_first_notification_message
 * Get first message function for non partitioned (brute force) messages
 * @param notification_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_notification * get_first_notification_message(xmachine_message_notification_list* notification_messages);

/** get_next_notification_message
 * Get first message function for non partitioned (brute force) messages
 * @param current the current message struct
 * @param notification_messages message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_notification * get_next_notification_message(xmachine_message_notification* current, xmachine_message_notification_list* notification_messages);

  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_Man_agent
 * Adds a new continuous valued Man agent to the xmachine_memory_Man_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Man_list agent list
 * @param id	agent agent variable of type int
 * @param round	agent agent variable of type int
 * @param engaged_to	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_Man_agent(xmachine_memory_Man_list* agents, int id, int round, int engaged_to);

/** get_Man_agent_array_value
 *  Template function for accessing Man agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_Man_agent_array_value(T *array, unsigned int index);

/** set_Man_agent_array_value
 *  Template function for setting Man agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_Man_agent_array_value(T *array, unsigned int index, T value);


  

/** add_Woman_agent
 * Adds a new continuous valued Woman agent to the xmachine_memory_Woman_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Woman_list agent list
 * @param id	agent agent variable of type int
 * @param current_suitor	agent agent variable of type int
 * @param current_suitor_rank	agent agent variable of type int
 */
__FLAME_GPU_FUNC__ void add_Woman_agent(xmachine_memory_Woman_list* agents, int id, int current_suitor, int current_suitor_rank);

/** get_Woman_agent_array_value
 *  Template function for accessing Woman agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_Woman_agent_array_value(T *array, unsigned int index);

/** set_Woman_agent_array_value
 *  Template function for setting Woman agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_Woman_agent_array_value(T *array, unsigned int index, T value);


  


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
 * @param h_Mans Pointer to agent list on the host
 * @param d_Mans Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Man_count Pointer to agent counter
 * @param h_Womans Pointer to agent list on the host
 * @param d_Womans Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Woman_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Man_list* h_Mans_unengaged, xmachine_memory_Man_list* d_Mans_unengaged, int h_xmachine_memory_Man_unengaged_count,xmachine_memory_Man_list* h_Mans_engaged, xmachine_memory_Man_list* d_Mans_engaged, int h_xmachine_memory_Man_engaged_count,xmachine_memory_Woman_list* h_Womans_default, xmachine_memory_Woman_list* d_Womans_default, int h_xmachine_memory_Woman_default_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_Mans Pointer to agent list on the host
 * @param h_xmachine_memory_Man_count Pointer to agent counter
 * @param h_Womans Pointer to agent list on the host
 * @param h_xmachine_memory_Woman_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_Man_list* h_Mans, int* h_xmachine_memory_Man_count,xmachine_memory_Woman_list* h_Womans, int* h_xmachine_memory_Woman_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_Man_MAX_count
 * Gets the max agent count for the Man agent type 
 * @return		the maximum Man agent count
 */
extern int get_agent_Man_MAX_count();



/** get_agent_Man_unengaged_count
 * Gets the agent count for the Man agent type in state unengaged
 * @return		the current Man agent count in state unengaged
 */
extern int get_agent_Man_unengaged_count();

/** reset_unengaged_count
 * Resets the agent count of the Man in state unengaged to 0. This is useful for interacting with some visualisations.
 */
extern void reset_Man_unengaged_count();

/** get_device_Man_unengaged_agents
 * Gets a pointer to xmachine_memory_Man_list on the GPU device
 * @return		a xmachine_memory_Man_list on the GPU device
 */
extern xmachine_memory_Man_list* get_device_Man_unengaged_agents();

/** get_host_Man_unengaged_agents
 * Gets a pointer to xmachine_memory_Man_list on the CPU host
 * @return		a xmachine_memory_Man_list on the CPU host
 */
extern xmachine_memory_Man_list* get_host_Man_unengaged_agents();


/** sort_Mans_unengaged
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Mans_unengaged(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Man_list* agents));


/** get_agent_Man_engaged_count
 * Gets the agent count for the Man agent type in state engaged
 * @return		the current Man agent count in state engaged
 */
extern int get_agent_Man_engaged_count();

/** reset_engaged_count
 * Resets the agent count of the Man in state engaged to 0. This is useful for interacting with some visualisations.
 */
extern void reset_Man_engaged_count();

/** get_device_Man_engaged_agents
 * Gets a pointer to xmachine_memory_Man_list on the GPU device
 * @return		a xmachine_memory_Man_list on the GPU device
 */
extern xmachine_memory_Man_list* get_device_Man_engaged_agents();

/** get_host_Man_engaged_agents
 * Gets a pointer to xmachine_memory_Man_list on the CPU host
 * @return		a xmachine_memory_Man_list on the CPU host
 */
extern xmachine_memory_Man_list* get_host_Man_engaged_agents();


/** sort_Mans_engaged
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Mans_engaged(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Man_list* agents));


    
/** get_agent_Woman_MAX_count
 * Gets the max agent count for the Woman agent type 
 * @return		the maximum Woman agent count
 */
extern int get_agent_Woman_MAX_count();



/** get_agent_Woman_default_count
 * Gets the agent count for the Woman agent type in state default
 * @return		the current Woman agent count in state default
 */
extern int get_agent_Woman_default_count();

/** reset_default_count
 * Resets the agent count of the Woman in state default to 0. This is useful for interacting with some visualisations.
 */
extern void reset_Woman_default_count();

/** get_device_Woman_default_agents
 * Gets a pointer to xmachine_memory_Woman_list on the GPU device
 * @return		a xmachine_memory_Woman_list on the GPU device
 */
extern xmachine_memory_Woman_list* get_device_Woman_default_agents();

/** get_host_Woman_default_agents
 * Gets a pointer to xmachine_memory_Woman_list on the CPU host
 * @return		a xmachine_memory_Woman_list on the CPU host
 */
extern xmachine_memory_Woman_list* get_host_Woman_default_agents();


/** sort_Womans_default
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Womans_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Woman_list* agents));



/* Host based access of agent variables*/

/** int get_Man_unengaged_variable_id(unsigned int index)
 * Gets the value of the id variable of an Man agent in the unengaged state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_Man_unengaged_variable_id(unsigned int index);

/** int get_Man_unengaged_variable_round(unsigned int index)
 * Gets the value of the round variable of an Man agent in the unengaged state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable round
 */
__host__ int get_Man_unengaged_variable_round(unsigned int index);

/** int get_Man_unengaged_variable_engaged_to(unsigned int index)
 * Gets the value of the engaged_to variable of an Man agent in the unengaged state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable engaged_to
 */
__host__ int get_Man_unengaged_variable_engaged_to(unsigned int index);

/** int get_Man_unengaged_variable_preferred_woman(unsigned int index, unsigned int element)
 * Gets the element-th value of the preferred_woman variable array of an Man agent in the unengaged state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable preferred_woman
 */
__host__ int get_Man_unengaged_variable_preferred_woman(unsigned int index, unsigned int element);

/** int get_Man_engaged_variable_id(unsigned int index)
 * Gets the value of the id variable of an Man agent in the engaged state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_Man_engaged_variable_id(unsigned int index);

/** int get_Man_engaged_variable_round(unsigned int index)
 * Gets the value of the round variable of an Man agent in the engaged state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable round
 */
__host__ int get_Man_engaged_variable_round(unsigned int index);

/** int get_Man_engaged_variable_engaged_to(unsigned int index)
 * Gets the value of the engaged_to variable of an Man agent in the engaged state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable engaged_to
 */
__host__ int get_Man_engaged_variable_engaged_to(unsigned int index);

/** int get_Man_engaged_variable_preferred_woman(unsigned int index, unsigned int element)
 * Gets the element-th value of the preferred_woman variable array of an Man agent in the engaged state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable preferred_woman
 */
__host__ int get_Man_engaged_variable_preferred_woman(unsigned int index, unsigned int element);

/** int get_Woman_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an Woman agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_Woman_default_variable_id(unsigned int index);

/** int get_Woman_default_variable_current_suitor(unsigned int index)
 * Gets the value of the current_suitor variable of an Woman agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_suitor
 */
__host__ int get_Woman_default_variable_current_suitor(unsigned int index);

/** int get_Woman_default_variable_current_suitor_rank(unsigned int index)
 * Gets the value of the current_suitor_rank variable of an Woman agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_suitor_rank
 */
__host__ int get_Woman_default_variable_current_suitor_rank(unsigned int index);

/** int get_Woman_default_variable_preferred_man(unsigned int index, unsigned int element)
 * Gets the element-th value of the preferred_man variable array of an Woman agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable preferred_man
 */
__host__ int get_Woman_default_variable_preferred_man(unsigned int index, unsigned int element);




/* Host based agent creation functions */

/** h_allocate_agent_Man
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated Man struct.
 */
xmachine_memory_Man* h_allocate_agent_Man();
/** h_free_agent_Man
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_Man(xmachine_memory_Man** agent);
/** h_allocate_agent_Man_array
 * Utility function to allocate an array of structs for  Man agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_Man** h_allocate_agent_Man_array(unsigned int count);
/** h_free_agent_Man_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_Man_array(xmachine_memory_Man*** agents, unsigned int count);


/** h_add_agent_Man_unengaged
 * Host function to add a single agent of type Man to the unengaged state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_Man_unengaged instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_Man_unengaged(xmachine_memory_Man* agent);

/** h_add_agents_Man_unengaged(
 * Host function to add multiple agents of type Man to the unengaged state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of Man agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_Man_unengaged(xmachine_memory_Man** agents, unsigned int count);


/** h_add_agent_Man_engaged
 * Host function to add a single agent of type Man to the engaged state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_Man_engaged instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_Man_engaged(xmachine_memory_Man* agent);

/** h_add_agents_Man_engaged(
 * Host function to add multiple agents of type Man to the engaged state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of Man agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_Man_engaged(xmachine_memory_Man** agents, unsigned int count);

/** h_allocate_agent_Woman
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated Woman struct.
 */
xmachine_memory_Woman* h_allocate_agent_Woman();
/** h_free_agent_Woman
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_Woman(xmachine_memory_Woman** agent);
/** h_allocate_agent_Woman_array
 * Utility function to allocate an array of structs for  Woman agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_Woman** h_allocate_agent_Woman_array(unsigned int count);
/** h_free_agent_Woman_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_Woman_array(xmachine_memory_Woman*** agents, unsigned int count);


/** h_add_agent_Woman_default
 * Host function to add a single agent of type Woman to the default state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_Woman_default instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_Woman_default(xmachine_memory_Woman* agent);

/** h_add_agents_Woman_default(
 * Host function to add multiple agents of type Woman to the default state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of Woman agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_Woman_default(xmachine_memory_Woman** agents, unsigned int count);

  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** int reduce_Man_unengaged_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Man_unengaged_id_variable();



/** int count_Man_unengaged_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Man_unengaged_id_variable(int count_value);

/** int min_Man_unengaged_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Man_unengaged_id_variable();
/** int max_Man_unengaged_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Man_unengaged_id_variable();

/** int reduce_Man_unengaged_round_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Man_unengaged_round_variable();



/** int count_Man_unengaged_round_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Man_unengaged_round_variable(int count_value);

/** int min_Man_unengaged_round_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Man_unengaged_round_variable();
/** int max_Man_unengaged_round_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Man_unengaged_round_variable();

/** int reduce_Man_unengaged_engaged_to_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Man_unengaged_engaged_to_variable();



/** int count_Man_unengaged_engaged_to_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Man_unengaged_engaged_to_variable(int count_value);

/** int min_Man_unengaged_engaged_to_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Man_unengaged_engaged_to_variable();
/** int max_Man_unengaged_engaged_to_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Man_unengaged_engaged_to_variable();

/** int reduce_Man_engaged_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Man_engaged_id_variable();



/** int count_Man_engaged_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Man_engaged_id_variable(int count_value);

/** int min_Man_engaged_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Man_engaged_id_variable();
/** int max_Man_engaged_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Man_engaged_id_variable();

/** int reduce_Man_engaged_round_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Man_engaged_round_variable();



/** int count_Man_engaged_round_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Man_engaged_round_variable(int count_value);

/** int min_Man_engaged_round_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Man_engaged_round_variable();
/** int max_Man_engaged_round_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Man_engaged_round_variable();

/** int reduce_Man_engaged_engaged_to_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Man_engaged_engaged_to_variable();



/** int count_Man_engaged_engaged_to_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Man_engaged_engaged_to_variable(int count_value);

/** int min_Man_engaged_engaged_to_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Man_engaged_engaged_to_variable();
/** int max_Man_engaged_engaged_to_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Man_engaged_engaged_to_variable();

/** int reduce_Woman_default_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Woman_default_id_variable();



/** int count_Woman_default_id_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Woman_default_id_variable(int count_value);

/** int min_Woman_default_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Woman_default_id_variable();
/** int max_Woman_default_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Woman_default_id_variable();

/** int reduce_Woman_default_current_suitor_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Woman_default_current_suitor_variable();



/** int count_Woman_default_current_suitor_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Woman_default_current_suitor_variable(int count_value);

/** int min_Woman_default_current_suitor_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Woman_default_current_suitor_variable();
/** int max_Woman_default_current_suitor_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Woman_default_current_suitor_variable();

/** int reduce_Woman_default_current_suitor_rank_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_Woman_default_current_suitor_rank_variable();



/** int count_Woman_default_current_suitor_rank_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
int count_Woman_default_current_suitor_rank_variable(int count_value);

/** int min_Woman_default_current_suitor_rank_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int min_Woman_default_current_suitor_rank_variable();
/** int max_Woman_default_current_suitor_rank_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
int max_Woman_default_current_suitor_rank_variable();


  
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
