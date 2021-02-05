
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
#define buffer_size_MAX 2048

//Maximum population size of xmachine_memory_Agent
#define xmachine_memory_Agent_MAX 2048


  
  
/* Message population size definitions */
//Maximum population size of xmachine_mmessage_location
#define xmachine_message_location_MAX 2048

//Maximum population size of xmachine_mmessage_intent
#define xmachine_message_intent_MAX 2048


/* Define preprocessor symbols for each message to specify the type, to simplify / improve portability */

#define xmachine_message_location_partitioningGraphEdge
#define xmachine_message_intent_partitioningGraphEdge

/* Spatial partitioning grid size definitions */

/* Static Graph size definitions*/
#define staticGraph_network_vertex_bufferSize 1024256
#define staticGraph_network_edge_bufferSize 256
  

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

/** struct xmachine_memory_Agent
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_Agent
{
    unsigned int id;    /**< X-machine memory variable id of type unsigned int.*/
    unsigned int currentEdge;    /**< X-machine memory variable currentEdge of type unsigned int.*/
    unsigned int nextEdge;    /**< X-machine memory variable nextEdge of type unsigned int.*/
    unsigned int nextEdgeRemainingCapacity;    /**< X-machine memory variable nextEdgeRemainingCapacity of type unsigned int.*/
    bool hasIntent;    /**< X-machine memory variable hasIntent of type bool.*/
    float position;    /**< X-machine memory variable position of type float.*/
    float distanceTravelled;    /**< X-machine memory variable distanceTravelled of type float.*/
    unsigned int blockedIterationCount;    /**< X-machine memory variable blockedIterationCount of type unsigned int.*/
    float speed;    /**< X-machine memory variable speed of type float.*/
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    float z;    /**< X-machine memory variable z of type float.*/
    float colour;    /**< X-machine memory variable colour of type float.*/
};



/* Message structures */

/** struct xmachine_message_location
 * Graph Edge Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_location
{	
    /* Graph Edge partitioning Variables */
    unsigned int _position;          /**< 1D position of message in linear message list.*/   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/  
    unsigned int edge_id;        /**< Message variable edge_id of type unsigned int.*/  
    float position;        /**< Message variable position of type float.*/
};

/** struct xmachine_message_intent
 * Graph Edge Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_intent
{	
    /* Graph Edge partitioning Variables */
    unsigned int _position;          /**< 1D position of message in linear message list.*/   
      
    unsigned int id;        /**< Message variable id of type unsigned int.*/  
    unsigned int edge_id;        /**< Message variable edge_id of type unsigned int.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_Agent_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_Agent_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_Agent_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_Agent_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list id of type unsigned int.*/
    unsigned int currentEdge [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list currentEdge of type unsigned int.*/
    unsigned int nextEdge [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list nextEdge of type unsigned int.*/
    unsigned int nextEdgeRemainingCapacity [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list nextEdgeRemainingCapacity of type unsigned int.*/
    bool hasIntent [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list hasIntent of type bool.*/
    float position [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list position of type float.*/
    float distanceTravelled [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list distanceTravelled of type float.*/
    unsigned int blockedIterationCount [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list blockedIterationCount of type unsigned int.*/
    float speed [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list speed of type float.*/
    float x [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list y of type float.*/
    float z [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list z of type float.*/
    float colour [xmachine_memory_Agent_MAX];    /**< X-machine memory variable list colour of type float.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_location_list
 * Graph Edge Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_location_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_location_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_location_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_location_MAX];    /**< Message memory variable list id of type unsigned int.*/
    unsigned int edge_id [xmachine_message_location_MAX];    /**< Message memory variable list edge_id of type unsigned int.*/
    float position [xmachine_message_location_MAX];    /**< Message memory variable list position of type float.*/
    
};

/** struct xmachine_message_intent_list
 * Graph Edge Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_intent_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_intent_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_intent_MAX];  /**< Used during parallel prefix sum */
    
    unsigned int id [xmachine_message_intent_MAX];    /**< Message memory variable list id of type unsigned int.*/
    unsigned int edge_id [xmachine_message_intent_MAX];    /**< Message memory variable list edge_id of type unsigned int.*/
    
};



/* Spatially Partitioned Message boundary Matrices */



/* Graph structures */



/*
 * struct staticGraph_memory_network
 * Graph Data structure for the static graph network. 
 * Struct of Array style storage for vertex data, edge data and CSR index per Vertex
 */
struct staticGraph_memory_network
{
    struct vertex
    {
        unsigned int count;
        unsigned int id[staticGraph_network_vertex_bufferSize];
        float x[staticGraph_network_vertex_bufferSize];
        float y[staticGraph_network_vertex_bufferSize];
        float z[staticGraph_network_vertex_bufferSize];
        
        // CSR data structure variable
        unsigned int first_edge_index[staticGraph_network_vertex_bufferSize + 1];
    } vertex;
    struct edge
    {
        unsigned int count;
        unsigned int id[staticGraph_network_edge_bufferSize];
        unsigned int source[staticGraph_network_edge_bufferSize];
        unsigned int destination[staticGraph_network_edge_bufferSize];
        float length[staticGraph_network_edge_bufferSize];
        unsigned int capacity[staticGraph_network_edge_bufferSize];
        
    } edge;
};


/* Graph Edge Partitioned message boundary structures */
/** struct xmachine_message_location_bounds
 * Graph Communication boundary data structure, used to access messages for the correct edge.
 * Contains an array of the first message index per edge, and an array containing the number of messages per edge. 
 */
struct xmachine_message_location_bounds
{
    unsigned int start[staticGraph_network_edge_bufferSize];
    unsigned int count[staticGraph_network_edge_bufferSize];
};

/** struct xmachine_message_location_scatterer
 * Graph Communication temporary data structure, used during the scattering of message data from the output location to the sorted location
 */
struct xmachine_message_location_scatterer
{
    unsigned int edge_local_index[xmachine_message_location_MAX];
    unsigned int unsorted_edge_index[xmachine_message_location_MAX];
};

/** struct xmachine_message_intent_bounds
 * Graph Communication boundary data structure, used to access messages for the correct edge.
 * Contains an array of the first message index per edge, and an array containing the number of messages per edge. 
 */
struct xmachine_message_intent_bounds
{
    unsigned int start[staticGraph_network_edge_bufferSize];
    unsigned int count[staticGraph_network_edge_bufferSize];
};

/** struct xmachine_message_intent_scatterer
 * Graph Communication temporary data structure, used during the scattering of message data from the output location to the sorted location
 */
struct xmachine_message_intent_scatterer
{
    unsigned int edge_local_index[xmachine_message_intent_MAX];
    unsigned int unsorted_edge_index[xmachine_message_intent_MAX];
};



/* Graph utility functions, usable in agent functions and implemented in FLAMEGPU_Kernels */

__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_vertex_count();
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_edge_count();
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_vertex_first_edge_index(unsigned int index);
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_vertex_num_edges(unsigned int index);

__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_vertex_id(unsigned int vertexIndex);
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ float get_staticGraph_network_vertex_x(unsigned int vertexIndex);
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ float get_staticGraph_network_vertex_y(unsigned int vertexIndex);
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ float get_staticGraph_network_vertex_z(unsigned int vertexIndex);
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_edge_id(unsigned int edgeIndex);
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_edge_source(unsigned int edgeIndex);
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_edge_destination(unsigned int edgeIndex);
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ float get_staticGraph_network_edge_length(unsigned int edgeIndex);
__FLAME_GPU_HOST_FUNC__ __FLAME_GPU_FUNC__ unsigned int get_staticGraph_network_edge_capacity(unsigned int edgeIndex);


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
 * output_location FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Agent. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.
 */
__FLAME_GPU_FUNC__ int output_location(xmachine_memory_Agent* agent, xmachine_message_location_list* location_messages);

/**
 * read_locations FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Agent. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param intent_messages Pointer to output message list of type xmachine_message_intent_list. Must be passed as an argument to the add_intent_message function ??.
 */
__FLAME_GPU_FUNC__ int read_locations(xmachine_memory_Agent* agent, xmachine_message_location_list* location_messages, xmachine_message_location_bounds* message_bounds, xmachine_message_intent_list* intent_messages);

/**
 * resolve_intent FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Agent. This represents a single agent instance and can be modified directly.
 * @param intent_messages  intent_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_intent_message and get_next_intent_message functions.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int resolve_intent(xmachine_memory_Agent* agent, xmachine_message_intent_list* intent_messages, xmachine_message_intent_bounds* message_bounds, RNG_rand48* rand48);

/**
 * move FLAMEGPU Agent Function
 * @param agent Pointer to an agent structure of type xmachine_memory_Agent. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_Agent* agent);

  
/* Message Function Prototypes for On-Graph Partitioned location message implemented in FLAMEGPU_Kernels */

/** add_location_message
 * Function for all types of message partitioning
 * Adds a new location agent to the xmachine_memory_location_list list using a linear mapping
 * @param agents	xmachine_memory_location_list agent list
 * @param id	message variable of type unsigned int
 * @param edge_id	message variable of type unsigned int
 * @param position	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_location_message(xmachine_message_location_list* location_messages, unsigned int id, unsigned int edge_id, float position);
 
/** get_first_location_message
 * Get first message function for graph edge partitioned communication messages
 * @param location_messages message list 
 * @param message_bounds boundary structure providing information about each message
 * @param edge_id edge index to retrieve messages for
 * @return returns the first message from the message list 
 */
__FLAME_GPU_FUNC__ xmachine_message_location * get_first_location_message(xmachine_message_location_list* location_messages, xmachine_message_location_bounds* message_bounds, unsigned int edge_id);

/** get_next_location_message
 * Get next message function for graph edge partitioned communication messages
 * @param current The current message
 * @param location_messages list of messages
 * @param message_bounds boundary structure providing information about each message
 * @return returns the next message from the message list 
 */
__FLAME_GPU_FUNC__ xmachine_message_location * get_next_location_message(xmachine_message_location* current, xmachine_message_location_list* location_messages, xmachine_message_location_bounds* message_bounds);

  
/* Message Function Prototypes for On-Graph Partitioned intent message implemented in FLAMEGPU_Kernels */

/** add_intent_message
 * Function for all types of message partitioning
 * Adds a new intent agent to the xmachine_memory_intent_list list using a linear mapping
 * @param agents	xmachine_memory_intent_list agent list
 * @param id	message variable of type unsigned int
 * @param edge_id	message variable of type unsigned int
 */
 
 __FLAME_GPU_FUNC__ void add_intent_message(xmachine_message_intent_list* intent_messages, unsigned int id, unsigned int edge_id);
 
/** get_first_intent_message
 * Get first message function for graph edge partitioned communication messages
 * @param intent_messages message list 
 * @param message_bounds boundary structure providing information about each message
 * @param edge_id edge index to retrieve messages for
 * @return returns the first message from the message list 
 */
__FLAME_GPU_FUNC__ xmachine_message_intent * get_first_intent_message(xmachine_message_intent_list* intent_messages, xmachine_message_intent_bounds* message_bounds, unsigned int edge_id);

/** get_next_intent_message
 * Get next message function for graph edge partitioned communication messages
 * @param current The current message
 * @param intent_messages list of messages
 * @param message_bounds boundary structure providing information about each message
 * @return returns the next message from the message list 
 */
__FLAME_GPU_FUNC__ xmachine_message_intent * get_next_intent_message(xmachine_message_intent* current, xmachine_message_intent_list* intent_messages, xmachine_message_intent_bounds* message_bounds);

  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_Agent_agent
 * Adds a new continuous valued Agent agent to the xmachine_memory_Agent_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_Agent_list agent list
 * @param id	agent agent variable of type unsigned int
 * @param currentEdge	agent agent variable of type unsigned int
 * @param nextEdge	agent agent variable of type unsigned int
 * @param nextEdgeRemainingCapacity	agent agent variable of type unsigned int
 * @param hasIntent	agent agent variable of type bool
 * @param position	agent agent variable of type float
 * @param distanceTravelled	agent agent variable of type float
 * @param blockedIterationCount	agent agent variable of type unsigned int
 * @param speed	agent agent variable of type float
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param z	agent agent variable of type float
 * @param colour	agent agent variable of type float
 */
__FLAME_GPU_FUNC__ void add_Agent_agent(xmachine_memory_Agent_list* agents, unsigned int id, unsigned int currentEdge, unsigned int nextEdge, unsigned int nextEdgeRemainingCapacity, bool hasIntent, float position, float distanceTravelled, unsigned int blockedIterationCount, float speed, float x, float y, float z, float colour);


/* Graph loading function prototypes implemented in io.cu */
void load_staticGraph_network_from_json(const char* file, staticGraph_memory_network* h_staticGraph_memory_network);



  
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
 * @param h_Agents Pointer to agent list on the host
 * @param d_Agents Pointer to agent list on the GPU device
 * @param h_xmachine_memory_Agent_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Agent_list* h_Agents_default, xmachine_memory_Agent_list* d_Agents_default, int h_xmachine_memory_Agent_default_count);


/** readInitialStates
 * Reads the current agent data from the device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_Agents Pointer to agent list on the host
 * @param h_xmachine_memory_Agent_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_Agent_list* h_Agents, int* h_xmachine_memory_Agent_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_Agent_MAX_count
 * Gets the max agent count for the Agent agent type 
 * @return		the maximum Agent agent count
 */
extern int get_agent_Agent_MAX_count();



/** get_agent_Agent_default_count
 * Gets the agent count for the Agent agent type in state default
 * @return		the current Agent agent count in state default
 */
extern int get_agent_Agent_default_count();

/** reset_default_count
 * Resets the agent count of the Agent in state default to 0. This is useful for interacting with some visualisations.
 */
extern void reset_Agent_default_count();

/** get_device_Agent_default_agents
 * Gets a pointer to xmachine_memory_Agent_list on the GPU device
 * @return		a xmachine_memory_Agent_list on the GPU device
 */
extern xmachine_memory_Agent_list* get_device_Agent_default_agents();

/** get_host_Agent_default_agents
 * Gets a pointer to xmachine_memory_Agent_list on the CPU host
 * @return		a xmachine_memory_Agent_list on the CPU host
 */
extern xmachine_memory_Agent_list* get_host_Agent_default_agents();


/** sort_Agents_default
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_Agents_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_Agent_list* agents));



/* Host based access of agent variables*/

/** unsigned int get_Agent_default_variable_id(unsigned int index)
 * Gets the value of the id variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_Agent_default_variable_id(unsigned int index);

/** unsigned int get_Agent_default_variable_currentEdge(unsigned int index)
 * Gets the value of the currentEdge variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable currentEdge
 */
__host__ unsigned int get_Agent_default_variable_currentEdge(unsigned int index);

/** unsigned int get_Agent_default_variable_nextEdge(unsigned int index)
 * Gets the value of the nextEdge variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable nextEdge
 */
__host__ unsigned int get_Agent_default_variable_nextEdge(unsigned int index);

/** unsigned int get_Agent_default_variable_nextEdgeRemainingCapacity(unsigned int index)
 * Gets the value of the nextEdgeRemainingCapacity variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable nextEdgeRemainingCapacity
 */
__host__ unsigned int get_Agent_default_variable_nextEdgeRemainingCapacity(unsigned int index);

/** bool get_Agent_default_variable_hasIntent(unsigned int index)
 * Gets the value of the hasIntent variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable hasIntent
 */
__host__ bool get_Agent_default_variable_hasIntent(unsigned int index);

/** float get_Agent_default_variable_position(unsigned int index)
 * Gets the value of the position variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable position
 */
__host__ float get_Agent_default_variable_position(unsigned int index);

/** float get_Agent_default_variable_distanceTravelled(unsigned int index)
 * Gets the value of the distanceTravelled variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable distanceTravelled
 */
__host__ float get_Agent_default_variable_distanceTravelled(unsigned int index);

/** unsigned int get_Agent_default_variable_blockedIterationCount(unsigned int index)
 * Gets the value of the blockedIterationCount variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable blockedIterationCount
 */
__host__ unsigned int get_Agent_default_variable_blockedIterationCount(unsigned int index);

/** float get_Agent_default_variable_speed(unsigned int index)
 * Gets the value of the speed variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable speed
 */
__host__ float get_Agent_default_variable_speed(unsigned int index);

/** float get_Agent_default_variable_x(unsigned int index)
 * Gets the value of the x variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ float get_Agent_default_variable_x(unsigned int index);

/** float get_Agent_default_variable_y(unsigned int index)
 * Gets the value of the y variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ float get_Agent_default_variable_y(unsigned int index);

/** float get_Agent_default_variable_z(unsigned int index)
 * Gets the value of the z variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable z
 */
__host__ float get_Agent_default_variable_z(unsigned int index);

/** float get_Agent_default_variable_colour(unsigned int index)
 * Gets the value of the colour variable of an Agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable colour
 */
__host__ float get_Agent_default_variable_colour(unsigned int index);




/* Host based agent creation functions */

/** h_allocate_agent_Agent
 * Utility function to allocate and initialise an agent struct on the host.
 * @return address of a host-allocated Agent struct.
 */
xmachine_memory_Agent* h_allocate_agent_Agent();
/** h_free_agent_Agent
 * Utility function to free a host-allocated agent struct.
 * This also deallocates any agent variable arrays, and sets the pointer to null
 * @param agent address of pointer to the host allocated struct
 */
void h_free_agent_Agent(xmachine_memory_Agent** agent);
/** h_allocate_agent_Agent_array
 * Utility function to allocate an array of structs for  Agent agents.
 * @param count the number of structs to allocate memory for.
 * @return pointer to the allocated array of structs
 */
xmachine_memory_Agent** h_allocate_agent_Agent_array(unsigned int count);
/** h_free_agent_Agent_array(
 * Utility function to deallocate a host array of agent structs, including agent variables, and set pointer values to NULL.
 * @param agents the address of the pointer to the host array of structs.
 * @param count the number of elements in the AoS, to deallocate individual elements.
 */
void h_free_agent_Agent_array(xmachine_memory_Agent*** agents, unsigned int count);


/** h_add_agent_Agent_default
 * Host function to add a single agent of type Agent to the default state on the device.
 * This invokes many cudaMempcy, and an append kernel launch. 
 * If multiple agents are to be created in a single iteration, consider h_add_agent_Agent_default instead.
 * @param agent pointer to agent struct on the host. Agent member arrays are supported.
 */
void h_add_agent_Agent_default(xmachine_memory_Agent* agent);

/** h_add_agents_Agent_default(
 * Host function to add multiple agents of type Agent to the default state on the device if possible.
 * This includes the transparent conversion from AoS to SoA, many calls to cudaMemcpy and an append kernel.
 * @param agents pointer to host struct of arrays of Agent agents
 * @param count the number of agents to copy from the host to the device.
 */
void h_add_agents_Agent_default(xmachine_memory_Agent** agents, unsigned int count);

  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** unsigned int reduce_Agent_default_id_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Agent_default_id_variable();



/** unsigned int count_Agent_default_id_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Agent_default_id_variable(unsigned int count_value);

/** unsigned int min_Agent_default_id_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Agent_default_id_variable();
/** unsigned int max_Agent_default_id_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Agent_default_id_variable();

/** unsigned int reduce_Agent_default_currentEdge_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Agent_default_currentEdge_variable();



/** unsigned int count_Agent_default_currentEdge_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Agent_default_currentEdge_variable(unsigned int count_value);

/** unsigned int min_Agent_default_currentEdge_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Agent_default_currentEdge_variable();
/** unsigned int max_Agent_default_currentEdge_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Agent_default_currentEdge_variable();

/** unsigned int reduce_Agent_default_nextEdge_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Agent_default_nextEdge_variable();



/** unsigned int count_Agent_default_nextEdge_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Agent_default_nextEdge_variable(unsigned int count_value);

/** unsigned int min_Agent_default_nextEdge_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Agent_default_nextEdge_variable();
/** unsigned int max_Agent_default_nextEdge_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Agent_default_nextEdge_variable();

/** unsigned int reduce_Agent_default_nextEdgeRemainingCapacity_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Agent_default_nextEdgeRemainingCapacity_variable();



/** unsigned int count_Agent_default_nextEdgeRemainingCapacity_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Agent_default_nextEdgeRemainingCapacity_variable(unsigned int count_value);

/** unsigned int min_Agent_default_nextEdgeRemainingCapacity_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Agent_default_nextEdgeRemainingCapacity_variable();
/** unsigned int max_Agent_default_nextEdgeRemainingCapacity_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Agent_default_nextEdgeRemainingCapacity_variable();

/** bool reduce_Agent_default_hasIntent_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
bool reduce_Agent_default_hasIntent_variable();



/** bool min_Agent_default_hasIntent_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
bool min_Agent_default_hasIntent_variable();
/** bool max_Agent_default_hasIntent_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
bool max_Agent_default_hasIntent_variable();

/** float reduce_Agent_default_position_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Agent_default_position_variable();



/** float min_Agent_default_position_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Agent_default_position_variable();
/** float max_Agent_default_position_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Agent_default_position_variable();

/** float reduce_Agent_default_distanceTravelled_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Agent_default_distanceTravelled_variable();



/** float min_Agent_default_distanceTravelled_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Agent_default_distanceTravelled_variable();
/** float max_Agent_default_distanceTravelled_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Agent_default_distanceTravelled_variable();

/** unsigned int reduce_Agent_default_blockedIterationCount_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
unsigned int reduce_Agent_default_blockedIterationCount_variable();



/** unsigned int count_Agent_default_blockedIterationCount_variable(unsigned int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state variable list
 */
unsigned int count_Agent_default_blockedIterationCount_variable(unsigned int count_value);

/** unsigned int min_Agent_default_blockedIterationCount_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int min_Agent_default_blockedIterationCount_variable();
/** unsigned int max_Agent_default_blockedIterationCount_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
unsigned int max_Agent_default_blockedIterationCount_variable();

/** float reduce_Agent_default_speed_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Agent_default_speed_variable();



/** float min_Agent_default_speed_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Agent_default_speed_variable();
/** float max_Agent_default_speed_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Agent_default_speed_variable();

/** float reduce_Agent_default_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Agent_default_x_variable();



/** float min_Agent_default_x_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Agent_default_x_variable();
/** float max_Agent_default_x_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Agent_default_x_variable();

/** float reduce_Agent_default_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Agent_default_y_variable();



/** float min_Agent_default_y_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Agent_default_y_variable();
/** float max_Agent_default_y_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Agent_default_y_variable();

/** float reduce_Agent_default_z_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Agent_default_z_variable();



/** float min_Agent_default_z_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Agent_default_z_variable();
/** float max_Agent_default_z_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Agent_default_z_variable();

/** float reduce_Agent_default_colour_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_Agent_default_colour_variable();



/** float min_Agent_default_colour_variable();
 * Min functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float min_Agent_default_colour_variable();
/** float max_Agent_default_colour_variable();
 * Max functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the minimum variable value of the specified agent name and state
 */
float max_Agent_default_colour_variable();


  
/* global constant variables */

__constant__ unsigned int SEED;

__constant__ unsigned int INIT_POPULATION;

__constant__ float PARAM_MIN_SPEED;

__constant__ float PARAM_MAX_SPEED;

/** set_SEED
 * Sets the constant variable SEED on the device which can then be used in the agent functions.
 * @param h_SEED value to set the variable
 */
extern void set_SEED(unsigned int* h_SEED);

extern const unsigned int* get_SEED();


extern unsigned int h_env_SEED;

/** set_INIT_POPULATION
 * Sets the constant variable INIT_POPULATION on the device which can then be used in the agent functions.
 * @param h_INIT_POPULATION value to set the variable
 */
extern void set_INIT_POPULATION(unsigned int* h_INIT_POPULATION);

extern const unsigned int* get_INIT_POPULATION();


extern unsigned int h_env_INIT_POPULATION;

/** set_PARAM_MIN_SPEED
 * Sets the constant variable PARAM_MIN_SPEED on the device which can then be used in the agent functions.
 * @param h_PARAM_MIN_SPEED value to set the variable
 */
extern void set_PARAM_MIN_SPEED(float* h_PARAM_MIN_SPEED);

extern const float* get_PARAM_MIN_SPEED();


extern float h_env_PARAM_MIN_SPEED;

/** set_PARAM_MAX_SPEED
 * Sets the constant variable PARAM_MAX_SPEED on the device which can then be used in the agent functions.
 * @param h_PARAM_MAX_SPEED value to set the variable
 */
extern void set_PARAM_MAX_SPEED(float* h_PARAM_MAX_SPEED);

extern const float* get_PARAM_MAX_SPEED();


extern float h_env_PARAM_MAX_SPEED;


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

