
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

/* chair Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_chair_list* d_chairs;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_chair_list* d_chairs_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_chair_list* d_chairs_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_chair_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_chair_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_chair_values;  /**< Agent sort identifiers value */

/* chair state variables */
xmachine_memory_chair_list* h_chairs_defaultChair;      /**< Pointer to agent list (population) on host*/
xmachine_memory_chair_list* d_chairs_defaultChair;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_chair_defaultChair_count;   /**< Agent population size counter */ 

/* doctor_manager Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_doctor_manager_list* d_doctor_managers;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_doctor_manager_list* d_doctor_managers_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_doctor_manager_list* d_doctor_managers_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_doctor_manager_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_doctor_manager_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_doctor_manager_values;  /**< Agent sort identifiers value */

/* doctor_manager state variables */
xmachine_memory_doctor_manager_list* h_doctor_managers_defaultDoctorManager;      /**< Pointer to agent list (population) on host*/
xmachine_memory_doctor_manager_list* d_doctor_managers_defaultDoctorManager;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_doctor_manager_defaultDoctorManager_count;   /**< Agent population size counter */ 

/* specialist_manager Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_specialist_manager_list* d_specialist_managers;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_specialist_manager_list* d_specialist_managers_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_specialist_manager_list* d_specialist_managers_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_specialist_manager_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_specialist_manager_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_specialist_manager_values;  /**< Agent sort identifiers value */

/* specialist_manager state variables */
xmachine_memory_specialist_manager_list* h_specialist_managers_defaultSpecialistManager;      /**< Pointer to agent list (population) on host*/
xmachine_memory_specialist_manager_list* d_specialist_managers_defaultSpecialistManager;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_specialist_manager_defaultSpecialistManager_count;   /**< Agent population size counter */ 

/* specialist Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_specialist_list* d_specialists;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_specialist_list* d_specialists_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_specialist_list* d_specialists_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_specialist_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_specialist_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_specialist_values;  /**< Agent sort identifiers value */

/* specialist state variables */
xmachine_memory_specialist_list* h_specialists_defaultSpecialist;      /**< Pointer to agent list (population) on host*/
xmachine_memory_specialist_list* d_specialists_defaultSpecialist;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_specialist_defaultSpecialist_count;   /**< Agent population size counter */ 

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

/* agent_generator Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_agent_generator_list* d_agent_generators;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_agent_generator_list* d_agent_generators_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_agent_generator_list* d_agent_generators_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_agent_generator_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_agent_generator_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_agent_generator_values;  /**< Agent sort identifiers value */

/* agent_generator state variables */
xmachine_memory_agent_generator_list* h_agent_generators_defaultGenerator;      /**< Pointer to agent list (population) on host*/
xmachine_memory_agent_generator_list* d_agent_generators_defaultGenerator;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_agent_generator_defaultGenerator_count;   /**< Agent population size counter */ 

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

/* box Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_box_list* d_boxs;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_box_list* d_boxs_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_box_list* d_boxs_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_box_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_box_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_box_values;  /**< Agent sort identifiers value */

/* box state variables */
xmachine_memory_box_list* h_boxs_defaultBox;      /**< Pointer to agent list (population) on host*/
xmachine_memory_box_list* d_boxs_defaultBox;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_box_defaultBox_count;   /**< Agent population size counter */ 

/* doctor Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_doctor_list* d_doctors;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_doctor_list* d_doctors_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_doctor_list* d_doctors_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_doctor_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_doctor_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_doctor_values;  /**< Agent sort identifiers value */

/* doctor state variables */
xmachine_memory_doctor_list* h_doctors_defaultDoctor;      /**< Pointer to agent list (population) on host*/
xmachine_memory_doctor_list* d_doctors_defaultDoctor;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_doctor_defaultDoctor_count;   /**< Agent population size counter */ 

/* triage Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_triage_list* d_triages;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_triage_list* d_triages_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_triage_list* d_triages_new;  /**< Pointer to new agent list on the device (used to hold new agents before they are appended to the population)*/
int h_xmachine_memory_triage_count;   /**< Agent population size counter */ 
uint * d_xmachine_memory_triage_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_triage_values;  /**< Agent sort identifiers value */

/* triage state variables */
xmachine_memory_triage_list* h_triages_defaultTriage;      /**< Pointer to agent list (population) on host*/
xmachine_memory_triage_list* d_triages_defaultTriage;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_triage_defaultTriage_count;   /**< Agent population size counter */ 


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
unsigned int h_agents_default_variable_checkpoint_data_iteration;
unsigned int h_agents_default_variable_chair_no_data_iteration;
unsigned int h_agents_default_variable_box_no_data_iteration;
unsigned int h_agents_default_variable_doctor_no_data_iteration;
unsigned int h_agents_default_variable_specialist_no_data_iteration;
unsigned int h_agents_default_variable_priority_data_iteration;
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
unsigned int h_chairs_defaultChair_variable_id_data_iteration;
unsigned int h_chairs_defaultChair_variable_x_data_iteration;
unsigned int h_chairs_defaultChair_variable_y_data_iteration;
unsigned int h_chairs_defaultChair_variable_state_data_iteration;
unsigned int h_doctor_managers_defaultDoctorManager_variable_tick_data_iteration;
unsigned int h_doctor_managers_defaultDoctorManager_variable_rear_data_iteration;
unsigned int h_doctor_managers_defaultDoctorManager_variable_size_data_iteration;
unsigned int h_doctor_managers_defaultDoctorManager_variable_doctors_occupied_data_iteration;
unsigned int h_doctor_managers_defaultDoctorManager_variable_free_doctors_data_iteration;
unsigned int h_doctor_managers_defaultDoctorManager_variable_patientQueue_data_iteration;
unsigned int h_specialist_managers_defaultSpecialistManager_variable_id_data_iteration;
unsigned int h_specialist_managers_defaultSpecialistManager_variable_tick_data_iteration;
unsigned int h_specialist_managers_defaultSpecialistManager_variable_free_specialist_data_iteration;
unsigned int h_specialist_managers_defaultSpecialistManager_variable_rear_data_iteration;
unsigned int h_specialist_managers_defaultSpecialistManager_variable_size_data_iteration;
unsigned int h_specialist_managers_defaultSpecialistManager_variable_surgicalQueue_data_iteration;
unsigned int h_specialist_managers_defaultSpecialistManager_variable_pediatricsQueue_data_iteration;
unsigned int h_specialist_managers_defaultSpecialistManager_variable_gynecologistQueue_data_iteration;
unsigned int h_specialist_managers_defaultSpecialistManager_variable_geriatricsQueue_data_iteration;
unsigned int h_specialist_managers_defaultSpecialistManager_variable_psychiatristQueue_data_iteration;
unsigned int h_specialists_defaultSpecialist_variable_id_data_iteration;
unsigned int h_specialists_defaultSpecialist_variable_current_patient_data_iteration;
unsigned int h_specialists_defaultSpecialist_variable_tick_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_x_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_y_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_patientQueue_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_front_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_rear_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_size_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_tick_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_current_patient_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_attend_patient_data_iteration;
unsigned int h_receptionists_defaultReceptionist_variable_estado_data_iteration;
unsigned int h_agent_generators_defaultGenerator_variable_chairs_generated_data_iteration;
unsigned int h_agent_generators_defaultGenerator_variable_boxes_generated_data_iteration;
unsigned int h_agent_generators_defaultGenerator_variable_doctors_generated_data_iteration;
unsigned int h_agent_generators_defaultGenerator_variable_specialists_generated_data_iteration;
unsigned int h_chair_admins_defaultAdmin_variable_id_data_iteration;
unsigned int h_chair_admins_defaultAdmin_variable_chairArray_data_iteration;
unsigned int h_boxs_defaultBox_variable_id_data_iteration;
unsigned int h_boxs_defaultBox_variable_attending_data_iteration;
unsigned int h_boxs_defaultBox_variable_tick_data_iteration;
unsigned int h_doctors_defaultDoctor_variable_id_data_iteration;
unsigned int h_doctors_defaultDoctor_variable_current_patient_data_iteration;
unsigned int h_doctors_defaultDoctor_variable_tick_data_iteration;
unsigned int h_triages_defaultTriage_variable_front_data_iteration;
unsigned int h_triages_defaultTriage_variable_rear_data_iteration;
unsigned int h_triages_defaultTriage_variable_size_data_iteration;
unsigned int h_triages_defaultTriage_variable_tick_data_iteration;
unsigned int h_triages_defaultTriage_variable_boxArray_data_iteration;
unsigned int h_triages_defaultTriage_variable_patientQueue_data_iteration;


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

/* check_in_response Message variables */
xmachine_message_check_in_response_list* h_check_in_responses;         /**< Pointer to message list on host*/
xmachine_message_check_in_response_list* d_check_in_responses;         /**< Pointer to message list on device*/
xmachine_message_check_in_response_list* d_check_in_responses_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_check_in_response_count;         /**< message list counter*/
int h_message_check_in_response_output_type;   /**< message output type (single or optional)*/

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

/* chair_state Message variables */
xmachine_message_chair_state_list* h_chair_states;         /**< Pointer to message list on host*/
xmachine_message_chair_state_list* d_chair_states;         /**< Pointer to message list on device*/
xmachine_message_chair_state_list* d_chair_states_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_chair_state_count;         /**< message list counter*/
int h_message_chair_state_output_type;   /**< message output type (single or optional)*/

/* free_chair Message variables */
xmachine_message_free_chair_list* h_free_chairs;         /**< Pointer to message list on host*/
xmachine_message_free_chair_list* d_free_chairs;         /**< Pointer to message list on device*/
xmachine_message_free_chair_list* d_free_chairs_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_free_chair_count;         /**< message list counter*/
int h_message_free_chair_output_type;   /**< message output type (single or optional)*/

/* chair_contact Message variables */
xmachine_message_chair_contact_list* h_chair_contacts;         /**< Pointer to message list on host*/
xmachine_message_chair_contact_list* d_chair_contacts;         /**< Pointer to message list on device*/
xmachine_message_chair_contact_list* d_chair_contacts_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_chair_contact_count;         /**< message list counter*/
int h_message_chair_contact_output_type;   /**< message output type (single or optional)*/

/* box_petition Message variables */
xmachine_message_box_petition_list* h_box_petitions;         /**< Pointer to message list on host*/
xmachine_message_box_petition_list* d_box_petitions;         /**< Pointer to message list on device*/
xmachine_message_box_petition_list* d_box_petitions_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_box_petition_count;         /**< message list counter*/
int h_message_box_petition_output_type;   /**< message output type (single or optional)*/

/* box_response Message variables */
xmachine_message_box_response_list* h_box_responses;         /**< Pointer to message list on host*/
xmachine_message_box_response_list* d_box_responses;         /**< Pointer to message list on device*/
xmachine_message_box_response_list* d_box_responses_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_box_response_count;         /**< message list counter*/
int h_message_box_response_output_type;   /**< message output type (single or optional)*/

/* specialist_reached Message variables */
xmachine_message_specialist_reached_list* h_specialist_reacheds;         /**< Pointer to message list on host*/
xmachine_message_specialist_reached_list* d_specialist_reacheds;         /**< Pointer to message list on device*/
xmachine_message_specialist_reached_list* d_specialist_reacheds_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_specialist_reached_count;         /**< message list counter*/
int h_message_specialist_reached_output_type;   /**< message output type (single or optional)*/

/* specialist_petition Message variables */
xmachine_message_specialist_petition_list* h_specialist_petitions;         /**< Pointer to message list on host*/
xmachine_message_specialist_petition_list* d_specialist_petitions;         /**< Pointer to message list on device*/
xmachine_message_specialist_petition_list* d_specialist_petitions_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_specialist_petition_count;         /**< message list counter*/
int h_message_specialist_petition_output_type;   /**< message output type (single or optional)*/

/* doctor_reached Message variables */
xmachine_message_doctor_reached_list* h_doctor_reacheds;         /**< Pointer to message list on host*/
xmachine_message_doctor_reached_list* d_doctor_reacheds;         /**< Pointer to message list on device*/
xmachine_message_doctor_reached_list* d_doctor_reacheds_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_doctor_reached_count;         /**< message list counter*/
int h_message_doctor_reached_output_type;   /**< message output type (single or optional)*/

/* free_doctor Message variables */
xmachine_message_free_doctor_list* h_free_doctors;         /**< Pointer to message list on host*/
xmachine_message_free_doctor_list* d_free_doctors;         /**< Pointer to message list on device*/
xmachine_message_free_doctor_list* d_free_doctors_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_free_doctor_count;         /**< message list counter*/
int h_message_free_doctor_output_type;   /**< message output type (single or optional)*/

/* attention_terminated Message variables */
xmachine_message_attention_terminated_list* h_attention_terminateds;         /**< Pointer to message list on host*/
xmachine_message_attention_terminated_list* d_attention_terminateds;         /**< Pointer to message list on device*/
xmachine_message_attention_terminated_list* d_attention_terminateds_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_attention_terminated_count;         /**< message list counter*/
int h_message_attention_terminated_output_type;   /**< message output type (single or optional)*/

/* doctor_petition Message variables */
xmachine_message_doctor_petition_list* h_doctor_petitions;         /**< Pointer to message list on host*/
xmachine_message_doctor_petition_list* d_doctor_petitions;         /**< Pointer to message list on device*/
xmachine_message_doctor_petition_list* d_doctor_petitions_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_doctor_petition_count;         /**< message list counter*/
int h_message_doctor_petition_output_type;   /**< message output type (single or optional)*/

/* doctor_response Message variables */
xmachine_message_doctor_response_list* h_doctor_responses;         /**< Pointer to message list on host*/
xmachine_message_doctor_response_list* d_doctor_responses;         /**< Pointer to message list on device*/
xmachine_message_doctor_response_list* d_doctor_responses_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_doctor_response_count;         /**< message list counter*/
int h_message_doctor_response_output_type;   /**< message output type (single or optional)*/

/* specialist_response Message variables */
xmachine_message_specialist_response_list* h_specialist_responses;         /**< Pointer to message list on host*/
xmachine_message_specialist_response_list* d_specialist_responses;         /**< Pointer to message list on device*/
xmachine_message_specialist_response_list* d_specialist_responses_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_specialist_response_count;         /**< message list counter*/
int h_message_specialist_response_output_type;   /**< message output type (single or optional)*/

/* triage_petition Message variables */
xmachine_message_triage_petition_list* h_triage_petitions;         /**< Pointer to message list on host*/
xmachine_message_triage_petition_list* d_triage_petitions;         /**< Pointer to message list on device*/
xmachine_message_triage_petition_list* d_triage_petitions_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_triage_petition_count;         /**< message list counter*/
int h_message_triage_petition_output_type;   /**< message output type (single or optional)*/

/* triage_response Message variables */
xmachine_message_triage_response_list* h_triage_responses;         /**< Pointer to message list on host*/
xmachine_message_triage_response_list* d_triage_responses;         /**< Pointer to message list on device*/
xmachine_message_triage_response_list* d_triage_responses_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/
/* Non partitioned and spatial partitioned message variables  */
int h_message_triage_response_count;         /**< message list counter*/
int h_message_triage_response_output_type;   /**< message output type (single or optional)*/

  
/* CUDA Streams for function layers */
cudaStream_t stream1;
cudaStream_t stream2;
cudaStream_t stream3;
cudaStream_t stream4;
cudaStream_t stream5;
cudaStream_t stream6;
cudaStream_t stream7;
cudaStream_t stream8;
cudaStream_t stream9;

/* Device memory and sizes for CUB values */

void * d_temp_scan_storage_agent;
size_t temp_scan_storage_bytes_agent;

void * d_temp_scan_storage_navmap;
size_t temp_scan_storage_bytes_navmap;

void * d_temp_scan_storage_chair;
size_t temp_scan_storage_bytes_chair;

void * d_temp_scan_storage_doctor_manager;
size_t temp_scan_storage_bytes_doctor_manager;

void * d_temp_scan_storage_specialist_manager;
size_t temp_scan_storage_bytes_specialist_manager;

void * d_temp_scan_storage_specialist;
size_t temp_scan_storage_bytes_specialist;

void * d_temp_scan_storage_receptionist;
size_t temp_scan_storage_bytes_receptionist;

void * d_temp_scan_storage_agent_generator;
size_t temp_scan_storage_bytes_agent_generator;

void * d_temp_scan_storage_chair_admin;
size_t temp_scan_storage_bytes_chair_admin;

void * d_temp_scan_storage_box;
size_t temp_scan_storage_bytes_box;

void * d_temp_scan_storage_doctor;
size_t temp_scan_storage_bytes_doctor;

void * d_temp_scan_storage_triage;
size_t temp_scan_storage_bytes_triage;


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

/** agent_receive_chair_state
 * Agent function prototype for receive_chair_state function of agent agent
 */
void agent_receive_chair_state(cudaStream_t &stream);

/** agent_output_chair_contact
 * Agent function prototype for output_chair_contact function of agent agent
 */
void agent_output_chair_contact(cudaStream_t &stream);

/** agent_output_free_chair
 * Agent function prototype for output_free_chair function of agent agent
 */
void agent_output_free_chair(cudaStream_t &stream);

/** agent_output_chair_petition
 * Agent function prototype for output_chair_petition function of agent agent
 */
void agent_output_chair_petition(cudaStream_t &stream);

/** agent_receive_chair_response
 * Agent function prototype for receive_chair_response function of agent agent
 */
void agent_receive_chair_response(cudaStream_t &stream);

/** agent_receive_check_in_response
 * Agent function prototype for receive_check_in_response function of agent agent
 */
void agent_receive_check_in_response(cudaStream_t &stream);

/** agent_output_box_petition
 * Agent function prototype for output_box_petition function of agent agent
 */
void agent_output_box_petition(cudaStream_t &stream);

/** agent_receive_box_response
 * Agent function prototype for receive_box_response function of agent agent
 */
void agent_receive_box_response(cudaStream_t &stream);

/** agent_output_doctor_petition
 * Agent function prototype for output_doctor_petition function of agent agent
 */
void agent_output_doctor_petition(cudaStream_t &stream);

/** agent_receive_doctor_response
 * Agent function prototype for receive_doctor_response function of agent agent
 */
void agent_receive_doctor_response(cudaStream_t &stream);

/** agent_receive_attention_terminated
 * Agent function prototype for receive_attention_terminated function of agent agent
 */
void agent_receive_attention_terminated(cudaStream_t &stream);

/** agent_output_doctor_reached
 * Agent function prototype for output_doctor_reached function of agent agent
 */
void agent_output_doctor_reached(cudaStream_t &stream);

/** agent_receive_specialist_response
 * Agent function prototype for receive_specialist_response function of agent agent
 */
void agent_receive_specialist_response(cudaStream_t &stream);

/** agent_output_specialist_petition
 * Agent function prototype for output_specialist_petition function of agent agent
 */
void agent_output_specialist_petition(cudaStream_t &stream);

/** agent_output_specialist_reached
 * Agent function prototype for output_specialist_reached function of agent agent
 */
void agent_output_specialist_reached(cudaStream_t &stream);

/** agent_output_triage_petition
 * Agent function prototype for output_triage_petition function of agent agent
 */
void agent_output_triage_petition(cudaStream_t &stream);

/** agent_receive_triage_response
 * Agent function prototype for receive_triage_response function of agent agent
 */
void agent_receive_triage_response(cudaStream_t &stream);

/** navmap_output_navmap_cells
 * Agent function prototype for output_navmap_cells function of navmap agent
 */
void navmap_output_navmap_cells(cudaStream_t &stream);

/** navmap_generate_pedestrians
 * Agent function prototype for generate_pedestrians function of navmap agent
 */
void navmap_generate_pedestrians(cudaStream_t &stream);

/** chair_output_chair_state
 * Agent function prototype for output_chair_state function of chair agent
 */
void chair_output_chair_state(cudaStream_t &stream);

/** doctor_manager_receive_doctor_petitions
 * Agent function prototype for receive_doctor_petitions function of doctor_manager agent
 */
void doctor_manager_receive_doctor_petitions(cudaStream_t &stream);

/** doctor_manager_receive_free_doctors
 * Agent function prototype for receive_free_doctors function of doctor_manager agent
 */
void doctor_manager_receive_free_doctors(cudaStream_t &stream);

/** specialist_manager_receive_specialist_petitions
 * Agent function prototype for receive_specialist_petitions function of specialist_manager agent
 */
void specialist_manager_receive_specialist_petitions(cudaStream_t &stream);

/** specialist_receive_specialist_reached
 * Agent function prototype for receive_specialist_reached function of specialist agent
 */
void specialist_receive_specialist_reached(cudaStream_t &stream);

/** receptionist_receptionServer
 * Agent function prototype for receptionServer function of receptionist agent
 */
void receptionist_receptionServer(cudaStream_t &stream);

/** receptionist_infect_receptionist
 * Agent function prototype for infect_receptionist function of receptionist agent
 */
void receptionist_infect_receptionist(cudaStream_t &stream);

/** agent_generator_generate_chairs
 * Agent function prototype for generate_chairs function of agent_generator agent
 */
void agent_generator_generate_chairs(cudaStream_t &stream);

/** agent_generator_generate_boxes
 * Agent function prototype for generate_boxes function of agent_generator agent
 */
void agent_generator_generate_boxes(cudaStream_t &stream);

/** agent_generator_generate_doctors
 * Agent function prototype for generate_doctors function of agent_generator agent
 */
void agent_generator_generate_doctors(cudaStream_t &stream);

/** agent_generator_generate_specialists
 * Agent function prototype for generate_specialists function of agent_generator agent
 */
void agent_generator_generate_specialists(cudaStream_t &stream);

/** chair_admin_attend_chair_petitions
 * Agent function prototype for attend_chair_petitions function of chair_admin agent
 */
void chair_admin_attend_chair_petitions(cudaStream_t &stream);

/** chair_admin_receive_free_chair
 * Agent function prototype for receive_free_chair function of chair_admin agent
 */
void chair_admin_receive_free_chair(cudaStream_t &stream);

/** box_box_server
 * Agent function prototype for box_server function of box agent
 */
void box_box_server(cudaStream_t &stream);

/** box_attend_box_patient
 * Agent function prototype for attend_box_patient function of box agent
 */
void box_attend_box_patient(cudaStream_t &stream);

/** doctor_doctor_server
 * Agent function prototype for doctor_server function of doctor agent
 */
void doctor_doctor_server(cudaStream_t &stream);

/** triage_receive_triage_petitions
 * Agent function prototype for receive_triage_petitions function of triage agent
 */
void triage_receive_triage_petitions(cudaStream_t &stream);

  
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
    h_agents_default_variable_checkpoint_data_iteration = 0;
    h_agents_default_variable_chair_no_data_iteration = 0;
    h_agents_default_variable_box_no_data_iteration = 0;
    h_agents_default_variable_doctor_no_data_iteration = 0;
    h_agents_default_variable_specialist_no_data_iteration = 0;
    h_agents_default_variable_priority_data_iteration = 0;
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
    h_chairs_defaultChair_variable_id_data_iteration = 0;
    h_chairs_defaultChair_variable_x_data_iteration = 0;
    h_chairs_defaultChair_variable_y_data_iteration = 0;
    h_chairs_defaultChair_variable_state_data_iteration = 0;
    h_doctor_managers_defaultDoctorManager_variable_tick_data_iteration = 0;
    h_doctor_managers_defaultDoctorManager_variable_rear_data_iteration = 0;
    h_doctor_managers_defaultDoctorManager_variable_size_data_iteration = 0;
    h_doctor_managers_defaultDoctorManager_variable_doctors_occupied_data_iteration = 0;
    h_doctor_managers_defaultDoctorManager_variable_free_doctors_data_iteration = 0;
    h_doctor_managers_defaultDoctorManager_variable_patientQueue_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_id_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_tick_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_free_specialist_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_rear_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_size_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_surgicalQueue_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_pediatricsQueue_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_gynecologistQueue_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_geriatricsQueue_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_psychiatristQueue_data_iteration = 0;
    h_specialists_defaultSpecialist_variable_id_data_iteration = 0;
    h_specialists_defaultSpecialist_variable_current_patient_data_iteration = 0;
    h_specialists_defaultSpecialist_variable_tick_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_x_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_y_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_patientQueue_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_front_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_rear_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_size_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_tick_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_current_patient_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_attend_patient_data_iteration = 0;
    h_receptionists_defaultReceptionist_variable_estado_data_iteration = 0;
    h_agent_generators_defaultGenerator_variable_chairs_generated_data_iteration = 0;
    h_agent_generators_defaultGenerator_variable_boxes_generated_data_iteration = 0;
    h_agent_generators_defaultGenerator_variable_doctors_generated_data_iteration = 0;
    h_agent_generators_defaultGenerator_variable_specialists_generated_data_iteration = 0;
    h_chair_admins_defaultAdmin_variable_id_data_iteration = 0;
    h_chair_admins_defaultAdmin_variable_chairArray_data_iteration = 0;
    h_boxs_defaultBox_variable_id_data_iteration = 0;
    h_boxs_defaultBox_variable_attending_data_iteration = 0;
    h_boxs_defaultBox_variable_tick_data_iteration = 0;
    h_doctors_defaultDoctor_variable_id_data_iteration = 0;
    h_doctors_defaultDoctor_variable_current_patient_data_iteration = 0;
    h_doctors_defaultDoctor_variable_tick_data_iteration = 0;
    h_triages_defaultTriage_variable_front_data_iteration = 0;
    h_triages_defaultTriage_variable_rear_data_iteration = 0;
    h_triages_defaultTriage_variable_size_data_iteration = 0;
    h_triages_defaultTriage_variable_tick_data_iteration = 0;
    h_triages_defaultTriage_variable_boxArray_data_iteration = 0;
    h_triages_defaultTriage_variable_patientQueue_data_iteration = 0;
    



	printf("Allocating Host and Device memory\n");
    PROFILE_PUSH_RANGE("allocate host");
	/* Agent memory allocation (CPU) */
	int xmachine_agent_SoA_size = sizeof(xmachine_memory_agent_list);
	h_agents_default = (xmachine_memory_agent_list*)malloc(xmachine_agent_SoA_size);
	int xmachine_navmap_SoA_size = sizeof(xmachine_memory_navmap_list);
	h_navmaps_static = (xmachine_memory_navmap_list*)malloc(xmachine_navmap_SoA_size);
	int xmachine_chair_SoA_size = sizeof(xmachine_memory_chair_list);
	h_chairs_defaultChair = (xmachine_memory_chair_list*)malloc(xmachine_chair_SoA_size);
	int xmachine_doctor_manager_SoA_size = sizeof(xmachine_memory_doctor_manager_list);
	h_doctor_managers_defaultDoctorManager = (xmachine_memory_doctor_manager_list*)malloc(xmachine_doctor_manager_SoA_size);
	int xmachine_specialist_manager_SoA_size = sizeof(xmachine_memory_specialist_manager_list);
	h_specialist_managers_defaultSpecialistManager = (xmachine_memory_specialist_manager_list*)malloc(xmachine_specialist_manager_SoA_size);
	int xmachine_specialist_SoA_size = sizeof(xmachine_memory_specialist_list);
	h_specialists_defaultSpecialist = (xmachine_memory_specialist_list*)malloc(xmachine_specialist_SoA_size);
	int xmachine_receptionist_SoA_size = sizeof(xmachine_memory_receptionist_list);
	h_receptionists_defaultReceptionist = (xmachine_memory_receptionist_list*)malloc(xmachine_receptionist_SoA_size);
	int xmachine_agent_generator_SoA_size = sizeof(xmachine_memory_agent_generator_list);
	h_agent_generators_defaultGenerator = (xmachine_memory_agent_generator_list*)malloc(xmachine_agent_generator_SoA_size);
	int xmachine_chair_admin_SoA_size = sizeof(xmachine_memory_chair_admin_list);
	h_chair_admins_defaultAdmin = (xmachine_memory_chair_admin_list*)malloc(xmachine_chair_admin_SoA_size);
	int xmachine_box_SoA_size = sizeof(xmachine_memory_box_list);
	h_boxs_defaultBox = (xmachine_memory_box_list*)malloc(xmachine_box_SoA_size);
	int xmachine_doctor_SoA_size = sizeof(xmachine_memory_doctor_list);
	h_doctors_defaultDoctor = (xmachine_memory_doctor_list*)malloc(xmachine_doctor_SoA_size);
	int xmachine_triage_SoA_size = sizeof(xmachine_memory_triage_list);
	h_triages_defaultTriage = (xmachine_memory_triage_list*)malloc(xmachine_triage_SoA_size);

	/* Message memory allocation (CPU) */
	int message_pedestrian_location_SoA_size = sizeof(xmachine_message_pedestrian_location_list);
	h_pedestrian_locations = (xmachine_message_pedestrian_location_list*)malloc(message_pedestrian_location_SoA_size);
	int message_pedestrian_state_SoA_size = sizeof(xmachine_message_pedestrian_state_list);
	h_pedestrian_states = (xmachine_message_pedestrian_state_list*)malloc(message_pedestrian_state_SoA_size);
	int message_navmap_cell_SoA_size = sizeof(xmachine_message_navmap_cell_list);
	h_navmap_cells = (xmachine_message_navmap_cell_list*)malloc(message_navmap_cell_SoA_size);
	int message_check_in_SoA_size = sizeof(xmachine_message_check_in_list);
	h_check_ins = (xmachine_message_check_in_list*)malloc(message_check_in_SoA_size);
	int message_check_in_response_SoA_size = sizeof(xmachine_message_check_in_response_list);
	h_check_in_responses = (xmachine_message_check_in_response_list*)malloc(message_check_in_response_SoA_size);
	int message_chair_petition_SoA_size = sizeof(xmachine_message_chair_petition_list);
	h_chair_petitions = (xmachine_message_chair_petition_list*)malloc(message_chair_petition_SoA_size);
	int message_chair_response_SoA_size = sizeof(xmachine_message_chair_response_list);
	h_chair_responses = (xmachine_message_chair_response_list*)malloc(message_chair_response_SoA_size);
	int message_chair_state_SoA_size = sizeof(xmachine_message_chair_state_list);
	h_chair_states = (xmachine_message_chair_state_list*)malloc(message_chair_state_SoA_size);
	int message_free_chair_SoA_size = sizeof(xmachine_message_free_chair_list);
	h_free_chairs = (xmachine_message_free_chair_list*)malloc(message_free_chair_SoA_size);
	int message_chair_contact_SoA_size = sizeof(xmachine_message_chair_contact_list);
	h_chair_contacts = (xmachine_message_chair_contact_list*)malloc(message_chair_contact_SoA_size);
	int message_box_petition_SoA_size = sizeof(xmachine_message_box_petition_list);
	h_box_petitions = (xmachine_message_box_petition_list*)malloc(message_box_petition_SoA_size);
	int message_box_response_SoA_size = sizeof(xmachine_message_box_response_list);
	h_box_responses = (xmachine_message_box_response_list*)malloc(message_box_response_SoA_size);
	int message_specialist_reached_SoA_size = sizeof(xmachine_message_specialist_reached_list);
	h_specialist_reacheds = (xmachine_message_specialist_reached_list*)malloc(message_specialist_reached_SoA_size);
	int message_specialist_petition_SoA_size = sizeof(xmachine_message_specialist_petition_list);
	h_specialist_petitions = (xmachine_message_specialist_petition_list*)malloc(message_specialist_petition_SoA_size);
	int message_doctor_reached_SoA_size = sizeof(xmachine_message_doctor_reached_list);
	h_doctor_reacheds = (xmachine_message_doctor_reached_list*)malloc(message_doctor_reached_SoA_size);
	int message_free_doctor_SoA_size = sizeof(xmachine_message_free_doctor_list);
	h_free_doctors = (xmachine_message_free_doctor_list*)malloc(message_free_doctor_SoA_size);
	int message_attention_terminated_SoA_size = sizeof(xmachine_message_attention_terminated_list);
	h_attention_terminateds = (xmachine_message_attention_terminated_list*)malloc(message_attention_terminated_SoA_size);
	int message_doctor_petition_SoA_size = sizeof(xmachine_message_doctor_petition_list);
	h_doctor_petitions = (xmachine_message_doctor_petition_list*)malloc(message_doctor_petition_SoA_size);
	int message_doctor_response_SoA_size = sizeof(xmachine_message_doctor_response_list);
	h_doctor_responses = (xmachine_message_doctor_response_list*)malloc(message_doctor_response_SoA_size);
	int message_specialist_response_SoA_size = sizeof(xmachine_message_specialist_response_list);
	h_specialist_responses = (xmachine_message_specialist_response_list*)malloc(message_specialist_response_SoA_size);
	int message_triage_petition_SoA_size = sizeof(xmachine_message_triage_petition_list);
	h_triage_petitions = (xmachine_message_triage_petition_list*)malloc(message_triage_petition_SoA_size);
	int message_triage_response_SoA_size = sizeof(xmachine_message_triage_response_list);
	h_triage_responses = (xmachine_message_triage_response_list*)malloc(message_triage_response_SoA_size);

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
	readInitialStates(inputfile, h_agents_default, &h_xmachine_memory_agent_default_count, h_navmaps_static, &h_xmachine_memory_navmap_static_count, h_chairs_defaultChair, &h_xmachine_memory_chair_defaultChair_count, h_doctor_managers_defaultDoctorManager, &h_xmachine_memory_doctor_manager_defaultDoctorManager_count, h_specialist_managers_defaultSpecialistManager, &h_xmachine_memory_specialist_manager_defaultSpecialistManager_count, h_specialists_defaultSpecialist, &h_xmachine_memory_specialist_defaultSpecialist_count, h_receptionists_defaultReceptionist, &h_xmachine_memory_receptionist_defaultReceptionist_count, h_agent_generators_defaultGenerator, &h_xmachine_memory_agent_generator_defaultGenerator_count, h_chair_admins_defaultAdmin, &h_xmachine_memory_chair_admin_defaultAdmin_count, h_boxs_defaultBox, &h_xmachine_memory_box_defaultBox_count, h_doctors_defaultDoctor, &h_xmachine_memory_doctor_defaultDoctor_count, h_triages_defaultTriage, &h_xmachine_memory_triage_defaultTriage_count);

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
    
	/* navmap Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_navmaps, xmachine_navmap_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_navmaps_swap, xmachine_navmap_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_navmaps_new, xmachine_navmap_SoA_size));
    
	/* static memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_navmaps_static, xmachine_navmap_SoA_size));
	gpuErrchk( cudaMemcpy( d_navmaps_static, h_navmaps_static, xmachine_navmap_SoA_size, cudaMemcpyHostToDevice));
    
	/* chair Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_chairs, xmachine_chair_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_chairs_swap, xmachine_chair_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_chairs_new, xmachine_chair_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_chair_keys, xmachine_memory_chair_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_chair_values, xmachine_memory_chair_MAX* sizeof(uint)));
	/* defaultChair memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_chairs_defaultChair, xmachine_chair_SoA_size));
	gpuErrchk( cudaMemcpy( d_chairs_defaultChair, h_chairs_defaultChair, xmachine_chair_SoA_size, cudaMemcpyHostToDevice));
    
	/* doctor_manager Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_doctor_managers, xmachine_doctor_manager_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_doctor_managers_swap, xmachine_doctor_manager_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_doctor_managers_new, xmachine_doctor_manager_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_doctor_manager_keys, xmachine_memory_doctor_manager_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_doctor_manager_values, xmachine_memory_doctor_manager_MAX* sizeof(uint)));
	/* defaultDoctorManager memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_doctor_managers_defaultDoctorManager, xmachine_doctor_manager_SoA_size));
	gpuErrchk( cudaMemcpy( d_doctor_managers_defaultDoctorManager, h_doctor_managers_defaultDoctorManager, xmachine_doctor_manager_SoA_size, cudaMemcpyHostToDevice));
    
	/* specialist_manager Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_specialist_managers, xmachine_specialist_manager_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_specialist_managers_swap, xmachine_specialist_manager_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_specialist_managers_new, xmachine_specialist_manager_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_specialist_manager_keys, xmachine_memory_specialist_manager_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_specialist_manager_values, xmachine_memory_specialist_manager_MAX* sizeof(uint)));
	/* defaultSpecialistManager memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_specialist_managers_defaultSpecialistManager, xmachine_specialist_manager_SoA_size));
	gpuErrchk( cudaMemcpy( d_specialist_managers_defaultSpecialistManager, h_specialist_managers_defaultSpecialistManager, xmachine_specialist_manager_SoA_size, cudaMemcpyHostToDevice));
    
	/* specialist Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_specialists, xmachine_specialist_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_specialists_swap, xmachine_specialist_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_specialists_new, xmachine_specialist_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_specialist_keys, xmachine_memory_specialist_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_specialist_values, xmachine_memory_specialist_MAX* sizeof(uint)));
	/* defaultSpecialist memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_specialists_defaultSpecialist, xmachine_specialist_SoA_size));
	gpuErrchk( cudaMemcpy( d_specialists_defaultSpecialist, h_specialists_defaultSpecialist, xmachine_specialist_SoA_size, cudaMemcpyHostToDevice));
    
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
    
	/* agent_generator Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_agent_generators, xmachine_agent_generator_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_agent_generators_swap, xmachine_agent_generator_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_agent_generators_new, xmachine_agent_generator_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_agent_generator_keys, xmachine_memory_agent_generator_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_agent_generator_values, xmachine_memory_agent_generator_MAX* sizeof(uint)));
	/* defaultGenerator memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_agent_generators_defaultGenerator, xmachine_agent_generator_SoA_size));
	gpuErrchk( cudaMemcpy( d_agent_generators_defaultGenerator, h_agent_generators_defaultGenerator, xmachine_agent_generator_SoA_size, cudaMemcpyHostToDevice));
    
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
    
	/* box Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_boxs, xmachine_box_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_boxs_swap, xmachine_box_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_boxs_new, xmachine_box_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_box_keys, xmachine_memory_box_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_box_values, xmachine_memory_box_MAX* sizeof(uint)));
	/* defaultBox memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_boxs_defaultBox, xmachine_box_SoA_size));
	gpuErrchk( cudaMemcpy( d_boxs_defaultBox, h_boxs_defaultBox, xmachine_box_SoA_size, cudaMemcpyHostToDevice));
    
	/* doctor Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_doctors, xmachine_doctor_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_doctors_swap, xmachine_doctor_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_doctors_new, xmachine_doctor_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_doctor_keys, xmachine_memory_doctor_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_doctor_values, xmachine_memory_doctor_MAX* sizeof(uint)));
	/* defaultDoctor memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_doctors_defaultDoctor, xmachine_doctor_SoA_size));
	gpuErrchk( cudaMemcpy( d_doctors_defaultDoctor, h_doctors_defaultDoctor, xmachine_doctor_SoA_size, cudaMemcpyHostToDevice));
    
	/* triage Agent memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_triages, xmachine_triage_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_triages_swap, xmachine_triage_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_triages_new, xmachine_triage_SoA_size));
    //continuous agent sort identifiers
  gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_triage_keys, xmachine_memory_triage_MAX* sizeof(uint)));
	gpuErrchk( cudaMalloc( (void**) &d_xmachine_memory_triage_values, xmachine_memory_triage_MAX* sizeof(uint)));
	/* defaultTriage memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_triages_defaultTriage, xmachine_triage_SoA_size));
	gpuErrchk( cudaMemcpy( d_triages_defaultTriage, h_triages_defaultTriage, xmachine_triage_SoA_size, cudaMemcpyHostToDevice));
    
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
	
	/* check_in_response Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_check_in_responses, message_check_in_response_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_check_in_responses_swap, message_check_in_response_SoA_size));
	gpuErrchk( cudaMemcpy( d_check_in_responses, h_check_in_responses, message_check_in_response_SoA_size, cudaMemcpyHostToDevice));
	
	/* chair_petition Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_chair_petitions, message_chair_petition_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_chair_petitions_swap, message_chair_petition_SoA_size));
	gpuErrchk( cudaMemcpy( d_chair_petitions, h_chair_petitions, message_chair_petition_SoA_size, cudaMemcpyHostToDevice));
	
	/* chair_response Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_chair_responses, message_chair_response_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_chair_responses_swap, message_chair_response_SoA_size));
	gpuErrchk( cudaMemcpy( d_chair_responses, h_chair_responses, message_chair_response_SoA_size, cudaMemcpyHostToDevice));
	
	/* chair_state Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_chair_states, message_chair_state_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_chair_states_swap, message_chair_state_SoA_size));
	gpuErrchk( cudaMemcpy( d_chair_states, h_chair_states, message_chair_state_SoA_size, cudaMemcpyHostToDevice));
	
	/* free_chair Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_free_chairs, message_free_chair_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_free_chairs_swap, message_free_chair_SoA_size));
	gpuErrchk( cudaMemcpy( d_free_chairs, h_free_chairs, message_free_chair_SoA_size, cudaMemcpyHostToDevice));
	
	/* chair_contact Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_chair_contacts, message_chair_contact_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_chair_contacts_swap, message_chair_contact_SoA_size));
	gpuErrchk( cudaMemcpy( d_chair_contacts, h_chair_contacts, message_chair_contact_SoA_size, cudaMemcpyHostToDevice));
	
	/* box_petition Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_box_petitions, message_box_petition_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_box_petitions_swap, message_box_petition_SoA_size));
	gpuErrchk( cudaMemcpy( d_box_petitions, h_box_petitions, message_box_petition_SoA_size, cudaMemcpyHostToDevice));
	
	/* box_response Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_box_responses, message_box_response_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_box_responses_swap, message_box_response_SoA_size));
	gpuErrchk( cudaMemcpy( d_box_responses, h_box_responses, message_box_response_SoA_size, cudaMemcpyHostToDevice));
	
	/* specialist_reached Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_specialist_reacheds, message_specialist_reached_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_specialist_reacheds_swap, message_specialist_reached_SoA_size));
	gpuErrchk( cudaMemcpy( d_specialist_reacheds, h_specialist_reacheds, message_specialist_reached_SoA_size, cudaMemcpyHostToDevice));
	
	/* specialist_petition Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_specialist_petitions, message_specialist_petition_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_specialist_petitions_swap, message_specialist_petition_SoA_size));
	gpuErrchk( cudaMemcpy( d_specialist_petitions, h_specialist_petitions, message_specialist_petition_SoA_size, cudaMemcpyHostToDevice));
	
	/* doctor_reached Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_doctor_reacheds, message_doctor_reached_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_doctor_reacheds_swap, message_doctor_reached_SoA_size));
	gpuErrchk( cudaMemcpy( d_doctor_reacheds, h_doctor_reacheds, message_doctor_reached_SoA_size, cudaMemcpyHostToDevice));
	
	/* free_doctor Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_free_doctors, message_free_doctor_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_free_doctors_swap, message_free_doctor_SoA_size));
	gpuErrchk( cudaMemcpy( d_free_doctors, h_free_doctors, message_free_doctor_SoA_size, cudaMemcpyHostToDevice));
	
	/* attention_terminated Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_attention_terminateds, message_attention_terminated_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_attention_terminateds_swap, message_attention_terminated_SoA_size));
	gpuErrchk( cudaMemcpy( d_attention_terminateds, h_attention_terminateds, message_attention_terminated_SoA_size, cudaMemcpyHostToDevice));
	
	/* doctor_petition Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_doctor_petitions, message_doctor_petition_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_doctor_petitions_swap, message_doctor_petition_SoA_size));
	gpuErrchk( cudaMemcpy( d_doctor_petitions, h_doctor_petitions, message_doctor_petition_SoA_size, cudaMemcpyHostToDevice));
	
	/* doctor_response Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_doctor_responses, message_doctor_response_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_doctor_responses_swap, message_doctor_response_SoA_size));
	gpuErrchk( cudaMemcpy( d_doctor_responses, h_doctor_responses, message_doctor_response_SoA_size, cudaMemcpyHostToDevice));
	
	/* specialist_response Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_specialist_responses, message_specialist_response_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_specialist_responses_swap, message_specialist_response_SoA_size));
	gpuErrchk( cudaMemcpy( d_specialist_responses, h_specialist_responses, message_specialist_response_SoA_size, cudaMemcpyHostToDevice));
	
	/* triage_petition Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_triage_petitions, message_triage_petition_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_triage_petitions_swap, message_triage_petition_SoA_size));
	gpuErrchk( cudaMemcpy( d_triage_petitions, h_triage_petitions, message_triage_petition_SoA_size, cudaMemcpyHostToDevice));
	
	/* triage_response Message memory allocation (GPU) */
	gpuErrchk( cudaMalloc( (void**) &d_triage_responses, message_triage_response_SoA_size));
	gpuErrchk( cudaMalloc( (void**) &d_triage_responses_swap, message_triage_response_SoA_size));
	gpuErrchk( cudaMemcpy( d_triage_responses, h_triage_responses, message_triage_response_SoA_size, cudaMemcpyHostToDevice));
		


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
    
    d_temp_scan_storage_chair = nullptr;
    temp_scan_storage_bytes_chair = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_chair, 
        temp_scan_storage_bytes_chair, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_chair_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_chair, temp_scan_storage_bytes_chair));
    
    d_temp_scan_storage_doctor_manager = nullptr;
    temp_scan_storage_bytes_doctor_manager = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_doctor_manager, 
        temp_scan_storage_bytes_doctor_manager, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_doctor_manager_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_doctor_manager, temp_scan_storage_bytes_doctor_manager));
    
    d_temp_scan_storage_specialist_manager = nullptr;
    temp_scan_storage_bytes_specialist_manager = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_specialist_manager, 
        temp_scan_storage_bytes_specialist_manager, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_specialist_manager_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_specialist_manager, temp_scan_storage_bytes_specialist_manager));
    
    d_temp_scan_storage_specialist = nullptr;
    temp_scan_storage_bytes_specialist = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_specialist, 
        temp_scan_storage_bytes_specialist, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_specialist_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_specialist, temp_scan_storage_bytes_specialist));
    
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
    
    d_temp_scan_storage_agent_generator = nullptr;
    temp_scan_storage_bytes_agent_generator = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent_generator, 
        temp_scan_storage_bytes_agent_generator, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_agent_generator_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_agent_generator, temp_scan_storage_bytes_agent_generator));
    
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
    
    d_temp_scan_storage_box = nullptr;
    temp_scan_storage_bytes_box = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_box, 
        temp_scan_storage_bytes_box, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_box_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_box, temp_scan_storage_bytes_box));
    
    d_temp_scan_storage_doctor = nullptr;
    temp_scan_storage_bytes_doctor = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_doctor, 
        temp_scan_storage_bytes_doctor, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_doctor_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_doctor, temp_scan_storage_bytes_doctor));
    
    d_temp_scan_storage_triage = nullptr;
    temp_scan_storage_bytes_triage = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_triage, 
        temp_scan_storage_bytes_triage, 
        (int*) nullptr, 
        (int*) nullptr, 
        xmachine_memory_triage_MAX
    );
    gpuErrchk(cudaMalloc(&d_temp_scan_storage_triage, temp_scan_storage_bytes_triage));
    

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
  gpuErrchk(cudaStreamCreate(&stream5));
  gpuErrchk(cudaStreamCreate(&stream6));
  gpuErrchk(cudaStreamCreate(&stream7));
  gpuErrchk(cudaStreamCreate(&stream8));
  gpuErrchk(cudaStreamCreate(&stream9));

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("Init agent_agent_default_count: %u\n",get_agent_agent_default_count());
	
		printf("Init agent_navmap_static_count: %u\n",get_agent_navmap_static_count());
	
		printf("Init agent_chair_defaultChair_count: %u\n",get_agent_chair_defaultChair_count());
	
		printf("Init agent_doctor_manager_defaultDoctorManager_count: %u\n",get_agent_doctor_manager_defaultDoctorManager_count());
	
		printf("Init agent_specialist_manager_defaultSpecialistManager_count: %u\n",get_agent_specialist_manager_defaultSpecialistManager_count());
	
		printf("Init agent_specialist_defaultSpecialist_count: %u\n",get_agent_specialist_defaultSpecialist_count());
	
		printf("Init agent_receptionist_defaultReceptionist_count: %u\n",get_agent_receptionist_defaultReceptionist_count());
	
		printf("Init agent_agent_generator_defaultGenerator_count: %u\n",get_agent_agent_generator_defaultGenerator_count());
	
		printf("Init agent_chair_admin_defaultAdmin_count: %u\n",get_agent_chair_admin_defaultAdmin_count());
	
		printf("Init agent_box_defaultBox_count: %u\n",get_agent_box_defaultBox_count());
	
		printf("Init agent_doctor_defaultDoctor_count: %u\n",get_agent_doctor_defaultDoctor_count());
	
		printf("Init agent_triage_defaultTriage_count: %u\n",get_agent_triage_defaultTriage_count());
	
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

void sort_chairs_defaultChair(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_chair_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_chair_defaultChair_count); 
	gridSize = (h_xmachine_memory_chair_defaultChair_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_chair_keys, d_xmachine_memory_chair_values, d_chairs_defaultChair);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_chair_keys),  thrust::device_pointer_cast(d_xmachine_memory_chair_keys) + h_xmachine_memory_chair_defaultChair_count,  thrust::device_pointer_cast(d_xmachine_memory_chair_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_chair_agents, no_sm, h_xmachine_memory_chair_defaultChair_count); 
	gridSize = (h_xmachine_memory_chair_defaultChair_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_chair_agents<<<gridSize, blockSize>>>(d_xmachine_memory_chair_values, d_chairs_defaultChair, d_chairs_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_chair_list* d_chairs_temp = d_chairs_defaultChair;
	d_chairs_defaultChair = d_chairs_swap;
	d_chairs_swap = d_chairs_temp;	
}

void sort_doctor_managers_defaultDoctorManager(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_doctor_manager_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_doctor_manager_defaultDoctorManager_count); 
	gridSize = (h_xmachine_memory_doctor_manager_defaultDoctorManager_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_doctor_manager_keys, d_xmachine_memory_doctor_manager_values, d_doctor_managers_defaultDoctorManager);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_doctor_manager_keys),  thrust::device_pointer_cast(d_xmachine_memory_doctor_manager_keys) + h_xmachine_memory_doctor_manager_defaultDoctorManager_count,  thrust::device_pointer_cast(d_xmachine_memory_doctor_manager_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_doctor_manager_agents, no_sm, h_xmachine_memory_doctor_manager_defaultDoctorManager_count); 
	gridSize = (h_xmachine_memory_doctor_manager_defaultDoctorManager_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_doctor_manager_agents<<<gridSize, blockSize>>>(d_xmachine_memory_doctor_manager_values, d_doctor_managers_defaultDoctorManager, d_doctor_managers_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_doctor_manager_list* d_doctor_managers_temp = d_doctor_managers_defaultDoctorManager;
	d_doctor_managers_defaultDoctorManager = d_doctor_managers_swap;
	d_doctor_managers_swap = d_doctor_managers_temp;	
}

void sort_specialist_managers_defaultSpecialistManager(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_specialist_manager_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_specialist_manager_defaultSpecialistManager_count); 
	gridSize = (h_xmachine_memory_specialist_manager_defaultSpecialistManager_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_specialist_manager_keys, d_xmachine_memory_specialist_manager_values, d_specialist_managers_defaultSpecialistManager);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_specialist_manager_keys),  thrust::device_pointer_cast(d_xmachine_memory_specialist_manager_keys) + h_xmachine_memory_specialist_manager_defaultSpecialistManager_count,  thrust::device_pointer_cast(d_xmachine_memory_specialist_manager_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_specialist_manager_agents, no_sm, h_xmachine_memory_specialist_manager_defaultSpecialistManager_count); 
	gridSize = (h_xmachine_memory_specialist_manager_defaultSpecialistManager_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_specialist_manager_agents<<<gridSize, blockSize>>>(d_xmachine_memory_specialist_manager_values, d_specialist_managers_defaultSpecialistManager, d_specialist_managers_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_specialist_manager_list* d_specialist_managers_temp = d_specialist_managers_defaultSpecialistManager;
	d_specialist_managers_defaultSpecialistManager = d_specialist_managers_swap;
	d_specialist_managers_swap = d_specialist_managers_temp;	
}

void sort_specialists_defaultSpecialist(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_specialist_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_specialist_defaultSpecialist_count); 
	gridSize = (h_xmachine_memory_specialist_defaultSpecialist_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_specialist_keys, d_xmachine_memory_specialist_values, d_specialists_defaultSpecialist);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_specialist_keys),  thrust::device_pointer_cast(d_xmachine_memory_specialist_keys) + h_xmachine_memory_specialist_defaultSpecialist_count,  thrust::device_pointer_cast(d_xmachine_memory_specialist_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_specialist_agents, no_sm, h_xmachine_memory_specialist_defaultSpecialist_count); 
	gridSize = (h_xmachine_memory_specialist_defaultSpecialist_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_specialist_agents<<<gridSize, blockSize>>>(d_xmachine_memory_specialist_values, d_specialists_defaultSpecialist, d_specialists_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_specialist_list* d_specialists_temp = d_specialists_defaultSpecialist;
	d_specialists_defaultSpecialist = d_specialists_swap;
	d_specialists_swap = d_specialists_temp;	
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

void sort_agent_generators_defaultGenerator(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_agent_generator_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_agent_generator_defaultGenerator_count); 
	gridSize = (h_xmachine_memory_agent_generator_defaultGenerator_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_agent_generator_keys, d_xmachine_memory_agent_generator_values, d_agent_generators_defaultGenerator);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_agent_generator_keys),  thrust::device_pointer_cast(d_xmachine_memory_agent_generator_keys) + h_xmachine_memory_agent_generator_defaultGenerator_count,  thrust::device_pointer_cast(d_xmachine_memory_agent_generator_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_agent_generator_agents, no_sm, h_xmachine_memory_agent_generator_defaultGenerator_count); 
	gridSize = (h_xmachine_memory_agent_generator_defaultGenerator_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_agent_generator_agents<<<gridSize, blockSize>>>(d_xmachine_memory_agent_generator_values, d_agent_generators_defaultGenerator, d_agent_generators_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_agent_generator_list* d_agent_generators_temp = d_agent_generators_defaultGenerator;
	d_agent_generators_defaultGenerator = d_agent_generators_swap;
	d_agent_generators_swap = d_agent_generators_temp;	
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

void sort_boxs_defaultBox(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_box_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_box_defaultBox_count); 
	gridSize = (h_xmachine_memory_box_defaultBox_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_box_keys, d_xmachine_memory_box_values, d_boxs_defaultBox);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_box_keys),  thrust::device_pointer_cast(d_xmachine_memory_box_keys) + h_xmachine_memory_box_defaultBox_count,  thrust::device_pointer_cast(d_xmachine_memory_box_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_box_agents, no_sm, h_xmachine_memory_box_defaultBox_count); 
	gridSize = (h_xmachine_memory_box_defaultBox_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_box_agents<<<gridSize, blockSize>>>(d_xmachine_memory_box_values, d_boxs_defaultBox, d_boxs_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_box_list* d_boxs_temp = d_boxs_defaultBox;
	d_boxs_defaultBox = d_boxs_swap;
	d_boxs_swap = d_boxs_temp;	
}

void sort_doctors_defaultDoctor(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_doctor_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_doctor_defaultDoctor_count); 
	gridSize = (h_xmachine_memory_doctor_defaultDoctor_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_doctor_keys, d_xmachine_memory_doctor_values, d_doctors_defaultDoctor);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_doctor_keys),  thrust::device_pointer_cast(d_xmachine_memory_doctor_keys) + h_xmachine_memory_doctor_defaultDoctor_count,  thrust::device_pointer_cast(d_xmachine_memory_doctor_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_doctor_agents, no_sm, h_xmachine_memory_doctor_defaultDoctor_count); 
	gridSize = (h_xmachine_memory_doctor_defaultDoctor_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_doctor_agents<<<gridSize, blockSize>>>(d_xmachine_memory_doctor_values, d_doctors_defaultDoctor, d_doctors_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_doctor_list* d_doctors_temp = d_doctors_defaultDoctor;
	d_doctors_defaultDoctor = d_doctors_swap;
	d_doctors_swap = d_doctors_temp;	
}

void sort_triages_defaultTriage(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_triage_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_triage_defaultTriage_count); 
	gridSize = (h_xmachine_memory_triage_defaultTriage_count + blockSize - 1) / blockSize;    // Round up according to array size 
	generate_key_value_pairs<<<gridSize, blockSize>>>(d_xmachine_memory_triage_keys, d_xmachine_memory_triage_values, d_triages_defaultTriage);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key( thrust::device_pointer_cast(d_xmachine_memory_triage_keys),  thrust::device_pointer_cast(d_xmachine_memory_triage_keys) + h_xmachine_memory_triage_defaultTriage_count,  thrust::device_pointer_cast(d_xmachine_memory_triage_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reorder_triage_agents, no_sm, h_xmachine_memory_triage_defaultTriage_count); 
	gridSize = (h_xmachine_memory_triage_defaultTriage_count + blockSize - 1) / blockSize;    // Round up according to array size 
	reorder_triage_agents<<<gridSize, blockSize>>>(d_xmachine_memory_triage_values, d_triages_defaultTriage, d_triages_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_triage_list* d_triages_temp = d_triages_defaultTriage;
	d_triages_defaultTriage = d_triages_swap;
	d_triages_swap = d_triages_temp;	
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
	
	/* navmap Agent variables */
	gpuErrchk(cudaFree(d_navmaps));
	gpuErrchk(cudaFree(d_navmaps_swap));
	gpuErrchk(cudaFree(d_navmaps_new));
	
	free( h_navmaps_static);
	gpuErrchk(cudaFree(d_navmaps_static));
	
	/* chair Agent variables */
	gpuErrchk(cudaFree(d_chairs));
	gpuErrchk(cudaFree(d_chairs_swap));
	gpuErrchk(cudaFree(d_chairs_new));
	
	free( h_chairs_defaultChair);
	gpuErrchk(cudaFree(d_chairs_defaultChair));
	
	/* doctor_manager Agent variables */
	gpuErrchk(cudaFree(d_doctor_managers));
	gpuErrchk(cudaFree(d_doctor_managers_swap));
	gpuErrchk(cudaFree(d_doctor_managers_new));
	
	free( h_doctor_managers_defaultDoctorManager);
	gpuErrchk(cudaFree(d_doctor_managers_defaultDoctorManager));
	
	/* specialist_manager Agent variables */
	gpuErrchk(cudaFree(d_specialist_managers));
	gpuErrchk(cudaFree(d_specialist_managers_swap));
	gpuErrchk(cudaFree(d_specialist_managers_new));
	
	free( h_specialist_managers_defaultSpecialistManager);
	gpuErrchk(cudaFree(d_specialist_managers_defaultSpecialistManager));
	
	/* specialist Agent variables */
	gpuErrchk(cudaFree(d_specialists));
	gpuErrchk(cudaFree(d_specialists_swap));
	gpuErrchk(cudaFree(d_specialists_new));
	
	free( h_specialists_defaultSpecialist);
	gpuErrchk(cudaFree(d_specialists_defaultSpecialist));
	
	/* receptionist Agent variables */
	gpuErrchk(cudaFree(d_receptionists));
	gpuErrchk(cudaFree(d_receptionists_swap));
	gpuErrchk(cudaFree(d_receptionists_new));
	
	free( h_receptionists_defaultReceptionist);
	gpuErrchk(cudaFree(d_receptionists_defaultReceptionist));
	
	/* agent_generator Agent variables */
	gpuErrchk(cudaFree(d_agent_generators));
	gpuErrchk(cudaFree(d_agent_generators_swap));
	gpuErrchk(cudaFree(d_agent_generators_new));
	
	free( h_agent_generators_defaultGenerator);
	gpuErrchk(cudaFree(d_agent_generators_defaultGenerator));
	
	/* chair_admin Agent variables */
	gpuErrchk(cudaFree(d_chair_admins));
	gpuErrchk(cudaFree(d_chair_admins_swap));
	gpuErrchk(cudaFree(d_chair_admins_new));
	
	free( h_chair_admins_defaultAdmin);
	gpuErrchk(cudaFree(d_chair_admins_defaultAdmin));
	
	/* box Agent variables */
	gpuErrchk(cudaFree(d_boxs));
	gpuErrchk(cudaFree(d_boxs_swap));
	gpuErrchk(cudaFree(d_boxs_new));
	
	free( h_boxs_defaultBox);
	gpuErrchk(cudaFree(d_boxs_defaultBox));
	
	/* doctor Agent variables */
	gpuErrchk(cudaFree(d_doctors));
	gpuErrchk(cudaFree(d_doctors_swap));
	gpuErrchk(cudaFree(d_doctors_new));
	
	free( h_doctors_defaultDoctor);
	gpuErrchk(cudaFree(d_doctors_defaultDoctor));
	
	/* triage Agent variables */
	gpuErrchk(cudaFree(d_triages));
	gpuErrchk(cudaFree(d_triages_swap));
	gpuErrchk(cudaFree(d_triages_new));
	
	free( h_triages_defaultTriage);
	gpuErrchk(cudaFree(d_triages_defaultTriage));
	

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
	
	/* check_in_response Message variables */
	free( h_check_in_responses);
	gpuErrchk(cudaFree(d_check_in_responses));
	gpuErrchk(cudaFree(d_check_in_responses_swap));
	
	/* chair_petition Message variables */
	free( h_chair_petitions);
	gpuErrchk(cudaFree(d_chair_petitions));
	gpuErrchk(cudaFree(d_chair_petitions_swap));
	
	/* chair_response Message variables */
	free( h_chair_responses);
	gpuErrchk(cudaFree(d_chair_responses));
	gpuErrchk(cudaFree(d_chair_responses_swap));
	
	/* chair_state Message variables */
	free( h_chair_states);
	gpuErrchk(cudaFree(d_chair_states));
	gpuErrchk(cudaFree(d_chair_states_swap));
	
	/* free_chair Message variables */
	free( h_free_chairs);
	gpuErrchk(cudaFree(d_free_chairs));
	gpuErrchk(cudaFree(d_free_chairs_swap));
	
	/* chair_contact Message variables */
	free( h_chair_contacts);
	gpuErrchk(cudaFree(d_chair_contacts));
	gpuErrchk(cudaFree(d_chair_contacts_swap));
	
	/* box_petition Message variables */
	free( h_box_petitions);
	gpuErrchk(cudaFree(d_box_petitions));
	gpuErrchk(cudaFree(d_box_petitions_swap));
	
	/* box_response Message variables */
	free( h_box_responses);
	gpuErrchk(cudaFree(d_box_responses));
	gpuErrchk(cudaFree(d_box_responses_swap));
	
	/* specialist_reached Message variables */
	free( h_specialist_reacheds);
	gpuErrchk(cudaFree(d_specialist_reacheds));
	gpuErrchk(cudaFree(d_specialist_reacheds_swap));
	
	/* specialist_petition Message variables */
	free( h_specialist_petitions);
	gpuErrchk(cudaFree(d_specialist_petitions));
	gpuErrchk(cudaFree(d_specialist_petitions_swap));
	
	/* doctor_reached Message variables */
	free( h_doctor_reacheds);
	gpuErrchk(cudaFree(d_doctor_reacheds));
	gpuErrchk(cudaFree(d_doctor_reacheds_swap));
	
	/* free_doctor Message variables */
	free( h_free_doctors);
	gpuErrchk(cudaFree(d_free_doctors));
	gpuErrchk(cudaFree(d_free_doctors_swap));
	
	/* attention_terminated Message variables */
	free( h_attention_terminateds);
	gpuErrchk(cudaFree(d_attention_terminateds));
	gpuErrchk(cudaFree(d_attention_terminateds_swap));
	
	/* doctor_petition Message variables */
	free( h_doctor_petitions);
	gpuErrchk(cudaFree(d_doctor_petitions));
	gpuErrchk(cudaFree(d_doctor_petitions_swap));
	
	/* doctor_response Message variables */
	free( h_doctor_responses);
	gpuErrchk(cudaFree(d_doctor_responses));
	gpuErrchk(cudaFree(d_doctor_responses_swap));
	
	/* specialist_response Message variables */
	free( h_specialist_responses);
	gpuErrchk(cudaFree(d_specialist_responses));
	gpuErrchk(cudaFree(d_specialist_responses_swap));
	
	/* triage_petition Message variables */
	free( h_triage_petitions);
	gpuErrchk(cudaFree(d_triage_petitions));
	gpuErrchk(cudaFree(d_triage_petitions_swap));
	
	/* triage_response Message variables */
	free( h_triage_responses);
	gpuErrchk(cudaFree(d_triage_responses));
	gpuErrchk(cudaFree(d_triage_responses_swap));
	

    /* Free temporary CUB memory if required. */
    
    if(d_temp_scan_storage_agent != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_agent));
      d_temp_scan_storage_agent = nullptr;
      temp_scan_storage_bytes_agent = 0;
    }
    
    if(d_temp_scan_storage_navmap != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_navmap));
      d_temp_scan_storage_navmap = nullptr;
      temp_scan_storage_bytes_navmap = 0;
    }
    
    if(d_temp_scan_storage_chair != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_chair));
      d_temp_scan_storage_chair = nullptr;
      temp_scan_storage_bytes_chair = 0;
    }
    
    if(d_temp_scan_storage_doctor_manager != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_doctor_manager));
      d_temp_scan_storage_doctor_manager = nullptr;
      temp_scan_storage_bytes_doctor_manager = 0;
    }
    
    if(d_temp_scan_storage_specialist_manager != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_specialist_manager));
      d_temp_scan_storage_specialist_manager = nullptr;
      temp_scan_storage_bytes_specialist_manager = 0;
    }
    
    if(d_temp_scan_storage_specialist != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_specialist));
      d_temp_scan_storage_specialist = nullptr;
      temp_scan_storage_bytes_specialist = 0;
    }
    
    if(d_temp_scan_storage_receptionist != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_receptionist));
      d_temp_scan_storage_receptionist = nullptr;
      temp_scan_storage_bytes_receptionist = 0;
    }
    
    if(d_temp_scan_storage_agent_generator != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_agent_generator));
      d_temp_scan_storage_agent_generator = nullptr;
      temp_scan_storage_bytes_agent_generator = 0;
    }
    
    if(d_temp_scan_storage_chair_admin != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_chair_admin));
      d_temp_scan_storage_chair_admin = nullptr;
      temp_scan_storage_bytes_chair_admin = 0;
    }
    
    if(d_temp_scan_storage_box != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_box));
      d_temp_scan_storage_box = nullptr;
      temp_scan_storage_bytes_box = 0;
    }
    
    if(d_temp_scan_storage_doctor != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_doctor));
      d_temp_scan_storage_doctor = nullptr;
      temp_scan_storage_bytes_doctor = 0;
    }
    
    if(d_temp_scan_storage_triage != nullptr){
      gpuErrchk(cudaFree(d_temp_scan_storage_triage));
      d_temp_scan_storage_triage = nullptr;
      temp_scan_storage_bytes_triage = 0;
    }
    

  /* Graph data free */
  
  
  /* CUDA Streams for function layers */
  
  gpuErrchk(cudaStreamDestroy(stream1));
  gpuErrchk(cudaStreamDestroy(stream2));
  gpuErrchk(cudaStreamDestroy(stream3));
  gpuErrchk(cudaStreamDestroy(stream4));
  gpuErrchk(cudaStreamDestroy(stream5));
  gpuErrchk(cudaStreamDestroy(stream6));
  gpuErrchk(cudaStreamDestroy(stream7));
  gpuErrchk(cudaStreamDestroy(stream8));
  gpuErrchk(cudaStreamDestroy(stream9));

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
	
	h_message_check_in_response_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_check_in_response_count, &h_message_check_in_response_count, sizeof(int)));
	
	h_message_chair_petition_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_chair_petition_count, &h_message_chair_petition_count, sizeof(int)));
	
	h_message_chair_response_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_chair_response_count, &h_message_chair_response_count, sizeof(int)));
	
	h_message_chair_state_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_chair_state_count, &h_message_chair_state_count, sizeof(int)));
	
	h_message_free_chair_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_free_chair_count, &h_message_free_chair_count, sizeof(int)));
	
	h_message_chair_contact_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_chair_contact_count, &h_message_chair_contact_count, sizeof(int)));
	
	h_message_box_petition_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_box_petition_count, &h_message_box_petition_count, sizeof(int)));
	
	h_message_box_response_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_box_response_count, &h_message_box_response_count, sizeof(int)));
	
	h_message_specialist_reached_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_specialist_reached_count, &h_message_specialist_reached_count, sizeof(int)));
	
	h_message_specialist_petition_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_specialist_petition_count, &h_message_specialist_petition_count, sizeof(int)));
	
	h_message_doctor_reached_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_doctor_reached_count, &h_message_doctor_reached_count, sizeof(int)));
	
	h_message_free_doctor_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_free_doctor_count, &h_message_free_doctor_count, sizeof(int)));
	
	h_message_attention_terminated_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_attention_terminated_count, &h_message_attention_terminated_count, sizeof(int)));
	
	h_message_doctor_petition_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_doctor_petition_count, &h_message_doctor_petition_count, sizeof(int)));
	
	h_message_doctor_response_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_doctor_response_count, &h_message_doctor_response_count, sizeof(int)));
	
	h_message_specialist_response_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_specialist_response_count, &h_message_specialist_response_count, sizeof(int)));
	
	h_message_triage_petition_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_triage_petition_count, &h_message_triage_petition_count, sizeof(int)));
	
	h_message_triage_response_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol( d_message_triage_response_count, &h_message_triage_response_count, sizeof(int)));
	

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
	
    PROFILE_PUSH_RANGE("agent_generator_generate_chairs");
	agent_generator_generate_chairs(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_generator_generate_chairs = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_generator_generate_boxes");
	agent_generator_generate_boxes(stream3);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_generator_generate_boxes = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_generator_generate_doctors");
	agent_generator_generate_doctors(stream4);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_generator_generate_doctors = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_generator_generate_specialists");
	agent_generator_generate_specialists(stream5);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_generator_generate_specialists = %f (ms)\n", instrument_milliseconds);
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
	
    PROFILE_PUSH_RANGE("agent_output_chair_contact");
	agent_output_chair_contact(stream4);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_output_chair_contact = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_output_chair_petition");
	agent_output_chair_petition(stream5);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_output_chair_petition = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_output_triage_petition");
	agent_output_triage_petition(stream6);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_output_triage_petition = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_output_doctor_petition");
	agent_output_doctor_petition(stream7);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_output_doctor_petition = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_output_doctor_reached");
	agent_output_doctor_reached(stream8);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_output_doctor_reached = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_output_box_petition");
	agent_output_box_petition(stream9);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_output_box_petition = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 3*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_output_specialist_reached");
	agent_output_specialist_reached(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_output_specialist_reached = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_output_free_chair");
	agent_output_free_chair(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_output_free_chair = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_output_specialist_petition");
	agent_output_specialist_petition(stream3);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_output_specialist_petition = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_avoid_pedestrians");
	agent_avoid_pedestrians(stream4);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_avoid_pedestrians = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("chair_output_chair_state");
	chair_output_chair_state(stream5);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: chair_output_chair_state = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("triage_receive_triage_petitions");
	triage_receive_triage_petitions(stream6);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: triage_receive_triage_petitions = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("doctor_doctor_server");
	doctor_doctor_server(stream7);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: doctor_doctor_server = %f (ms)\n", instrument_milliseconds);
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
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_receive_chair_state");
	agent_receive_chair_state(stream3);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_receive_chair_state = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_receive_triage_response");
	agent_receive_triage_response(stream4);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_receive_triage_response = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("doctor_manager_receive_doctor_petitions");
	doctor_manager_receive_doctor_petitions(stream5);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: doctor_manager_receive_doctor_petitions = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("specialist_receive_specialist_reached");
	specialist_receive_specialist_reached(stream6);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: specialist_receive_specialist_reached = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("chair_admin_receive_free_chair");
	chair_admin_receive_free_chair(stream7);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: chair_admin_receive_free_chair = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 5*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("specialist_manager_receive_specialist_petitions");
	specialist_manager_receive_specialist_petitions(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: specialist_manager_receive_specialist_petitions = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_receive_attention_terminated");
	agent_receive_attention_terminated(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_receive_attention_terminated = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_move");
	agent_move(stream3);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_move = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_receive_doctor_response");
	agent_receive_doctor_response(stream4);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_receive_doctor_response = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_receive_specialist_response");
	agent_receive_specialist_response(stream5);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_receive_specialist_response = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("doctor_manager_receive_free_doctors");
	doctor_manager_receive_free_doctors(stream6);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: doctor_manager_receive_free_doctors = %f (ms)\n", instrument_milliseconds);
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
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("box_box_server");
	box_box_server(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: box_box_server = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("box_attend_box_patient");
	box_attend_box_patient(stream3);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: box_attend_box_patient = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_receive_check_in_response");
	agent_receive_check_in_response(stream4);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_receive_check_in_response = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("chair_admin_attend_chair_petitions");
	chair_admin_attend_chair_petitions(stream5);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: chair_admin_attend_chair_petitions = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
	/* Layer 7*/
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_receive_chair_response");
	agent_receive_chair_response(stream1);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_receive_chair_response = %f (ms)\n", instrument_milliseconds);
#endif
	
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_start);
#endif
	
    PROFILE_PUSH_RANGE("agent_receive_box_response");
	agent_receive_box_response(stream2);
    PROFILE_POP_RANGE();
#if defined(INSTRUMENT_AGENT_FUNCTIONS) && INSTRUMENT_AGENT_FUNCTIONS
	cudaEventRecord(instrument_stop);
	cudaEventSynchronize(instrument_stop);
	cudaEventElapsedTime(&instrument_milliseconds, instrument_start, instrument_stop);
	printf("Instrumentation: agent_receive_box_response = %f (ms)\n", instrument_milliseconds);
#endif
	cudaDeviceSynchronize();
  
    
    /* Call all step functions */
	

#if defined(OUTPUT_POPULATION_PER_ITERATION) && OUTPUT_POPULATION_PER_ITERATION
	// Print the agent population size of all agents in all states
	
		printf("agent_agent_default_count: %u\n",get_agent_agent_default_count());
	
		printf("agent_navmap_static_count: %u\n",get_agent_navmap_static_count());
	
		printf("agent_chair_defaultChair_count: %u\n",get_agent_chair_defaultChair_count());
	
		printf("agent_doctor_manager_defaultDoctorManager_count: %u\n",get_agent_doctor_manager_defaultDoctorManager_count());
	
		printf("agent_specialist_manager_defaultSpecialistManager_count: %u\n",get_agent_specialist_manager_defaultSpecialistManager_count());
	
		printf("agent_specialist_defaultSpecialist_count: %u\n",get_agent_specialist_defaultSpecialist_count());
	
		printf("agent_receptionist_defaultReceptionist_count: %u\n",get_agent_receptionist_defaultReceptionist_count());
	
		printf("agent_agent_generator_defaultGenerator_count: %u\n",get_agent_agent_generator_defaultGenerator_count());
	
		printf("agent_chair_admin_defaultAdmin_count: %u\n",get_agent_chair_admin_defaultAdmin_count());
	
		printf("agent_box_defaultBox_count: %u\n",get_agent_box_defaultBox_count());
	
		printf("agent_doctor_defaultDoctor_count: %u\n",get_agent_doctor_defaultDoctor_count());
	
		printf("agent_triage_defaultTriage_count: %u\n",get_agent_triage_defaultTriage_count());
	
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

    
int get_agent_chair_MAX_count(){
    return xmachine_memory_chair_MAX;
}


int get_agent_chair_defaultChair_count(){
	//continuous agent
	return h_xmachine_memory_chair_defaultChair_count;
	
}

xmachine_memory_chair_list* get_device_chair_defaultChair_agents(){
	return d_chairs_defaultChair;
}

xmachine_memory_chair_list* get_host_chair_defaultChair_agents(){
	return h_chairs_defaultChair;
}

    
int get_agent_doctor_manager_MAX_count(){
    return xmachine_memory_doctor_manager_MAX;
}


int get_agent_doctor_manager_defaultDoctorManager_count(){
	//continuous agent
	return h_xmachine_memory_doctor_manager_defaultDoctorManager_count;
	
}

xmachine_memory_doctor_manager_list* get_device_doctor_manager_defaultDoctorManager_agents(){
	return d_doctor_managers_defaultDoctorManager;
}

xmachine_memory_doctor_manager_list* get_host_doctor_manager_defaultDoctorManager_agents(){
	return h_doctor_managers_defaultDoctorManager;
}

    
int get_agent_specialist_manager_MAX_count(){
    return xmachine_memory_specialist_manager_MAX;
}


int get_agent_specialist_manager_defaultSpecialistManager_count(){
	//continuous agent
	return h_xmachine_memory_specialist_manager_defaultSpecialistManager_count;
	
}

xmachine_memory_specialist_manager_list* get_device_specialist_manager_defaultSpecialistManager_agents(){
	return d_specialist_managers_defaultSpecialistManager;
}

xmachine_memory_specialist_manager_list* get_host_specialist_manager_defaultSpecialistManager_agents(){
	return h_specialist_managers_defaultSpecialistManager;
}

    
int get_agent_specialist_MAX_count(){
    return xmachine_memory_specialist_MAX;
}


int get_agent_specialist_defaultSpecialist_count(){
	//continuous agent
	return h_xmachine_memory_specialist_defaultSpecialist_count;
	
}

xmachine_memory_specialist_list* get_device_specialist_defaultSpecialist_agents(){
	return d_specialists_defaultSpecialist;
}

xmachine_memory_specialist_list* get_host_specialist_defaultSpecialist_agents(){
	return h_specialists_defaultSpecialist;
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

    
int get_agent_agent_generator_MAX_count(){
    return xmachine_memory_agent_generator_MAX;
}


int get_agent_agent_generator_defaultGenerator_count(){
	//continuous agent
	return h_xmachine_memory_agent_generator_defaultGenerator_count;
	
}

xmachine_memory_agent_generator_list* get_device_agent_generator_defaultGenerator_agents(){
	return d_agent_generators_defaultGenerator;
}

xmachine_memory_agent_generator_list* get_host_agent_generator_defaultGenerator_agents(){
	return h_agent_generators_defaultGenerator;
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

    
int get_agent_box_MAX_count(){
    return xmachine_memory_box_MAX;
}


int get_agent_box_defaultBox_count(){
	//continuous agent
	return h_xmachine_memory_box_defaultBox_count;
	
}

xmachine_memory_box_list* get_device_box_defaultBox_agents(){
	return d_boxs_defaultBox;
}

xmachine_memory_box_list* get_host_box_defaultBox_agents(){
	return h_boxs_defaultBox;
}

    
int get_agent_doctor_MAX_count(){
    return xmachine_memory_doctor_MAX;
}


int get_agent_doctor_defaultDoctor_count(){
	//continuous agent
	return h_xmachine_memory_doctor_defaultDoctor_count;
	
}

xmachine_memory_doctor_list* get_device_doctor_defaultDoctor_agents(){
	return d_doctors_defaultDoctor;
}

xmachine_memory_doctor_list* get_host_doctor_defaultDoctor_agents(){
	return h_doctors_defaultDoctor;
}

    
int get_agent_triage_MAX_count(){
    return xmachine_memory_triage_MAX;
}


int get_agent_triage_defaultTriage_count(){
	//continuous agent
	return h_xmachine_memory_triage_defaultTriage_count;
	
}

xmachine_memory_triage_list* get_device_triage_defaultTriage_agents(){
	return d_triages_defaultTriage;
}

xmachine_memory_triage_list* get_host_triage_defaultTriage_agents(){
	return h_triages_defaultTriage;
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

/** unsigned int get_agent_default_variable_checkpoint(unsigned int index)
 * Gets the value of the checkpoint variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable checkpoint
 */
__host__ unsigned int get_agent_default_variable_checkpoint(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_checkpoint_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->checkpoint,
                    d_agents_default->checkpoint,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_checkpoint_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->checkpoint[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access checkpoint for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_default_variable_chair_no(unsigned int index)
 * Gets the value of the chair_no variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable chair_no
 */
__host__ int get_agent_default_variable_chair_no(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_chair_no_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->chair_no,
                    d_agents_default->chair_no,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_chair_no_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->chair_no[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access chair_no for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_agent_default_variable_box_no(unsigned int index)
 * Gets the value of the box_no variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable box_no
 */
__host__ unsigned int get_agent_default_variable_box_no(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_box_no_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->box_no,
                    d_agents_default->box_no,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_box_no_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->box_no[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access box_no for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_agent_default_variable_doctor_no(unsigned int index)
 * Gets the value of the doctor_no variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable doctor_no
 */
__host__ unsigned int get_agent_default_variable_doctor_no(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_doctor_no_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->doctor_no,
                    d_agents_default->doctor_no,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_doctor_no_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->doctor_no[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access doctor_no for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_agent_default_variable_specialist_no(unsigned int index)
 * Gets the value of the specialist_no variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable specialist_no
 */
__host__ unsigned int get_agent_default_variable_specialist_no(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_specialist_no_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->specialist_no,
                    d_agents_default->specialist_no,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_specialist_no_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->specialist_no[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access specialist_no for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_agent_default_variable_priority(unsigned int index)
 * Gets the value of the priority variable of an agent agent in the default state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable priority
 */
__host__ unsigned int get_agent_default_variable_priority(unsigned int index){
    unsigned int count = get_agent_agent_default_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agents_default_variable_priority_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agents_default->priority,
                    d_agents_default->priority,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agents_default_variable_priority_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agents_default->priority[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access priority for the %u th member of agent_default. count is %u at iteration %u\n", index, count, currentIteration);
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

/** int get_chair_defaultChair_variable_id(unsigned int index)
 * Gets the value of the id variable of an chair agent in the defaultChair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ int get_chair_defaultChair_variable_id(unsigned int index){
    unsigned int count = get_agent_chair_defaultChair_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_chairs_defaultChair_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_chairs_defaultChair->id,
                    d_chairs_defaultChair->id,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_chairs_defaultChair_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_chairs_defaultChair->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of chair_defaultChair. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_chair_defaultChair_variable_x(unsigned int index)
 * Gets the value of the x variable of an chair agent in the defaultChair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable x
 */
__host__ int get_chair_defaultChair_variable_x(unsigned int index){
    unsigned int count = get_agent_chair_defaultChair_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_chairs_defaultChair_variable_x_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_chairs_defaultChair->x,
                    d_chairs_defaultChair->x,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_chairs_defaultChair_variable_x_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_chairs_defaultChair->x[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access x for the %u th member of chair_defaultChair. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_chair_defaultChair_variable_y(unsigned int index)
 * Gets the value of the y variable of an chair agent in the defaultChair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable y
 */
__host__ int get_chair_defaultChair_variable_y(unsigned int index){
    unsigned int count = get_agent_chair_defaultChair_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_chairs_defaultChair_variable_y_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_chairs_defaultChair->y,
                    d_chairs_defaultChair->y,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_chairs_defaultChair_variable_y_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_chairs_defaultChair->y[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access y for the %u th member of chair_defaultChair. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_chair_defaultChair_variable_state(unsigned int index)
 * Gets the value of the state variable of an chair agent in the defaultChair state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable state
 */
__host__ int get_chair_defaultChair_variable_state(unsigned int index){
    unsigned int count = get_agent_chair_defaultChair_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_chairs_defaultChair_variable_state_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_chairs_defaultChair->state,
                    d_chairs_defaultChair->state,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_chairs_defaultChair_variable_state_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_chairs_defaultChair->state[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access state for the %u th member of chair_defaultChair. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_doctor_manager_defaultDoctorManager_variable_tick(unsigned int index)
 * Gets the value of the tick variable of an doctor_manager agent in the defaultDoctorManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable tick
 */
__host__ unsigned int get_doctor_manager_defaultDoctorManager_variable_tick(unsigned int index){
    unsigned int count = get_agent_doctor_manager_defaultDoctorManager_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_doctor_managers_defaultDoctorManager_variable_tick_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_doctor_managers_defaultDoctorManager->tick,
                    d_doctor_managers_defaultDoctorManager->tick,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_doctor_managers_defaultDoctorManager_variable_tick_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_doctor_managers_defaultDoctorManager->tick[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access tick for the %u th member of doctor_manager_defaultDoctorManager. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_doctor_manager_defaultDoctorManager_variable_rear(unsigned int index)
 * Gets the value of the rear variable of an doctor_manager agent in the defaultDoctorManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable rear
 */
__host__ unsigned int get_doctor_manager_defaultDoctorManager_variable_rear(unsigned int index){
    unsigned int count = get_agent_doctor_manager_defaultDoctorManager_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_doctor_managers_defaultDoctorManager_variable_rear_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_doctor_managers_defaultDoctorManager->rear,
                    d_doctor_managers_defaultDoctorManager->rear,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_doctor_managers_defaultDoctorManager_variable_rear_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_doctor_managers_defaultDoctorManager->rear[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access rear for the %u th member of doctor_manager_defaultDoctorManager. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_doctor_manager_defaultDoctorManager_variable_size(unsigned int index)
 * Gets the value of the size variable of an doctor_manager agent in the defaultDoctorManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable size
 */
__host__ unsigned int get_doctor_manager_defaultDoctorManager_variable_size(unsigned int index){
    unsigned int count = get_agent_doctor_manager_defaultDoctorManager_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_doctor_managers_defaultDoctorManager_variable_size_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_doctor_managers_defaultDoctorManager->size,
                    d_doctor_managers_defaultDoctorManager->size,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_doctor_managers_defaultDoctorManager_variable_size_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_doctor_managers_defaultDoctorManager->size[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access size for the %u th member of doctor_manager_defaultDoctorManager. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_doctor_manager_defaultDoctorManager_variable_doctors_occupied(unsigned int index, unsigned int element)
 * Gets the element-th value of the doctors_occupied variable array of an doctor_manager agent in the defaultDoctorManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable doctors_occupied
 */
__host__ int get_doctor_manager_defaultDoctorManager_variable_doctors_occupied(unsigned int index, unsigned int element){
    unsigned int count = get_agent_doctor_manager_defaultDoctorManager_count();
    unsigned int numElements = 4;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_doctor_managers_defaultDoctorManager_variable_doctors_occupied_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_doctor_managers_defaultDoctorManager->doctors_occupied + (e * xmachine_memory_doctor_manager_MAX),
                        d_doctor_managers_defaultDoctorManager->doctors_occupied + (e * xmachine_memory_doctor_manager_MAX), 
                        count * sizeof(int), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_doctor_managers_defaultDoctorManager_variable_doctors_occupied_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_doctor_managers_defaultDoctorManager->doctors_occupied[index + (element * xmachine_memory_doctor_manager_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of doctors_occupied for the %u th member of doctor_manager_defaultDoctorManager. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_doctor_manager_defaultDoctorManager_variable_free_doctors(unsigned int index)
 * Gets the value of the free_doctors variable of an doctor_manager agent in the defaultDoctorManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable free_doctors
 */
__host__ unsigned int get_doctor_manager_defaultDoctorManager_variable_free_doctors(unsigned int index){
    unsigned int count = get_agent_doctor_manager_defaultDoctorManager_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_doctor_managers_defaultDoctorManager_variable_free_doctors_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_doctor_managers_defaultDoctorManager->free_doctors,
                    d_doctor_managers_defaultDoctorManager->free_doctors,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_doctor_managers_defaultDoctorManager_variable_free_doctors_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_doctor_managers_defaultDoctorManager->free_doctors[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access free_doctors for the %u th member of doctor_manager_defaultDoctorManager. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** ivec2 get_doctor_manager_defaultDoctorManager_variable_patientQueue(unsigned int index, unsigned int element)
 * Gets the element-th value of the patientQueue variable array of an doctor_manager agent in the defaultDoctorManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable patientQueue
 */
__host__ ivec2 get_doctor_manager_defaultDoctorManager_variable_patientQueue(unsigned int index, unsigned int element){
    unsigned int count = get_agent_doctor_manager_defaultDoctorManager_count();
    unsigned int numElements = 35;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_doctor_managers_defaultDoctorManager_variable_patientQueue_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_doctor_managers_defaultDoctorManager->patientQueue + (e * xmachine_memory_doctor_manager_MAX),
                        d_doctor_managers_defaultDoctorManager->patientQueue + (e * xmachine_memory_doctor_manager_MAX), 
                        count * sizeof(ivec2), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_doctor_managers_defaultDoctorManager_variable_patientQueue_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_doctor_managers_defaultDoctorManager->patientQueue[index + (element * xmachine_memory_doctor_manager_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of patientQueue for the %u th member of doctor_manager_defaultDoctorManager. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return {0,0};

    }
}

/** unsigned int get_specialist_manager_defaultSpecialistManager_variable_id(unsigned int index)
 * Gets the value of the id variable of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_specialist_manager_defaultSpecialistManager_variable_id(unsigned int index){
    unsigned int count = get_agent_specialist_manager_defaultSpecialistManager_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_specialist_managers_defaultSpecialistManager_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_specialist_managers_defaultSpecialistManager->id,
                    d_specialist_managers_defaultSpecialistManager->id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_specialist_managers_defaultSpecialistManager_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_specialist_managers_defaultSpecialistManager->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of specialist_manager_defaultSpecialistManager. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_specialist_manager_defaultSpecialistManager_variable_tick(unsigned int index, unsigned int element)
 * Gets the element-th value of the tick variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable tick
 */
__host__ unsigned int get_specialist_manager_defaultSpecialistManager_variable_tick(unsigned int index, unsigned int element){
    unsigned int count = get_agent_specialist_manager_defaultSpecialistManager_count();
    unsigned int numElements = 5;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_specialist_managers_defaultSpecialistManager_variable_tick_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_specialist_managers_defaultSpecialistManager->tick + (e * xmachine_memory_specialist_manager_MAX),
                        d_specialist_managers_defaultSpecialistManager->tick + (e * xmachine_memory_specialist_manager_MAX), 
                        count * sizeof(unsigned int), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_specialist_managers_defaultSpecialistManager_variable_tick_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_specialist_managers_defaultSpecialistManager->tick[index + (element * xmachine_memory_specialist_manager_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of tick for the %u th member of specialist_manager_defaultSpecialistManager. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_specialist_manager_defaultSpecialistManager_variable_free_specialist(unsigned int index, unsigned int element)
 * Gets the element-th value of the free_specialist variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable free_specialist
 */
__host__ unsigned int get_specialist_manager_defaultSpecialistManager_variable_free_specialist(unsigned int index, unsigned int element){
    unsigned int count = get_agent_specialist_manager_defaultSpecialistManager_count();
    unsigned int numElements = 5;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_specialist_managers_defaultSpecialistManager_variable_free_specialist_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_specialist_managers_defaultSpecialistManager->free_specialist + (e * xmachine_memory_specialist_manager_MAX),
                        d_specialist_managers_defaultSpecialistManager->free_specialist + (e * xmachine_memory_specialist_manager_MAX), 
                        count * sizeof(unsigned int), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_specialist_managers_defaultSpecialistManager_variable_free_specialist_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_specialist_managers_defaultSpecialistManager->free_specialist[index + (element * xmachine_memory_specialist_manager_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of free_specialist for the %u th member of specialist_manager_defaultSpecialistManager. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_specialist_manager_defaultSpecialistManager_variable_rear(unsigned int index, unsigned int element)
 * Gets the element-th value of the rear variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable rear
 */
__host__ unsigned int get_specialist_manager_defaultSpecialistManager_variable_rear(unsigned int index, unsigned int element){
    unsigned int count = get_agent_specialist_manager_defaultSpecialistManager_count();
    unsigned int numElements = 5;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_specialist_managers_defaultSpecialistManager_variable_rear_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_specialist_managers_defaultSpecialistManager->rear + (e * xmachine_memory_specialist_manager_MAX),
                        d_specialist_managers_defaultSpecialistManager->rear + (e * xmachine_memory_specialist_manager_MAX), 
                        count * sizeof(unsigned int), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_specialist_managers_defaultSpecialistManager_variable_rear_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_specialist_managers_defaultSpecialistManager->rear[index + (element * xmachine_memory_specialist_manager_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of rear for the %u th member of specialist_manager_defaultSpecialistManager. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_specialist_manager_defaultSpecialistManager_variable_size(unsigned int index, unsigned int element)
 * Gets the element-th value of the size variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable size
 */
__host__ unsigned int get_specialist_manager_defaultSpecialistManager_variable_size(unsigned int index, unsigned int element){
    unsigned int count = get_agent_specialist_manager_defaultSpecialistManager_count();
    unsigned int numElements = 5;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_specialist_managers_defaultSpecialistManager_variable_size_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_specialist_managers_defaultSpecialistManager->size + (e * xmachine_memory_specialist_manager_MAX),
                        d_specialist_managers_defaultSpecialistManager->size + (e * xmachine_memory_specialist_manager_MAX), 
                        count * sizeof(unsigned int), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_specialist_managers_defaultSpecialistManager_variable_size_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_specialist_managers_defaultSpecialistManager->size[index + (element * xmachine_memory_specialist_manager_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of size for the %u th member of specialist_manager_defaultSpecialistManager. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** ivec2 get_specialist_manager_defaultSpecialistManager_variable_surgicalQueue(unsigned int index, unsigned int element)
 * Gets the element-th value of the surgicalQueue variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable surgicalQueue
 */
__host__ ivec2 get_specialist_manager_defaultSpecialistManager_variable_surgicalQueue(unsigned int index, unsigned int element){
    unsigned int count = get_agent_specialist_manager_defaultSpecialistManager_count();
    unsigned int numElements = 35;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_specialist_managers_defaultSpecialistManager_variable_surgicalQueue_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_specialist_managers_defaultSpecialistManager->surgicalQueue + (e * xmachine_memory_specialist_manager_MAX),
                        d_specialist_managers_defaultSpecialistManager->surgicalQueue + (e * xmachine_memory_specialist_manager_MAX), 
                        count * sizeof(ivec2), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_specialist_managers_defaultSpecialistManager_variable_surgicalQueue_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_specialist_managers_defaultSpecialistManager->surgicalQueue[index + (element * xmachine_memory_specialist_manager_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of surgicalQueue for the %u th member of specialist_manager_defaultSpecialistManager. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return {0,0};

    }
}

/** ivec2 get_specialist_manager_defaultSpecialistManager_variable_pediatricsQueue(unsigned int index, unsigned int element)
 * Gets the element-th value of the pediatricsQueue variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable pediatricsQueue
 */
__host__ ivec2 get_specialist_manager_defaultSpecialistManager_variable_pediatricsQueue(unsigned int index, unsigned int element){
    unsigned int count = get_agent_specialist_manager_defaultSpecialistManager_count();
    unsigned int numElements = 35;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_specialist_managers_defaultSpecialistManager_variable_pediatricsQueue_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_specialist_managers_defaultSpecialistManager->pediatricsQueue + (e * xmachine_memory_specialist_manager_MAX),
                        d_specialist_managers_defaultSpecialistManager->pediatricsQueue + (e * xmachine_memory_specialist_manager_MAX), 
                        count * sizeof(ivec2), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_specialist_managers_defaultSpecialistManager_variable_pediatricsQueue_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_specialist_managers_defaultSpecialistManager->pediatricsQueue[index + (element * xmachine_memory_specialist_manager_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of pediatricsQueue for the %u th member of specialist_manager_defaultSpecialistManager. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return {0,0};

    }
}

/** ivec2 get_specialist_manager_defaultSpecialistManager_variable_gynecologistQueue(unsigned int index, unsigned int element)
 * Gets the element-th value of the gynecologistQueue variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable gynecologistQueue
 */
__host__ ivec2 get_specialist_manager_defaultSpecialistManager_variable_gynecologistQueue(unsigned int index, unsigned int element){
    unsigned int count = get_agent_specialist_manager_defaultSpecialistManager_count();
    unsigned int numElements = 35;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_specialist_managers_defaultSpecialistManager_variable_gynecologistQueue_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_specialist_managers_defaultSpecialistManager->gynecologistQueue + (e * xmachine_memory_specialist_manager_MAX),
                        d_specialist_managers_defaultSpecialistManager->gynecologistQueue + (e * xmachine_memory_specialist_manager_MAX), 
                        count * sizeof(ivec2), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_specialist_managers_defaultSpecialistManager_variable_gynecologistQueue_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_specialist_managers_defaultSpecialistManager->gynecologistQueue[index + (element * xmachine_memory_specialist_manager_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of gynecologistQueue for the %u th member of specialist_manager_defaultSpecialistManager. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return {0,0};

    }
}

/** ivec2 get_specialist_manager_defaultSpecialistManager_variable_geriatricsQueue(unsigned int index, unsigned int element)
 * Gets the element-th value of the geriatricsQueue variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable geriatricsQueue
 */
__host__ ivec2 get_specialist_manager_defaultSpecialistManager_variable_geriatricsQueue(unsigned int index, unsigned int element){
    unsigned int count = get_agent_specialist_manager_defaultSpecialistManager_count();
    unsigned int numElements = 35;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_specialist_managers_defaultSpecialistManager_variable_geriatricsQueue_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_specialist_managers_defaultSpecialistManager->geriatricsQueue + (e * xmachine_memory_specialist_manager_MAX),
                        d_specialist_managers_defaultSpecialistManager->geriatricsQueue + (e * xmachine_memory_specialist_manager_MAX), 
                        count * sizeof(ivec2), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_specialist_managers_defaultSpecialistManager_variable_geriatricsQueue_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_specialist_managers_defaultSpecialistManager->geriatricsQueue[index + (element * xmachine_memory_specialist_manager_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of geriatricsQueue for the %u th member of specialist_manager_defaultSpecialistManager. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return {0,0};

    }
}

/** ivec2 get_specialist_manager_defaultSpecialistManager_variable_psychiatristQueue(unsigned int index, unsigned int element)
 * Gets the element-th value of the psychiatristQueue variable array of an specialist_manager agent in the defaultSpecialistManager state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable psychiatristQueue
 */
__host__ ivec2 get_specialist_manager_defaultSpecialistManager_variable_psychiatristQueue(unsigned int index, unsigned int element){
    unsigned int count = get_agent_specialist_manager_defaultSpecialistManager_count();
    unsigned int numElements = 35;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_specialist_managers_defaultSpecialistManager_variable_psychiatristQueue_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_specialist_managers_defaultSpecialistManager->psychiatristQueue + (e * xmachine_memory_specialist_manager_MAX),
                        d_specialist_managers_defaultSpecialistManager->psychiatristQueue + (e * xmachine_memory_specialist_manager_MAX), 
                        count * sizeof(ivec2), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_specialist_managers_defaultSpecialistManager_variable_psychiatristQueue_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_specialist_managers_defaultSpecialistManager->psychiatristQueue[index + (element * xmachine_memory_specialist_manager_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of psychiatristQueue for the %u th member of specialist_manager_defaultSpecialistManager. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return {0,0};

    }
}

/** unsigned int get_specialist_defaultSpecialist_variable_id(unsigned int index)
 * Gets the value of the id variable of an specialist agent in the defaultSpecialist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_specialist_defaultSpecialist_variable_id(unsigned int index){
    unsigned int count = get_agent_specialist_defaultSpecialist_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_specialists_defaultSpecialist_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_specialists_defaultSpecialist->id,
                    d_specialists_defaultSpecialist->id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_specialists_defaultSpecialist_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_specialists_defaultSpecialist->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of specialist_defaultSpecialist. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_specialist_defaultSpecialist_variable_current_patient(unsigned int index)
 * Gets the value of the current_patient variable of an specialist agent in the defaultSpecialist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_patient
 */
__host__ unsigned int get_specialist_defaultSpecialist_variable_current_patient(unsigned int index){
    unsigned int count = get_agent_specialist_defaultSpecialist_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_specialists_defaultSpecialist_variable_current_patient_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_specialists_defaultSpecialist->current_patient,
                    d_specialists_defaultSpecialist->current_patient,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_specialists_defaultSpecialist_variable_current_patient_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_specialists_defaultSpecialist->current_patient[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access current_patient for the %u th member of specialist_defaultSpecialist. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_specialist_defaultSpecialist_variable_tick(unsigned int index)
 * Gets the value of the tick variable of an specialist agent in the defaultSpecialist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable tick
 */
__host__ unsigned int get_specialist_defaultSpecialist_variable_tick(unsigned int index){
    unsigned int count = get_agent_specialist_defaultSpecialist_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_specialists_defaultSpecialist_variable_tick_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_specialists_defaultSpecialist->tick,
                    d_specialists_defaultSpecialist->tick,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_specialists_defaultSpecialist_variable_tick_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_specialists_defaultSpecialist->tick[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access tick for the %u th member of specialist_defaultSpecialist. count is %u at iteration %u\n", index, count, currentIteration);
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

/** unsigned int get_receptionist_defaultReceptionist_variable_patientQueue(unsigned int index, unsigned int element)
 * Gets the element-th value of the patientQueue variable array of an receptionist agent in the defaultReceptionist state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable patientQueue
 */
__host__ unsigned int get_receptionist_defaultReceptionist_variable_patientQueue(unsigned int index, unsigned int element){
    unsigned int count = get_agent_receptionist_defaultReceptionist_count();
    unsigned int numElements = 100;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_receptionists_defaultReceptionist_variable_patientQueue_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_receptionists_defaultReceptionist->patientQueue + (e * xmachine_memory_receptionist_MAX),
                        d_receptionists_defaultReceptionist->patientQueue + (e * xmachine_memory_receptionist_MAX), 
                        count * sizeof(unsigned int), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_receptionists_defaultReceptionist_variable_patientQueue_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_receptionists_defaultReceptionist->patientQueue[index + (element * xmachine_memory_receptionist_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of patientQueue for the %u th member of receptionist_defaultReceptionist. count is %u at iteration %u\n", element, index, count, currentIteration);
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

/** int get_agent_generator_defaultGenerator_variable_chairs_generated(unsigned int index)
 * Gets the value of the chairs_generated variable of an agent_generator agent in the defaultGenerator state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable chairs_generated
 */
__host__ int get_agent_generator_defaultGenerator_variable_chairs_generated(unsigned int index){
    unsigned int count = get_agent_agent_generator_defaultGenerator_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agent_generators_defaultGenerator_variable_chairs_generated_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agent_generators_defaultGenerator->chairs_generated,
                    d_agent_generators_defaultGenerator->chairs_generated,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agent_generators_defaultGenerator_variable_chairs_generated_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agent_generators_defaultGenerator->chairs_generated[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access chairs_generated for the %u th member of agent_generator_defaultGenerator. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_generator_defaultGenerator_variable_boxes_generated(unsigned int index)
 * Gets the value of the boxes_generated variable of an agent_generator agent in the defaultGenerator state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable boxes_generated
 */
__host__ int get_agent_generator_defaultGenerator_variable_boxes_generated(unsigned int index){
    unsigned int count = get_agent_agent_generator_defaultGenerator_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agent_generators_defaultGenerator_variable_boxes_generated_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agent_generators_defaultGenerator->boxes_generated,
                    d_agent_generators_defaultGenerator->boxes_generated,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agent_generators_defaultGenerator_variable_boxes_generated_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agent_generators_defaultGenerator->boxes_generated[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access boxes_generated for the %u th member of agent_generator_defaultGenerator. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_generator_defaultGenerator_variable_doctors_generated(unsigned int index)
 * Gets the value of the doctors_generated variable of an agent_generator agent in the defaultGenerator state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable doctors_generated
 */
__host__ int get_agent_generator_defaultGenerator_variable_doctors_generated(unsigned int index){
    unsigned int count = get_agent_agent_generator_defaultGenerator_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agent_generators_defaultGenerator_variable_doctors_generated_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agent_generators_defaultGenerator->doctors_generated,
                    d_agent_generators_defaultGenerator->doctors_generated,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agent_generators_defaultGenerator_variable_doctors_generated_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agent_generators_defaultGenerator->doctors_generated[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access doctors_generated for the %u th member of agent_generator_defaultGenerator. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_agent_generator_defaultGenerator_variable_specialists_generated(unsigned int index)
 * Gets the value of the specialists_generated variable of an agent_generator agent in the defaultGenerator state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable specialists_generated
 */
__host__ int get_agent_generator_defaultGenerator_variable_specialists_generated(unsigned int index){
    unsigned int count = get_agent_agent_generator_defaultGenerator_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_agent_generators_defaultGenerator_variable_specialists_generated_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_agent_generators_defaultGenerator->specialists_generated,
                    d_agent_generators_defaultGenerator->specialists_generated,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_agent_generators_defaultGenerator_variable_specialists_generated_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_agent_generators_defaultGenerator->specialists_generated[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access specialists_generated for the %u th member of agent_generator_defaultGenerator. count is %u at iteration %u\n", index, count, currentIteration);
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

/** unsigned int get_box_defaultBox_variable_id(unsigned int index)
 * Gets the value of the id variable of an box agent in the defaultBox state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_box_defaultBox_variable_id(unsigned int index){
    unsigned int count = get_agent_box_defaultBox_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_boxs_defaultBox_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_boxs_defaultBox->id,
                    d_boxs_defaultBox->id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_boxs_defaultBox_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_boxs_defaultBox->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of box_defaultBox. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_box_defaultBox_variable_attending(unsigned int index)
 * Gets the value of the attending variable of an box agent in the defaultBox state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable attending
 */
__host__ unsigned int get_box_defaultBox_variable_attending(unsigned int index){
    unsigned int count = get_agent_box_defaultBox_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_boxs_defaultBox_variable_attending_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_boxs_defaultBox->attending,
                    d_boxs_defaultBox->attending,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_boxs_defaultBox_variable_attending_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_boxs_defaultBox->attending[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access attending for the %u th member of box_defaultBox. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_box_defaultBox_variable_tick(unsigned int index)
 * Gets the value of the tick variable of an box agent in the defaultBox state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable tick
 */
__host__ unsigned int get_box_defaultBox_variable_tick(unsigned int index){
    unsigned int count = get_agent_box_defaultBox_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_boxs_defaultBox_variable_tick_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_boxs_defaultBox->tick,
                    d_boxs_defaultBox->tick,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_boxs_defaultBox_variable_tick_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_boxs_defaultBox->tick[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access tick for the %u th member of box_defaultBox. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_doctor_defaultDoctor_variable_id(unsigned int index)
 * Gets the value of the id variable of an doctor agent in the defaultDoctor state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable id
 */
__host__ unsigned int get_doctor_defaultDoctor_variable_id(unsigned int index){
    unsigned int count = get_agent_doctor_defaultDoctor_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_doctors_defaultDoctor_variable_id_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_doctors_defaultDoctor->id,
                    d_doctors_defaultDoctor->id,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_doctors_defaultDoctor_variable_id_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_doctors_defaultDoctor->id[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access id for the %u th member of doctor_defaultDoctor. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** int get_doctor_defaultDoctor_variable_current_patient(unsigned int index)
 * Gets the value of the current_patient variable of an doctor agent in the defaultDoctor state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable current_patient
 */
__host__ int get_doctor_defaultDoctor_variable_current_patient(unsigned int index){
    unsigned int count = get_agent_doctor_defaultDoctor_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_doctors_defaultDoctor_variable_current_patient_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_doctors_defaultDoctor->current_patient,
                    d_doctors_defaultDoctor->current_patient,
                    count * sizeof(int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_doctors_defaultDoctor_variable_current_patient_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_doctors_defaultDoctor->current_patient[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access current_patient for the %u th member of doctor_defaultDoctor. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_doctor_defaultDoctor_variable_tick(unsigned int index)
 * Gets the value of the tick variable of an doctor agent in the defaultDoctor state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable tick
 */
__host__ unsigned int get_doctor_defaultDoctor_variable_tick(unsigned int index){
    unsigned int count = get_agent_doctor_defaultDoctor_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_doctors_defaultDoctor_variable_tick_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_doctors_defaultDoctor->tick,
                    d_doctors_defaultDoctor->tick,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_doctors_defaultDoctor_variable_tick_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_doctors_defaultDoctor->tick[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access tick for the %u th member of doctor_defaultDoctor. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_triage_defaultTriage_variable_front(unsigned int index)
 * Gets the value of the front variable of an triage agent in the defaultTriage state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable front
 */
__host__ unsigned int get_triage_defaultTriage_variable_front(unsigned int index){
    unsigned int count = get_agent_triage_defaultTriage_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_triages_defaultTriage_variable_front_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_triages_defaultTriage->front,
                    d_triages_defaultTriage->front,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_triages_defaultTriage_variable_front_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_triages_defaultTriage->front[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access front for the %u th member of triage_defaultTriage. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_triage_defaultTriage_variable_rear(unsigned int index)
 * Gets the value of the rear variable of an triage agent in the defaultTriage state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable rear
 */
__host__ unsigned int get_triage_defaultTriage_variable_rear(unsigned int index){
    unsigned int count = get_agent_triage_defaultTriage_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_triages_defaultTriage_variable_rear_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_triages_defaultTriage->rear,
                    d_triages_defaultTriage->rear,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_triages_defaultTriage_variable_rear_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_triages_defaultTriage->rear[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access rear for the %u th member of triage_defaultTriage. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_triage_defaultTriage_variable_size(unsigned int index)
 * Gets the value of the size variable of an triage agent in the defaultTriage state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable size
 */
__host__ unsigned int get_triage_defaultTriage_variable_size(unsigned int index){
    unsigned int count = get_agent_triage_defaultTriage_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_triages_defaultTriage_variable_size_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_triages_defaultTriage->size,
                    d_triages_defaultTriage->size,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_triages_defaultTriage_variable_size_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_triages_defaultTriage->size[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access size for the %u th member of triage_defaultTriage. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_triage_defaultTriage_variable_tick(unsigned int index)
 * Gets the value of the tick variable of an triage agent in the defaultTriage state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @return value of agent variable tick
 */
__host__ unsigned int get_triage_defaultTriage_variable_tick(unsigned int index){
    unsigned int count = get_agent_triage_defaultTriage_count();
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_triages_defaultTriage_variable_tick_data_iteration != currentIteration){
            gpuErrchk(
                cudaMemcpy(
                    h_triages_defaultTriage->tick,
                    d_triages_defaultTriage->tick,
                    count * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                )
            );
            // Update some global value indicating what data is currently present in that host array.
            h_triages_defaultTriage_variable_tick_data_iteration = currentIteration;
        }

        // Return the value of the index-th element of the relevant host array.
        return h_triages_defaultTriage->tick[index];

    } else {
        fprintf(stderr, "Warning: Attempting to access tick for the %u th member of triage_defaultTriage. count is %u at iteration %u\n", index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_triage_defaultTriage_variable_boxArray(unsigned int index, unsigned int element)
 * Gets the element-th value of the boxArray variable array of an triage agent in the defaultTriage state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable boxArray
 */
__host__ unsigned int get_triage_defaultTriage_variable_boxArray(unsigned int index, unsigned int element){
    unsigned int count = get_agent_triage_defaultTriage_count();
    unsigned int numElements = 3;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_triages_defaultTriage_variable_boxArray_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_triages_defaultTriage->boxArray + (e * xmachine_memory_triage_MAX),
                        d_triages_defaultTriage->boxArray + (e * xmachine_memory_triage_MAX), 
                        count * sizeof(unsigned int), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_triages_defaultTriage_variable_boxArray_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_triages_defaultTriage->boxArray[index + (element * xmachine_memory_triage_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of boxArray for the %u th member of triage_defaultTriage. count is %u at iteration %u\n", element, index, count, currentIteration);
        // Otherwise we return a default value
        return 0;

    }
}

/** unsigned int get_triage_defaultTriage_variable_patientQueue(unsigned int index, unsigned int element)
 * Gets the element-th value of the patientQueue variable array of an triage agent in the defaultTriage state on the host. 
 * If the data is not currently on the host, a memcpy of the data of all agents in that state list will be issued, via a global.
 * This has a potentially significant performance impact if used improperly.
 * @param index the index of the agent within the list.
 * @param element the element index within the variable array
 * @return element-th value of agent variable patientQueue
 */
__host__ unsigned int get_triage_defaultTriage_variable_patientQueue(unsigned int index, unsigned int element){
    unsigned int count = get_agent_triage_defaultTriage_count();
    unsigned int numElements = 100;
    unsigned int currentIteration = getIterationNumber();
    
    // If the index is within bounds - no need to check >= 0 due to unsigned.
    if(count > 0 && index < count && element < numElements ){
        // If necessary, copy agent data from the device to the host in the default stream
        if(h_triages_defaultTriage_variable_patientQueue_data_iteration != currentIteration){
            
            for(unsigned int e = 0; e < numElements; e++){
                gpuErrchk(
                    cudaMemcpy(
                        h_triages_defaultTriage->patientQueue + (e * xmachine_memory_triage_MAX),
                        d_triages_defaultTriage->patientQueue + (e * xmachine_memory_triage_MAX), 
                        count * sizeof(unsigned int), 
                        cudaMemcpyDeviceToHost
                    )
                );
                // Update some global value indicating what data is currently present in that host array.
                h_triages_defaultTriage_variable_patientQueue_data_iteration = currentIteration;
            }
        }

        // Return the value of the index-th element of the relevant host array.
        return h_triages_defaultTriage->patientQueue[index + (element * xmachine_memory_triage_MAX)];

    } else {
        fprintf(stderr, "Warning: Attempting to access the %u-th element of patientQueue for the %u th member of triage_defaultTriage. count is %u at iteration %u\n", element, index, count, currentIteration);
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
 
		gpuErrchk(cudaMemcpy(d_dst->checkpoint, &h_agent->checkpoint, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->chair_no, &h_agent->chair_no, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->box_no, &h_agent->box_no, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->doctor_no, &h_agent->doctor_no, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->specialist_no, &h_agent->specialist_no, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->priority, &h_agent->priority, sizeof(unsigned int), cudaMemcpyHostToDevice));

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
 
		gpuErrchk(cudaMemcpy(d_dst->checkpoint, h_src->checkpoint, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->chair_no, h_src->chair_no, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->box_no, h_src->box_no, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->doctor_no, h_src->doctor_no, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->specialist_no, h_src->specialist_no, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->priority, h_src->priority, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_chair_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_chair_hostToDevice(xmachine_memory_chair_list * d_dst, xmachine_memory_chair * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, &h_agent->x, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, &h_agent->y, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->state, &h_agent->state, sizeof(int), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_chair_hostToDevice(xmachine_memory_chair_list * d_dst, xmachine_memory_chair_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->x, h_src->x, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->y, h_src->y, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->state, h_src->state, count * sizeof(int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_doctor_manager_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_doctor_manager_hostToDevice(xmachine_memory_doctor_manager_list * d_dst, xmachine_memory_doctor_manager * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->tick, &h_agent->tick, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->rear, &h_agent->rear, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->size, &h_agent->size, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
	for(unsigned int i = 0; i < 4; i++){
		gpuErrchk(cudaMemcpy(d_dst->doctors_occupied + (i * xmachine_memory_doctor_manager_MAX), h_agent->doctors_occupied + i, sizeof(int), cudaMemcpyHostToDevice));
    }
 
		gpuErrchk(cudaMemcpy(d_dst->free_doctors, &h_agent->free_doctors, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
	for(unsigned int i = 0; i < 35; i++){
		gpuErrchk(cudaMemcpy(d_dst->patientQueue + (i * xmachine_memory_doctor_manager_MAX), h_agent->patientQueue + i, sizeof(ivec2), cudaMemcpyHostToDevice));
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
void copy_partial_xmachine_memory_doctor_manager_hostToDevice(xmachine_memory_doctor_manager_list * d_dst, xmachine_memory_doctor_manager_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->tick, h_src->tick, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->rear, h_src->rear, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->size, h_src->size, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		for(unsigned int i = 0; i < 4; i++){
			gpuErrchk(cudaMemcpy(d_dst->doctors_occupied + (i * xmachine_memory_doctor_manager_MAX), h_src->doctors_occupied + (i * xmachine_memory_doctor_manager_MAX), count * sizeof(int), cudaMemcpyHostToDevice));
        }

 
		gpuErrchk(cudaMemcpy(d_dst->free_doctors, h_src->free_doctors, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		for(unsigned int i = 0; i < 35; i++){
			gpuErrchk(cudaMemcpy(d_dst->patientQueue + (i * xmachine_memory_doctor_manager_MAX), h_src->patientQueue + (i * xmachine_memory_doctor_manager_MAX), count * sizeof(ivec2), cudaMemcpyHostToDevice));
        }


    }
}


/* copy_single_xmachine_memory_specialist_manager_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_specialist_manager_hostToDevice(xmachine_memory_specialist_manager_list * d_dst, xmachine_memory_specialist_manager * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
	for(unsigned int i = 0; i < 5; i++){
		gpuErrchk(cudaMemcpy(d_dst->tick + (i * xmachine_memory_specialist_manager_MAX), h_agent->tick + i, sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
 
	for(unsigned int i = 0; i < 5; i++){
		gpuErrchk(cudaMemcpy(d_dst->free_specialist + (i * xmachine_memory_specialist_manager_MAX), h_agent->free_specialist + i, sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
 
	for(unsigned int i = 0; i < 5; i++){
		gpuErrchk(cudaMemcpy(d_dst->rear + (i * xmachine_memory_specialist_manager_MAX), h_agent->rear + i, sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
 
	for(unsigned int i = 0; i < 5; i++){
		gpuErrchk(cudaMemcpy(d_dst->size + (i * xmachine_memory_specialist_manager_MAX), h_agent->size + i, sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
 
	for(unsigned int i = 0; i < 35; i++){
		gpuErrchk(cudaMemcpy(d_dst->surgicalQueue + (i * xmachine_memory_specialist_manager_MAX), h_agent->surgicalQueue + i, sizeof(ivec2), cudaMemcpyHostToDevice));
    }
 
	for(unsigned int i = 0; i < 35; i++){
		gpuErrchk(cudaMemcpy(d_dst->pediatricsQueue + (i * xmachine_memory_specialist_manager_MAX), h_agent->pediatricsQueue + i, sizeof(ivec2), cudaMemcpyHostToDevice));
    }
 
	for(unsigned int i = 0; i < 35; i++){
		gpuErrchk(cudaMemcpy(d_dst->gynecologistQueue + (i * xmachine_memory_specialist_manager_MAX), h_agent->gynecologistQueue + i, sizeof(ivec2), cudaMemcpyHostToDevice));
    }
 
	for(unsigned int i = 0; i < 35; i++){
		gpuErrchk(cudaMemcpy(d_dst->geriatricsQueue + (i * xmachine_memory_specialist_manager_MAX), h_agent->geriatricsQueue + i, sizeof(ivec2), cudaMemcpyHostToDevice));
    }
 
	for(unsigned int i = 0; i < 35; i++){
		gpuErrchk(cudaMemcpy(d_dst->psychiatristQueue + (i * xmachine_memory_specialist_manager_MAX), h_agent->psychiatristQueue + i, sizeof(ivec2), cudaMemcpyHostToDevice));
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
void copy_partial_xmachine_memory_specialist_manager_hostToDevice(xmachine_memory_specialist_manager_list * d_dst, xmachine_memory_specialist_manager_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		for(unsigned int i = 0; i < 5; i++){
			gpuErrchk(cudaMemcpy(d_dst->tick + (i * xmachine_memory_specialist_manager_MAX), h_src->tick + (i * xmachine_memory_specialist_manager_MAX), count * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }

 
		for(unsigned int i = 0; i < 5; i++){
			gpuErrchk(cudaMemcpy(d_dst->free_specialist + (i * xmachine_memory_specialist_manager_MAX), h_src->free_specialist + (i * xmachine_memory_specialist_manager_MAX), count * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }

 
		for(unsigned int i = 0; i < 5; i++){
			gpuErrchk(cudaMemcpy(d_dst->rear + (i * xmachine_memory_specialist_manager_MAX), h_src->rear + (i * xmachine_memory_specialist_manager_MAX), count * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }

 
		for(unsigned int i = 0; i < 5; i++){
			gpuErrchk(cudaMemcpy(d_dst->size + (i * xmachine_memory_specialist_manager_MAX), h_src->size + (i * xmachine_memory_specialist_manager_MAX), count * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }

 
		for(unsigned int i = 0; i < 35; i++){
			gpuErrchk(cudaMemcpy(d_dst->surgicalQueue + (i * xmachine_memory_specialist_manager_MAX), h_src->surgicalQueue + (i * xmachine_memory_specialist_manager_MAX), count * sizeof(ivec2), cudaMemcpyHostToDevice));
        }

 
		for(unsigned int i = 0; i < 35; i++){
			gpuErrchk(cudaMemcpy(d_dst->pediatricsQueue + (i * xmachine_memory_specialist_manager_MAX), h_src->pediatricsQueue + (i * xmachine_memory_specialist_manager_MAX), count * sizeof(ivec2), cudaMemcpyHostToDevice));
        }

 
		for(unsigned int i = 0; i < 35; i++){
			gpuErrchk(cudaMemcpy(d_dst->gynecologistQueue + (i * xmachine_memory_specialist_manager_MAX), h_src->gynecologistQueue + (i * xmachine_memory_specialist_manager_MAX), count * sizeof(ivec2), cudaMemcpyHostToDevice));
        }

 
		for(unsigned int i = 0; i < 35; i++){
			gpuErrchk(cudaMemcpy(d_dst->geriatricsQueue + (i * xmachine_memory_specialist_manager_MAX), h_src->geriatricsQueue + (i * xmachine_memory_specialist_manager_MAX), count * sizeof(ivec2), cudaMemcpyHostToDevice));
        }

 
		for(unsigned int i = 0; i < 35; i++){
			gpuErrchk(cudaMemcpy(d_dst->psychiatristQueue + (i * xmachine_memory_specialist_manager_MAX), h_src->psychiatristQueue + (i * xmachine_memory_specialist_manager_MAX), count * sizeof(ivec2), cudaMemcpyHostToDevice));
        }


    }
}


/* copy_single_xmachine_memory_specialist_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_specialist_hostToDevice(xmachine_memory_specialist_list * d_dst, xmachine_memory_specialist * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->current_patient, &h_agent->current_patient, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->tick, &h_agent->tick, sizeof(unsigned int), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_specialist_hostToDevice(xmachine_memory_specialist_list * d_dst, xmachine_memory_specialist_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->current_patient, h_src->current_patient, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->tick, h_src->tick, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

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
 
	for(unsigned int i = 0; i < 100; i++){
		gpuErrchk(cudaMemcpy(d_dst->patientQueue + (i * xmachine_memory_receptionist_MAX), h_agent->patientQueue + i, sizeof(unsigned int), cudaMemcpyHostToDevice));
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
 
		for(unsigned int i = 0; i < 100; i++){
			gpuErrchk(cudaMemcpy(d_dst->patientQueue + (i * xmachine_memory_receptionist_MAX), h_src->patientQueue + (i * xmachine_memory_receptionist_MAX), count * sizeof(unsigned int), cudaMemcpyHostToDevice));
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


/* copy_single_xmachine_memory_agent_generator_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_agent_generator_hostToDevice(xmachine_memory_agent_generator_list * d_dst, xmachine_memory_agent_generator * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->chairs_generated, &h_agent->chairs_generated, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->boxes_generated, &h_agent->boxes_generated, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->doctors_generated, &h_agent->doctors_generated, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->specialists_generated, &h_agent->specialists_generated, sizeof(int), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_agent_generator_hostToDevice(xmachine_memory_agent_generator_list * d_dst, xmachine_memory_agent_generator_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->chairs_generated, h_src->chairs_generated, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->boxes_generated, h_src->boxes_generated, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->doctors_generated, h_src->doctors_generated, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->specialists_generated, h_src->specialists_generated, count * sizeof(int), cudaMemcpyHostToDevice));

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


/* copy_single_xmachine_memory_box_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_box_hostToDevice(xmachine_memory_box_list * d_dst, xmachine_memory_box * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->attending, &h_agent->attending, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->tick, &h_agent->tick, sizeof(unsigned int), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_box_hostToDevice(xmachine_memory_box_list * d_dst, xmachine_memory_box_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->attending, h_src->attending, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->tick, h_src->tick, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_doctor_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_doctor_hostToDevice(xmachine_memory_doctor_list * d_dst, xmachine_memory_doctor * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->id, &h_agent->id, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->current_patient, &h_agent->current_patient, sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->tick, &h_agent->tick, sizeof(unsigned int), cudaMemcpyHostToDevice));

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
void copy_partial_xmachine_memory_doctor_hostToDevice(xmachine_memory_doctor_list * d_dst, xmachine_memory_doctor_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->id, h_src->id, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->current_patient, h_src->current_patient, count * sizeof(int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->tick, h_src->tick, count * sizeof(unsigned int), cudaMemcpyHostToDevice));

    }
}


/* copy_single_xmachine_memory_triage_hostToDevice
 * Private function to copy a host agent struct into a device SoA agent list.
 * @param d_dst destination agent state list
 * @param h_agent agent struct
 */
void copy_single_xmachine_memory_triage_hostToDevice(xmachine_memory_triage_list * d_dst, xmachine_memory_triage * h_agent){
 
		gpuErrchk(cudaMemcpy(d_dst->front, &h_agent->front, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->rear, &h_agent->rear, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->size, &h_agent->size, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->tick, &h_agent->tick, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
	for(unsigned int i = 0; i < 3; i++){
		gpuErrchk(cudaMemcpy(d_dst->boxArray + (i * xmachine_memory_triage_MAX), h_agent->boxArray + i, sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
 
	for(unsigned int i = 0; i < 100; i++){
		gpuErrchk(cudaMemcpy(d_dst->patientQueue + (i * xmachine_memory_triage_MAX), h_agent->patientQueue + i, sizeof(unsigned int), cudaMemcpyHostToDevice));
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
void copy_partial_xmachine_memory_triage_hostToDevice(xmachine_memory_triage_list * d_dst, xmachine_memory_triage_list * h_src, unsigned int count){
    // Only copy elements if there is data to move.
    if (count > 0){
	 
		gpuErrchk(cudaMemcpy(d_dst->front, h_src->front, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->rear, h_src->rear, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->size, h_src->size, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		gpuErrchk(cudaMemcpy(d_dst->tick, h_src->tick, count * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
		for(unsigned int i = 0; i < 3; i++){
			gpuErrchk(cudaMemcpy(d_dst->boxArray + (i * xmachine_memory_triage_MAX), h_src->boxArray + (i * xmachine_memory_triage_MAX), count * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }

 
		for(unsigned int i = 0; i < 100; i++){
			gpuErrchk(cudaMemcpy(d_dst->patientQueue + (i * xmachine_memory_triage_MAX), h_src->patientQueue + (i * xmachine_memory_triage_MAX), count * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }


    }
}

xmachine_memory_agent* h_allocate_agent_agent(){
	xmachine_memory_agent* agent = (xmachine_memory_agent*)malloc(sizeof(xmachine_memory_agent));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_agent));

    agent->estado_movimiento = 0;

    agent->checkpoint = 0;

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
			 
			dst->checkpoint[i] = src[i]->checkpoint;
			 
			dst->chair_no[i] = src[i]->chair_no;
			 
			dst->box_no[i] = src[i]->box_no;
			 
			dst->doctor_no[i] = src[i]->doctor_no;
			 
			dst->specialist_no[i] = src[i]->specialist_no;
			 
			dst->priority[i] = src[i]->priority;
			
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
    h_agents_default_variable_checkpoint_data_iteration = 0;
    h_agents_default_variable_chair_no_data_iteration = 0;
    h_agents_default_variable_box_no_data_iteration = 0;
    h_agents_default_variable_doctor_no_data_iteration = 0;
    h_agents_default_variable_specialist_no_data_iteration = 0;
    h_agents_default_variable_priority_data_iteration = 0;
    

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
        h_agents_default_variable_checkpoint_data_iteration = 0;
        h_agents_default_variable_chair_no_data_iteration = 0;
        h_agents_default_variable_box_no_data_iteration = 0;
        h_agents_default_variable_doctor_no_data_iteration = 0;
        h_agents_default_variable_specialist_no_data_iteration = 0;
        h_agents_default_variable_priority_data_iteration = 0;
        

	}
}

xmachine_memory_chair* h_allocate_agent_chair(){
	xmachine_memory_chair* agent = (xmachine_memory_chair*)malloc(sizeof(xmachine_memory_chair));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_chair));

    agent->state = 0;

	return agent;
}
void h_free_agent_chair(xmachine_memory_chair** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_chair** h_allocate_agent_chair_array(unsigned int count){
	xmachine_memory_chair ** agents = (xmachine_memory_chair**)malloc(count * sizeof(xmachine_memory_chair*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_chair();
	}
	return agents;
}
void h_free_agent_chair_array(xmachine_memory_chair*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_chair(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_chair_AoS_to_SoA(xmachine_memory_chair_list * dst, xmachine_memory_chair** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->x[i] = src[i]->x;
			 
			dst->y[i] = src[i]->y;
			 
			dst->state[i] = src[i]->state;
			
		}
	}
}


void h_add_agent_chair_defaultChair(xmachine_memory_chair* agent){
	if (h_xmachine_memory_chair_count + 1 > xmachine_memory_chair_MAX){
		printf("Error: Buffer size of chair agents in state defaultChair will be exceeded by h_add_agent_chair_defaultChair\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_chair_hostToDevice(d_chairs_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_chair_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_chair_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_chairs_defaultChair, d_chairs_new, h_xmachine_memory_chair_defaultChair_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_chair_defaultChair_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_chair_defaultChair_count, &h_xmachine_memory_chair_defaultChair_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_chairs_defaultChair_variable_id_data_iteration = 0;
    h_chairs_defaultChair_variable_x_data_iteration = 0;
    h_chairs_defaultChair_variable_y_data_iteration = 0;
    h_chairs_defaultChair_variable_state_data_iteration = 0;
    

}
void h_add_agents_chair_defaultChair(xmachine_memory_chair** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_chair_count + count > xmachine_memory_chair_MAX){
			printf("Error: Buffer size of chair agents in state defaultChair will be exceeded by h_add_agents_chair_defaultChair\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_chair_AoS_to_SoA(h_chairs_defaultChair, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_chair_hostToDevice(d_chairs_new, h_chairs_defaultChair, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_chair_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_chair_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_chairs_defaultChair, d_chairs_new, h_xmachine_memory_chair_defaultChair_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_chair_defaultChair_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_chair_defaultChair_count, &h_xmachine_memory_chair_defaultChair_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_chairs_defaultChair_variable_id_data_iteration = 0;
        h_chairs_defaultChair_variable_x_data_iteration = 0;
        h_chairs_defaultChair_variable_y_data_iteration = 0;
        h_chairs_defaultChair_variable_state_data_iteration = 0;
        

	}
}

xmachine_memory_doctor_manager* h_allocate_agent_doctor_manager(){
	xmachine_memory_doctor_manager* agent = (xmachine_memory_doctor_manager*)malloc(sizeof(xmachine_memory_doctor_manager));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_doctor_manager));
	// Agent variable arrays must be allocated
    agent->doctors_occupied = (int*)malloc(4 * sizeof(int));
	// If we have a default value, set each element correctly.
	for(unsigned int index = 0; index < 4; index++){
		agent->doctors_occupied[index] = 0;
	}
    agent->free_doctors = 4;
	// Agent variable arrays must be allocated
    agent->patientQueue = (ivec2*)malloc(35 * sizeof(ivec2));
	// If we have a default value, set each element correctly.
	for(unsigned int index = 0; index < 35; index++){
		agent->patientQueue[index] = {-1,-1};
	}
	return agent;
}
void h_free_agent_doctor_manager(xmachine_memory_doctor_manager** agent){

    free((*agent)->doctors_occupied);

    free((*agent)->patientQueue);
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_doctor_manager** h_allocate_agent_doctor_manager_array(unsigned int count){
	xmachine_memory_doctor_manager ** agents = (xmachine_memory_doctor_manager**)malloc(count * sizeof(xmachine_memory_doctor_manager*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_doctor_manager();
	}
	return agents;
}
void h_free_agent_doctor_manager_array(xmachine_memory_doctor_manager*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_doctor_manager(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_doctor_manager_AoS_to_SoA(xmachine_memory_doctor_manager_list * dst, xmachine_memory_doctor_manager** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->tick[i] = src[i]->tick;
			 
			dst->rear[i] = src[i]->rear;
			 
			dst->size[i] = src[i]->size;
			 
			for(unsigned int j = 0; j < 4; j++){
				dst->doctors_occupied[(j * xmachine_memory_doctor_manager_MAX) + i] = src[i]->doctors_occupied[j];
			}
			 
			dst->free_doctors[i] = src[i]->free_doctors;
			 
			for(unsigned int j = 0; j < 35; j++){
				dst->patientQueue[(j * xmachine_memory_doctor_manager_MAX) + i] = src[i]->patientQueue[j];
			}
			
		}
	}
}


void h_add_agent_doctor_manager_defaultDoctorManager(xmachine_memory_doctor_manager* agent){
	if (h_xmachine_memory_doctor_manager_count + 1 > xmachine_memory_doctor_manager_MAX){
		printf("Error: Buffer size of doctor_manager agents in state defaultDoctorManager will be exceeded by h_add_agent_doctor_manager_defaultDoctorManager\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_doctor_manager_hostToDevice(d_doctor_managers_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_doctor_manager_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_doctor_manager_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_doctor_managers_defaultDoctorManager, d_doctor_managers_new, h_xmachine_memory_doctor_manager_defaultDoctorManager_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_doctor_manager_defaultDoctorManager_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_doctor_manager_defaultDoctorManager_count, &h_xmachine_memory_doctor_manager_defaultDoctorManager_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_doctor_managers_defaultDoctorManager_variable_tick_data_iteration = 0;
    h_doctor_managers_defaultDoctorManager_variable_rear_data_iteration = 0;
    h_doctor_managers_defaultDoctorManager_variable_size_data_iteration = 0;
    h_doctor_managers_defaultDoctorManager_variable_doctors_occupied_data_iteration = 0;
    h_doctor_managers_defaultDoctorManager_variable_free_doctors_data_iteration = 0;
    h_doctor_managers_defaultDoctorManager_variable_patientQueue_data_iteration = 0;
    

}
void h_add_agents_doctor_manager_defaultDoctorManager(xmachine_memory_doctor_manager** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_doctor_manager_count + count > xmachine_memory_doctor_manager_MAX){
			printf("Error: Buffer size of doctor_manager agents in state defaultDoctorManager will be exceeded by h_add_agents_doctor_manager_defaultDoctorManager\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_doctor_manager_AoS_to_SoA(h_doctor_managers_defaultDoctorManager, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_doctor_manager_hostToDevice(d_doctor_managers_new, h_doctor_managers_defaultDoctorManager, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_doctor_manager_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_doctor_manager_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_doctor_managers_defaultDoctorManager, d_doctor_managers_new, h_xmachine_memory_doctor_manager_defaultDoctorManager_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_doctor_manager_defaultDoctorManager_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_doctor_manager_defaultDoctorManager_count, &h_xmachine_memory_doctor_manager_defaultDoctorManager_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_doctor_managers_defaultDoctorManager_variable_tick_data_iteration = 0;
        h_doctor_managers_defaultDoctorManager_variable_rear_data_iteration = 0;
        h_doctor_managers_defaultDoctorManager_variable_size_data_iteration = 0;
        h_doctor_managers_defaultDoctorManager_variable_doctors_occupied_data_iteration = 0;
        h_doctor_managers_defaultDoctorManager_variable_free_doctors_data_iteration = 0;
        h_doctor_managers_defaultDoctorManager_variable_patientQueue_data_iteration = 0;
        

	}
}

xmachine_memory_specialist_manager* h_allocate_agent_specialist_manager(){
	xmachine_memory_specialist_manager* agent = (xmachine_memory_specialist_manager*)malloc(sizeof(xmachine_memory_specialist_manager));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_specialist_manager));
	// Agent variable arrays must be allocated
    agent->tick = (unsigned int*)malloc(5 * sizeof(unsigned int));
	
    // If there is no default value, memset to 0.
    memset(agent->tick, 0, sizeof(unsigned int)*5);	// Agent variable arrays must be allocated
    agent->free_specialist = (unsigned int*)malloc(5 * sizeof(unsigned int));
	// If we have a default value, set each element correctly.
	for(unsigned int index = 0; index < 5; index++){
		agent->free_specialist[index] = 1;
	}	// Agent variable arrays must be allocated
    agent->rear = (unsigned int*)malloc(5 * sizeof(unsigned int));
	
    // If there is no default value, memset to 0.
    memset(agent->rear, 0, sizeof(unsigned int)*5);	// Agent variable arrays must be allocated
    agent->size = (unsigned int*)malloc(5 * sizeof(unsigned int));
	
    // If there is no default value, memset to 0.
    memset(agent->size, 0, sizeof(unsigned int)*5);	// Agent variable arrays must be allocated
    agent->surgicalQueue = (ivec2*)malloc(35 * sizeof(ivec2));
	// If we have a default value, set each element correctly.
	for(unsigned int index = 0; index < 35; index++){
		agent->surgicalQueue[index] = {-1,-1};
	}	// Agent variable arrays must be allocated
    agent->pediatricsQueue = (ivec2*)malloc(35 * sizeof(ivec2));
	// If we have a default value, set each element correctly.
	for(unsigned int index = 0; index < 35; index++){
		agent->pediatricsQueue[index] = {-1,-1};
	}	// Agent variable arrays must be allocated
    agent->gynecologistQueue = (ivec2*)malloc(35 * sizeof(ivec2));
	// If we have a default value, set each element correctly.
	for(unsigned int index = 0; index < 35; index++){
		agent->gynecologistQueue[index] = {-1,-1};
	}	// Agent variable arrays must be allocated
    agent->geriatricsQueue = (ivec2*)malloc(35 * sizeof(ivec2));
	// If we have a default value, set each element correctly.
	for(unsigned int index = 0; index < 35; index++){
		agent->geriatricsQueue[index] = {-1,-1};
	}	// Agent variable arrays must be allocated
    agent->psychiatristQueue = (ivec2*)malloc(35 * sizeof(ivec2));
	// If we have a default value, set each element correctly.
	for(unsigned int index = 0; index < 35; index++){
		agent->psychiatristQueue[index] = {-1,-1};
	}
	return agent;
}
void h_free_agent_specialist_manager(xmachine_memory_specialist_manager** agent){

    free((*agent)->tick);

    free((*agent)->free_specialist);

    free((*agent)->rear);

    free((*agent)->size);

    free((*agent)->surgicalQueue);

    free((*agent)->pediatricsQueue);

    free((*agent)->gynecologistQueue);

    free((*agent)->geriatricsQueue);

    free((*agent)->psychiatristQueue);
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_specialist_manager** h_allocate_agent_specialist_manager_array(unsigned int count){
	xmachine_memory_specialist_manager ** agents = (xmachine_memory_specialist_manager**)malloc(count * sizeof(xmachine_memory_specialist_manager*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_specialist_manager();
	}
	return agents;
}
void h_free_agent_specialist_manager_array(xmachine_memory_specialist_manager*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_specialist_manager(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_specialist_manager_AoS_to_SoA(xmachine_memory_specialist_manager_list * dst, xmachine_memory_specialist_manager** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			for(unsigned int j = 0; j < 5; j++){
				dst->tick[(j * xmachine_memory_specialist_manager_MAX) + i] = src[i]->tick[j];
			}
			 
			for(unsigned int j = 0; j < 5; j++){
				dst->free_specialist[(j * xmachine_memory_specialist_manager_MAX) + i] = src[i]->free_specialist[j];
			}
			 
			for(unsigned int j = 0; j < 5; j++){
				dst->rear[(j * xmachine_memory_specialist_manager_MAX) + i] = src[i]->rear[j];
			}
			 
			for(unsigned int j = 0; j < 5; j++){
				dst->size[(j * xmachine_memory_specialist_manager_MAX) + i] = src[i]->size[j];
			}
			 
			for(unsigned int j = 0; j < 35; j++){
				dst->surgicalQueue[(j * xmachine_memory_specialist_manager_MAX) + i] = src[i]->surgicalQueue[j];
			}
			 
			for(unsigned int j = 0; j < 35; j++){
				dst->pediatricsQueue[(j * xmachine_memory_specialist_manager_MAX) + i] = src[i]->pediatricsQueue[j];
			}
			 
			for(unsigned int j = 0; j < 35; j++){
				dst->gynecologistQueue[(j * xmachine_memory_specialist_manager_MAX) + i] = src[i]->gynecologistQueue[j];
			}
			 
			for(unsigned int j = 0; j < 35; j++){
				dst->geriatricsQueue[(j * xmachine_memory_specialist_manager_MAX) + i] = src[i]->geriatricsQueue[j];
			}
			 
			for(unsigned int j = 0; j < 35; j++){
				dst->psychiatristQueue[(j * xmachine_memory_specialist_manager_MAX) + i] = src[i]->psychiatristQueue[j];
			}
			
		}
	}
}


void h_add_agent_specialist_manager_defaultSpecialistManager(xmachine_memory_specialist_manager* agent){
	if (h_xmachine_memory_specialist_manager_count + 1 > xmachine_memory_specialist_manager_MAX){
		printf("Error: Buffer size of specialist_manager agents in state defaultSpecialistManager will be exceeded by h_add_agent_specialist_manager_defaultSpecialistManager\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_specialist_manager_hostToDevice(d_specialist_managers_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_specialist_manager_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_specialist_manager_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_specialist_managers_defaultSpecialistManager, d_specialist_managers_new, h_xmachine_memory_specialist_manager_defaultSpecialistManager_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_specialist_manager_defaultSpecialistManager_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_specialist_manager_defaultSpecialistManager_count, &h_xmachine_memory_specialist_manager_defaultSpecialistManager_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_specialist_managers_defaultSpecialistManager_variable_id_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_tick_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_free_specialist_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_rear_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_size_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_surgicalQueue_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_pediatricsQueue_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_gynecologistQueue_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_geriatricsQueue_data_iteration = 0;
    h_specialist_managers_defaultSpecialistManager_variable_psychiatristQueue_data_iteration = 0;
    

}
void h_add_agents_specialist_manager_defaultSpecialistManager(xmachine_memory_specialist_manager** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_specialist_manager_count + count > xmachine_memory_specialist_manager_MAX){
			printf("Error: Buffer size of specialist_manager agents in state defaultSpecialistManager will be exceeded by h_add_agents_specialist_manager_defaultSpecialistManager\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_specialist_manager_AoS_to_SoA(h_specialist_managers_defaultSpecialistManager, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_specialist_manager_hostToDevice(d_specialist_managers_new, h_specialist_managers_defaultSpecialistManager, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_specialist_manager_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_specialist_manager_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_specialist_managers_defaultSpecialistManager, d_specialist_managers_new, h_xmachine_memory_specialist_manager_defaultSpecialistManager_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_specialist_manager_defaultSpecialistManager_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_specialist_manager_defaultSpecialistManager_count, &h_xmachine_memory_specialist_manager_defaultSpecialistManager_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_specialist_managers_defaultSpecialistManager_variable_id_data_iteration = 0;
        h_specialist_managers_defaultSpecialistManager_variable_tick_data_iteration = 0;
        h_specialist_managers_defaultSpecialistManager_variable_free_specialist_data_iteration = 0;
        h_specialist_managers_defaultSpecialistManager_variable_rear_data_iteration = 0;
        h_specialist_managers_defaultSpecialistManager_variable_size_data_iteration = 0;
        h_specialist_managers_defaultSpecialistManager_variable_surgicalQueue_data_iteration = 0;
        h_specialist_managers_defaultSpecialistManager_variable_pediatricsQueue_data_iteration = 0;
        h_specialist_managers_defaultSpecialistManager_variable_gynecologistQueue_data_iteration = 0;
        h_specialist_managers_defaultSpecialistManager_variable_geriatricsQueue_data_iteration = 0;
        h_specialist_managers_defaultSpecialistManager_variable_psychiatristQueue_data_iteration = 0;
        

	}
}

xmachine_memory_specialist* h_allocate_agent_specialist(){
	xmachine_memory_specialist* agent = (xmachine_memory_specialist*)malloc(sizeof(xmachine_memory_specialist));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_specialist));

	return agent;
}
void h_free_agent_specialist(xmachine_memory_specialist** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_specialist** h_allocate_agent_specialist_array(unsigned int count){
	xmachine_memory_specialist ** agents = (xmachine_memory_specialist**)malloc(count * sizeof(xmachine_memory_specialist*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_specialist();
	}
	return agents;
}
void h_free_agent_specialist_array(xmachine_memory_specialist*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_specialist(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_specialist_AoS_to_SoA(xmachine_memory_specialist_list * dst, xmachine_memory_specialist** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->current_patient[i] = src[i]->current_patient;
			 
			dst->tick[i] = src[i]->tick;
			
		}
	}
}


void h_add_agent_specialist_defaultSpecialist(xmachine_memory_specialist* agent){
	if (h_xmachine_memory_specialist_count + 1 > xmachine_memory_specialist_MAX){
		printf("Error: Buffer size of specialist agents in state defaultSpecialist will be exceeded by h_add_agent_specialist_defaultSpecialist\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_specialist_hostToDevice(d_specialists_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_specialist_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_specialist_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_specialists_defaultSpecialist, d_specialists_new, h_xmachine_memory_specialist_defaultSpecialist_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_specialist_defaultSpecialist_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_specialist_defaultSpecialist_count, &h_xmachine_memory_specialist_defaultSpecialist_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_specialists_defaultSpecialist_variable_id_data_iteration = 0;
    h_specialists_defaultSpecialist_variable_current_patient_data_iteration = 0;
    h_specialists_defaultSpecialist_variable_tick_data_iteration = 0;
    

}
void h_add_agents_specialist_defaultSpecialist(xmachine_memory_specialist** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_specialist_count + count > xmachine_memory_specialist_MAX){
			printf("Error: Buffer size of specialist agents in state defaultSpecialist will be exceeded by h_add_agents_specialist_defaultSpecialist\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_specialist_AoS_to_SoA(h_specialists_defaultSpecialist, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_specialist_hostToDevice(d_specialists_new, h_specialists_defaultSpecialist, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_specialist_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_specialist_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_specialists_defaultSpecialist, d_specialists_new, h_xmachine_memory_specialist_defaultSpecialist_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_specialist_defaultSpecialist_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_specialist_defaultSpecialist_count, &h_xmachine_memory_specialist_defaultSpecialist_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_specialists_defaultSpecialist_variable_id_data_iteration = 0;
        h_specialists_defaultSpecialist_variable_current_patient_data_iteration = 0;
        h_specialists_defaultSpecialist_variable_tick_data_iteration = 0;
        

	}
}

xmachine_memory_receptionist* h_allocate_agent_receptionist(){
	xmachine_memory_receptionist* agent = (xmachine_memory_receptionist*)malloc(sizeof(xmachine_memory_receptionist));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_receptionist));

    agent->x = 0.093750;

    agent->y = -0.375000;
	// Agent variable arrays must be allocated
    agent->patientQueue = (unsigned int*)malloc(100 * sizeof(unsigned int));
	
    // If there is no default value, memset to 0.
    memset(agent->patientQueue, 0, sizeof(unsigned int)*100);
    agent->tick = 0;

    agent->current_patient = -1;

    agent->attend_patient = 0;

    agent->estado = 0;

	return agent;
}
void h_free_agent_receptionist(xmachine_memory_receptionist** agent){

    free((*agent)->patientQueue);
 
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
			 
			for(unsigned int j = 0; j < 100; j++){
				dst->patientQueue[(j * xmachine_memory_receptionist_MAX) + i] = src[i]->patientQueue[j];
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
    h_receptionists_defaultReceptionist_variable_patientQueue_data_iteration = 0;
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
        h_receptionists_defaultReceptionist_variable_patientQueue_data_iteration = 0;
        h_receptionists_defaultReceptionist_variable_front_data_iteration = 0;
        h_receptionists_defaultReceptionist_variable_rear_data_iteration = 0;
        h_receptionists_defaultReceptionist_variable_size_data_iteration = 0;
        h_receptionists_defaultReceptionist_variable_tick_data_iteration = 0;
        h_receptionists_defaultReceptionist_variable_current_patient_data_iteration = 0;
        h_receptionists_defaultReceptionist_variable_attend_patient_data_iteration = 0;
        h_receptionists_defaultReceptionist_variable_estado_data_iteration = 0;
        

	}
}

xmachine_memory_agent_generator* h_allocate_agent_agent_generator(){
	xmachine_memory_agent_generator* agent = (xmachine_memory_agent_generator*)malloc(sizeof(xmachine_memory_agent_generator));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_agent_generator));

    agent->chairs_generated = 0;

    agent->boxes_generated = 0;

    agent->doctors_generated = 0;

    agent->specialists_generated = 0;

	return agent;
}
void h_free_agent_agent_generator(xmachine_memory_agent_generator** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_agent_generator** h_allocate_agent_agent_generator_array(unsigned int count){
	xmachine_memory_agent_generator ** agents = (xmachine_memory_agent_generator**)malloc(count * sizeof(xmachine_memory_agent_generator*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_agent_generator();
	}
	return agents;
}
void h_free_agent_agent_generator_array(xmachine_memory_agent_generator*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_agent_generator(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_agent_generator_AoS_to_SoA(xmachine_memory_agent_generator_list * dst, xmachine_memory_agent_generator** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->chairs_generated[i] = src[i]->chairs_generated;
			 
			dst->boxes_generated[i] = src[i]->boxes_generated;
			 
			dst->doctors_generated[i] = src[i]->doctors_generated;
			 
			dst->specialists_generated[i] = src[i]->specialists_generated;
			
		}
	}
}


void h_add_agent_agent_generator_defaultGenerator(xmachine_memory_agent_generator* agent){
	if (h_xmachine_memory_agent_generator_count + 1 > xmachine_memory_agent_generator_MAX){
		printf("Error: Buffer size of agent_generator agents in state defaultGenerator will be exceeded by h_add_agent_agent_generator_defaultGenerator\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_agent_generator_hostToDevice(d_agent_generators_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_agent_generator_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_agent_generator_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_agent_generators_defaultGenerator, d_agent_generators_new, h_xmachine_memory_agent_generator_defaultGenerator_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_agent_generator_defaultGenerator_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_agent_generator_defaultGenerator_count, &h_xmachine_memory_agent_generator_defaultGenerator_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_agent_generators_defaultGenerator_variable_chairs_generated_data_iteration = 0;
    h_agent_generators_defaultGenerator_variable_boxes_generated_data_iteration = 0;
    h_agent_generators_defaultGenerator_variable_doctors_generated_data_iteration = 0;
    h_agent_generators_defaultGenerator_variable_specialists_generated_data_iteration = 0;
    

}
void h_add_agents_agent_generator_defaultGenerator(xmachine_memory_agent_generator** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_agent_generator_count + count > xmachine_memory_agent_generator_MAX){
			printf("Error: Buffer size of agent_generator agents in state defaultGenerator will be exceeded by h_add_agents_agent_generator_defaultGenerator\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_agent_generator_AoS_to_SoA(h_agent_generators_defaultGenerator, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_agent_generator_hostToDevice(d_agent_generators_new, h_agent_generators_defaultGenerator, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_agent_generator_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_agent_generator_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_agent_generators_defaultGenerator, d_agent_generators_new, h_xmachine_memory_agent_generator_defaultGenerator_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_agent_generator_defaultGenerator_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_agent_generator_defaultGenerator_count, &h_xmachine_memory_agent_generator_defaultGenerator_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_agent_generators_defaultGenerator_variable_chairs_generated_data_iteration = 0;
        h_agent_generators_defaultGenerator_variable_boxes_generated_data_iteration = 0;
        h_agent_generators_defaultGenerator_variable_doctors_generated_data_iteration = 0;
        h_agent_generators_defaultGenerator_variable_specialists_generated_data_iteration = 0;
        

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

xmachine_memory_box* h_allocate_agent_box(){
	xmachine_memory_box* agent = (xmachine_memory_box*)malloc(sizeof(xmachine_memory_box));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_box));

	return agent;
}
void h_free_agent_box(xmachine_memory_box** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_box** h_allocate_agent_box_array(unsigned int count){
	xmachine_memory_box ** agents = (xmachine_memory_box**)malloc(count * sizeof(xmachine_memory_box*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_box();
	}
	return agents;
}
void h_free_agent_box_array(xmachine_memory_box*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_box(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_box_AoS_to_SoA(xmachine_memory_box_list * dst, xmachine_memory_box** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->attending[i] = src[i]->attending;
			 
			dst->tick[i] = src[i]->tick;
			
		}
	}
}


void h_add_agent_box_defaultBox(xmachine_memory_box* agent){
	if (h_xmachine_memory_box_count + 1 > xmachine_memory_box_MAX){
		printf("Error: Buffer size of box agents in state defaultBox will be exceeded by h_add_agent_box_defaultBox\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_box_hostToDevice(d_boxs_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_box_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_box_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_boxs_defaultBox, d_boxs_new, h_xmachine_memory_box_defaultBox_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_box_defaultBox_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_box_defaultBox_count, &h_xmachine_memory_box_defaultBox_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_boxs_defaultBox_variable_id_data_iteration = 0;
    h_boxs_defaultBox_variable_attending_data_iteration = 0;
    h_boxs_defaultBox_variable_tick_data_iteration = 0;
    

}
void h_add_agents_box_defaultBox(xmachine_memory_box** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_box_count + count > xmachine_memory_box_MAX){
			printf("Error: Buffer size of box agents in state defaultBox will be exceeded by h_add_agents_box_defaultBox\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_box_AoS_to_SoA(h_boxs_defaultBox, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_box_hostToDevice(d_boxs_new, h_boxs_defaultBox, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_box_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_box_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_boxs_defaultBox, d_boxs_new, h_xmachine_memory_box_defaultBox_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_box_defaultBox_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_box_defaultBox_count, &h_xmachine_memory_box_defaultBox_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_boxs_defaultBox_variable_id_data_iteration = 0;
        h_boxs_defaultBox_variable_attending_data_iteration = 0;
        h_boxs_defaultBox_variable_tick_data_iteration = 0;
        

	}
}

xmachine_memory_doctor* h_allocate_agent_doctor(){
	xmachine_memory_doctor* agent = (xmachine_memory_doctor*)malloc(sizeof(xmachine_memory_doctor));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_doctor));

	return agent;
}
void h_free_agent_doctor(xmachine_memory_doctor** agent){
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_doctor** h_allocate_agent_doctor_array(unsigned int count){
	xmachine_memory_doctor ** agents = (xmachine_memory_doctor**)malloc(count * sizeof(xmachine_memory_doctor*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_doctor();
	}
	return agents;
}
void h_free_agent_doctor_array(xmachine_memory_doctor*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_doctor(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_doctor_AoS_to_SoA(xmachine_memory_doctor_list * dst, xmachine_memory_doctor** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->id[i] = src[i]->id;
			 
			dst->current_patient[i] = src[i]->current_patient;
			 
			dst->tick[i] = src[i]->tick;
			
		}
	}
}


void h_add_agent_doctor_defaultDoctor(xmachine_memory_doctor* agent){
	if (h_xmachine_memory_doctor_count + 1 > xmachine_memory_doctor_MAX){
		printf("Error: Buffer size of doctor agents in state defaultDoctor will be exceeded by h_add_agent_doctor_defaultDoctor\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_doctor_hostToDevice(d_doctors_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_doctor_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_doctor_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_doctors_defaultDoctor, d_doctors_new, h_xmachine_memory_doctor_defaultDoctor_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_doctor_defaultDoctor_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_doctor_defaultDoctor_count, &h_xmachine_memory_doctor_defaultDoctor_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_doctors_defaultDoctor_variable_id_data_iteration = 0;
    h_doctors_defaultDoctor_variable_current_patient_data_iteration = 0;
    h_doctors_defaultDoctor_variable_tick_data_iteration = 0;
    

}
void h_add_agents_doctor_defaultDoctor(xmachine_memory_doctor** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_doctor_count + count > xmachine_memory_doctor_MAX){
			printf("Error: Buffer size of doctor agents in state defaultDoctor will be exceeded by h_add_agents_doctor_defaultDoctor\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_doctor_AoS_to_SoA(h_doctors_defaultDoctor, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_doctor_hostToDevice(d_doctors_new, h_doctors_defaultDoctor, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_doctor_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_doctor_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_doctors_defaultDoctor, d_doctors_new, h_xmachine_memory_doctor_defaultDoctor_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_doctor_defaultDoctor_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_doctor_defaultDoctor_count, &h_xmachine_memory_doctor_defaultDoctor_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_doctors_defaultDoctor_variable_id_data_iteration = 0;
        h_doctors_defaultDoctor_variable_current_patient_data_iteration = 0;
        h_doctors_defaultDoctor_variable_tick_data_iteration = 0;
        

	}
}

xmachine_memory_triage* h_allocate_agent_triage(){
	xmachine_memory_triage* agent = (xmachine_memory_triage*)malloc(sizeof(xmachine_memory_triage));
	// Memset the whole agent strcuture
    memset(agent, 0, sizeof(xmachine_memory_triage));

    agent->tick = 0;
	// Agent variable arrays must be allocated
    agent->boxArray = (unsigned int*)malloc(3 * sizeof(unsigned int));
	// If we have a default value, set each element correctly.
	for(unsigned int index = 0; index < 3; index++){
		agent->boxArray[index] = 0;
	}	// Agent variable arrays must be allocated
    agent->patientQueue = (unsigned int*)malloc(100 * sizeof(unsigned int));
	
    // If there is no default value, memset to 0.
    memset(agent->patientQueue, 0, sizeof(unsigned int)*100);
	return agent;
}
void h_free_agent_triage(xmachine_memory_triage** agent){

    free((*agent)->boxArray);

    free((*agent)->patientQueue);
 
	free((*agent));
	(*agent) = NULL;
}
xmachine_memory_triage** h_allocate_agent_triage_array(unsigned int count){
	xmachine_memory_triage ** agents = (xmachine_memory_triage**)malloc(count * sizeof(xmachine_memory_triage*));
	for (unsigned int i = 0; i < count; i++) {
		agents[i] = h_allocate_agent_triage();
	}
	return agents;
}
void h_free_agent_triage_array(xmachine_memory_triage*** agents, unsigned int count){
	for (unsigned int i = 0; i < count; i++) {
		h_free_agent_triage(&((*agents)[i]));
	}
	free((*agents));
	(*agents) = NULL;
}

void h_unpack_agents_triage_AoS_to_SoA(xmachine_memory_triage_list * dst, xmachine_memory_triage** src, unsigned int count){
	if(count > 0){
		for(unsigned int i = 0; i < count; i++){
			 
			dst->front[i] = src[i]->front;
			 
			dst->rear[i] = src[i]->rear;
			 
			dst->size[i] = src[i]->size;
			 
			dst->tick[i] = src[i]->tick;
			 
			for(unsigned int j = 0; j < 3; j++){
				dst->boxArray[(j * xmachine_memory_triage_MAX) + i] = src[i]->boxArray[j];
			}
			 
			for(unsigned int j = 0; j < 100; j++){
				dst->patientQueue[(j * xmachine_memory_triage_MAX) + i] = src[i]->patientQueue[j];
			}
			
		}
	}
}


void h_add_agent_triage_defaultTriage(xmachine_memory_triage* agent){
	if (h_xmachine_memory_triage_count + 1 > xmachine_memory_triage_MAX){
		printf("Error: Buffer size of triage agents in state defaultTriage will be exceeded by h_add_agent_triage_defaultTriage\n");
		exit(EXIT_FAILURE);
	}	

	int blockSize;
	int minGridSize;
	int gridSize;
	unsigned int count = 1;
	
	// Copy data from host struct to device SoA for target state
	copy_single_xmachine_memory_triage_hostToDevice(d_triages_new, agent);

	// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_triage_Agents, no_sm, count);
	gridSize = (count + blockSize - 1) / blockSize;
	append_triage_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_triages_defaultTriage, d_triages_new, h_xmachine_memory_triage_defaultTriage_count, count);
	gpuErrchkLaunch();
	// Update the number of agents in this state.
	h_xmachine_memory_triage_defaultTriage_count += count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_triage_defaultTriage_count, &h_xmachine_memory_triage_defaultTriage_count, sizeof(int)));
	cudaDeviceSynchronize();

    // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
    h_triages_defaultTriage_variable_front_data_iteration = 0;
    h_triages_defaultTriage_variable_rear_data_iteration = 0;
    h_triages_defaultTriage_variable_size_data_iteration = 0;
    h_triages_defaultTriage_variable_tick_data_iteration = 0;
    h_triages_defaultTriage_variable_boxArray_data_iteration = 0;
    h_triages_defaultTriage_variable_patientQueue_data_iteration = 0;
    

}
void h_add_agents_triage_defaultTriage(xmachine_memory_triage** agents, unsigned int count){
	if(count > 0){
		int blockSize;
		int minGridSize;
		int gridSize;

		if (h_xmachine_memory_triage_count + count > xmachine_memory_triage_MAX){
			printf("Error: Buffer size of triage agents in state defaultTriage will be exceeded by h_add_agents_triage_defaultTriage\n");
			exit(EXIT_FAILURE);
		}

		// Unpack data from AoS into the pre-existing SoA
		h_unpack_agents_triage_AoS_to_SoA(h_triages_defaultTriage, agents, count);

		// Copy data from the host SoA to the device SoA for the target state
		copy_partial_xmachine_memory_triage_hostToDevice(d_triages_new, h_triages_defaultTriage, count);

		// Use append kernel (@optimisation - This can be replaced with a pointer swap if the target state list is empty)
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_triage_Agents, no_sm, count);
		gridSize = (count + blockSize - 1) / blockSize;
		append_triage_Agents <<<gridSize, blockSize, 0, stream1 >>>(d_triages_defaultTriage, d_triages_new, h_xmachine_memory_triage_defaultTriage_count, count);
		gpuErrchkLaunch();
		// Update the number of agents in this state.
		h_xmachine_memory_triage_defaultTriage_count += count;
		gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_triage_defaultTriage_count, &h_xmachine_memory_triage_defaultTriage_count, sizeof(int)));
		cudaDeviceSynchronize();

        // Reset host variable status flags for the relevant agent state list as the device state list has been modified.
        h_triages_defaultTriage_variable_front_data_iteration = 0;
        h_triages_defaultTriage_variable_rear_data_iteration = 0;
        h_triages_defaultTriage_variable_size_data_iteration = 0;
        h_triages_defaultTriage_variable_tick_data_iteration = 0;
        h_triages_defaultTriage_variable_boxArray_data_iteration = 0;
        h_triages_defaultTriage_variable_patientQueue_data_iteration = 0;
        

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
unsigned int reduce_agent_default_checkpoint_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->checkpoint),  thrust::device_pointer_cast(d_agents_default->checkpoint) + h_xmachine_memory_agent_default_count);
}

unsigned int count_agent_default_checkpoint_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_agents_default->checkpoint),  thrust::device_pointer_cast(d_agents_default->checkpoint) + h_xmachine_memory_agent_default_count, count_value);
}
unsigned int min_agent_default_checkpoint_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->checkpoint);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_agent_default_checkpoint_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->checkpoint);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_default_chair_no_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->chair_no),  thrust::device_pointer_cast(d_agents_default->chair_no) + h_xmachine_memory_agent_default_count);
}

int count_agent_default_chair_no_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agents_default->chair_no),  thrust::device_pointer_cast(d_agents_default->chair_no) + h_xmachine_memory_agent_default_count, count_value);
}
int min_agent_default_chair_no_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->chair_no);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_default_chair_no_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->chair_no);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_agent_default_box_no_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->box_no),  thrust::device_pointer_cast(d_agents_default->box_no) + h_xmachine_memory_agent_default_count);
}

unsigned int count_agent_default_box_no_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_agents_default->box_no),  thrust::device_pointer_cast(d_agents_default->box_no) + h_xmachine_memory_agent_default_count, count_value);
}
unsigned int min_agent_default_box_no_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->box_no);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_agent_default_box_no_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->box_no);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_agent_default_doctor_no_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->doctor_no),  thrust::device_pointer_cast(d_agents_default->doctor_no) + h_xmachine_memory_agent_default_count);
}

unsigned int count_agent_default_doctor_no_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_agents_default->doctor_no),  thrust::device_pointer_cast(d_agents_default->doctor_no) + h_xmachine_memory_agent_default_count, count_value);
}
unsigned int min_agent_default_doctor_no_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->doctor_no);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_agent_default_doctor_no_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->doctor_no);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_agent_default_specialist_no_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->specialist_no),  thrust::device_pointer_cast(d_agents_default->specialist_no) + h_xmachine_memory_agent_default_count);
}

unsigned int count_agent_default_specialist_no_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_agents_default->specialist_no),  thrust::device_pointer_cast(d_agents_default->specialist_no) + h_xmachine_memory_agent_default_count, count_value);
}
unsigned int min_agent_default_specialist_no_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->specialist_no);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_agent_default_specialist_no_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->specialist_no);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_agent_default_priority_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agents_default->priority),  thrust::device_pointer_cast(d_agents_default->priority) + h_xmachine_memory_agent_default_count);
}

unsigned int count_agent_default_priority_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_agents_default->priority),  thrust::device_pointer_cast(d_agents_default->priority) + h_xmachine_memory_agent_default_count, count_value);
}
unsigned int min_agent_default_priority_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->priority);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_agent_default_priority_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_agents_default->priority);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_default_count) - thrust_ptr;
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
int reduce_chair_defaultChair_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_chairs_defaultChair->id),  thrust::device_pointer_cast(d_chairs_defaultChair->id) + h_xmachine_memory_chair_defaultChair_count);
}

int count_chair_defaultChair_id_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_chairs_defaultChair->id),  thrust::device_pointer_cast(d_chairs_defaultChair->id) + h_xmachine_memory_chair_defaultChair_count, count_value);
}
int min_chair_defaultChair_id_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_chairs_defaultChair->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_chair_defaultChair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_chair_defaultChair_id_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_chairs_defaultChair->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_chair_defaultChair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_chair_defaultChair_x_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_chairs_defaultChair->x),  thrust::device_pointer_cast(d_chairs_defaultChair->x) + h_xmachine_memory_chair_defaultChair_count);
}

int count_chair_defaultChair_x_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_chairs_defaultChair->x),  thrust::device_pointer_cast(d_chairs_defaultChair->x) + h_xmachine_memory_chair_defaultChair_count, count_value);
}
int min_chair_defaultChair_x_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_chairs_defaultChair->x);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_chair_defaultChair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_chair_defaultChair_x_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_chairs_defaultChair->x);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_chair_defaultChair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_chair_defaultChair_y_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_chairs_defaultChair->y),  thrust::device_pointer_cast(d_chairs_defaultChair->y) + h_xmachine_memory_chair_defaultChair_count);
}

int count_chair_defaultChair_y_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_chairs_defaultChair->y),  thrust::device_pointer_cast(d_chairs_defaultChair->y) + h_xmachine_memory_chair_defaultChair_count, count_value);
}
int min_chair_defaultChair_y_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_chairs_defaultChair->y);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_chair_defaultChair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_chair_defaultChair_y_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_chairs_defaultChair->y);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_chair_defaultChair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_chair_defaultChair_state_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_chairs_defaultChair->state),  thrust::device_pointer_cast(d_chairs_defaultChair->state) + h_xmachine_memory_chair_defaultChair_count);
}

int count_chair_defaultChair_state_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_chairs_defaultChair->state),  thrust::device_pointer_cast(d_chairs_defaultChair->state) + h_xmachine_memory_chair_defaultChair_count, count_value);
}
int min_chair_defaultChair_state_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_chairs_defaultChair->state);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_chair_defaultChair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_chair_defaultChair_state_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_chairs_defaultChair->state);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_chair_defaultChair_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_doctor_manager_defaultDoctorManager_tick_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->tick),  thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->tick) + h_xmachine_memory_doctor_manager_defaultDoctorManager_count);
}

unsigned int count_doctor_manager_defaultDoctorManager_tick_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->tick),  thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->tick) + h_xmachine_memory_doctor_manager_defaultDoctorManager_count, count_value);
}
unsigned int min_doctor_manager_defaultDoctorManager_tick_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->tick);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_doctor_manager_defaultDoctorManager_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_doctor_manager_defaultDoctorManager_tick_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->tick);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_doctor_manager_defaultDoctorManager_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_doctor_manager_defaultDoctorManager_rear_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->rear),  thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->rear) + h_xmachine_memory_doctor_manager_defaultDoctorManager_count);
}

unsigned int count_doctor_manager_defaultDoctorManager_rear_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->rear),  thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->rear) + h_xmachine_memory_doctor_manager_defaultDoctorManager_count, count_value);
}
unsigned int min_doctor_manager_defaultDoctorManager_rear_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->rear);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_doctor_manager_defaultDoctorManager_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_doctor_manager_defaultDoctorManager_rear_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->rear);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_doctor_manager_defaultDoctorManager_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_doctor_manager_defaultDoctorManager_size_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->size),  thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->size) + h_xmachine_memory_doctor_manager_defaultDoctorManager_count);
}

unsigned int count_doctor_manager_defaultDoctorManager_size_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->size),  thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->size) + h_xmachine_memory_doctor_manager_defaultDoctorManager_count, count_value);
}
unsigned int min_doctor_manager_defaultDoctorManager_size_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->size);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_doctor_manager_defaultDoctorManager_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_doctor_manager_defaultDoctorManager_size_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->size);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_doctor_manager_defaultDoctorManager_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_doctor_manager_defaultDoctorManager_free_doctors_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->free_doctors),  thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->free_doctors) + h_xmachine_memory_doctor_manager_defaultDoctorManager_count);
}

unsigned int count_doctor_manager_defaultDoctorManager_free_doctors_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->free_doctors),  thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->free_doctors) + h_xmachine_memory_doctor_manager_defaultDoctorManager_count, count_value);
}
unsigned int min_doctor_manager_defaultDoctorManager_free_doctors_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->free_doctors);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_doctor_manager_defaultDoctorManager_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_doctor_manager_defaultDoctorManager_free_doctors_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_doctor_managers_defaultDoctorManager->free_doctors);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_doctor_manager_defaultDoctorManager_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_specialist_manager_defaultSpecialistManager_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_specialist_managers_defaultSpecialistManager->id),  thrust::device_pointer_cast(d_specialist_managers_defaultSpecialistManager->id) + h_xmachine_memory_specialist_manager_defaultSpecialistManager_count);
}

unsigned int count_specialist_manager_defaultSpecialistManager_id_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_specialist_managers_defaultSpecialistManager->id),  thrust::device_pointer_cast(d_specialist_managers_defaultSpecialistManager->id) + h_xmachine_memory_specialist_manager_defaultSpecialistManager_count, count_value);
}
unsigned int min_specialist_manager_defaultSpecialistManager_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_specialist_managers_defaultSpecialistManager->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_specialist_manager_defaultSpecialistManager_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_specialist_manager_defaultSpecialistManager_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_specialist_managers_defaultSpecialistManager->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_specialist_manager_defaultSpecialistManager_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_specialist_defaultSpecialist_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_specialists_defaultSpecialist->id),  thrust::device_pointer_cast(d_specialists_defaultSpecialist->id) + h_xmachine_memory_specialist_defaultSpecialist_count);
}

unsigned int count_specialist_defaultSpecialist_id_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_specialists_defaultSpecialist->id),  thrust::device_pointer_cast(d_specialists_defaultSpecialist->id) + h_xmachine_memory_specialist_defaultSpecialist_count, count_value);
}
unsigned int min_specialist_defaultSpecialist_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_specialists_defaultSpecialist->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_specialist_defaultSpecialist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_specialist_defaultSpecialist_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_specialists_defaultSpecialist->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_specialist_defaultSpecialist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_specialist_defaultSpecialist_current_patient_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_specialists_defaultSpecialist->current_patient),  thrust::device_pointer_cast(d_specialists_defaultSpecialist->current_patient) + h_xmachine_memory_specialist_defaultSpecialist_count);
}

unsigned int count_specialist_defaultSpecialist_current_patient_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_specialists_defaultSpecialist->current_patient),  thrust::device_pointer_cast(d_specialists_defaultSpecialist->current_patient) + h_xmachine_memory_specialist_defaultSpecialist_count, count_value);
}
unsigned int min_specialist_defaultSpecialist_current_patient_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_specialists_defaultSpecialist->current_patient);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_specialist_defaultSpecialist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_specialist_defaultSpecialist_current_patient_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_specialists_defaultSpecialist->current_patient);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_specialist_defaultSpecialist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_specialist_defaultSpecialist_tick_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_specialists_defaultSpecialist->tick),  thrust::device_pointer_cast(d_specialists_defaultSpecialist->tick) + h_xmachine_memory_specialist_defaultSpecialist_count);
}

unsigned int count_specialist_defaultSpecialist_tick_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_specialists_defaultSpecialist->tick),  thrust::device_pointer_cast(d_specialists_defaultSpecialist->tick) + h_xmachine_memory_specialist_defaultSpecialist_count, count_value);
}
unsigned int min_specialist_defaultSpecialist_tick_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_specialists_defaultSpecialist->tick);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_specialist_defaultSpecialist_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_specialist_defaultSpecialist_tick_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_specialists_defaultSpecialist->tick);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_specialist_defaultSpecialist_count) - thrust_ptr;
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
int reduce_agent_generator_defaultGenerator_chairs_generated_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agent_generators_defaultGenerator->chairs_generated),  thrust::device_pointer_cast(d_agent_generators_defaultGenerator->chairs_generated) + h_xmachine_memory_agent_generator_defaultGenerator_count);
}

int count_agent_generator_defaultGenerator_chairs_generated_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agent_generators_defaultGenerator->chairs_generated),  thrust::device_pointer_cast(d_agent_generators_defaultGenerator->chairs_generated) + h_xmachine_memory_agent_generator_defaultGenerator_count, count_value);
}
int min_agent_generator_defaultGenerator_chairs_generated_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agent_generators_defaultGenerator->chairs_generated);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_generator_defaultGenerator_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_generator_defaultGenerator_chairs_generated_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agent_generators_defaultGenerator->chairs_generated);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_generator_defaultGenerator_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_generator_defaultGenerator_boxes_generated_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agent_generators_defaultGenerator->boxes_generated),  thrust::device_pointer_cast(d_agent_generators_defaultGenerator->boxes_generated) + h_xmachine_memory_agent_generator_defaultGenerator_count);
}

int count_agent_generator_defaultGenerator_boxes_generated_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agent_generators_defaultGenerator->boxes_generated),  thrust::device_pointer_cast(d_agent_generators_defaultGenerator->boxes_generated) + h_xmachine_memory_agent_generator_defaultGenerator_count, count_value);
}
int min_agent_generator_defaultGenerator_boxes_generated_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agent_generators_defaultGenerator->boxes_generated);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_generator_defaultGenerator_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_generator_defaultGenerator_boxes_generated_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agent_generators_defaultGenerator->boxes_generated);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_generator_defaultGenerator_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_generator_defaultGenerator_doctors_generated_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agent_generators_defaultGenerator->doctors_generated),  thrust::device_pointer_cast(d_agent_generators_defaultGenerator->doctors_generated) + h_xmachine_memory_agent_generator_defaultGenerator_count);
}

int count_agent_generator_defaultGenerator_doctors_generated_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agent_generators_defaultGenerator->doctors_generated),  thrust::device_pointer_cast(d_agent_generators_defaultGenerator->doctors_generated) + h_xmachine_memory_agent_generator_defaultGenerator_count, count_value);
}
int min_agent_generator_defaultGenerator_doctors_generated_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agent_generators_defaultGenerator->doctors_generated);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_generator_defaultGenerator_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_generator_defaultGenerator_doctors_generated_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agent_generators_defaultGenerator->doctors_generated);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_generator_defaultGenerator_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_agent_generator_defaultGenerator_specialists_generated_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_agent_generators_defaultGenerator->specialists_generated),  thrust::device_pointer_cast(d_agent_generators_defaultGenerator->specialists_generated) + h_xmachine_memory_agent_generator_defaultGenerator_count);
}

int count_agent_generator_defaultGenerator_specialists_generated_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_agent_generators_defaultGenerator->specialists_generated),  thrust::device_pointer_cast(d_agent_generators_defaultGenerator->specialists_generated) + h_xmachine_memory_agent_generator_defaultGenerator_count, count_value);
}
int min_agent_generator_defaultGenerator_specialists_generated_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agent_generators_defaultGenerator->specialists_generated);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_generator_defaultGenerator_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_agent_generator_defaultGenerator_specialists_generated_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_agent_generators_defaultGenerator->specialists_generated);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_agent_generator_defaultGenerator_count) - thrust_ptr;
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
unsigned int reduce_box_defaultBox_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_boxs_defaultBox->id),  thrust::device_pointer_cast(d_boxs_defaultBox->id) + h_xmachine_memory_box_defaultBox_count);
}

unsigned int count_box_defaultBox_id_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_boxs_defaultBox->id),  thrust::device_pointer_cast(d_boxs_defaultBox->id) + h_xmachine_memory_box_defaultBox_count, count_value);
}
unsigned int min_box_defaultBox_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_boxs_defaultBox->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_box_defaultBox_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_box_defaultBox_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_boxs_defaultBox->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_box_defaultBox_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_box_defaultBox_attending_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_boxs_defaultBox->attending),  thrust::device_pointer_cast(d_boxs_defaultBox->attending) + h_xmachine_memory_box_defaultBox_count);
}

unsigned int count_box_defaultBox_attending_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_boxs_defaultBox->attending),  thrust::device_pointer_cast(d_boxs_defaultBox->attending) + h_xmachine_memory_box_defaultBox_count, count_value);
}
unsigned int min_box_defaultBox_attending_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_boxs_defaultBox->attending);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_box_defaultBox_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_box_defaultBox_attending_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_boxs_defaultBox->attending);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_box_defaultBox_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_box_defaultBox_tick_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_boxs_defaultBox->tick),  thrust::device_pointer_cast(d_boxs_defaultBox->tick) + h_xmachine_memory_box_defaultBox_count);
}

unsigned int count_box_defaultBox_tick_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_boxs_defaultBox->tick),  thrust::device_pointer_cast(d_boxs_defaultBox->tick) + h_xmachine_memory_box_defaultBox_count, count_value);
}
unsigned int min_box_defaultBox_tick_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_boxs_defaultBox->tick);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_box_defaultBox_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_box_defaultBox_tick_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_boxs_defaultBox->tick);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_box_defaultBox_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_doctor_defaultDoctor_id_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_doctors_defaultDoctor->id),  thrust::device_pointer_cast(d_doctors_defaultDoctor->id) + h_xmachine_memory_doctor_defaultDoctor_count);
}

unsigned int count_doctor_defaultDoctor_id_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_doctors_defaultDoctor->id),  thrust::device_pointer_cast(d_doctors_defaultDoctor->id) + h_xmachine_memory_doctor_defaultDoctor_count, count_value);
}
unsigned int min_doctor_defaultDoctor_id_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_doctors_defaultDoctor->id);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_doctor_defaultDoctor_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_doctor_defaultDoctor_id_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_doctors_defaultDoctor->id);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_doctor_defaultDoctor_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int reduce_doctor_defaultDoctor_current_patient_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_doctors_defaultDoctor->current_patient),  thrust::device_pointer_cast(d_doctors_defaultDoctor->current_patient) + h_xmachine_memory_doctor_defaultDoctor_count);
}

int count_doctor_defaultDoctor_current_patient_variable(int count_value){
    //count in default stream
    return (int)thrust::count(thrust::device_pointer_cast(d_doctors_defaultDoctor->current_patient),  thrust::device_pointer_cast(d_doctors_defaultDoctor->current_patient) + h_xmachine_memory_doctor_defaultDoctor_count, count_value);
}
int min_doctor_defaultDoctor_current_patient_variable(){
    //min in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_doctors_defaultDoctor->current_patient);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_doctor_defaultDoctor_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
int max_doctor_defaultDoctor_current_patient_variable(){
    //max in default stream
    thrust::device_ptr<int> thrust_ptr = thrust::device_pointer_cast(d_doctors_defaultDoctor->current_patient);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_doctor_defaultDoctor_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_doctor_defaultDoctor_tick_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_doctors_defaultDoctor->tick),  thrust::device_pointer_cast(d_doctors_defaultDoctor->tick) + h_xmachine_memory_doctor_defaultDoctor_count);
}

unsigned int count_doctor_defaultDoctor_tick_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_doctors_defaultDoctor->tick),  thrust::device_pointer_cast(d_doctors_defaultDoctor->tick) + h_xmachine_memory_doctor_defaultDoctor_count, count_value);
}
unsigned int min_doctor_defaultDoctor_tick_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_doctors_defaultDoctor->tick);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_doctor_defaultDoctor_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_doctor_defaultDoctor_tick_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_doctors_defaultDoctor->tick);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_doctor_defaultDoctor_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_triage_defaultTriage_front_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_triages_defaultTriage->front),  thrust::device_pointer_cast(d_triages_defaultTriage->front) + h_xmachine_memory_triage_defaultTriage_count);
}

unsigned int count_triage_defaultTriage_front_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_triages_defaultTriage->front),  thrust::device_pointer_cast(d_triages_defaultTriage->front) + h_xmachine_memory_triage_defaultTriage_count, count_value);
}
unsigned int min_triage_defaultTriage_front_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_triages_defaultTriage->front);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_triage_defaultTriage_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_triage_defaultTriage_front_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_triages_defaultTriage->front);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_triage_defaultTriage_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_triage_defaultTriage_rear_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_triages_defaultTriage->rear),  thrust::device_pointer_cast(d_triages_defaultTriage->rear) + h_xmachine_memory_triage_defaultTriage_count);
}

unsigned int count_triage_defaultTriage_rear_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_triages_defaultTriage->rear),  thrust::device_pointer_cast(d_triages_defaultTriage->rear) + h_xmachine_memory_triage_defaultTriage_count, count_value);
}
unsigned int min_triage_defaultTriage_rear_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_triages_defaultTriage->rear);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_triage_defaultTriage_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_triage_defaultTriage_rear_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_triages_defaultTriage->rear);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_triage_defaultTriage_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_triage_defaultTriage_size_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_triages_defaultTriage->size),  thrust::device_pointer_cast(d_triages_defaultTriage->size) + h_xmachine_memory_triage_defaultTriage_count);
}

unsigned int count_triage_defaultTriage_size_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_triages_defaultTriage->size),  thrust::device_pointer_cast(d_triages_defaultTriage->size) + h_xmachine_memory_triage_defaultTriage_count, count_value);
}
unsigned int min_triage_defaultTriage_size_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_triages_defaultTriage->size);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_triage_defaultTriage_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_triage_defaultTriage_size_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_triages_defaultTriage->size);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_triage_defaultTriage_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int reduce_triage_defaultTriage_tick_variable(){
    //reduce in default stream
    return thrust::reduce(thrust::device_pointer_cast(d_triages_defaultTriage->tick),  thrust::device_pointer_cast(d_triages_defaultTriage->tick) + h_xmachine_memory_triage_defaultTriage_count);
}

unsigned int count_triage_defaultTriage_tick_variable(unsigned int count_value){
    //count in default stream
    return (unsigned int)thrust::count(thrust::device_pointer_cast(d_triages_defaultTriage->tick),  thrust::device_pointer_cast(d_triages_defaultTriage->tick) + h_xmachine_memory_triage_defaultTriage_count, count_value);
}
unsigned int min_triage_defaultTriage_tick_variable(){
    //min in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_triages_defaultTriage->tick);
    size_t result_offset = thrust::min_element(thrust_ptr, thrust_ptr + h_xmachine_memory_triage_defaultTriage_count) - thrust_ptr;
    return *(thrust_ptr + result_offset);
}
unsigned int max_triage_defaultTriage_tick_variable(){
    //max in default stream
    thrust::device_ptr<unsigned int> thrust_ptr = thrust::device_pointer_cast(d_triages_defaultTriage->tick);
    size_t result_offset = thrust::max_element(thrust_ptr, thrust_ptr + h_xmachine_memory_triage_defaultTriage_count) - thrust_ptr;
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
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_agent_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agents);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (move)
	//Reallocate   : true
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
	xmachine_memory_agent_list* move_agents_temp = d_agents;
	d_agents = d_agents_swap;
	d_agents_swap = move_agents_temp;
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
		printf("Error: Buffer size of move agents in state default will be exceeded moving working agents to next state in function move\n");
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
int agent_receive_chair_state_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_chair_state));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** agent_receive_chair_state
 * Agent function prototype for receive_chair_state function of agent agent
 */
void agent_receive_chair_state(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_receive_chair_state, agent_receive_chair_state_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_receive_chair_state_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (receive_chair_state)
	//Reallocate   : false
	//Input        : chair_state
	//Output       : 
	//Agent Output : 
	GPUFLAME_receive_chair_state<<<g, b, sm_size, stream>>>(d_agents, d_chair_states, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of receive_chair_state agents in state default will be exceeded moving working agents to next state in function receive_chair_state\n");
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
int agent_output_chair_contact_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_output_chair_contact
 * Agent function prototype for output_chair_contact function of agent agent
 */
void agent_output_chair_contact(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, output_chair_contact_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	output_chair_contact_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
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
	if (h_message_chair_contact_count + h_xmachine_memory_agent_count > xmachine_message_chair_contact_MAX){
		printf("Error: Buffer size of chair_contact message will be exceeded in function output_chair_contact\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_chair_contact, agent_output_chair_contact_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_output_chair_contact_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_chair_contact_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_chair_contact_output_type, &h_message_chair_contact_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_chair_contact_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_chair_contact_swaps<<<gridSize, blockSize, 0, stream>>>(d_chair_contacts); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (output_chair_contact)
	//Reallocate   : false
	//Input        : 
	//Output       : chair_contact
	//Agent Output : 
	GPUFLAME_output_chair_contact<<<g, b, sm_size, stream>>>(d_agents, d_chair_contacts);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//chair_contact Message Type Prefix Sum
	
	//swap output
	xmachine_message_chair_contact_list* d_chair_contacts_scanswap_temp = d_chair_contacts;
	d_chair_contacts = d_chair_contacts_swap;
	d_chair_contacts_swap = d_chair_contacts_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_chair_contacts_swap->_scan_input,
        d_chair_contacts_swap->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_chair_contact_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_chair_contact_messages<<<gridSize, blockSize, 0, stream>>>(d_chair_contacts, d_chair_contacts_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_chair_contacts_swap->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_chair_contacts_swap->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_chair_contact_count += scan_last_sum+1;
	}else{
		h_message_chair_contact_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_chair_contact_count, &h_message_chair_contact_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of output_chair_contact agents in state default will be exceeded moving working agents to next state in function output_chair_contact\n");
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
int agent_output_free_chair_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_output_free_chair
 * Agent function prototype for output_free_chair function of agent agent
 */
void agent_output_free_chair(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, output_free_chair_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	output_free_chair_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
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
	if (h_message_free_chair_count + h_xmachine_memory_agent_count > xmachine_message_free_chair_MAX){
		printf("Error: Buffer size of free_chair message will be exceeded in function output_free_chair\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_free_chair, agent_output_free_chair_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_output_free_chair_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_free_chair_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_free_chair_output_type, &h_message_free_chair_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_free_chair_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_free_chair_swaps<<<gridSize, blockSize, 0, stream>>>(d_free_chairs); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (output_free_chair)
	//Reallocate   : false
	//Input        : 
	//Output       : free_chair
	//Agent Output : 
	GPUFLAME_output_free_chair<<<g, b, sm_size, stream>>>(d_agents, d_free_chairs);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//free_chair Message Type Prefix Sum
	
	//swap output
	xmachine_message_free_chair_list* d_free_chairs_scanswap_temp = d_free_chairs;
	d_free_chairs = d_free_chairs_swap;
	d_free_chairs_swap = d_free_chairs_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_free_chairs_swap->_scan_input,
        d_free_chairs_swap->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_free_chair_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_free_chair_messages<<<gridSize, blockSize, 0, stream>>>(d_free_chairs, d_free_chairs_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_free_chairs_swap->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_free_chairs_swap->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_free_chair_count += scan_last_sum+1;
	}else{
		h_message_free_chair_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_free_chair_count, &h_message_free_chair_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of output_free_chair agents in state default will be exceeded moving working agents to next state in function output_free_chair\n");
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
int agent_receive_chair_response_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_chair_response));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** agent_receive_chair_response
 * Agent function prototype for receive_chair_response function of agent agent
 */
void agent_receive_chair_response(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, receive_chair_response_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	receive_chair_response_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_receive_chair_response, agent_receive_chair_response_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_receive_chair_response_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_agent_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_agent_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agents);
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (receive_chair_response)
	//Reallocate   : true
	//Input        : chair_response
	//Output       : 
	//Agent Output : 
	GPUFLAME_receive_chair_response<<<g, b, sm_size, stream>>>(d_agents, d_chair_responses);
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
	xmachine_memory_agent_list* receive_chair_response_agents_temp = d_agents;
	d_agents = d_agents_swap;
	d_agents_swap = receive_chair_response_agents_temp;
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
		printf("Error: Buffer size of receive_chair_response agents in state default will be exceeded moving working agents to next state in function receive_chair_response\n");
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
int agent_receive_check_in_response_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_check_in_response));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** agent_receive_check_in_response
 * Agent function prototype for receive_check_in_response function of agent agent
 */
void agent_receive_check_in_response(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, receive_check_in_response_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	receive_check_in_response_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
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
		printf("Error: Buffer size of chair_petition message will be exceeded in function receive_check_in_response\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_receive_check_in_response, agent_receive_check_in_response_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_receive_check_in_response_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_chair_petition_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_chair_petition_output_type, &h_message_chair_petition_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_chair_petition_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_chair_petition_swaps<<<gridSize, blockSize, 0, stream>>>(d_chair_petitions); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (receive_check_in_response)
	//Reallocate   : false
	//Input        : check_in_response
	//Output       : chair_petition
	//Agent Output : 
	GPUFLAME_receive_check_in_response<<<g, b, sm_size, stream>>>(d_agents, d_check_in_responses, d_chair_petitions);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
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
		printf("Error: Buffer size of receive_check_in_response agents in state default will be exceeded moving working agents to next state in function receive_check_in_response\n");
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
int agent_output_box_petition_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_output_box_petition
 * Agent function prototype for output_box_petition function of agent agent
 */
void agent_output_box_petition(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, output_box_petition_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	output_box_petition_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
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
	if (h_message_box_petition_count + h_xmachine_memory_agent_count > xmachine_message_box_petition_MAX){
		printf("Error: Buffer size of box_petition message will be exceeded in function output_box_petition\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_box_petition, agent_output_box_petition_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_output_box_petition_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_box_petition_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_box_petition_output_type, &h_message_box_petition_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_box_petition_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_box_petition_swaps<<<gridSize, blockSize, 0, stream>>>(d_box_petitions); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (output_box_petition)
	//Reallocate   : false
	//Input        : 
	//Output       : box_petition
	//Agent Output : 
	GPUFLAME_output_box_petition<<<g, b, sm_size, stream>>>(d_agents, d_box_petitions);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//box_petition Message Type Prefix Sum
	
	//swap output
	xmachine_message_box_petition_list* d_box_petitions_scanswap_temp = d_box_petitions;
	d_box_petitions = d_box_petitions_swap;
	d_box_petitions_swap = d_box_petitions_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_box_petitions_swap->_scan_input,
        d_box_petitions_swap->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_box_petition_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_box_petition_messages<<<gridSize, blockSize, 0, stream>>>(d_box_petitions, d_box_petitions_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_box_petitions_swap->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_box_petitions_swap->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_box_petition_count += scan_last_sum+1;
	}else{
		h_message_box_petition_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_box_petition_count, &h_message_box_petition_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of output_box_petition agents in state default will be exceeded moving working agents to next state in function output_box_petition\n");
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
int agent_receive_box_response_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_box_response));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** agent_receive_box_response
 * Agent function prototype for receive_box_response function of agent agent
 */
void agent_receive_box_response(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_receive_box_response, agent_receive_box_response_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_receive_box_response_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (receive_box_response)
	//Reallocate   : false
	//Input        : box_response
	//Output       : 
	//Agent Output : 
	GPUFLAME_receive_box_response<<<g, b, sm_size, stream>>>(d_agents, d_box_responses);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of receive_box_response agents in state default will be exceeded moving working agents to next state in function receive_box_response\n");
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
int agent_output_doctor_petition_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_output_doctor_petition
 * Agent function prototype for output_doctor_petition function of agent agent
 */
void agent_output_doctor_petition(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, output_doctor_petition_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	output_doctor_petition_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
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
	if (h_message_doctor_petition_count + h_xmachine_memory_agent_count > xmachine_message_doctor_petition_MAX){
		printf("Error: Buffer size of doctor_petition message will be exceeded in function output_doctor_petition\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_doctor_petition, agent_output_doctor_petition_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_output_doctor_petition_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_doctor_petition_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_doctor_petition_output_type, &h_message_doctor_petition_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_doctor_petition_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_doctor_petition_swaps<<<gridSize, blockSize, 0, stream>>>(d_doctor_petitions); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (output_doctor_petition)
	//Reallocate   : false
	//Input        : 
	//Output       : doctor_petition
	//Agent Output : 
	GPUFLAME_output_doctor_petition<<<g, b, sm_size, stream>>>(d_agents, d_doctor_petitions);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//doctor_petition Message Type Prefix Sum
	
	//swap output
	xmachine_message_doctor_petition_list* d_doctor_petitions_scanswap_temp = d_doctor_petitions;
	d_doctor_petitions = d_doctor_petitions_swap;
	d_doctor_petitions_swap = d_doctor_petitions_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_doctor_petitions_swap->_scan_input,
        d_doctor_petitions_swap->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_doctor_petition_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_doctor_petition_messages<<<gridSize, blockSize, 0, stream>>>(d_doctor_petitions, d_doctor_petitions_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_doctor_petitions_swap->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_doctor_petitions_swap->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_doctor_petition_count += scan_last_sum+1;
	}else{
		h_message_doctor_petition_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_doctor_petition_count, &h_message_doctor_petition_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of output_doctor_petition agents in state default will be exceeded moving working agents to next state in function output_doctor_petition\n");
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
int agent_receive_doctor_response_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_doctor_response));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** agent_receive_doctor_response
 * Agent function prototype for receive_doctor_response function of agent agent
 */
void agent_receive_doctor_response(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, receive_doctor_response_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	receive_doctor_response_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_receive_doctor_response, agent_receive_doctor_response_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_receive_doctor_response_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (receive_doctor_response)
	//Reallocate   : false
	//Input        : doctor_response
	//Output       : 
	//Agent Output : 
	GPUFLAME_receive_doctor_response<<<g, b, sm_size, stream>>>(d_agents, d_doctor_responses);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of receive_doctor_response agents in state default will be exceeded moving working agents to next state in function receive_doctor_response\n");
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
int agent_receive_attention_terminated_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_attention_terminated));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** agent_receive_attention_terminated
 * Agent function prototype for receive_attention_terminated function of agent agent
 */
void agent_receive_attention_terminated(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, receive_attention_terminated_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	receive_attention_terminated_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
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
	if (h_message_free_doctor_count + h_xmachine_memory_agent_count > xmachine_message_free_doctor_MAX){
		printf("Error: Buffer size of free_doctor message will be exceeded in function receive_attention_terminated\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_receive_attention_terminated, agent_receive_attention_terminated_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_receive_attention_terminated_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_free_doctor_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_free_doctor_output_type, &h_message_free_doctor_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_free_doctor_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_free_doctor_swaps<<<gridSize, blockSize, 0, stream>>>(d_free_doctors); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (receive_attention_terminated)
	//Reallocate   : false
	//Input        : attention_terminated
	//Output       : free_doctor
	//Agent Output : 
	GPUFLAME_receive_attention_terminated<<<g, b, sm_size, stream>>>(d_agents, d_attention_terminateds, d_free_doctors);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//free_doctor Message Type Prefix Sum
	
	//swap output
	xmachine_message_free_doctor_list* d_free_doctors_scanswap_temp = d_free_doctors;
	d_free_doctors = d_free_doctors_swap;
	d_free_doctors_swap = d_free_doctors_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_free_doctors_swap->_scan_input,
        d_free_doctors_swap->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_free_doctor_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_free_doctor_messages<<<gridSize, blockSize, 0, stream>>>(d_free_doctors, d_free_doctors_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_free_doctors_swap->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_free_doctors_swap->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_free_doctor_count += scan_last_sum+1;
	}else{
		h_message_free_doctor_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_free_doctor_count, &h_message_free_doctor_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of receive_attention_terminated agents in state default will be exceeded moving working agents to next state in function receive_attention_terminated\n");
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
int agent_output_doctor_reached_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_output_doctor_reached
 * Agent function prototype for output_doctor_reached function of agent agent
 */
void agent_output_doctor_reached(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, output_doctor_reached_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	output_doctor_reached_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
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
	if (h_message_doctor_reached_count + h_xmachine_memory_agent_count > xmachine_message_doctor_reached_MAX){
		printf("Error: Buffer size of doctor_reached message will be exceeded in function output_doctor_reached\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_doctor_reached, agent_output_doctor_reached_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_output_doctor_reached_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_doctor_reached_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_doctor_reached_output_type, &h_message_doctor_reached_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_doctor_reached_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_doctor_reached_swaps<<<gridSize, blockSize, 0, stream>>>(d_doctor_reacheds); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (output_doctor_reached)
	//Reallocate   : false
	//Input        : 
	//Output       : doctor_reached
	//Agent Output : 
	GPUFLAME_output_doctor_reached<<<g, b, sm_size, stream>>>(d_agents, d_doctor_reacheds);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//doctor_reached Message Type Prefix Sum
	
	//swap output
	xmachine_message_doctor_reached_list* d_doctor_reacheds_scanswap_temp = d_doctor_reacheds;
	d_doctor_reacheds = d_doctor_reacheds_swap;
	d_doctor_reacheds_swap = d_doctor_reacheds_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_doctor_reacheds_swap->_scan_input,
        d_doctor_reacheds_swap->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_doctor_reached_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_doctor_reached_messages<<<gridSize, blockSize, 0, stream>>>(d_doctor_reacheds, d_doctor_reacheds_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_doctor_reacheds_swap->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_doctor_reacheds_swap->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_doctor_reached_count += scan_last_sum+1;
	}else{
		h_message_doctor_reached_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_doctor_reached_count, &h_message_doctor_reached_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of output_doctor_reached agents in state default will be exceeded moving working agents to next state in function output_doctor_reached\n");
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
int agent_receive_specialist_response_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_specialist_response));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** agent_receive_specialist_response
 * Agent function prototype for receive_specialist_response function of agent agent
 */
void agent_receive_specialist_response(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, receive_specialist_response_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	receive_specialist_response_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_receive_specialist_response, agent_receive_specialist_response_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_receive_specialist_response_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (receive_specialist_response)
	//Reallocate   : false
	//Input        : specialist_response
	//Output       : 
	//Agent Output : 
	GPUFLAME_receive_specialist_response<<<g, b, sm_size, stream>>>(d_agents, d_specialist_responses);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of receive_specialist_response agents in state default will be exceeded moving working agents to next state in function receive_specialist_response\n");
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
int agent_output_specialist_petition_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_output_specialist_petition
 * Agent function prototype for output_specialist_petition function of agent agent
 */
void agent_output_specialist_petition(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, output_specialist_petition_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	output_specialist_petition_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
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
	if (h_message_specialist_petition_count + h_xmachine_memory_agent_count > xmachine_message_specialist_petition_MAX){
		printf("Error: Buffer size of specialist_petition message will be exceeded in function output_specialist_petition\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_specialist_petition, agent_output_specialist_petition_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_output_specialist_petition_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_specialist_petition_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_specialist_petition_output_type, &h_message_specialist_petition_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_specialist_petition_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_specialist_petition_swaps<<<gridSize, blockSize, 0, stream>>>(d_specialist_petitions); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (output_specialist_petition)
	//Reallocate   : false
	//Input        : 
	//Output       : specialist_petition
	//Agent Output : 
	GPUFLAME_output_specialist_petition<<<g, b, sm_size, stream>>>(d_agents, d_specialist_petitions);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//specialist_petition Message Type Prefix Sum
	
	//swap output
	xmachine_message_specialist_petition_list* d_specialist_petitions_scanswap_temp = d_specialist_petitions;
	d_specialist_petitions = d_specialist_petitions_swap;
	d_specialist_petitions_swap = d_specialist_petitions_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_specialist_petitions_swap->_scan_input,
        d_specialist_petitions_swap->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_specialist_petition_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_specialist_petition_messages<<<gridSize, blockSize, 0, stream>>>(d_specialist_petitions, d_specialist_petitions_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_specialist_petitions_swap->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_specialist_petitions_swap->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_specialist_petition_count += scan_last_sum+1;
	}else{
		h_message_specialist_petition_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_specialist_petition_count, &h_message_specialist_petition_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of output_specialist_petition agents in state default will be exceeded moving working agents to next state in function output_specialist_petition\n");
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
int agent_output_specialist_reached_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_output_specialist_reached
 * Agent function prototype for output_specialist_reached function of agent agent
 */
void agent_output_specialist_reached(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, output_specialist_reached_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	output_specialist_reached_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
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
	if (h_message_specialist_reached_count + h_xmachine_memory_agent_count > xmachine_message_specialist_reached_MAX){
		printf("Error: Buffer size of specialist_reached message will be exceeded in function output_specialist_reached\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_specialist_reached, agent_output_specialist_reached_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_output_specialist_reached_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_specialist_reached_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_specialist_reached_output_type, &h_message_specialist_reached_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_specialist_reached_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_specialist_reached_swaps<<<gridSize, blockSize, 0, stream>>>(d_specialist_reacheds); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (output_specialist_reached)
	//Reallocate   : false
	//Input        : 
	//Output       : specialist_reached
	//Agent Output : 
	GPUFLAME_output_specialist_reached<<<g, b, sm_size, stream>>>(d_agents, d_specialist_reacheds);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//specialist_reached Message Type Prefix Sum
	
	//swap output
	xmachine_message_specialist_reached_list* d_specialist_reacheds_scanswap_temp = d_specialist_reacheds;
	d_specialist_reacheds = d_specialist_reacheds_swap;
	d_specialist_reacheds_swap = d_specialist_reacheds_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_specialist_reacheds_swap->_scan_input,
        d_specialist_reacheds_swap->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_specialist_reached_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_specialist_reached_messages<<<gridSize, blockSize, 0, stream>>>(d_specialist_reacheds, d_specialist_reacheds_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_specialist_reacheds_swap->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_specialist_reacheds_swap->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_specialist_reached_count += scan_last_sum+1;
	}else{
		h_message_specialist_reached_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_specialist_reached_count, &h_message_specialist_reached_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of output_specialist_reached agents in state default will be exceeded moving working agents to next state in function output_specialist_reached\n");
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
int agent_output_triage_petition_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_output_triage_petition
 * Agent function prototype for output_triage_petition function of agent agent
 */
void agent_output_triage_petition(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, output_triage_petition_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	output_triage_petition_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
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
	if (h_message_triage_petition_count + h_xmachine_memory_agent_count > xmachine_message_triage_petition_MAX){
		printf("Error: Buffer size of triage_petition message will be exceeded in function output_triage_petition\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_triage_petition, agent_output_triage_petition_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_output_triage_petition_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_triage_petition_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_triage_petition_output_type, &h_message_triage_petition_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_triage_petition_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_triage_petition_swaps<<<gridSize, blockSize, 0, stream>>>(d_triage_petitions); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (output_triage_petition)
	//Reallocate   : false
	//Input        : 
	//Output       : triage_petition
	//Agent Output : 
	GPUFLAME_output_triage_petition<<<g, b, sm_size, stream>>>(d_agents, d_triage_petitions);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//triage_petition Message Type Prefix Sum
	
	//swap output
	xmachine_message_triage_petition_list* d_triage_petitions_scanswap_temp = d_triage_petitions;
	d_triage_petitions = d_triage_petitions_swap;
	d_triage_petitions_swap = d_triage_petitions_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent, 
        temp_scan_storage_bytes_agent, 
        d_triage_petitions_swap->_scan_input,
        d_triage_petitions_swap->_position,
        h_xmachine_memory_agent_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_triage_petition_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_triage_petition_messages<<<gridSize, blockSize, 0, stream>>>(d_triage_petitions, d_triage_petitions_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_triage_petitions_swap->_position[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_triage_petitions_swap->_scan_input[h_xmachine_memory_agent_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_triage_petition_count += scan_last_sum+1;
	}else{
		h_message_triage_petition_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_triage_petition_count, &h_message_triage_petition_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_default_count+h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
		printf("Error: Buffer size of output_triage_petition agents in state default will be exceeded moving working agents to next state in function output_triage_petition\n");
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
int agent_receive_triage_response_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_triage_response));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** agent_receive_triage_response
 * Agent function prototype for receive_triage_response function of agent agent
 */
void agent_receive_triage_response(cudaStream_t &stream){

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
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, receive_triage_response_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	receive_triage_response_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agents_default, d_agents);
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
		printf("Error: Buffer size of chair_petition message will be exceeded in function receive_triage_response\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_receive_triage_response, agent_receive_triage_response_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_receive_triage_response_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_chair_petition_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_chair_petition_output_type, &h_message_chair_petition_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_chair_petition_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_chair_petition_swaps<<<gridSize, blockSize, 0, stream>>>(d_chair_petitions); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (receive_triage_response)
	//Reallocate   : false
	//Input        : triage_response
	//Output       : chair_petition
	//Agent Output : 
	GPUFLAME_receive_triage_response<<<g, b, sm_size, stream>>>(d_agents, d_triage_responses, d_chair_petitions);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
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
		printf("Error: Buffer size of receive_triage_response agents in state default will be exceeded moving working agents to next state in function receive_triage_response\n");
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
int chair_output_chair_state_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_chair_contact));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** chair_output_chair_state
 * Agent function prototype for output_chair_state function of chair agent
 */
void chair_output_chair_state(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_chair_defaultChair_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_chair_defaultChair_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_chair_list* chairs_defaultChair_temp = d_chairs;
	d_chairs = d_chairs_defaultChair;
	d_chairs_defaultChair = chairs_defaultChair_temp;
	//set working count to current state count
	h_xmachine_memory_chair_count = h_xmachine_memory_chair_defaultChair_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_chair_count, &h_xmachine_memory_chair_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_chair_defaultChair_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_chair_defaultChair_count, &h_xmachine_memory_chair_defaultChair_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_chair_state_count + h_xmachine_memory_chair_count > xmachine_message_chair_state_MAX){
		printf("Error: Buffer size of chair_state message will be exceeded in function output_chair_state\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_output_chair_state, chair_output_chair_state_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = chair_output_chair_state_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_chair_state_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_chair_state_output_type, &h_message_chair_state_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_chair_state_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_chair_state_swaps<<<gridSize, blockSize, 0, stream>>>(d_chair_states); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (output_chair_state)
	//Reallocate   : false
	//Input        : chair_contact
	//Output       : chair_state
	//Agent Output : 
	GPUFLAME_output_chair_state<<<g, b, sm_size, stream>>>(d_chairs, d_chair_contacts, d_chair_states, d_rand48);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//chair_state Message Type Prefix Sum
	
	//swap output
	xmachine_message_chair_state_list* d_chair_states_scanswap_temp = d_chair_states;
	d_chair_states = d_chair_states_swap;
	d_chair_states_swap = d_chair_states_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_chair, 
        temp_scan_storage_bytes_chair, 
        d_chair_states_swap->_scan_input,
        d_chair_states_swap->_position,
        h_xmachine_memory_chair_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_chair_state_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_chair_state_messages<<<gridSize, blockSize, 0, stream>>>(d_chair_states, d_chair_states_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_chair_states_swap->_position[h_xmachine_memory_chair_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_chair_states_swap->_scan_input[h_xmachine_memory_chair_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_chair_state_count += scan_last_sum+1;
	}else{
		h_message_chair_state_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_chair_state_count, &h_message_chair_state_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_chair_defaultChair_count+h_xmachine_memory_chair_count > xmachine_memory_chair_MAX){
		printf("Error: Buffer size of output_chair_state agents in state defaultChair will be exceeded moving working agents to next state in function output_chair_state\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  chairs_defaultChair_temp = d_chairs;
  d_chairs = d_chairs_defaultChair;
  d_chairs_defaultChair = chairs_defaultChair_temp;
        
	//update new state agent size
	h_xmachine_memory_chair_defaultChair_count += h_xmachine_memory_chair_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_chair_defaultChair_count, &h_xmachine_memory_chair_defaultChair_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int doctor_manager_receive_doctor_petitions_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_doctor_petition));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** doctor_manager_receive_doctor_petitions
 * Agent function prototype for receive_doctor_petitions function of doctor_manager agent
 */
void doctor_manager_receive_doctor_petitions(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_doctor_manager_defaultDoctorManager_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_doctor_manager_defaultDoctorManager_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_doctor_manager_list* doctor_managers_defaultDoctorManager_temp = d_doctor_managers;
	d_doctor_managers = d_doctor_managers_defaultDoctorManager;
	d_doctor_managers_defaultDoctorManager = doctor_managers_defaultDoctorManager_temp;
	//set working count to current state count
	h_xmachine_memory_doctor_manager_count = h_xmachine_memory_doctor_manager_defaultDoctorManager_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_doctor_manager_count, &h_xmachine_memory_doctor_manager_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_doctor_manager_defaultDoctorManager_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_doctor_manager_defaultDoctorManager_count, &h_xmachine_memory_doctor_manager_defaultDoctorManager_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_doctor_response_count + h_xmachine_memory_doctor_manager_count > xmachine_message_doctor_response_MAX){
		printf("Error: Buffer size of doctor_response message will be exceeded in function receive_doctor_petitions\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_receive_doctor_petitions, doctor_manager_receive_doctor_petitions_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = doctor_manager_receive_doctor_petitions_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_doctor_response_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_doctor_response_output_type, &h_message_doctor_response_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_doctor_response_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_doctor_response_swaps<<<gridSize, blockSize, 0, stream>>>(d_doctor_responses); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (receive_doctor_petitions)
	//Reallocate   : false
	//Input        : doctor_petition
	//Output       : doctor_response
	//Agent Output : 
	GPUFLAME_receive_doctor_petitions<<<g, b, sm_size, stream>>>(d_doctor_managers, d_doctor_petitions, d_doctor_responses);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//doctor_response Message Type Prefix Sum
	
	//swap output
	xmachine_message_doctor_response_list* d_doctor_responses_scanswap_temp = d_doctor_responses;
	d_doctor_responses = d_doctor_responses_swap;
	d_doctor_responses_swap = d_doctor_responses_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_doctor_manager, 
        temp_scan_storage_bytes_doctor_manager, 
        d_doctor_responses_swap->_scan_input,
        d_doctor_responses_swap->_position,
        h_xmachine_memory_doctor_manager_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_doctor_response_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_doctor_response_messages<<<gridSize, blockSize, 0, stream>>>(d_doctor_responses, d_doctor_responses_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_doctor_responses_swap->_position[h_xmachine_memory_doctor_manager_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_doctor_responses_swap->_scan_input[h_xmachine_memory_doctor_manager_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_doctor_response_count += scan_last_sum+1;
	}else{
		h_message_doctor_response_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_doctor_response_count, &h_message_doctor_response_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_doctor_manager_defaultDoctorManager_count+h_xmachine_memory_doctor_manager_count > xmachine_memory_doctor_manager_MAX){
		printf("Error: Buffer size of receive_doctor_petitions agents in state defaultDoctorManager will be exceeded moving working agents to next state in function receive_doctor_petitions\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  doctor_managers_defaultDoctorManager_temp = d_doctor_managers;
  d_doctor_managers = d_doctor_managers_defaultDoctorManager;
  d_doctor_managers_defaultDoctorManager = doctor_managers_defaultDoctorManager_temp;
        
	//update new state agent size
	h_xmachine_memory_doctor_manager_defaultDoctorManager_count += h_xmachine_memory_doctor_manager_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_doctor_manager_defaultDoctorManager_count, &h_xmachine_memory_doctor_manager_defaultDoctorManager_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int doctor_manager_receive_free_doctors_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_free_doctor));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** doctor_manager_receive_free_doctors
 * Agent function prototype for receive_free_doctors function of doctor_manager agent
 */
void doctor_manager_receive_free_doctors(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_doctor_manager_defaultDoctorManager_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_doctor_manager_defaultDoctorManager_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_doctor_manager_list* doctor_managers_defaultDoctorManager_temp = d_doctor_managers;
	d_doctor_managers = d_doctor_managers_defaultDoctorManager;
	d_doctor_managers_defaultDoctorManager = doctor_managers_defaultDoctorManager_temp;
	//set working count to current state count
	h_xmachine_memory_doctor_manager_count = h_xmachine_memory_doctor_manager_defaultDoctorManager_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_doctor_manager_count, &h_xmachine_memory_doctor_manager_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_doctor_manager_defaultDoctorManager_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_doctor_manager_defaultDoctorManager_count, &h_xmachine_memory_doctor_manager_defaultDoctorManager_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_receive_free_doctors, doctor_manager_receive_free_doctors_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = doctor_manager_receive_free_doctors_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (receive_free_doctors)
	//Reallocate   : false
	//Input        : free_doctor
	//Output       : 
	//Agent Output : 
	GPUFLAME_receive_free_doctors<<<g, b, sm_size, stream>>>(d_doctor_managers, d_free_doctors);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_doctor_manager_defaultDoctorManager_count+h_xmachine_memory_doctor_manager_count > xmachine_memory_doctor_manager_MAX){
		printf("Error: Buffer size of receive_free_doctors agents in state defaultDoctorManager will be exceeded moving working agents to next state in function receive_free_doctors\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  doctor_managers_defaultDoctorManager_temp = d_doctor_managers;
  d_doctor_managers = d_doctor_managers_defaultDoctorManager;
  d_doctor_managers_defaultDoctorManager = doctor_managers_defaultDoctorManager_temp;
        
	//update new state agent size
	h_xmachine_memory_doctor_manager_defaultDoctorManager_count += h_xmachine_memory_doctor_manager_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_doctor_manager_defaultDoctorManager_count, &h_xmachine_memory_doctor_manager_defaultDoctorManager_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int specialist_manager_receive_specialist_petitions_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_specialist_petition));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** specialist_manager_receive_specialist_petitions
 * Agent function prototype for receive_specialist_petitions function of specialist_manager agent
 */
void specialist_manager_receive_specialist_petitions(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_specialist_manager_defaultSpecialistManager_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_specialist_manager_defaultSpecialistManager_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_specialist_manager_list* specialist_managers_defaultSpecialistManager_temp = d_specialist_managers;
	d_specialist_managers = d_specialist_managers_defaultSpecialistManager;
	d_specialist_managers_defaultSpecialistManager = specialist_managers_defaultSpecialistManager_temp;
	//set working count to current state count
	h_xmachine_memory_specialist_manager_count = h_xmachine_memory_specialist_manager_defaultSpecialistManager_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_specialist_manager_count, &h_xmachine_memory_specialist_manager_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_specialist_manager_defaultSpecialistManager_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_specialist_manager_defaultSpecialistManager_count, &h_xmachine_memory_specialist_manager_defaultSpecialistManager_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_specialist_response_count + h_xmachine_memory_specialist_manager_count > xmachine_message_specialist_response_MAX){
		printf("Error: Buffer size of specialist_response message will be exceeded in function receive_specialist_petitions\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_receive_specialist_petitions, specialist_manager_receive_specialist_petitions_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = specialist_manager_receive_specialist_petitions_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_specialist_response_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_specialist_response_output_type, &h_message_specialist_response_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_specialist_response_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_specialist_response_swaps<<<gridSize, blockSize, 0, stream>>>(d_specialist_responses); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (receive_specialist_petitions)
	//Reallocate   : false
	//Input        : specialist_petition
	//Output       : specialist_response
	//Agent Output : 
	GPUFLAME_receive_specialist_petitions<<<g, b, sm_size, stream>>>(d_specialist_managers, d_specialist_petitions, d_specialist_responses);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//specialist_response Message Type Prefix Sum
	
	//swap output
	xmachine_message_specialist_response_list* d_specialist_responses_scanswap_temp = d_specialist_responses;
	d_specialist_responses = d_specialist_responses_swap;
	d_specialist_responses_swap = d_specialist_responses_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_specialist_manager, 
        temp_scan_storage_bytes_specialist_manager, 
        d_specialist_responses_swap->_scan_input,
        d_specialist_responses_swap->_position,
        h_xmachine_memory_specialist_manager_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_specialist_response_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_specialist_response_messages<<<gridSize, blockSize, 0, stream>>>(d_specialist_responses, d_specialist_responses_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_specialist_responses_swap->_position[h_xmachine_memory_specialist_manager_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_specialist_responses_swap->_scan_input[h_xmachine_memory_specialist_manager_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_specialist_response_count += scan_last_sum+1;
	}else{
		h_message_specialist_response_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_specialist_response_count, &h_message_specialist_response_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_specialist_manager_defaultSpecialistManager_count+h_xmachine_memory_specialist_manager_count > xmachine_memory_specialist_manager_MAX){
		printf("Error: Buffer size of receive_specialist_petitions agents in state defaultSpecialistManager will be exceeded moving working agents to next state in function receive_specialist_petitions\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  specialist_managers_defaultSpecialistManager_temp = d_specialist_managers;
  d_specialist_managers = d_specialist_managers_defaultSpecialistManager;
  d_specialist_managers_defaultSpecialistManager = specialist_managers_defaultSpecialistManager_temp;
        
	//update new state agent size
	h_xmachine_memory_specialist_manager_defaultSpecialistManager_count += h_xmachine_memory_specialist_manager_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_specialist_manager_defaultSpecialistManager_count, &h_xmachine_memory_specialist_manager_defaultSpecialistManager_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int specialist_receive_specialist_reached_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_specialist_reached));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** specialist_receive_specialist_reached
 * Agent function prototype for receive_specialist_reached function of specialist agent
 */
void specialist_receive_specialist_reached(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_specialist_defaultSpecialist_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_specialist_defaultSpecialist_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_specialist_list* specialists_defaultSpecialist_temp = d_specialists;
	d_specialists = d_specialists_defaultSpecialist;
	d_specialists_defaultSpecialist = specialists_defaultSpecialist_temp;
	//set working count to current state count
	h_xmachine_memory_specialist_count = h_xmachine_memory_specialist_defaultSpecialist_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_specialist_count, &h_xmachine_memory_specialist_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_specialist_defaultSpecialist_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_specialist_defaultSpecialist_count, &h_xmachine_memory_specialist_defaultSpecialist_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_attention_terminated_count + h_xmachine_memory_specialist_count > xmachine_message_attention_terminated_MAX){
		printf("Error: Buffer size of attention_terminated message will be exceeded in function receive_specialist_reached\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_receive_specialist_reached, specialist_receive_specialist_reached_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = specialist_receive_specialist_reached_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_attention_terminated_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_attention_terminated_output_type, &h_message_attention_terminated_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_attention_terminated_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_attention_terminated_swaps<<<gridSize, blockSize, 0, stream>>>(d_attention_terminateds); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (receive_specialist_reached)
	//Reallocate   : false
	//Input        : specialist_reached
	//Output       : attention_terminated
	//Agent Output : 
	GPUFLAME_receive_specialist_reached<<<g, b, sm_size, stream>>>(d_specialists, d_specialist_reacheds, d_attention_terminateds);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//attention_terminated Message Type Prefix Sum
	
	//swap output
	xmachine_message_attention_terminated_list* d_attention_terminateds_scanswap_temp = d_attention_terminateds;
	d_attention_terminateds = d_attention_terminateds_swap;
	d_attention_terminateds_swap = d_attention_terminateds_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_specialist, 
        temp_scan_storage_bytes_specialist, 
        d_attention_terminateds_swap->_scan_input,
        d_attention_terminateds_swap->_position,
        h_xmachine_memory_specialist_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_attention_terminated_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_attention_terminated_messages<<<gridSize, blockSize, 0, stream>>>(d_attention_terminateds, d_attention_terminateds_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_attention_terminateds_swap->_position[h_xmachine_memory_specialist_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_attention_terminateds_swap->_scan_input[h_xmachine_memory_specialist_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_attention_terminated_count += scan_last_sum+1;
	}else{
		h_message_attention_terminated_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_attention_terminated_count, &h_message_attention_terminated_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_specialist_defaultSpecialist_count+h_xmachine_memory_specialist_count > xmachine_memory_specialist_MAX){
		printf("Error: Buffer size of receive_specialist_reached agents in state defaultSpecialist will be exceeded moving working agents to next state in function receive_specialist_reached\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  specialists_defaultSpecialist_temp = d_specialists;
  d_specialists = d_specialists_defaultSpecialist;
  d_specialists_defaultSpecialist = specialists_defaultSpecialist_temp;
        
	//update new state agent size
	h_xmachine_memory_specialist_defaultSpecialist_count += h_xmachine_memory_specialist_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_specialist_defaultSpecialist_count, &h_xmachine_memory_specialist_defaultSpecialist_count, sizeof(int)));	
	
	
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
	if (h_message_check_in_response_count + h_xmachine_memory_receptionist_count > xmachine_message_check_in_response_MAX){
		printf("Error: Buffer size of check_in_response message will be exceeded in function receptionServer\n");
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
	h_message_check_in_response_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_check_in_response_output_type, &h_message_check_in_response_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_check_in_response_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_check_in_response_swaps<<<gridSize, blockSize, 0, stream>>>(d_check_in_responses); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (receptionServer)
	//Reallocate   : false
	//Input        : check_in
	//Output       : check_in_response
	//Agent Output : 
	GPUFLAME_receptionServer<<<g, b, sm_size, stream>>>(d_receptionists, d_check_ins, d_check_in_responses);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//check_in_response Message Type Prefix Sum
	
	//swap output
	xmachine_message_check_in_response_list* d_check_in_responses_scanswap_temp = d_check_in_responses;
	d_check_in_responses = d_check_in_responses_swap;
	d_check_in_responses_swap = d_check_in_responses_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_receptionist, 
        temp_scan_storage_bytes_receptionist, 
        d_check_in_responses_swap->_scan_input,
        d_check_in_responses_swap->_position,
        h_xmachine_memory_receptionist_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_check_in_response_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_check_in_response_messages<<<gridSize, blockSize, 0, stream>>>(d_check_in_responses, d_check_in_responses_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_check_in_responses_swap->_position[h_xmachine_memory_receptionist_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_check_in_responses_swap->_scan_input[h_xmachine_memory_receptionist_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_check_in_response_count += scan_last_sum+1;
	}else{
		h_message_check_in_response_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_check_in_response_count, &h_message_check_in_response_count, sizeof(int)));	
	
	
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
int agent_generator_generate_chairs_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_generator_generate_chairs
 * Agent function prototype for generate_chairs function of agent_generator agent
 */
void agent_generator_generate_chairs(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_agent_generator_defaultGenerator_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_generator_defaultGenerator_count;

	
	//FOR chair AGENT OUTPUT, RESET THE AGENT NEW LIST SCAN INPUT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_chair_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_chair_scan_input<<<gridSize, blockSize, 0, stream>>>(d_chairs_new);
	gpuErrchkLaunch();
	

	//******************************** AGENT FUNCTION CONDITION *********************
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_agent_generator_count = h_xmachine_memory_agent_generator_defaultGenerator_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_generator_count, &h_xmachine_memory_agent_generator_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_agent_generator_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_agent_generator_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_defaultGenerator);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_agent_generator_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agent_generators);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_chairs_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	generate_chairs_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_defaultGenerator, d_agent_generators);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_agent_generator_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent_generator, 
        temp_scan_storage_bytes_agent_generator, 
        d_agent_generators_defaultGenerator->_scan_input,
        d_agent_generators_defaultGenerator->_position,
        h_xmachine_memory_agent_generator_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agent_generators_defaultGenerator->_position[h_xmachine_memory_agent_generator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agent_generators_defaultGenerator->_scan_input[h_xmachine_memory_agent_generator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_agent_generator_defaultGenerator_count = scan_last_sum+1;
	else		
		h_xmachine_memory_agent_generator_defaultGenerator_count = scan_last_sum;
	//Scatter into swap
	scatter_agent_generator_Agents<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_swap, d_agent_generators_defaultGenerator, 0, h_xmachine_memory_agent_generator_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_agent_generator_list* agent_generators_defaultGenerator_temp = d_agent_generators_defaultGenerator;
	d_agent_generators_defaultGenerator = d_agent_generators_swap;
	d_agent_generators_swap = agent_generators_defaultGenerator_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_generator_defaultGenerator_count, &h_xmachine_memory_agent_generator_defaultGenerator_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent_generator, 
        temp_scan_storage_bytes_agent_generator, 
        d_agent_generators->_scan_input,
        d_agent_generators->_position,
        h_xmachine_memory_agent_generator_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agent_generators->_position[h_xmachine_memory_agent_generator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agent_generators->_scan_input[h_xmachine_memory_agent_generator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_agent_generator_Agents<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_swap, d_agent_generators, 0, h_xmachine_memory_agent_generator_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_agent_generator_count = scan_last_sum+1;
	else		
		h_xmachine_memory_agent_generator_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_agent_generator_list* agent_generators_temp = d_agent_generators;
	d_agent_generators = d_agent_generators_swap;
	d_agent_generators_swap = agent_generators_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_generator_count, &h_xmachine_memory_agent_generator_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_agent_generator_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_agent_generator_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_generate_chairs, agent_generator_generate_chairs_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_generator_generate_chairs_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (generate_chairs)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : chair
	GPUFLAME_generate_chairs<<<g, b, sm_size, stream>>>(d_agent_generators, d_chairs_new);
	gpuErrchkLaunch();
	
	
    //COPY ANY AGENT COUNT BEFORE agent_generator AGENTS ARE KILLED (needed for scatter)
	int agent_generators_pre_death_count = h_xmachine_memory_agent_generator_count;
	
	//FOR chair AGENT OUTPUT SCATTER AGENTS 

    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_chair, 
        temp_scan_storage_bytes_chair, 
        d_chairs_new->_scan_input, 
        d_chairs_new->_position, 
        agent_generators_pre_death_count,
        stream
    );

	//reset agent count
	int chair_after_birth_count;
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_chairs_new->_position[agent_generators_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_chairs_new->_scan_input[agent_generators_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		chair_after_birth_count = h_xmachine_memory_chair_defaultChair_count + scan_last_sum+1;
	else
		chair_after_birth_count = h_xmachine_memory_chair_defaultChair_count + scan_last_sum;
	//check buffer is not exceeded
	if (chair_after_birth_count > xmachine_memory_chair_MAX){
		printf("Error: Buffer size of chair agents in state defaultChair will be exceeded writing new agents in function generate_chairs\n");
		exit(EXIT_FAILURE);
	}
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_chair_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_chair_Agents<<<gridSize, blockSize, 0, stream>>>(d_chairs_defaultChair, d_chairs_new, h_xmachine_memory_chair_defaultChair_count, agent_generators_pre_death_count);
	gpuErrchkLaunch();
	//Copy count to device
	h_xmachine_memory_chair_defaultChair_count = chair_after_birth_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_chair_defaultChair_count, &h_xmachine_memory_chair_defaultChair_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_generator_defaultGenerator_count+h_xmachine_memory_agent_generator_count > xmachine_memory_agent_generator_MAX){
		printf("Error: Buffer size of generate_chairs agents in state defaultGenerator will be exceeded moving working agents to next state in function generate_chairs\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_agent_generator_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_agent_generator_Agents<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_defaultGenerator, d_agent_generators, h_xmachine_memory_agent_generator_defaultGenerator_count, h_xmachine_memory_agent_generator_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_agent_generator_defaultGenerator_count += h_xmachine_memory_agent_generator_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_generator_defaultGenerator_count, &h_xmachine_memory_agent_generator_defaultGenerator_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_generator_generate_boxes_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_generator_generate_boxes
 * Agent function prototype for generate_boxes function of agent_generator agent
 */
void agent_generator_generate_boxes(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_agent_generator_defaultGenerator_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_generator_defaultGenerator_count;

	
	//FOR box AGENT OUTPUT, RESET THE AGENT NEW LIST SCAN INPUT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_box_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_box_scan_input<<<gridSize, blockSize, 0, stream>>>(d_boxs_new);
	gpuErrchkLaunch();
	

	//******************************** AGENT FUNCTION CONDITION *********************
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_agent_generator_count = h_xmachine_memory_agent_generator_defaultGenerator_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_generator_count, &h_xmachine_memory_agent_generator_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_agent_generator_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_agent_generator_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_defaultGenerator);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_agent_generator_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agent_generators);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_boxes_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	generate_boxes_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_defaultGenerator, d_agent_generators);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_agent_generator_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent_generator, 
        temp_scan_storage_bytes_agent_generator, 
        d_agent_generators_defaultGenerator->_scan_input,
        d_agent_generators_defaultGenerator->_position,
        h_xmachine_memory_agent_generator_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agent_generators_defaultGenerator->_position[h_xmachine_memory_agent_generator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agent_generators_defaultGenerator->_scan_input[h_xmachine_memory_agent_generator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_agent_generator_defaultGenerator_count = scan_last_sum+1;
	else		
		h_xmachine_memory_agent_generator_defaultGenerator_count = scan_last_sum;
	//Scatter into swap
	scatter_agent_generator_Agents<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_swap, d_agent_generators_defaultGenerator, 0, h_xmachine_memory_agent_generator_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_agent_generator_list* agent_generators_defaultGenerator_temp = d_agent_generators_defaultGenerator;
	d_agent_generators_defaultGenerator = d_agent_generators_swap;
	d_agent_generators_swap = agent_generators_defaultGenerator_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_generator_defaultGenerator_count, &h_xmachine_memory_agent_generator_defaultGenerator_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent_generator, 
        temp_scan_storage_bytes_agent_generator, 
        d_agent_generators->_scan_input,
        d_agent_generators->_position,
        h_xmachine_memory_agent_generator_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agent_generators->_position[h_xmachine_memory_agent_generator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agent_generators->_scan_input[h_xmachine_memory_agent_generator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_agent_generator_Agents<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_swap, d_agent_generators, 0, h_xmachine_memory_agent_generator_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_agent_generator_count = scan_last_sum+1;
	else		
		h_xmachine_memory_agent_generator_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_agent_generator_list* agent_generators_temp = d_agent_generators;
	d_agent_generators = d_agent_generators_swap;
	d_agent_generators_swap = agent_generators_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_generator_count, &h_xmachine_memory_agent_generator_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_agent_generator_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_agent_generator_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_generate_boxes, agent_generator_generate_boxes_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_generator_generate_boxes_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (generate_boxes)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : box
	GPUFLAME_generate_boxes<<<g, b, sm_size, stream>>>(d_agent_generators, d_boxs_new);
	gpuErrchkLaunch();
	
	
    //COPY ANY AGENT COUNT BEFORE agent_generator AGENTS ARE KILLED (needed for scatter)
	int agent_generators_pre_death_count = h_xmachine_memory_agent_generator_count;
	
	//FOR box AGENT OUTPUT SCATTER AGENTS 

    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_box, 
        temp_scan_storage_bytes_box, 
        d_boxs_new->_scan_input, 
        d_boxs_new->_position, 
        agent_generators_pre_death_count,
        stream
    );

	//reset agent count
	int box_after_birth_count;
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_boxs_new->_position[agent_generators_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_boxs_new->_scan_input[agent_generators_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		box_after_birth_count = h_xmachine_memory_box_defaultBox_count + scan_last_sum+1;
	else
		box_after_birth_count = h_xmachine_memory_box_defaultBox_count + scan_last_sum;
	//check buffer is not exceeded
	if (box_after_birth_count > xmachine_memory_box_MAX){
		printf("Error: Buffer size of box agents in state defaultBox will be exceeded writing new agents in function generate_boxes\n");
		exit(EXIT_FAILURE);
	}
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_box_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_box_Agents<<<gridSize, blockSize, 0, stream>>>(d_boxs_defaultBox, d_boxs_new, h_xmachine_memory_box_defaultBox_count, agent_generators_pre_death_count);
	gpuErrchkLaunch();
	//Copy count to device
	h_xmachine_memory_box_defaultBox_count = box_after_birth_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_box_defaultBox_count, &h_xmachine_memory_box_defaultBox_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_generator_defaultGenerator_count+h_xmachine_memory_agent_generator_count > xmachine_memory_agent_generator_MAX){
		printf("Error: Buffer size of generate_boxes agents in state defaultGenerator will be exceeded moving working agents to next state in function generate_boxes\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_agent_generator_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_agent_generator_Agents<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_defaultGenerator, d_agent_generators, h_xmachine_memory_agent_generator_defaultGenerator_count, h_xmachine_memory_agent_generator_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_agent_generator_defaultGenerator_count += h_xmachine_memory_agent_generator_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_generator_defaultGenerator_count, &h_xmachine_memory_agent_generator_defaultGenerator_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_generator_generate_doctors_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_generator_generate_doctors
 * Agent function prototype for generate_doctors function of agent_generator agent
 */
void agent_generator_generate_doctors(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_agent_generator_defaultGenerator_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_generator_defaultGenerator_count;

	
	//FOR doctor AGENT OUTPUT, RESET THE AGENT NEW LIST SCAN INPUT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_doctor_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_doctor_scan_input<<<gridSize, blockSize, 0, stream>>>(d_doctors_new);
	gpuErrchkLaunch();
	

	//******************************** AGENT FUNCTION CONDITION *********************
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_agent_generator_count = h_xmachine_memory_agent_generator_defaultGenerator_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_generator_count, &h_xmachine_memory_agent_generator_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_agent_generator_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_agent_generator_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_defaultGenerator);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_agent_generator_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agent_generators);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_doctors_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	generate_doctors_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_defaultGenerator, d_agent_generators);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_agent_generator_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent_generator, 
        temp_scan_storage_bytes_agent_generator, 
        d_agent_generators_defaultGenerator->_scan_input,
        d_agent_generators_defaultGenerator->_position,
        h_xmachine_memory_agent_generator_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agent_generators_defaultGenerator->_position[h_xmachine_memory_agent_generator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agent_generators_defaultGenerator->_scan_input[h_xmachine_memory_agent_generator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_agent_generator_defaultGenerator_count = scan_last_sum+1;
	else		
		h_xmachine_memory_agent_generator_defaultGenerator_count = scan_last_sum;
	//Scatter into swap
	scatter_agent_generator_Agents<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_swap, d_agent_generators_defaultGenerator, 0, h_xmachine_memory_agent_generator_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_agent_generator_list* agent_generators_defaultGenerator_temp = d_agent_generators_defaultGenerator;
	d_agent_generators_defaultGenerator = d_agent_generators_swap;
	d_agent_generators_swap = agent_generators_defaultGenerator_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_generator_defaultGenerator_count, &h_xmachine_memory_agent_generator_defaultGenerator_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent_generator, 
        temp_scan_storage_bytes_agent_generator, 
        d_agent_generators->_scan_input,
        d_agent_generators->_position,
        h_xmachine_memory_agent_generator_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agent_generators->_position[h_xmachine_memory_agent_generator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agent_generators->_scan_input[h_xmachine_memory_agent_generator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_agent_generator_Agents<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_swap, d_agent_generators, 0, h_xmachine_memory_agent_generator_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_agent_generator_count = scan_last_sum+1;
	else		
		h_xmachine_memory_agent_generator_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_agent_generator_list* agent_generators_temp = d_agent_generators;
	d_agent_generators = d_agent_generators_swap;
	d_agent_generators_swap = agent_generators_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_generator_count, &h_xmachine_memory_agent_generator_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_agent_generator_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_agent_generator_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_generate_doctors, agent_generator_generate_doctors_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_generator_generate_doctors_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (generate_doctors)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : doctor
	GPUFLAME_generate_doctors<<<g, b, sm_size, stream>>>(d_agent_generators, d_doctors_new);
	gpuErrchkLaunch();
	
	
    //COPY ANY AGENT COUNT BEFORE agent_generator AGENTS ARE KILLED (needed for scatter)
	int agent_generators_pre_death_count = h_xmachine_memory_agent_generator_count;
	
	//FOR doctor AGENT OUTPUT SCATTER AGENTS 

    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_doctor, 
        temp_scan_storage_bytes_doctor, 
        d_doctors_new->_scan_input, 
        d_doctors_new->_position, 
        agent_generators_pre_death_count,
        stream
    );

	//reset agent count
	int doctor_after_birth_count;
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_doctors_new->_position[agent_generators_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_doctors_new->_scan_input[agent_generators_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		doctor_after_birth_count = h_xmachine_memory_doctor_defaultDoctor_count + scan_last_sum+1;
	else
		doctor_after_birth_count = h_xmachine_memory_doctor_defaultDoctor_count + scan_last_sum;
	//check buffer is not exceeded
	if (doctor_after_birth_count > xmachine_memory_doctor_MAX){
		printf("Error: Buffer size of doctor agents in state defaultDoctor will be exceeded writing new agents in function generate_doctors\n");
		exit(EXIT_FAILURE);
	}
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_doctor_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_doctor_Agents<<<gridSize, blockSize, 0, stream>>>(d_doctors_defaultDoctor, d_doctors_new, h_xmachine_memory_doctor_defaultDoctor_count, agent_generators_pre_death_count);
	gpuErrchkLaunch();
	//Copy count to device
	h_xmachine_memory_doctor_defaultDoctor_count = doctor_after_birth_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_doctor_defaultDoctor_count, &h_xmachine_memory_doctor_defaultDoctor_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_generator_defaultGenerator_count+h_xmachine_memory_agent_generator_count > xmachine_memory_agent_generator_MAX){
		printf("Error: Buffer size of generate_doctors agents in state defaultGenerator will be exceeded moving working agents to next state in function generate_doctors\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_agent_generator_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_agent_generator_Agents<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_defaultGenerator, d_agent_generators, h_xmachine_memory_agent_generator_defaultGenerator_count, h_xmachine_memory_agent_generator_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_agent_generator_defaultGenerator_count += h_xmachine_memory_agent_generator_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_generator_defaultGenerator_count, &h_xmachine_memory_agent_generator_defaultGenerator_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int agent_generator_generate_specialists_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** agent_generator_generate_specialists
 * Agent function prototype for generate_specialists function of agent_generator agent
 */
void agent_generator_generate_specialists(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_agent_generator_defaultGenerator_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_agent_generator_defaultGenerator_count;

	
	//FOR specialist AGENT OUTPUT, RESET THE AGENT NEW LIST SCAN INPUT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_specialist_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_specialist_scan_input<<<gridSize, blockSize, 0, stream>>>(d_specialists_new);
	gpuErrchkLaunch();
	

	//******************************** AGENT FUNCTION CONDITION *********************
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_agent_generator_count = h_xmachine_memory_agent_generator_defaultGenerator_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_generator_count, &h_xmachine_memory_agent_generator_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_agent_generator_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_agent_generator_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_defaultGenerator);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_agent_generator_scan_input<<<gridSize, blockSize, 0, stream>>>(d_agent_generators);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, generate_specialists_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	generate_specialists_function_filter<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_defaultGenerator, d_agent_generators);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_agent_generator_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent_generator, 
        temp_scan_storage_bytes_agent_generator, 
        d_agent_generators_defaultGenerator->_scan_input,
        d_agent_generators_defaultGenerator->_position,
        h_xmachine_memory_agent_generator_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agent_generators_defaultGenerator->_position[h_xmachine_memory_agent_generator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agent_generators_defaultGenerator->_scan_input[h_xmachine_memory_agent_generator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_agent_generator_defaultGenerator_count = scan_last_sum+1;
	else		
		h_xmachine_memory_agent_generator_defaultGenerator_count = scan_last_sum;
	//Scatter into swap
	scatter_agent_generator_Agents<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_swap, d_agent_generators_defaultGenerator, 0, h_xmachine_memory_agent_generator_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_agent_generator_list* agent_generators_defaultGenerator_temp = d_agent_generators_defaultGenerator;
	d_agent_generators_defaultGenerator = d_agent_generators_swap;
	d_agent_generators_swap = agent_generators_defaultGenerator_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_generator_defaultGenerator_count, &h_xmachine_memory_agent_generator_defaultGenerator_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_agent_generator, 
        temp_scan_storage_bytes_agent_generator, 
        d_agent_generators->_scan_input,
        d_agent_generators->_position,
        h_xmachine_memory_agent_generator_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_agent_generators->_position[h_xmachine_memory_agent_generator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_agent_generators->_scan_input[h_xmachine_memory_agent_generator_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_agent_generator_Agents<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_swap, d_agent_generators, 0, h_xmachine_memory_agent_generator_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_agent_generator_count = scan_last_sum+1;
	else		
		h_xmachine_memory_agent_generator_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_agent_generator_list* agent_generators_temp = d_agent_generators;
	d_agent_generators = d_agent_generators_swap;
	d_agent_generators_swap = agent_generators_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_generator_count, &h_xmachine_memory_agent_generator_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_agent_generator_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_agent_generator_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_generate_specialists, agent_generator_generate_specialists_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = agent_generator_generate_specialists_sm_size(blockSize);
	
	
	
	
	//MAIN XMACHINE FUNCTION CALL (generate_specialists)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : specialist
	GPUFLAME_generate_specialists<<<g, b, sm_size, stream>>>(d_agent_generators, d_specialists_new);
	gpuErrchkLaunch();
	
	
    //COPY ANY AGENT COUNT BEFORE agent_generator AGENTS ARE KILLED (needed for scatter)
	int agent_generators_pre_death_count = h_xmachine_memory_agent_generator_count;
	
	//FOR specialist AGENT OUTPUT SCATTER AGENTS 

    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_specialist, 
        temp_scan_storage_bytes_specialist, 
        d_specialists_new->_scan_input, 
        d_specialists_new->_position, 
        agent_generators_pre_death_count,
        stream
    );

	//reset agent count
	int specialist_after_birth_count;
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_specialists_new->_position[agent_generators_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_specialists_new->_scan_input[agent_generators_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		specialist_after_birth_count = h_xmachine_memory_specialist_defaultSpecialist_count + scan_last_sum+1;
	else
		specialist_after_birth_count = h_xmachine_memory_specialist_defaultSpecialist_count + scan_last_sum;
	//check buffer is not exceeded
	if (specialist_after_birth_count > xmachine_memory_specialist_MAX){
		printf("Error: Buffer size of specialist agents in state defaultSpecialist will be exceeded writing new agents in function generate_specialists\n");
		exit(EXIT_FAILURE);
	}
	//Scatter into swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_specialist_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_specialist_Agents<<<gridSize, blockSize, 0, stream>>>(d_specialists_defaultSpecialist, d_specialists_new, h_xmachine_memory_specialist_defaultSpecialist_count, agent_generators_pre_death_count);
	gpuErrchkLaunch();
	//Copy count to device
	h_xmachine_memory_specialist_defaultSpecialist_count = specialist_after_birth_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_specialist_defaultSpecialist_count, &h_xmachine_memory_specialist_defaultSpecialist_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_agent_generator_defaultGenerator_count+h_xmachine_memory_agent_generator_count > xmachine_memory_agent_generator_MAX){
		printf("Error: Buffer size of generate_specialists agents in state defaultGenerator will be exceeded moving working agents to next state in function generate_specialists\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_agent_generator_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_agent_generator_Agents<<<gridSize, blockSize, 0, stream>>>(d_agent_generators_defaultGenerator, d_agent_generators, h_xmachine_memory_agent_generator_defaultGenerator_count, h_xmachine_memory_agent_generator_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_agent_generator_defaultGenerator_count += h_xmachine_memory_agent_generator_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_agent_generator_defaultGenerator_count, &h_xmachine_memory_agent_generator_defaultGenerator_count, sizeof(int)));	
	
	
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
int chair_admin_receive_free_chair_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_free_chair));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** chair_admin_receive_free_chair
 * Agent function prototype for receive_free_chair function of chair_admin agent
 */
void chair_admin_receive_free_chair(cudaStream_t &stream){

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

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_receive_free_chair, chair_admin_receive_free_chair_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = chair_admin_receive_free_chair_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (receive_free_chair)
	//Reallocate   : false
	//Input        : free_chair
	//Output       : 
	//Agent Output : 
	GPUFLAME_receive_free_chair<<<g, b, sm_size, stream>>>(d_chair_admins, d_free_chairs);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_chair_admin_defaultAdmin_count+h_xmachine_memory_chair_admin_count > xmachine_memory_chair_admin_MAX){
		printf("Error: Buffer size of receive_free_chair agents in state defaultAdmin will be exceeded moving working agents to next state in function receive_free_chair\n");
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
int box_box_server_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_box_petition));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** box_box_server
 * Agent function prototype for box_server function of box agent
 */
void box_box_server(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_box_defaultBox_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_box_defaultBox_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_box_count = h_xmachine_memory_box_defaultBox_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_box_count, &h_xmachine_memory_box_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_box_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_box_scan_input<<<gridSize, blockSize, 0, stream>>>(d_boxs_defaultBox);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_box_scan_input<<<gridSize, blockSize, 0, stream>>>(d_boxs);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, box_server_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	box_server_function_filter<<<gridSize, blockSize, 0, stream>>>(d_boxs_defaultBox, d_boxs);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_box_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_box, 
        temp_scan_storage_bytes_box, 
        d_boxs_defaultBox->_scan_input,
        d_boxs_defaultBox->_position,
        h_xmachine_memory_box_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_boxs_defaultBox->_position[h_xmachine_memory_box_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_boxs_defaultBox->_scan_input[h_xmachine_memory_box_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_box_defaultBox_count = scan_last_sum+1;
	else		
		h_xmachine_memory_box_defaultBox_count = scan_last_sum;
	//Scatter into swap
	scatter_box_Agents<<<gridSize, blockSize, 0, stream>>>(d_boxs_swap, d_boxs_defaultBox, 0, h_xmachine_memory_box_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_box_list* boxs_defaultBox_temp = d_boxs_defaultBox;
	d_boxs_defaultBox = d_boxs_swap;
	d_boxs_swap = boxs_defaultBox_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_box_defaultBox_count, &h_xmachine_memory_box_defaultBox_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_box, 
        temp_scan_storage_bytes_box, 
        d_boxs->_scan_input,
        d_boxs->_position,
        h_xmachine_memory_box_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_boxs->_position[h_xmachine_memory_box_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_boxs->_scan_input[h_xmachine_memory_box_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_box_Agents<<<gridSize, blockSize, 0, stream>>>(d_boxs_swap, d_boxs, 0, h_xmachine_memory_box_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_box_count = scan_last_sum+1;
	else		
		h_xmachine_memory_box_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_box_list* boxs_temp = d_boxs;
	d_boxs = d_boxs_swap;
	d_boxs_swap = boxs_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_box_count, &h_xmachine_memory_box_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_box_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_box_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_box_server, box_box_server_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = box_box_server_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	
	//MAIN XMACHINE FUNCTION CALL (box_server)
	//Reallocate   : false
	//Input        : box_petition
	//Output       : 
	//Agent Output : 
	GPUFLAME_box_server<<<g, b, sm_size, stream>>>(d_boxs, d_box_petitions);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_box_defaultBox_count+h_xmachine_memory_box_count > xmachine_memory_box_MAX){
		printf("Error: Buffer size of box_server agents in state defaultBox will be exceeded moving working agents to next state in function box_server\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_box_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_box_Agents<<<gridSize, blockSize, 0, stream>>>(d_boxs_defaultBox, d_boxs, h_xmachine_memory_box_defaultBox_count, h_xmachine_memory_box_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_box_defaultBox_count += h_xmachine_memory_box_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_box_defaultBox_count, &h_xmachine_memory_box_defaultBox_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int box_attend_box_patient_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  
	return sm_size;
}

/** box_attend_box_patient
 * Agent function prototype for attend_box_patient function of box agent
 */
void box_attend_box_patient(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_box_defaultBox_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_box_defaultBox_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_box_count = h_xmachine_memory_box_defaultBox_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_box_count, &h_xmachine_memory_box_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_box_scan_input, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_box_scan_input<<<gridSize, blockSize, 0, stream>>>(d_boxs_defaultBox);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_box_scan_input<<<gridSize, blockSize, 0, stream>>>(d_boxs);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, attend_box_patient_function_filter, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	attend_box_patient_function_filter<<<gridSize, blockSize, 0, stream>>>(d_boxs_defaultBox, d_boxs);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_box_Agents, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	
	//COMPACT CURRENT STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_box, 
        temp_scan_storage_bytes_box, 
        d_boxs_defaultBox->_scan_input,
        d_boxs_defaultBox->_position,
        h_xmachine_memory_box_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_boxs_defaultBox->_position[h_xmachine_memory_box_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_boxs_defaultBox->_scan_input[h_xmachine_memory_box_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_box_defaultBox_count = scan_last_sum+1;
	else		
		h_xmachine_memory_box_defaultBox_count = scan_last_sum;
	//Scatter into swap
	scatter_box_Agents<<<gridSize, blockSize, 0, stream>>>(d_boxs_swap, d_boxs_defaultBox, 0, h_xmachine_memory_box_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_box_list* boxs_defaultBox_temp = d_boxs_defaultBox;
	d_boxs_defaultBox = d_boxs_swap;
	d_boxs_swap = boxs_defaultBox_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_box_defaultBox_count, &h_xmachine_memory_box_defaultBox_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_box, 
        temp_scan_storage_bytes_box, 
        d_boxs->_scan_input,
        d_boxs->_position,
        h_xmachine_memory_box_count, 
        stream
    );

	//reset agent count
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_boxs->_position[h_xmachine_memory_box_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_boxs->_scan_input[h_xmachine_memory_box_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_box_Agents<<<gridSize, blockSize, 0, stream>>>(d_boxs_swap, d_boxs, 0, h_xmachine_memory_box_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_box_count = scan_last_sum+1;
	else		
		h_xmachine_memory_box_count = scan_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_box_list* boxs_temp = d_boxs;
	d_boxs = d_boxs_swap;
	d_boxs_swap = boxs_temp;
	//update the device count
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_box_count, &h_xmachine_memory_box_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_box_count == 0)
	{
		return;
	}
	
	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_box_count;
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_box_response_count + h_xmachine_memory_box_count > xmachine_message_box_response_MAX){
		printf("Error: Buffer size of box_response message will be exceeded in function attend_box_patient\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_attend_box_patient, box_attend_box_patient_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = box_attend_box_patient_sm_size(blockSize);
	
	
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_box_response_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_box_response_output_type, &h_message_box_response_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_box_response_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_box_response_swaps<<<gridSize, blockSize, 0, stream>>>(d_box_responses); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (attend_box_patient)
	//Reallocate   : false
	//Input        : 
	//Output       : box_response
	//Agent Output : 
	GPUFLAME_attend_box_patient<<<g, b, sm_size, stream>>>(d_boxs, d_box_responses, d_rand48);
	gpuErrchkLaunch();
	
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//box_response Message Type Prefix Sum
	
	//swap output
	xmachine_message_box_response_list* d_box_responses_scanswap_temp = d_box_responses;
	d_box_responses = d_box_responses_swap;
	d_box_responses_swap = d_box_responses_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_box, 
        temp_scan_storage_bytes_box, 
        d_box_responses_swap->_scan_input,
        d_box_responses_swap->_position,
        h_xmachine_memory_box_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_box_response_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_box_response_messages<<<gridSize, blockSize, 0, stream>>>(d_box_responses, d_box_responses_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_box_responses_swap->_position[h_xmachine_memory_box_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_box_responses_swap->_scan_input[h_xmachine_memory_box_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_box_response_count += scan_last_sum+1;
	}else{
		h_message_box_response_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_box_response_count, &h_message_box_response_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_box_defaultBox_count+h_xmachine_memory_box_count > xmachine_memory_box_MAX){
		printf("Error: Buffer size of attend_box_patient agents in state defaultBox will be exceeded moving working agents to next state in function attend_box_patient\n");
      exit(EXIT_FAILURE);
      }
      
  //append agents to next state list
  cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_box_Agents, no_sm, state_list_size);
  gridSize = (state_list_size + blockSize - 1) / blockSize;
  append_box_Agents<<<gridSize, blockSize, 0, stream>>>(d_boxs_defaultBox, d_boxs, h_xmachine_memory_box_defaultBox_count, h_xmachine_memory_box_count);
  gpuErrchkLaunch();
        
	//update new state agent size
	h_xmachine_memory_box_defaultBox_count += h_xmachine_memory_box_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_box_defaultBox_count, &h_xmachine_memory_box_defaultBox_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int doctor_doctor_server_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_doctor_reached));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** doctor_doctor_server
 * Agent function prototype for doctor_server function of doctor agent
 */
void doctor_doctor_server(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_doctor_defaultDoctor_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_doctor_defaultDoctor_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_doctor_list* doctors_defaultDoctor_temp = d_doctors;
	d_doctors = d_doctors_defaultDoctor;
	d_doctors_defaultDoctor = doctors_defaultDoctor_temp;
	//set working count to current state count
	h_xmachine_memory_doctor_count = h_xmachine_memory_doctor_defaultDoctor_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_doctor_count, &h_xmachine_memory_doctor_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_doctor_defaultDoctor_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_doctor_defaultDoctor_count, &h_xmachine_memory_doctor_defaultDoctor_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_attention_terminated_count + h_xmachine_memory_doctor_count > xmachine_message_attention_terminated_MAX){
		printf("Error: Buffer size of attention_terminated message will be exceeded in function doctor_server\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_doctor_server, doctor_doctor_server_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = doctor_doctor_server_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_attention_terminated_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_attention_terminated_output_type, &h_message_attention_terminated_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_attention_terminated_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_attention_terminated_swaps<<<gridSize, blockSize, 0, stream>>>(d_attention_terminateds); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (doctor_server)
	//Reallocate   : false
	//Input        : doctor_reached
	//Output       : attention_terminated
	//Agent Output : 
	GPUFLAME_doctor_server<<<g, b, sm_size, stream>>>(d_doctors, d_doctor_reacheds, d_attention_terminateds);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//attention_terminated Message Type Prefix Sum
	
	//swap output
	xmachine_message_attention_terminated_list* d_attention_terminateds_scanswap_temp = d_attention_terminateds;
	d_attention_terminateds = d_attention_terminateds_swap;
	d_attention_terminateds_swap = d_attention_terminateds_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_doctor, 
        temp_scan_storage_bytes_doctor, 
        d_attention_terminateds_swap->_scan_input,
        d_attention_terminateds_swap->_position,
        h_xmachine_memory_doctor_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_attention_terminated_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_attention_terminated_messages<<<gridSize, blockSize, 0, stream>>>(d_attention_terminateds, d_attention_terminateds_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_attention_terminateds_swap->_position[h_xmachine_memory_doctor_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_attention_terminateds_swap->_scan_input[h_xmachine_memory_doctor_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_attention_terminated_count += scan_last_sum+1;
	}else{
		h_message_attention_terminated_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_attention_terminated_count, &h_message_attention_terminated_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_doctor_defaultDoctor_count+h_xmachine_memory_doctor_count > xmachine_memory_doctor_MAX){
		printf("Error: Buffer size of doctor_server agents in state defaultDoctor will be exceeded moving working agents to next state in function doctor_server\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  doctors_defaultDoctor_temp = d_doctors;
  d_doctors = d_doctors_defaultDoctor;
  d_doctors_defaultDoctor = doctors_defaultDoctor_temp;
        
	//update new state agent size
	h_xmachine_memory_doctor_defaultDoctor_count += h_xmachine_memory_doctor_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_doctor_defaultDoctor_count, &h_xmachine_memory_doctor_defaultDoctor_count, sizeof(int)));	
	
	
}



	
/* Shared memory size calculator for agent function */
int triage_receive_triage_petitions_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;
  //Continuous agent and message input has no partitioning
	sm_size += (blockSize * sizeof(xmachine_message_triage_petition));
	
	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);
	
	return sm_size;
}

/** triage_receive_triage_petitions
 * Agent function prototype for receive_triage_petitions function of triage agent
 */
void triage_receive_triage_petitions(cudaStream_t &stream){

    int sm_size;
    int blockSize;
    int minGridSize;
    int gridSize;
    int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func

	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	
	if (h_xmachine_memory_triage_defaultTriage_count == 0)
	{
		return;
	}
	
	
	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_triage_defaultTriage_count;

	

	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_triage_list* triages_defaultTriage_temp = d_triages;
	d_triages = d_triages_defaultTriage;
	d_triages_defaultTriage = triages_defaultTriage_temp;
	//set working count to current state count
	h_xmachine_memory_triage_count = h_xmachine_memory_triage_defaultTriage_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_triage_count, &h_xmachine_memory_triage_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_triage_defaultTriage_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_triage_defaultTriage_count, &h_xmachine_memory_triage_defaultTriage_count, sizeof(int)));	
	
 

	//******************************** AGENT FUNCTION *******************************

	
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_triage_response_count + h_xmachine_memory_triage_count > xmachine_message_triage_response_MAX){
		printf("Error: Buffer size of triage_response message will be exceeded in function receive_triage_petitions\n");
		exit(EXIT_FAILURE);
	}
	
	
	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, GPUFLAME_receive_triage_petitions, triage_receive_triage_petitions_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;
	
	sm_size = triage_receive_triage_petitions_sm_size(blockSize);
	
	
	
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	
	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS
	//Set the message_type for non partitioned, spatially partitioned and On-Graph Partitioned message outputs
	h_message_triage_response_output_type = optional_message;
	gpuErrchk( cudaMemcpyToSymbol( d_message_triage_response_output_type, &h_message_triage_response_output_type, sizeof(int)));
	//message is optional so reset the swap
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, reset_triage_response_swaps, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_triage_response_swaps<<<gridSize, blockSize, 0, stream>>>(d_triage_responses); 
	gpuErrchkLaunch();
	
	
	//MAIN XMACHINE FUNCTION CALL (receive_triage_petitions)
	//Reallocate   : false
	//Input        : triage_petition
	//Output       : triage_response
	//Agent Output : 
	GPUFLAME_receive_triage_petitions<<<g, b, sm_size, stream>>>(d_triages, d_triage_petitions, d_triage_responses);
	gpuErrchkLaunch();
	
	
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	//triage_response Message Type Prefix Sum
	
	//swap output
	xmachine_message_triage_response_list* d_triage_responses_scanswap_temp = d_triage_responses;
	d_triage_responses = d_triage_responses_swap;
	d_triage_responses_swap = d_triage_responses_scanswap_temp;
	
    cub::DeviceScan::ExclusiveSum(
        d_temp_scan_storage_triage, 
        temp_scan_storage_bytes_triage, 
        d_triage_responses_swap->_scan_input,
        d_triage_responses_swap->_position,
        h_xmachine_memory_triage_count, 
        stream
    );

	//Scatter
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, scatter_optional_triage_response_messages, no_sm, state_list_size); 
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	scatter_optional_triage_response_messages<<<gridSize, blockSize, 0, stream>>>(d_triage_responses, d_triage_responses_swap);
	gpuErrchkLaunch();
	
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	gpuErrchk( cudaMemcpy( &scan_last_sum, &d_triage_responses_swap->_position[h_xmachine_memory_triage_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaMemcpy( &scan_last_included, &d_triage_responses_swap->_scan_input[h_xmachine_memory_triage_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (scan_last_included == 1){
		h_message_triage_response_count += scan_last_sum+1;
	}else{
		h_message_triage_response_count += scan_last_sum;
	}
    //Copy count to device
	gpuErrchk( cudaMemcpyToSymbol( d_message_triage_response_count, &h_message_triage_response_count, sizeof(int)));	
	
	
	//************************ MOVE AGENTS TO NEXT STATE ****************************
    
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_triage_defaultTriage_count+h_xmachine_memory_triage_count > xmachine_memory_triage_MAX){
		printf("Error: Buffer size of receive_triage_petitions agents in state defaultTriage will be exceeded moving working agents to next state in function receive_triage_petitions\n");
      exit(EXIT_FAILURE);
      }
      
  //pointer swap the updated data
  triages_defaultTriage_temp = d_triages;
  d_triages = d_triages_defaultTriage;
  d_triages_defaultTriage = triages_defaultTriage_temp;
        
	//update new state agent size
	h_xmachine_memory_triage_defaultTriage_count += h_xmachine_memory_triage_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_triage_defaultTriage_count, &h_xmachine_memory_triage_defaultTriage_count, sizeof(int)));	
	
	
}


 
extern void reset_agent_default_count()
{
    h_xmachine_memory_agent_default_count = 0;
}
 
extern void reset_navmap_static_count()
{
    h_xmachine_memory_navmap_static_count = 0;
}
 
extern void reset_chair_defaultChair_count()
{
    h_xmachine_memory_chair_defaultChair_count = 0;
}
 
extern void reset_doctor_manager_defaultDoctorManager_count()
{
    h_xmachine_memory_doctor_manager_defaultDoctorManager_count = 0;
}
 
extern void reset_specialist_manager_defaultSpecialistManager_count()
{
    h_xmachine_memory_specialist_manager_defaultSpecialistManager_count = 0;
}
 
extern void reset_specialist_defaultSpecialist_count()
{
    h_xmachine_memory_specialist_defaultSpecialist_count = 0;
}
 
extern void reset_receptionist_defaultReceptionist_count()
{
    h_xmachine_memory_receptionist_defaultReceptionist_count = 0;
}
 
extern void reset_agent_generator_defaultGenerator_count()
{
    h_xmachine_memory_agent_generator_defaultGenerator_count = 0;
}
 
extern void reset_chair_admin_defaultAdmin_count()
{
    h_xmachine_memory_chair_admin_defaultAdmin_count = 0;
}
 
extern void reset_box_defaultBox_count()
{
    h_xmachine_memory_box_defaultBox_count = 0;
}
 
extern void reset_doctor_defaultDoctor_count()
{
    h_xmachine_memory_doctor_defaultDoctor_count = 0;
}
 
extern void reset_triage_defaultTriage_count()
{
    h_xmachine_memory_triage_defaultTriage_count = 0;
}
