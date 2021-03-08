
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


#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <limits.h>
#include <algorithm>
#include <string>
#include <vector>



#ifdef _WIN32
#define strtok_r strtok_s
#endif

// include header
#include "header.h"

glm::vec3 agent_maximum;
glm::vec3 agent_minimum;

int fpgu_strtol(const char* str){
    return (int)strtol(str, NULL, 0);
}

unsigned int fpgu_strtoul(const char* str){
    return (unsigned int)strtoul(str, NULL, 0);
}

long long int fpgu_strtoll(const char* str){
    return strtoll(str, NULL, 0);
}

unsigned long long int fpgu_strtoull(const char* str){
    return strtoull(str, NULL, 0);
}

double fpgu_strtod(const char* str){
    return strtod(str, NULL);
}

float fgpu_atof(const char* str){
    return (float)atof(str);
}


//templated class function to read array inputs from supported types
template <class T>
void readArrayInput( T (*parseFunc)(const char*), char* buffer, T *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;
    char * end_str;

    token = strtok_r(buffer, s, &end_str);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: variable array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        array[i++] = (T)parseFunc(token);
        
        token = strtok_r(NULL, s, &end_str);
    }
    if (i != expected_items){
        printf("Error: variable array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
    }
}

//templated class function to read array inputs from supported types
template <class T, class BASE_T, unsigned int D>
void readArrayInputVectorType( BASE_T (*parseFunc)(const char*), char* buffer, T *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = "|";
    char * token;
    char * end_str;

    token = strtok_r(buffer, s, &end_str);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(EXIT_FAILURE);
        }
        
        //read vector type as an array
        T vec;
        readArrayInput<BASE_T>(parseFunc, token, (BASE_T*) &vec, D);
        array[i++] = vec;
        
        token = strtok_r(NULL, s, &end_str);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(EXIT_FAILURE);
    }
}

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_agent_list* h_agents_default, xmachine_memory_agent_list* d_agents_default, int h_xmachine_memory_agent_default_count,xmachine_memory_navmap_list* h_navmaps_static, xmachine_memory_navmap_list* d_navmaps_static, int h_xmachine_memory_navmap_static_count,xmachine_memory_chair_list* h_chairs_defaultChair, xmachine_memory_chair_list* d_chairs_defaultChair, int h_xmachine_memory_chair_defaultChair_count,xmachine_memory_doctor_manager_list* h_doctor_managers_defaultDoctorManager, xmachine_memory_doctor_manager_list* d_doctor_managers_defaultDoctorManager, int h_xmachine_memory_doctor_manager_defaultDoctorManager_count,xmachine_memory_receptionist_list* h_receptionists_defaultReceptionist, xmachine_memory_receptionist_list* d_receptionists_defaultReceptionist, int h_xmachine_memory_receptionist_defaultReceptionist_count,xmachine_memory_agent_generator_list* h_agent_generators_defaultGenerator, xmachine_memory_agent_generator_list* d_agent_generators_defaultGenerator, int h_xmachine_memory_agent_generator_defaultGenerator_count,xmachine_memory_chair_admin_list* h_chair_admins_defaultAdmin, xmachine_memory_chair_admin_list* d_chair_admins_defaultAdmin, int h_xmachine_memory_chair_admin_defaultAdmin_count,xmachine_memory_box_list* h_boxs_defaultBox, xmachine_memory_box_list* d_boxs_defaultBox, int h_xmachine_memory_box_defaultBox_count,xmachine_memory_doctor_list* h_doctors_defaultDoctor, xmachine_memory_doctor_list* d_doctors_defaultDoctor, int h_xmachine_memory_doctor_defaultDoctor_count,xmachine_memory_triage_list* h_triages_defaultTriage, xmachine_memory_triage_list* d_triages_defaultTriage, int h_xmachine_memory_triage_defaultTriage_count)
{
    PROFILE_SCOPED_RANGE("saveIterationData");
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_agents_default, d_agents_default, sizeof(xmachine_memory_agent_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying agent Agent default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_navmaps_static, d_navmaps_static, sizeof(xmachine_memory_navmap_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying navmap Agent static State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_chairs_defaultChair, d_chairs_defaultChair, sizeof(xmachine_memory_chair_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying chair Agent defaultChair State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_doctor_managers_defaultDoctorManager, d_doctor_managers_defaultDoctorManager, sizeof(xmachine_memory_doctor_manager_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying doctor_manager Agent defaultDoctorManager State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_receptionists_defaultReceptionist, d_receptionists_defaultReceptionist, sizeof(xmachine_memory_receptionist_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying receptionist Agent defaultReceptionist State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_agent_generators_defaultGenerator, d_agent_generators_defaultGenerator, sizeof(xmachine_memory_agent_generator_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying agent_generator Agent defaultGenerator State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_chair_admins_defaultAdmin, d_chair_admins_defaultAdmin, sizeof(xmachine_memory_chair_admin_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying chair_admin Agent defaultAdmin State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_boxs_defaultBox, d_boxs_defaultBox, sizeof(xmachine_memory_box_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying box Agent defaultBox State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_doctors_defaultDoctor, d_doctors_defaultDoctor, sizeof(xmachine_memory_doctor_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying doctor Agent defaultDoctor State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_triages_defaultTriage, d_triages_defaultTriage, sizeof(xmachine_memory_triage_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying triage Agent defaultTriage State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	
	/* Pointer to file */
	FILE *file;
	char data[100];

	sprintf(data, "%s%i.xml", outputpath, iteration_number);
	//printf("Writing iteration %i data to %s\n", iteration_number, data);
	file = fopen(data, "w");
    if(file == nullptr){
        printf("Error: Could not open file `%s` for output. Aborting.\n", data);
        exit(EXIT_FAILURE);
    }
    fputs("<states>\n<itno>", file);
    sprintf(data, "%i", iteration_number);
    fputs(data, file);
    fputs("</itno>\n", file);
    fputs("<environment>\n" , file);
    
    fputs("\t<EMMISION_RATE_EXIT1>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT1()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT1>\n", file);
    fputs("\t<EMMISION_RATE_EXIT2>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT2()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT2>\n", file);
    fputs("\t<EMMISION_RATE_EXIT3>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT3()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT3>\n", file);
    fputs("\t<EMMISION_RATE_EXIT4>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT4()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT4>\n", file);
    fputs("\t<EMMISION_RATE_EXIT5>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT5()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT5>\n", file);
    fputs("\t<EMMISION_RATE_EXIT6>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT6()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT6>\n", file);
    fputs("\t<EMMISION_RATE_EXIT7>", file);
    sprintf(data, "%f", (*get_EMMISION_RATE_EXIT7()));
    fputs(data, file);
    fputs("</EMMISION_RATE_EXIT7>\n", file);
    fputs("\t<EXIT1_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT1_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT1_PROBABILITY>\n", file);
    fputs("\t<EXIT2_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT2_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT2_PROBABILITY>\n", file);
    fputs("\t<EXIT3_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT3_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT3_PROBABILITY>\n", file);
    fputs("\t<EXIT4_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT4_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT4_PROBABILITY>\n", file);
    fputs("\t<EXIT5_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT5_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT5_PROBABILITY>\n", file);
    fputs("\t<EXIT6_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT6_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT6_PROBABILITY>\n", file);
    fputs("\t<EXIT7_PROBABILITY>", file);
    sprintf(data, "%d", (*get_EXIT7_PROBABILITY()));
    fputs(data, file);
    fputs("</EXIT7_PROBABILITY>\n", file);
    fputs("\t<EXIT1_STATE>", file);
    sprintf(data, "%d", (*get_EXIT1_STATE()));
    fputs(data, file);
    fputs("</EXIT1_STATE>\n", file);
    fputs("\t<EXIT2_STATE>", file);
    sprintf(data, "%d", (*get_EXIT2_STATE()));
    fputs(data, file);
    fputs("</EXIT2_STATE>\n", file);
    fputs("\t<EXIT3_STATE>", file);
    sprintf(data, "%d", (*get_EXIT3_STATE()));
    fputs(data, file);
    fputs("</EXIT3_STATE>\n", file);
    fputs("\t<EXIT4_STATE>", file);
    sprintf(data, "%d", (*get_EXIT4_STATE()));
    fputs(data, file);
    fputs("</EXIT4_STATE>\n", file);
    fputs("\t<EXIT5_STATE>", file);
    sprintf(data, "%d", (*get_EXIT5_STATE()));
    fputs(data, file);
    fputs("</EXIT5_STATE>\n", file);
    fputs("\t<EXIT6_STATE>", file);
    sprintf(data, "%d", (*get_EXIT6_STATE()));
    fputs(data, file);
    fputs("</EXIT6_STATE>\n", file);
    fputs("\t<EXIT7_STATE>", file);
    sprintf(data, "%d", (*get_EXIT7_STATE()));
    fputs(data, file);
    fputs("</EXIT7_STATE>\n", file);
    fputs("\t<EXIT1_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT1_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT1_CELL_COUNT>\n", file);
    fputs("\t<EXIT2_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT2_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT2_CELL_COUNT>\n", file);
    fputs("\t<EXIT3_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT3_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT3_CELL_COUNT>\n", file);
    fputs("\t<EXIT4_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT4_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT4_CELL_COUNT>\n", file);
    fputs("\t<EXIT5_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT5_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT5_CELL_COUNT>\n", file);
    fputs("\t<EXIT6_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT6_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT6_CELL_COUNT>\n", file);
    fputs("\t<EXIT7_CELL_COUNT>", file);
    sprintf(data, "%d", (*get_EXIT7_CELL_COUNT()));
    fputs(data, file);
    fputs("</EXIT7_CELL_COUNT>\n", file);
    fputs("\t<TIME_SCALER>", file);
    sprintf(data, "%f", (*get_TIME_SCALER()));
    fputs(data, file);
    fputs("</TIME_SCALER>\n", file);
    fputs("\t<STEER_WEIGHT>", file);
    sprintf(data, "%f", (*get_STEER_WEIGHT()));
    fputs(data, file);
    fputs("</STEER_WEIGHT>\n", file);
    fputs("\t<AVOID_WEIGHT>", file);
    sprintf(data, "%f", (*get_AVOID_WEIGHT()));
    fputs(data, file);
    fputs("</AVOID_WEIGHT>\n", file);
    fputs("\t<COLLISION_WEIGHT>", file);
    sprintf(data, "%f", (*get_COLLISION_WEIGHT()));
    fputs(data, file);
    fputs("</COLLISION_WEIGHT>\n", file);
    fputs("\t<GOAL_WEIGHT>", file);
    sprintf(data, "%f", (*get_GOAL_WEIGHT()));
    fputs(data, file);
    fputs("</GOAL_WEIGHT>\n", file);
	fputs("</environment>\n" , file);

	//Write each agent agent to xml
	for (int i=0; i<h_xmachine_memory_agent_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>agent</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_agents_default->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_agents_default->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_agents_default->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<velx>", file);
        sprintf(data, "%f", h_agents_default->velx[i]);
		fputs(data, file);
		fputs("</velx>\n", file);
        
		fputs("<vely>", file);
        sprintf(data, "%f", h_agents_default->vely[i]);
		fputs(data, file);
		fputs("</vely>\n", file);
        
		fputs("<steer_x>", file);
        sprintf(data, "%f", h_agents_default->steer_x[i]);
		fputs(data, file);
		fputs("</steer_x>\n", file);
        
		fputs("<steer_y>", file);
        sprintf(data, "%f", h_agents_default->steer_y[i]);
		fputs(data, file);
		fputs("</steer_y>\n", file);
        
		fputs("<height>", file);
        sprintf(data, "%f", h_agents_default->height[i]);
		fputs(data, file);
		fputs("</height>\n", file);
        
		fputs("<exit_no>", file);
        sprintf(data, "%d", h_agents_default->exit_no[i]);
		fputs(data, file);
		fputs("</exit_no>\n", file);
        
		fputs("<speed>", file);
        sprintf(data, "%f", h_agents_default->speed[i]);
		fputs(data, file);
		fputs("</speed>\n", file);
        
		fputs("<lod>", file);
        sprintf(data, "%d", h_agents_default->lod[i]);
		fputs(data, file);
		fputs("</lod>\n", file);
        
		fputs("<animate>", file);
        sprintf(data, "%f", h_agents_default->animate[i]);
		fputs(data, file);
		fputs("</animate>\n", file);
        
		fputs("<animate_dir>", file);
        sprintf(data, "%d", h_agents_default->animate_dir[i]);
		fputs(data, file);
		fputs("</animate_dir>\n", file);
        
		fputs("<estado>", file);
        sprintf(data, "%d", h_agents_default->estado[i]);
		fputs(data, file);
		fputs("</estado>\n", file);
        
		fputs("<tick>", file);
        sprintf(data, "%d", h_agents_default->tick[i]);
		fputs(data, file);
		fputs("</tick>\n", file);
        
		fputs("<estado_movimiento>", file);
        sprintf(data, "%u", h_agents_default->estado_movimiento[i]);
		fputs(data, file);
		fputs("</estado_movimiento>\n", file);
        
		fputs("<go_to_x>", file);
        sprintf(data, "%u", h_agents_default->go_to_x[i]);
		fputs(data, file);
		fputs("</go_to_x>\n", file);
        
		fputs("<go_to_y>", file);
        sprintf(data, "%u", h_agents_default->go_to_y[i]);
		fputs(data, file);
		fputs("</go_to_y>\n", file);
        
		fputs("<checkpoint>", file);
        sprintf(data, "%u", h_agents_default->checkpoint[i]);
		fputs(data, file);
		fputs("</checkpoint>\n", file);
        
		fputs("<chair_no>", file);
        sprintf(data, "%d", h_agents_default->chair_no[i]);
		fputs(data, file);
		fputs("</chair_no>\n", file);
        
		fputs("<box_no>", file);
        sprintf(data, "%u", h_agents_default->box_no[i]);
		fputs(data, file);
		fputs("</box_no>\n", file);
        
		fputs("<doctor_no>", file);
        sprintf(data, "%u", h_agents_default->doctor_no[i]);
		fputs(data, file);
		fputs("</doctor_no>\n", file);
        
		fputs("<priority>", file);
        sprintf(data, "%u", h_agents_default->priority[i]);
		fputs(data, file);
		fputs("</priority>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each navmap agent to xml
	for (int i=0; i<h_xmachine_memory_navmap_static_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>navmap</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%d", h_navmaps_static->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%d", h_navmaps_static->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<exit_no>", file);
        sprintf(data, "%d", h_navmaps_static->exit_no[i]);
		fputs(data, file);
		fputs("</exit_no>\n", file);
        
		fputs("<height>", file);
        sprintf(data, "%f", h_navmaps_static->height[i]);
		fputs(data, file);
		fputs("</height>\n", file);
        
		fputs("<collision_x>", file);
        sprintf(data, "%f", h_navmaps_static->collision_x[i]);
		fputs(data, file);
		fputs("</collision_x>\n", file);
        
		fputs("<collision_y>", file);
        sprintf(data, "%f", h_navmaps_static->collision_y[i]);
		fputs(data, file);
		fputs("</collision_y>\n", file);
        
		fputs("<exit0_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit0_x[i]);
		fputs(data, file);
		fputs("</exit0_x>\n", file);
        
		fputs("<exit0_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit0_y[i]);
		fputs(data, file);
		fputs("</exit0_y>\n", file);
        
		fputs("<exit1_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit1_x[i]);
		fputs(data, file);
		fputs("</exit1_x>\n", file);
        
		fputs("<exit1_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit1_y[i]);
		fputs(data, file);
		fputs("</exit1_y>\n", file);
        
		fputs("<exit2_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit2_x[i]);
		fputs(data, file);
		fputs("</exit2_x>\n", file);
        
		fputs("<exit2_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit2_y[i]);
		fputs(data, file);
		fputs("</exit2_y>\n", file);
        
		fputs("<exit3_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit3_x[i]);
		fputs(data, file);
		fputs("</exit3_x>\n", file);
        
		fputs("<exit3_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit3_y[i]);
		fputs(data, file);
		fputs("</exit3_y>\n", file);
        
		fputs("<exit4_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit4_x[i]);
		fputs(data, file);
		fputs("</exit4_x>\n", file);
        
		fputs("<exit4_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit4_y[i]);
		fputs(data, file);
		fputs("</exit4_y>\n", file);
        
		fputs("<exit5_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit5_x[i]);
		fputs(data, file);
		fputs("</exit5_x>\n", file);
        
		fputs("<exit5_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit5_y[i]);
		fputs(data, file);
		fputs("</exit5_y>\n", file);
        
		fputs("<exit6_x>", file);
        sprintf(data, "%f", h_navmaps_static->exit6_x[i]);
		fputs(data, file);
		fputs("</exit6_x>\n", file);
        
		fputs("<exit6_y>", file);
        sprintf(data, "%f", h_navmaps_static->exit6_y[i]);
		fputs(data, file);
		fputs("</exit6_y>\n", file);
        
		fputs("<cant_generados>", file);
        sprintf(data, "%u", h_navmaps_static->cant_generados[i]);
		fputs(data, file);
		fputs("</cant_generados>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each chair agent to xml
	for (int i=0; i<h_xmachine_memory_chair_defaultChair_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>chair</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%d", h_chairs_defaultChair->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%d", h_chairs_defaultChair->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%d", h_chairs_defaultChair->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<state>", file);
        sprintf(data, "%d", h_chairs_defaultChair->state[i]);
		fputs(data, file);
		fputs("</state>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each doctor_manager agent to xml
	for (int i=0; i<h_xmachine_memory_doctor_manager_defaultDoctorManager_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>doctor_manager</name>\n", file);
        
		fputs("<tick>", file);
        sprintf(data, "%u", h_doctor_managers_defaultDoctorManager->tick[i]);
		fputs(data, file);
		fputs("</tick>\n", file);
        
		fputs("<rear>", file);
        sprintf(data, "%u", h_doctor_managers_defaultDoctorManager->rear[i]);
		fputs(data, file);
		fputs("</rear>\n", file);
        
		fputs("<size>", file);
        sprintf(data, "%u", h_doctor_managers_defaultDoctorManager->size[i]);
		fputs(data, file);
		fputs("</size>\n", file);
        
		fputs("<doctors_occupied>", file);
        for (int j=0;j<4;j++){
            fprintf(file, "%d", h_doctor_managers_defaultDoctorManager->doctors_occupied[(j*xmachine_memory_doctor_manager_MAX)+i]);
            if(j!=(4-1))
                fprintf(file, ",");
        }
		fputs("</doctors_occupied>\n", file);
        
		fputs("<free_doctors>", file);
        sprintf(data, "%u", h_doctor_managers_defaultDoctorManager->free_doctors[i]);
		fputs(data, file);
		fputs("</free_doctors>\n", file);
        
		fputs("<patientQueue>", file);
        for (int j=0;j<35;j++){
            fprintf(file, "%d, %d", h_doctor_managers_defaultDoctorManager->patientQueue[(j*xmachine_memory_doctor_manager_MAX)+i].x, h_doctor_managers_defaultDoctorManager->patientQueue[(j*xmachine_memory_doctor_manager_MAX)+i].y);
            if(j!=(35-1))
                fprintf(file, "|");
        }
		fputs("</patientQueue>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each receptionist agent to xml
	for (int i=0; i<h_xmachine_memory_receptionist_defaultReceptionist_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>receptionist</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%d", h_receptionists_defaultReceptionist->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%d", h_receptionists_defaultReceptionist->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<patientQueue>", file);
        for (int j=0;j<100;j++){
            fprintf(file, "%u", h_receptionists_defaultReceptionist->patientQueue[(j*xmachine_memory_receptionist_MAX)+i]);
            if(j!=(100-1))
                fprintf(file, ",");
        }
		fputs("</patientQueue>\n", file);
        
		fputs("<front>", file);
        sprintf(data, "%u", h_receptionists_defaultReceptionist->front[i]);
		fputs(data, file);
		fputs("</front>\n", file);
        
		fputs("<rear>", file);
        sprintf(data, "%u", h_receptionists_defaultReceptionist->rear[i]);
		fputs(data, file);
		fputs("</rear>\n", file);
        
		fputs("<size>", file);
        sprintf(data, "%u", h_receptionists_defaultReceptionist->size[i]);
		fputs(data, file);
		fputs("</size>\n", file);
        
		fputs("<tick>", file);
        sprintf(data, "%u", h_receptionists_defaultReceptionist->tick[i]);
		fputs(data, file);
		fputs("</tick>\n", file);
        
		fputs("<current_patient>", file);
        sprintf(data, "%d", h_receptionists_defaultReceptionist->current_patient[i]);
		fputs(data, file);
		fputs("</current_patient>\n", file);
        
		fputs("<attend_patient>", file);
        sprintf(data, "%d", h_receptionists_defaultReceptionist->attend_patient[i]);
		fputs(data, file);
		fputs("</attend_patient>\n", file);
        
		fputs("<estado>", file);
        sprintf(data, "%d", h_receptionists_defaultReceptionist->estado[i]);
		fputs(data, file);
		fputs("</estado>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each agent_generator agent to xml
	for (int i=0; i<h_xmachine_memory_agent_generator_defaultGenerator_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>agent_generator</name>\n", file);
        
		fputs("<chairs_generated>", file);
        sprintf(data, "%d", h_agent_generators_defaultGenerator->chairs_generated[i]);
		fputs(data, file);
		fputs("</chairs_generated>\n", file);
        
		fputs("<boxes_generated>", file);
        sprintf(data, "%d", h_agent_generators_defaultGenerator->boxes_generated[i]);
		fputs(data, file);
		fputs("</boxes_generated>\n", file);
        
		fputs("<doctors_generated>", file);
        sprintf(data, "%d", h_agent_generators_defaultGenerator->doctors_generated[i]);
		fputs(data, file);
		fputs("</doctors_generated>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each chair_admin agent to xml
	for (int i=0; i<h_xmachine_memory_chair_admin_defaultAdmin_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>chair_admin</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_chair_admins_defaultAdmin->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<chairArray>", file);
        for (int j=0;j<35;j++){
            fprintf(file, "%u", h_chair_admins_defaultAdmin->chairArray[(j*xmachine_memory_chair_admin_MAX)+i]);
            if(j!=(35-1))
                fprintf(file, ",");
        }
		fputs("</chairArray>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each box agent to xml
	for (int i=0; i<h_xmachine_memory_box_defaultBox_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>box</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_boxs_defaultBox->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<attending>", file);
        sprintf(data, "%u", h_boxs_defaultBox->attending[i]);
		fputs(data, file);
		fputs("</attending>\n", file);
        
		fputs("<tick>", file);
        sprintf(data, "%u", h_boxs_defaultBox->tick[i]);
		fputs(data, file);
		fputs("</tick>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each doctor agent to xml
	for (int i=0; i<h_xmachine_memory_doctor_defaultDoctor_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>doctor</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_doctors_defaultDoctor->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<attending>", file);
        sprintf(data, "%d", h_doctors_defaultDoctor->attending[i]);
		fputs(data, file);
		fputs("</attending>\n", file);
        
		fputs("<tick>", file);
        sprintf(data, "%u", h_doctors_defaultDoctor->tick[i]);
		fputs(data, file);
		fputs("</tick>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each triage agent to xml
	for (int i=0; i<h_xmachine_memory_triage_defaultTriage_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>triage</name>\n", file);
        
		fputs("<front>", file);
        sprintf(data, "%u", h_triages_defaultTriage->front[i]);
		fputs(data, file);
		fputs("</front>\n", file);
        
		fputs("<rear>", file);
        sprintf(data, "%u", h_triages_defaultTriage->rear[i]);
		fputs(data, file);
		fputs("</rear>\n", file);
        
		fputs("<size>", file);
        sprintf(data, "%u", h_triages_defaultTriage->size[i]);
		fputs(data, file);
		fputs("</size>\n", file);
        
		fputs("<tick>", file);
        sprintf(data, "%u", h_triages_defaultTriage->tick[i]);
		fputs(data, file);
		fputs("</tick>\n", file);
        
		fputs("<boxArray>", file);
        for (int j=0;j<3;j++){
            fprintf(file, "%u", h_triages_defaultTriage->boxArray[(j*xmachine_memory_triage_MAX)+i]);
            if(j!=(3-1))
                fprintf(file, ",");
        }
		fputs("</boxArray>\n", file);
        
		fputs("<patientQueue>", file);
        for (int j=0;j<100;j++){
            fprintf(file, "%u", h_triages_defaultTriage->patientQueue[(j*xmachine_memory_triage_MAX)+i]);
            if(j!=(100-1))
                fprintf(file, ",");
        }
		fputs("</patientQueue>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);

}

void readInitialStates(char* inputpath, xmachine_memory_agent_list* h_agents, int* h_xmachine_memory_agent_count,xmachine_memory_navmap_list* h_navmaps, int* h_xmachine_memory_navmap_count,xmachine_memory_chair_list* h_chairs, int* h_xmachine_memory_chair_count,xmachine_memory_doctor_manager_list* h_doctor_managers, int* h_xmachine_memory_doctor_manager_count,xmachine_memory_receptionist_list* h_receptionists, int* h_xmachine_memory_receptionist_count,xmachine_memory_agent_generator_list* h_agent_generators, int* h_xmachine_memory_agent_generator_count,xmachine_memory_chair_admin_list* h_chair_admins, int* h_xmachine_memory_chair_admin_count,xmachine_memory_box_list* h_boxs, int* h_xmachine_memory_box_count,xmachine_memory_doctor_list* h_doctors, int* h_xmachine_memory_doctor_count,xmachine_memory_triage_list* h_triages, int* h_xmachine_memory_triage_count)
{
    PROFILE_SCOPED_RANGE("readInitialStates");

	int temp = 0;
	int* itno = &temp;

	/* Pointer to file */
	FILE *file;
	/* Char and char buffer for reading file to */
	char c = ' ';
	const int bufferSize = 10000;
	char buffer[bufferSize];
	char agentname[1000];

	/* Pointer to x-memory for initial state data */
	/*xmachine * current_xmachine;*/
	/* Variables for checking tags */
	int reading, i;
	int in_tag, in_itno, in_xagent, in_name, in_comment;
    int in_agent_id;
    int in_agent_x;
    int in_agent_y;
    int in_agent_velx;
    int in_agent_vely;
    int in_agent_steer_x;
    int in_agent_steer_y;
    int in_agent_height;
    int in_agent_exit_no;
    int in_agent_speed;
    int in_agent_lod;
    int in_agent_animate;
    int in_agent_animate_dir;
    int in_agent_estado;
    int in_agent_tick;
    int in_agent_estado_movimiento;
    int in_agent_go_to_x;
    int in_agent_go_to_y;
    int in_agent_checkpoint;
    int in_agent_chair_no;
    int in_agent_box_no;
    int in_agent_doctor_no;
    int in_agent_priority;
    int in_navmap_x;
    int in_navmap_y;
    int in_navmap_exit_no;
    int in_navmap_height;
    int in_navmap_collision_x;
    int in_navmap_collision_y;
    int in_navmap_exit0_x;
    int in_navmap_exit0_y;
    int in_navmap_exit1_x;
    int in_navmap_exit1_y;
    int in_navmap_exit2_x;
    int in_navmap_exit2_y;
    int in_navmap_exit3_x;
    int in_navmap_exit3_y;
    int in_navmap_exit4_x;
    int in_navmap_exit4_y;
    int in_navmap_exit5_x;
    int in_navmap_exit5_y;
    int in_navmap_exit6_x;
    int in_navmap_exit6_y;
    int in_navmap_cant_generados;
    int in_chair_id;
    int in_chair_x;
    int in_chair_y;
    int in_chair_state;
    int in_doctor_manager_tick;
    int in_doctor_manager_rear;
    int in_doctor_manager_size;
    int in_doctor_manager_doctors_occupied;
    int in_doctor_manager_free_doctors;
    int in_doctor_manager_patientQueue;
    int in_receptionist_x;
    int in_receptionist_y;
    int in_receptionist_patientQueue;
    int in_receptionist_front;
    int in_receptionist_rear;
    int in_receptionist_size;
    int in_receptionist_tick;
    int in_receptionist_current_patient;
    int in_receptionist_attend_patient;
    int in_receptionist_estado;
    int in_agent_generator_chairs_generated;
    int in_agent_generator_boxes_generated;
    int in_agent_generator_doctors_generated;
    int in_chair_admin_id;
    int in_chair_admin_chairArray;
    int in_box_id;
    int in_box_attending;
    int in_box_tick;
    int in_doctor_id;
    int in_doctor_attending;
    int in_doctor_tick;
    int in_triage_front;
    int in_triage_rear;
    int in_triage_size;
    int in_triage_tick;
    int in_triage_boxArray;
    int in_triage_patientQueue;
    
    /* tags for environment global variables */
    int in_env;
    int in_env_EMMISION_RATE_EXIT1;
    
    int in_env_EMMISION_RATE_EXIT2;
    
    int in_env_EMMISION_RATE_EXIT3;
    
    int in_env_EMMISION_RATE_EXIT4;
    
    int in_env_EMMISION_RATE_EXIT5;
    
    int in_env_EMMISION_RATE_EXIT6;
    
    int in_env_EMMISION_RATE_EXIT7;
    
    int in_env_EXIT1_PROBABILITY;
    
    int in_env_EXIT2_PROBABILITY;
    
    int in_env_EXIT3_PROBABILITY;
    
    int in_env_EXIT4_PROBABILITY;
    
    int in_env_EXIT5_PROBABILITY;
    
    int in_env_EXIT6_PROBABILITY;
    
    int in_env_EXIT7_PROBABILITY;
    
    int in_env_EXIT1_STATE;
    
    int in_env_EXIT2_STATE;
    
    int in_env_EXIT3_STATE;
    
    int in_env_EXIT4_STATE;
    
    int in_env_EXIT5_STATE;
    
    int in_env_EXIT6_STATE;
    
    int in_env_EXIT7_STATE;
    
    int in_env_EXIT1_CELL_COUNT;
    
    int in_env_EXIT2_CELL_COUNT;
    
    int in_env_EXIT3_CELL_COUNT;
    
    int in_env_EXIT4_CELL_COUNT;
    
    int in_env_EXIT5_CELL_COUNT;
    
    int in_env_EXIT6_CELL_COUNT;
    
    int in_env_EXIT7_CELL_COUNT;
    
    int in_env_TIME_SCALER;
    
    int in_env_STEER_WEIGHT;
    
    int in_env_AVOID_WEIGHT;
    
    int in_env_COLLISION_WEIGHT;
    
    int in_env_GOAL_WEIGHT;
    
	/* set agent count to zero */
	*h_xmachine_memory_agent_count = 0;
	*h_xmachine_memory_navmap_count = 0;
	*h_xmachine_memory_chair_count = 0;
	*h_xmachine_memory_doctor_manager_count = 0;
	*h_xmachine_memory_receptionist_count = 0;
	*h_xmachine_memory_agent_generator_count = 0;
	*h_xmachine_memory_chair_admin_count = 0;
	*h_xmachine_memory_box_count = 0;
	*h_xmachine_memory_doctor_count = 0;
	*h_xmachine_memory_triage_count = 0;
	
	/* Variables for initial state data */
	unsigned int agent_id;
	float agent_x;
	float agent_y;
	float agent_velx;
	float agent_vely;
	float agent_steer_x;
	float agent_steer_y;
	float agent_height;
	int agent_exit_no;
	float agent_speed;
	int agent_lod;
	float agent_animate;
	int agent_animate_dir;
	int agent_estado;
	int agent_tick;
	unsigned int agent_estado_movimiento;
	unsigned int agent_go_to_x;
	unsigned int agent_go_to_y;
	unsigned int agent_checkpoint;
	int agent_chair_no;
	unsigned int agent_box_no;
	unsigned int agent_doctor_no;
	unsigned int agent_priority;
	int navmap_x;
	int navmap_y;
	int navmap_exit_no;
	float navmap_height;
	float navmap_collision_x;
	float navmap_collision_y;
	float navmap_exit0_x;
	float navmap_exit0_y;
	float navmap_exit1_x;
	float navmap_exit1_y;
	float navmap_exit2_x;
	float navmap_exit2_y;
	float navmap_exit3_x;
	float navmap_exit3_y;
	float navmap_exit4_x;
	float navmap_exit4_y;
	float navmap_exit5_x;
	float navmap_exit5_y;
	float navmap_exit6_x;
	float navmap_exit6_y;
	unsigned int navmap_cant_generados;
	int chair_id;
	int chair_x;
	int chair_y;
	int chair_state;
	unsigned int doctor_manager_tick;
	unsigned int doctor_manager_rear;
	unsigned int doctor_manager_size;
    int doctor_manager_doctors_occupied[4];
	unsigned int doctor_manager_free_doctors;
    ivec2 doctor_manager_patientQueue[35];
	int receptionist_x;
	int receptionist_y;
    unsigned int receptionist_patientQueue[100];
	unsigned int receptionist_front;
	unsigned int receptionist_rear;
	unsigned int receptionist_size;
	unsigned int receptionist_tick;
	int receptionist_current_patient;
	int receptionist_attend_patient;
	int receptionist_estado;
	int agent_generator_chairs_generated;
	int agent_generator_boxes_generated;
	int agent_generator_doctors_generated;
	unsigned int chair_admin_id;
    unsigned int chair_admin_chairArray[35];
	unsigned int box_id;
	unsigned int box_attending;
	unsigned int box_tick;
	unsigned int doctor_id;
	int doctor_attending;
	unsigned int doctor_tick;
	unsigned int triage_front;
	unsigned int triage_rear;
	unsigned int triage_size;
	unsigned int triage_tick;
    unsigned int triage_boxArray[3];
    unsigned int triage_patientQueue[100];

    /* Variables for environment variables */
    float env_EMMISION_RATE_EXIT1;
    float env_EMMISION_RATE_EXIT2;
    float env_EMMISION_RATE_EXIT3;
    float env_EMMISION_RATE_EXIT4;
    float env_EMMISION_RATE_EXIT5;
    float env_EMMISION_RATE_EXIT6;
    float env_EMMISION_RATE_EXIT7;
    int env_EXIT1_PROBABILITY;
    int env_EXIT2_PROBABILITY;
    int env_EXIT3_PROBABILITY;
    int env_EXIT4_PROBABILITY;
    int env_EXIT5_PROBABILITY;
    int env_EXIT6_PROBABILITY;
    int env_EXIT7_PROBABILITY;
    int env_EXIT1_STATE;
    int env_EXIT2_STATE;
    int env_EXIT3_STATE;
    int env_EXIT4_STATE;
    int env_EXIT5_STATE;
    int env_EXIT6_STATE;
    int env_EXIT7_STATE;
    int env_EXIT1_CELL_COUNT;
    int env_EXIT2_CELL_COUNT;
    int env_EXIT3_CELL_COUNT;
    int env_EXIT4_CELL_COUNT;
    int env_EXIT5_CELL_COUNT;
    int env_EXIT6_CELL_COUNT;
    int env_EXIT7_CELL_COUNT;
    float env_TIME_SCALER;
    float env_STEER_WEIGHT;
    float env_AVOID_WEIGHT;
    float env_COLLISION_WEIGHT;
    float env_GOAL_WEIGHT;
    


	/* Initialise variables */
    agent_maximum.x = 0;
    agent_maximum.y = 0;
    agent_maximum.z = 0;
    agent_minimum.x = 0;
    agent_minimum.y = 0;
    agent_minimum.z = 0;
	reading = 1;
    in_comment = 0;
	in_tag = 0;
	in_itno = 0;
    in_env = 0;
    in_xagent = 0;
	in_name = 0;
	in_agent_id = 0;
	in_agent_x = 0;
	in_agent_y = 0;
	in_agent_velx = 0;
	in_agent_vely = 0;
	in_agent_steer_x = 0;
	in_agent_steer_y = 0;
	in_agent_height = 0;
	in_agent_exit_no = 0;
	in_agent_speed = 0;
	in_agent_lod = 0;
	in_agent_animate = 0;
	in_agent_animate_dir = 0;
	in_agent_estado = 0;
	in_agent_tick = 0;
	in_agent_estado_movimiento = 0;
	in_agent_go_to_x = 0;
	in_agent_go_to_y = 0;
	in_agent_checkpoint = 0;
	in_agent_chair_no = 0;
	in_agent_box_no = 0;
	in_agent_doctor_no = 0;
	in_agent_priority = 0;
	in_navmap_x = 0;
	in_navmap_y = 0;
	in_navmap_exit_no = 0;
	in_navmap_height = 0;
	in_navmap_collision_x = 0;
	in_navmap_collision_y = 0;
	in_navmap_exit0_x = 0;
	in_navmap_exit0_y = 0;
	in_navmap_exit1_x = 0;
	in_navmap_exit1_y = 0;
	in_navmap_exit2_x = 0;
	in_navmap_exit2_y = 0;
	in_navmap_exit3_x = 0;
	in_navmap_exit3_y = 0;
	in_navmap_exit4_x = 0;
	in_navmap_exit4_y = 0;
	in_navmap_exit5_x = 0;
	in_navmap_exit5_y = 0;
	in_navmap_exit6_x = 0;
	in_navmap_exit6_y = 0;
	in_navmap_cant_generados = 0;
	in_chair_id = 0;
	in_chair_x = 0;
	in_chair_y = 0;
	in_chair_state = 0;
	in_doctor_manager_tick = 0;
	in_doctor_manager_rear = 0;
	in_doctor_manager_size = 0;
	in_doctor_manager_doctors_occupied = 0;
	in_doctor_manager_free_doctors = 0;
	in_doctor_manager_patientQueue = 0;
	in_receptionist_x = 0;
	in_receptionist_y = 0;
	in_receptionist_patientQueue = 0;
	in_receptionist_front = 0;
	in_receptionist_rear = 0;
	in_receptionist_size = 0;
	in_receptionist_tick = 0;
	in_receptionist_current_patient = 0;
	in_receptionist_attend_patient = 0;
	in_receptionist_estado = 0;
	in_agent_generator_chairs_generated = 0;
	in_agent_generator_boxes_generated = 0;
	in_agent_generator_doctors_generated = 0;
	in_chair_admin_id = 0;
	in_chair_admin_chairArray = 0;
	in_box_id = 0;
	in_box_attending = 0;
	in_box_tick = 0;
	in_doctor_id = 0;
	in_doctor_attending = 0;
	in_doctor_tick = 0;
	in_triage_front = 0;
	in_triage_rear = 0;
	in_triage_size = 0;
	in_triage_tick = 0;
	in_triage_boxArray = 0;
	in_triage_patientQueue = 0;
    in_env_EMMISION_RATE_EXIT1 = 0;
    in_env_EMMISION_RATE_EXIT2 = 0;
    in_env_EMMISION_RATE_EXIT3 = 0;
    in_env_EMMISION_RATE_EXIT4 = 0;
    in_env_EMMISION_RATE_EXIT5 = 0;
    in_env_EMMISION_RATE_EXIT6 = 0;
    in_env_EMMISION_RATE_EXIT7 = 0;
    in_env_EXIT1_PROBABILITY = 0;
    in_env_EXIT2_PROBABILITY = 0;
    in_env_EXIT3_PROBABILITY = 0;
    in_env_EXIT4_PROBABILITY = 0;
    in_env_EXIT5_PROBABILITY = 0;
    in_env_EXIT6_PROBABILITY = 0;
    in_env_EXIT7_PROBABILITY = 0;
    in_env_EXIT1_STATE = 0;
    in_env_EXIT2_STATE = 0;
    in_env_EXIT3_STATE = 0;
    in_env_EXIT4_STATE = 0;
    in_env_EXIT5_STATE = 0;
    in_env_EXIT6_STATE = 0;
    in_env_EXIT7_STATE = 0;
    in_env_EXIT1_CELL_COUNT = 0;
    in_env_EXIT2_CELL_COUNT = 0;
    in_env_EXIT3_CELL_COUNT = 0;
    in_env_EXIT4_CELL_COUNT = 0;
    in_env_EXIT5_CELL_COUNT = 0;
    in_env_EXIT6_CELL_COUNT = 0;
    in_env_EXIT7_CELL_COUNT = 0;
    in_env_TIME_SCALER = 0;
    in_env_STEER_WEIGHT = 0;
    in_env_AVOID_WEIGHT = 0;
    in_env_COLLISION_WEIGHT = 0;
    in_env_GOAL_WEIGHT = 0;
	//set all agent values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_agent_MAX; k++)
	{	
		h_agents->id[k] = 0;
		h_agents->x[k] = 0;
		h_agents->y[k] = 0;
		h_agents->velx[k] = 0;
		h_agents->vely[k] = 0;
		h_agents->steer_x[k] = 0;
		h_agents->steer_y[k] = 0;
		h_agents->height[k] = 0;
		h_agents->exit_no[k] = 0;
		h_agents->speed[k] = 0;
		h_agents->lod[k] = 0;
		h_agents->animate[k] = 0;
		h_agents->animate_dir[k] = 0;
		h_agents->estado[k] = 0;
		h_agents->tick[k] = 0;
		h_agents->estado_movimiento[k] = 0;
		h_agents->go_to_x[k] = 0;
		h_agents->go_to_y[k] = 0;
		h_agents->checkpoint[k] = 0;
		h_agents->chair_no[k] = 0;
		h_agents->box_no[k] = 0;
		h_agents->doctor_no[k] = 0;
		h_agents->priority[k] = 0;
	}
	
	//set all navmap values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_navmap_MAX; k++)
	{	
		h_navmaps->x[k] = 0;
		h_navmaps->y[k] = 0;
		h_navmaps->exit_no[k] = 0;
		h_navmaps->height[k] = 0;
		h_navmaps->collision_x[k] = 0;
		h_navmaps->collision_y[k] = 0;
		h_navmaps->exit0_x[k] = 0;
		h_navmaps->exit0_y[k] = 0;
		h_navmaps->exit1_x[k] = 0;
		h_navmaps->exit1_y[k] = 0;
		h_navmaps->exit2_x[k] = 0;
		h_navmaps->exit2_y[k] = 0;
		h_navmaps->exit3_x[k] = 0;
		h_navmaps->exit3_y[k] = 0;
		h_navmaps->exit4_x[k] = 0;
		h_navmaps->exit4_y[k] = 0;
		h_navmaps->exit5_x[k] = 0;
		h_navmaps->exit5_y[k] = 0;
		h_navmaps->exit6_x[k] = 0;
		h_navmaps->exit6_y[k] = 0;
		h_navmaps->cant_generados[k] = 0;
	}
	
	//set all chair values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_chair_MAX; k++)
	{	
		h_chairs->id[k] = 0;
		h_chairs->x[k] = 0;
		h_chairs->y[k] = 0;
		h_chairs->state[k] = 0;
	}
	
	//set all doctor_manager values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_doctor_manager_MAX; k++)
	{	
		h_doctor_managers->tick[k] = 0;
		h_doctor_managers->rear[k] = 0;
		h_doctor_managers->size[k] = 0;
        for (i=0;i<4;i++){
            h_doctor_managers->doctors_occupied[(i*xmachine_memory_doctor_manager_MAX)+k] = 0;
        }
		h_doctor_managers->free_doctors[k] = 4;
        for (i=0;i<35;i++){
            h_doctor_managers->patientQueue[(i*xmachine_memory_doctor_manager_MAX)+k] = {-1,-1};
        }
	}
	
	//set all receptionist values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_receptionist_MAX; k++)
	{	
		h_receptionists->x[k] = 0.093750;
		h_receptionists->y[k] = -0.375000;
        for (i=0;i<100;i++){
            h_receptionists->patientQueue[(i*xmachine_memory_receptionist_MAX)+k] = 0;
        }
		h_receptionists->front[k] = 0;
		h_receptionists->rear[k] = 0;
		h_receptionists->size[k] = 0;
		h_receptionists->tick[k] = 0;
		h_receptionists->current_patient[k] = -1;
		h_receptionists->attend_patient[k] = 0;
		h_receptionists->estado[k] = 0;
	}
	
	//set all agent_generator values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_agent_generator_MAX; k++)
	{	
		h_agent_generators->chairs_generated[k] = 0;
		h_agent_generators->boxes_generated[k] = 0;
		h_agent_generators->doctors_generated[k] = 0;
	}
	
	//set all chair_admin values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_chair_admin_MAX; k++)
	{	
		h_chair_admins->id[k] = 0;
        for (i=0;i<35;i++){
            h_chair_admins->chairArray[(i*xmachine_memory_chair_admin_MAX)+k] = 0;
        }
	}
	
	//set all box values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_box_MAX; k++)
	{	
		h_boxs->id[k] = 0;
		h_boxs->attending[k] = 0;
		h_boxs->tick[k] = 0;
	}
	
	//set all doctor values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_doctor_MAX; k++)
	{	
		h_doctors->id[k] = 0;
		h_doctors->attending[k] = 0;
		h_doctors->tick[k] = 0;
	}
	
	//set all triage values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_triage_MAX; k++)
	{	
		h_triages->front[k] = 0;
		h_triages->rear[k] = 0;
		h_triages->size[k] = 0;
		h_triages->tick[k] = 0;
        for (i=0;i<3;i++){
            h_triages->boxArray[(i*xmachine_memory_triage_MAX)+k] = 0;
        }
        for (i=0;i<100;i++){
            h_triages->patientQueue[(i*xmachine_memory_triage_MAX)+k] = 0;
        }
	}
	

	/* Default variables for memory */
    agent_id = 0;
    agent_x = 0;
    agent_y = 0;
    agent_velx = 0;
    agent_vely = 0;
    agent_steer_x = 0;
    agent_steer_y = 0;
    agent_height = 0;
    agent_exit_no = 0;
    agent_speed = 0;
    agent_lod = 0;
    agent_animate = 0;
    agent_animate_dir = 0;
    agent_estado = 0;
    agent_tick = 0;
    agent_estado_movimiento = 0;
    agent_go_to_x = 0;
    agent_go_to_y = 0;
    agent_checkpoint = 0;
    agent_chair_no = 0;
    agent_box_no = 0;
    agent_doctor_no = 0;
    agent_priority = 0;
    navmap_x = 0;
    navmap_y = 0;
    navmap_exit_no = 0;
    navmap_height = 0;
    navmap_collision_x = 0;
    navmap_collision_y = 0;
    navmap_exit0_x = 0;
    navmap_exit0_y = 0;
    navmap_exit1_x = 0;
    navmap_exit1_y = 0;
    navmap_exit2_x = 0;
    navmap_exit2_y = 0;
    navmap_exit3_x = 0;
    navmap_exit3_y = 0;
    navmap_exit4_x = 0;
    navmap_exit4_y = 0;
    navmap_exit5_x = 0;
    navmap_exit5_y = 0;
    navmap_exit6_x = 0;
    navmap_exit6_y = 0;
    navmap_cant_generados = 0;
    chair_id = 0;
    chair_x = 0;
    chair_y = 0;
    chair_state = 0;
    doctor_manager_tick = 0;
    doctor_manager_rear = 0;
    doctor_manager_size = 0;
    for (i=0;i<4;i++){
        doctor_manager_doctors_occupied[i] = 0;
    }
    doctor_manager_free_doctors = 4;
    for (i=0;i<35;i++){
        doctor_manager_patientQueue[i] = {-1,-1};
    }
    receptionist_x = 0.093750;
    receptionist_y = -0.375000;
    for (i=0;i<100;i++){
        receptionist_patientQueue[i] = 0;
    }
    receptionist_front = 0;
    receptionist_rear = 0;
    receptionist_size = 0;
    receptionist_tick = 0;
    receptionist_current_patient = -1;
    receptionist_attend_patient = 0;
    receptionist_estado = 0;
    agent_generator_chairs_generated = 0;
    agent_generator_boxes_generated = 0;
    agent_generator_doctors_generated = 0;
    chair_admin_id = 0;
    for (i=0;i<35;i++){
        chair_admin_chairArray[i] = 0;
    }
    box_id = 0;
    box_attending = 0;
    box_tick = 0;
    doctor_id = 0;
    doctor_attending = 0;
    doctor_tick = 0;
    triage_front = 0;
    triage_rear = 0;
    triage_size = 0;
    triage_tick = 0;
    for (i=0;i<3;i++){
        triage_boxArray[i] = 0;
    }
    for (i=0;i<100;i++){
        triage_patientQueue[i] = 0;
    }

    /* Default variables for environment variables */
    env_EMMISION_RATE_EXIT1 = 0;
    env_EMMISION_RATE_EXIT2 = 0;
    env_EMMISION_RATE_EXIT3 = 0;
    env_EMMISION_RATE_EXIT4 = 0;
    env_EMMISION_RATE_EXIT5 = 0;
    env_EMMISION_RATE_EXIT6 = 0;
    env_EMMISION_RATE_EXIT7 = 0;
    env_EXIT1_PROBABILITY = 0;
    env_EXIT2_PROBABILITY = 0;
    env_EXIT3_PROBABILITY = 0;
    env_EXIT4_PROBABILITY = 0;
    env_EXIT5_PROBABILITY = 0;
    env_EXIT6_PROBABILITY = 0;
    env_EXIT7_PROBABILITY = 0;
    env_EXIT1_STATE = 0;
    env_EXIT2_STATE = 0;
    env_EXIT3_STATE = 0;
    env_EXIT4_STATE = 0;
    env_EXIT5_STATE = 0;
    env_EXIT6_STATE = 0;
    env_EXIT7_STATE = 0;
    env_EXIT1_CELL_COUNT = 0;
    env_EXIT2_CELL_COUNT = 0;
    env_EXIT3_CELL_COUNT = 0;
    env_EXIT4_CELL_COUNT = 0;
    env_EXIT5_CELL_COUNT = 0;
    env_EXIT6_CELL_COUNT = 0;
    env_EXIT7_CELL_COUNT = 0;
    env_TIME_SCALER = 0;
    env_STEER_WEIGHT = 0;
    env_AVOID_WEIGHT = 0;
    env_COLLISION_WEIGHT = 0;
    env_GOAL_WEIGHT = 0;
    
    
    // If no input path was specified, issue a message and return.
    if(inputpath[0] == '\0'){
        printf("No initial states file specified. Using default values.\n");
        return;
    }
    
    // Otherwise an input path was specified, and we have previously checked that it is (was) not a directory. 
    
	// Attempt to open the non directory path as read only.
	file = fopen(inputpath, "r");
    
    // If the file could not be opened, issue a message and return.
    if(file == nullptr)
    {
      printf("Could not open input file %s. Continuing with default values\n", inputpath);
      return;
    }
    // Otherwise we can iterate the file until the end of XML is reached.
    size_t bytesRead = 0;
    i = 0;
	while(reading==1)
	{
        // If I exceeds our buffer size we must abort
        if(i >= bufferSize){
            fprintf(stderr, "Error: XML Parsing failed Tag name or content too long (> %d characters)\n", bufferSize);
            exit(EXIT_FAILURE);
        }

		/* Get the next char from the file */
		c = (char)fgetc(file);

        // Check if we reached the end of the file.
        if(c == EOF){
            // Break out of the loop. This allows for empty files(which may or may not be)
            break;
        }
        // Increment byte counter.
        bytesRead++;

        /*If in a  comment, look for the end of a comment */
        if(in_comment){

            /* Look for an end tag following two (or more) hyphens.
               To support very long comments, we use the minimal amount of buffer we can. 
               If we see a hyphen, store it and increment i (but don't increment i)
               If we see a > check if we have a correct terminating comment
               If we see any other characters, reset i.
            */

            if(c == '-'){
                buffer[i] = c;
                i++;
            } else if(c == '>' && i >= 2){
                in_comment = 0;
                i = 0;
            } else {
                i = 0;
            }

            /*// If we see the end tag, check the preceding two characters for a close comment, if enough characters have been read for -->
            if(c == '>' && i >= 2 && buffer[i-1] == '-' && buffer[i-2] == '-'){
                in_comment = 0;
                buffer[0] = 0;
                i = 0;
            } else {
                // Otherwise just store it in the buffer so we can keep checking for close tags
                buffer[i] = c;
                i++;
            }*/
        }
		/* If the end of a tag */
		else if(c == '>')
		{
			/* Place 0 at end of buffer to make chars a string */
			buffer[i] = 0;

			if(strcmp(buffer, "states") == 0) reading = 1;
			if(strcmp(buffer, "/states") == 0) reading = 0;
			if(strcmp(buffer, "itno") == 0) in_itno = 1;
			if(strcmp(buffer, "/itno") == 0) in_itno = 0;
            if(strcmp(buffer, "environment") == 0) in_env = 1;
            if(strcmp(buffer, "/environment") == 0) in_env = 0;
			if(strcmp(buffer, "name") == 0) in_name = 1;
			if(strcmp(buffer, "/name") == 0) in_name = 0;
            if(strcmp(buffer, "xagent") == 0) in_xagent = 1;
			if(strcmp(buffer, "/xagent") == 0)
			{
				if(strcmp(agentname, "agent") == 0)
				{
					if (*h_xmachine_memory_agent_count > xmachine_memory_agent_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent agent exceeded whilst reading data\n", xmachine_memory_agent_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_agents->id[*h_xmachine_memory_agent_count] = agent_id;
					h_agents->x[*h_xmachine_memory_agent_count] = agent_x;//Check maximum x value
                    if(agent_maximum.x < agent_x)
                        agent_maximum.x = (float)agent_x;
                    //Check minimum x value
                    if(agent_minimum.x > agent_x)
                        agent_minimum.x = (float)agent_x;
                    
					h_agents->y[*h_xmachine_memory_agent_count] = agent_y;//Check maximum y value
                    if(agent_maximum.y < agent_y)
                        agent_maximum.y = (float)agent_y;
                    //Check minimum y value
                    if(agent_minimum.y > agent_y)
                        agent_minimum.y = (float)agent_y;
                    
					h_agents->velx[*h_xmachine_memory_agent_count] = agent_velx;
					h_agents->vely[*h_xmachine_memory_agent_count] = agent_vely;
					h_agents->steer_x[*h_xmachine_memory_agent_count] = agent_steer_x;
					h_agents->steer_y[*h_xmachine_memory_agent_count] = agent_steer_y;
					h_agents->height[*h_xmachine_memory_agent_count] = agent_height;
					h_agents->exit_no[*h_xmachine_memory_agent_count] = agent_exit_no;
					h_agents->speed[*h_xmachine_memory_agent_count] = agent_speed;
					h_agents->lod[*h_xmachine_memory_agent_count] = agent_lod;
					h_agents->animate[*h_xmachine_memory_agent_count] = agent_animate;
					h_agents->animate_dir[*h_xmachine_memory_agent_count] = agent_animate_dir;
					h_agents->estado[*h_xmachine_memory_agent_count] = agent_estado;
					h_agents->tick[*h_xmachine_memory_agent_count] = agent_tick;
					h_agents->estado_movimiento[*h_xmachine_memory_agent_count] = agent_estado_movimiento;
					h_agents->go_to_x[*h_xmachine_memory_agent_count] = agent_go_to_x;
					h_agents->go_to_y[*h_xmachine_memory_agent_count] = agent_go_to_y;
					h_agents->checkpoint[*h_xmachine_memory_agent_count] = agent_checkpoint;
					h_agents->chair_no[*h_xmachine_memory_agent_count] = agent_chair_no;
					h_agents->box_no[*h_xmachine_memory_agent_count] = agent_box_no;
					h_agents->doctor_no[*h_xmachine_memory_agent_count] = agent_doctor_no;
					h_agents->priority[*h_xmachine_memory_agent_count] = agent_priority;
					(*h_xmachine_memory_agent_count) ++;	
				}
				else if(strcmp(agentname, "navmap") == 0)
				{
					if (*h_xmachine_memory_navmap_count > xmachine_memory_navmap_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent navmap exceeded whilst reading data\n", xmachine_memory_navmap_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_navmaps->x[*h_xmachine_memory_navmap_count] = navmap_x;//Check maximum x value
                    if(agent_maximum.x < navmap_x)
                        agent_maximum.x = (float)navmap_x;
                    //Check minimum x value
                    if(agent_minimum.x > navmap_x)
                        agent_minimum.x = (float)navmap_x;
                    
					h_navmaps->y[*h_xmachine_memory_navmap_count] = navmap_y;//Check maximum y value
                    if(agent_maximum.y < navmap_y)
                        agent_maximum.y = (float)navmap_y;
                    //Check minimum y value
                    if(agent_minimum.y > navmap_y)
                        agent_minimum.y = (float)navmap_y;
                    
					h_navmaps->exit_no[*h_xmachine_memory_navmap_count] = navmap_exit_no;
					h_navmaps->height[*h_xmachine_memory_navmap_count] = navmap_height;
					h_navmaps->collision_x[*h_xmachine_memory_navmap_count] = navmap_collision_x;
					h_navmaps->collision_y[*h_xmachine_memory_navmap_count] = navmap_collision_y;
					h_navmaps->exit0_x[*h_xmachine_memory_navmap_count] = navmap_exit0_x;
					h_navmaps->exit0_y[*h_xmachine_memory_navmap_count] = navmap_exit0_y;
					h_navmaps->exit1_x[*h_xmachine_memory_navmap_count] = navmap_exit1_x;
					h_navmaps->exit1_y[*h_xmachine_memory_navmap_count] = navmap_exit1_y;
					h_navmaps->exit2_x[*h_xmachine_memory_navmap_count] = navmap_exit2_x;
					h_navmaps->exit2_y[*h_xmachine_memory_navmap_count] = navmap_exit2_y;
					h_navmaps->exit3_x[*h_xmachine_memory_navmap_count] = navmap_exit3_x;
					h_navmaps->exit3_y[*h_xmachine_memory_navmap_count] = navmap_exit3_y;
					h_navmaps->exit4_x[*h_xmachine_memory_navmap_count] = navmap_exit4_x;
					h_navmaps->exit4_y[*h_xmachine_memory_navmap_count] = navmap_exit4_y;
					h_navmaps->exit5_x[*h_xmachine_memory_navmap_count] = navmap_exit5_x;
					h_navmaps->exit5_y[*h_xmachine_memory_navmap_count] = navmap_exit5_y;
					h_navmaps->exit6_x[*h_xmachine_memory_navmap_count] = navmap_exit6_x;
					h_navmaps->exit6_y[*h_xmachine_memory_navmap_count] = navmap_exit6_y;
					h_navmaps->cant_generados[*h_xmachine_memory_navmap_count] = navmap_cant_generados;
					(*h_xmachine_memory_navmap_count) ++;	
				}
				else if(strcmp(agentname, "chair") == 0)
				{
					if (*h_xmachine_memory_chair_count > xmachine_memory_chair_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent chair exceeded whilst reading data\n", xmachine_memory_chair_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_chairs->id[*h_xmachine_memory_chair_count] = chair_id;
					h_chairs->x[*h_xmachine_memory_chair_count] = chair_x;//Check maximum x value
                    if(agent_maximum.x < chair_x)
                        agent_maximum.x = (float)chair_x;
                    //Check minimum x value
                    if(agent_minimum.x > chair_x)
                        agent_minimum.x = (float)chair_x;
                    
					h_chairs->y[*h_xmachine_memory_chair_count] = chair_y;//Check maximum y value
                    if(agent_maximum.y < chair_y)
                        agent_maximum.y = (float)chair_y;
                    //Check minimum y value
                    if(agent_minimum.y > chair_y)
                        agent_minimum.y = (float)chair_y;
                    
					h_chairs->state[*h_xmachine_memory_chair_count] = chair_state;
					(*h_xmachine_memory_chair_count) ++;	
				}
				else if(strcmp(agentname, "doctor_manager") == 0)
				{
					if (*h_xmachine_memory_doctor_manager_count > xmachine_memory_doctor_manager_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent doctor_manager exceeded whilst reading data\n", xmachine_memory_doctor_manager_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_doctor_managers->tick[*h_xmachine_memory_doctor_manager_count] = doctor_manager_tick;
					h_doctor_managers->rear[*h_xmachine_memory_doctor_manager_count] = doctor_manager_rear;
					h_doctor_managers->size[*h_xmachine_memory_doctor_manager_count] = doctor_manager_size;
                    for (int k=0;k<4;k++){
                        h_doctor_managers->doctors_occupied[(k*xmachine_memory_doctor_manager_MAX)+(*h_xmachine_memory_doctor_manager_count)] = doctor_manager_doctors_occupied[k];
                    }
					h_doctor_managers->free_doctors[*h_xmachine_memory_doctor_manager_count] = doctor_manager_free_doctors;
                    for (int k=0;k<35;k++){
                        h_doctor_managers->patientQueue[(k*xmachine_memory_doctor_manager_MAX)+(*h_xmachine_memory_doctor_manager_count)] = doctor_manager_patientQueue[k];
                    }
					(*h_xmachine_memory_doctor_manager_count) ++;	
				}
				else if(strcmp(agentname, "receptionist") == 0)
				{
					if (*h_xmachine_memory_receptionist_count > xmachine_memory_receptionist_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent receptionist exceeded whilst reading data\n", xmachine_memory_receptionist_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_receptionists->x[*h_xmachine_memory_receptionist_count] = receptionist_x;//Check maximum x value
                    if(agent_maximum.x < receptionist_x)
                        agent_maximum.x = (float)receptionist_x;
                    //Check minimum x value
                    if(agent_minimum.x > receptionist_x)
                        agent_minimum.x = (float)receptionist_x;
                    
					h_receptionists->y[*h_xmachine_memory_receptionist_count] = receptionist_y;//Check maximum y value
                    if(agent_maximum.y < receptionist_y)
                        agent_maximum.y = (float)receptionist_y;
                    //Check minimum y value
                    if(agent_minimum.y > receptionist_y)
                        agent_minimum.y = (float)receptionist_y;
                    
                    for (int k=0;k<100;k++){
                        h_receptionists->patientQueue[(k*xmachine_memory_receptionist_MAX)+(*h_xmachine_memory_receptionist_count)] = receptionist_patientQueue[k];
                    }
					h_receptionists->front[*h_xmachine_memory_receptionist_count] = receptionist_front;
					h_receptionists->rear[*h_xmachine_memory_receptionist_count] = receptionist_rear;
					h_receptionists->size[*h_xmachine_memory_receptionist_count] = receptionist_size;
					h_receptionists->tick[*h_xmachine_memory_receptionist_count] = receptionist_tick;
					h_receptionists->current_patient[*h_xmachine_memory_receptionist_count] = receptionist_current_patient;
					h_receptionists->attend_patient[*h_xmachine_memory_receptionist_count] = receptionist_attend_patient;
					h_receptionists->estado[*h_xmachine_memory_receptionist_count] = receptionist_estado;
					(*h_xmachine_memory_receptionist_count) ++;	
				}
				else if(strcmp(agentname, "agent_generator") == 0)
				{
					if (*h_xmachine_memory_agent_generator_count > xmachine_memory_agent_generator_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent agent_generator exceeded whilst reading data\n", xmachine_memory_agent_generator_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_agent_generators->chairs_generated[*h_xmachine_memory_agent_generator_count] = agent_generator_chairs_generated;
					h_agent_generators->boxes_generated[*h_xmachine_memory_agent_generator_count] = agent_generator_boxes_generated;
					h_agent_generators->doctors_generated[*h_xmachine_memory_agent_generator_count] = agent_generator_doctors_generated;
					(*h_xmachine_memory_agent_generator_count) ++;	
				}
				else if(strcmp(agentname, "chair_admin") == 0)
				{
					if (*h_xmachine_memory_chair_admin_count > xmachine_memory_chair_admin_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent chair_admin exceeded whilst reading data\n", xmachine_memory_chair_admin_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_chair_admins->id[*h_xmachine_memory_chair_admin_count] = chair_admin_id;
                    for (int k=0;k<35;k++){
                        h_chair_admins->chairArray[(k*xmachine_memory_chair_admin_MAX)+(*h_xmachine_memory_chair_admin_count)] = chair_admin_chairArray[k];
                    }
					(*h_xmachine_memory_chair_admin_count) ++;	
				}
				else if(strcmp(agentname, "box") == 0)
				{
					if (*h_xmachine_memory_box_count > xmachine_memory_box_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent box exceeded whilst reading data\n", xmachine_memory_box_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_boxs->id[*h_xmachine_memory_box_count] = box_id;
					h_boxs->attending[*h_xmachine_memory_box_count] = box_attending;
					h_boxs->tick[*h_xmachine_memory_box_count] = box_tick;
					(*h_xmachine_memory_box_count) ++;	
				}
				else if(strcmp(agentname, "doctor") == 0)
				{
					if (*h_xmachine_memory_doctor_count > xmachine_memory_doctor_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent doctor exceeded whilst reading data\n", xmachine_memory_doctor_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_doctors->id[*h_xmachine_memory_doctor_count] = doctor_id;
					h_doctors->attending[*h_xmachine_memory_doctor_count] = doctor_attending;
					h_doctors->tick[*h_xmachine_memory_doctor_count] = doctor_tick;
					(*h_xmachine_memory_doctor_count) ++;	
				}
				else if(strcmp(agentname, "triage") == 0)
				{
					if (*h_xmachine_memory_triage_count > xmachine_memory_triage_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent triage exceeded whilst reading data\n", xmachine_memory_triage_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_triages->front[*h_xmachine_memory_triage_count] = triage_front;
					h_triages->rear[*h_xmachine_memory_triage_count] = triage_rear;
					h_triages->size[*h_xmachine_memory_triage_count] = triage_size;
					h_triages->tick[*h_xmachine_memory_triage_count] = triage_tick;
                    for (int k=0;k<3;k++){
                        h_triages->boxArray[(k*xmachine_memory_triage_MAX)+(*h_xmachine_memory_triage_count)] = triage_boxArray[k];
                    }
                    for (int k=0;k<100;k++){
                        h_triages->patientQueue[(k*xmachine_memory_triage_MAX)+(*h_xmachine_memory_triage_count)] = triage_patientQueue[k];
                    }
					(*h_xmachine_memory_triage_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}



				/* Reset xagent variables */
                agent_id = 0;
                agent_x = 0;
                agent_y = 0;
                agent_velx = 0;
                agent_vely = 0;
                agent_steer_x = 0;
                agent_steer_y = 0;
                agent_height = 0;
                agent_exit_no = 0;
                agent_speed = 0;
                agent_lod = 0;
                agent_animate = 0;
                agent_animate_dir = 0;
                agent_estado = 0;
                agent_tick = 0;
                agent_estado_movimiento = 0;
                agent_go_to_x = 0;
                agent_go_to_y = 0;
                agent_checkpoint = 0;
                agent_chair_no = 0;
                agent_box_no = 0;
                agent_doctor_no = 0;
                agent_priority = 0;
                navmap_x = 0;
                navmap_y = 0;
                navmap_exit_no = 0;
                navmap_height = 0;
                navmap_collision_x = 0;
                navmap_collision_y = 0;
                navmap_exit0_x = 0;
                navmap_exit0_y = 0;
                navmap_exit1_x = 0;
                navmap_exit1_y = 0;
                navmap_exit2_x = 0;
                navmap_exit2_y = 0;
                navmap_exit3_x = 0;
                navmap_exit3_y = 0;
                navmap_exit4_x = 0;
                navmap_exit4_y = 0;
                navmap_exit5_x = 0;
                navmap_exit5_y = 0;
                navmap_exit6_x = 0;
                navmap_exit6_y = 0;
                navmap_cant_generados = 0;
                chair_id = 0;
                chair_x = 0;
                chair_y = 0;
                chair_state = 0;
                doctor_manager_tick = 0;
                doctor_manager_rear = 0;
                doctor_manager_size = 0;
                for (i=0;i<4;i++){
                    doctor_manager_doctors_occupied[i] = 0;
                }
                doctor_manager_free_doctors = 4;
                for (i=0;i<35;i++){
                    doctor_manager_patientQueue[i] = {-1,-1};
                }
                receptionist_x = 0.093750;
                receptionist_y = -0.375000;
                for (i=0;i<100;i++){
                    receptionist_patientQueue[i] = 0;
                }
                receptionist_front = 0;
                receptionist_rear = 0;
                receptionist_size = 0;
                receptionist_tick = 0;
                receptionist_current_patient = -1;
                receptionist_attend_patient = 0;
                receptionist_estado = 0;
                agent_generator_chairs_generated = 0;
                agent_generator_boxes_generated = 0;
                agent_generator_doctors_generated = 0;
                chair_admin_id = 0;
                for (i=0;i<35;i++){
                    chair_admin_chairArray[i] = 0;
                }
                box_id = 0;
                box_attending = 0;
                box_tick = 0;
                doctor_id = 0;
                doctor_attending = 0;
                doctor_tick = 0;
                triage_front = 0;
                triage_rear = 0;
                triage_size = 0;
                triage_tick = 0;
                for (i=0;i<3;i++){
                    triage_boxArray[i] = 0;
                }
                for (i=0;i<100;i++){
                    triage_patientQueue[i] = 0;
                }
                
                in_xagent = 0;
			}
			if(strcmp(buffer, "id") == 0) in_agent_id = 1;
			if(strcmp(buffer, "/id") == 0) in_agent_id = 0;
			if(strcmp(buffer, "x") == 0) in_agent_x = 1;
			if(strcmp(buffer, "/x") == 0) in_agent_x = 0;
			if(strcmp(buffer, "y") == 0) in_agent_y = 1;
			if(strcmp(buffer, "/y") == 0) in_agent_y = 0;
			if(strcmp(buffer, "velx") == 0) in_agent_velx = 1;
			if(strcmp(buffer, "/velx") == 0) in_agent_velx = 0;
			if(strcmp(buffer, "vely") == 0) in_agent_vely = 1;
			if(strcmp(buffer, "/vely") == 0) in_agent_vely = 0;
			if(strcmp(buffer, "steer_x") == 0) in_agent_steer_x = 1;
			if(strcmp(buffer, "/steer_x") == 0) in_agent_steer_x = 0;
			if(strcmp(buffer, "steer_y") == 0) in_agent_steer_y = 1;
			if(strcmp(buffer, "/steer_y") == 0) in_agent_steer_y = 0;
			if(strcmp(buffer, "height") == 0) in_agent_height = 1;
			if(strcmp(buffer, "/height") == 0) in_agent_height = 0;
			if(strcmp(buffer, "exit_no") == 0) in_agent_exit_no = 1;
			if(strcmp(buffer, "/exit_no") == 0) in_agent_exit_no = 0;
			if(strcmp(buffer, "speed") == 0) in_agent_speed = 1;
			if(strcmp(buffer, "/speed") == 0) in_agent_speed = 0;
			if(strcmp(buffer, "lod") == 0) in_agent_lod = 1;
			if(strcmp(buffer, "/lod") == 0) in_agent_lod = 0;
			if(strcmp(buffer, "animate") == 0) in_agent_animate = 1;
			if(strcmp(buffer, "/animate") == 0) in_agent_animate = 0;
			if(strcmp(buffer, "animate_dir") == 0) in_agent_animate_dir = 1;
			if(strcmp(buffer, "/animate_dir") == 0) in_agent_animate_dir = 0;
			if(strcmp(buffer, "estado") == 0) in_agent_estado = 1;
			if(strcmp(buffer, "/estado") == 0) in_agent_estado = 0;
			if(strcmp(buffer, "tick") == 0) in_agent_tick = 1;
			if(strcmp(buffer, "/tick") == 0) in_agent_tick = 0;
			if(strcmp(buffer, "estado_movimiento") == 0) in_agent_estado_movimiento = 1;
			if(strcmp(buffer, "/estado_movimiento") == 0) in_agent_estado_movimiento = 0;
			if(strcmp(buffer, "go_to_x") == 0) in_agent_go_to_x = 1;
			if(strcmp(buffer, "/go_to_x") == 0) in_agent_go_to_x = 0;
			if(strcmp(buffer, "go_to_y") == 0) in_agent_go_to_y = 1;
			if(strcmp(buffer, "/go_to_y") == 0) in_agent_go_to_y = 0;
			if(strcmp(buffer, "checkpoint") == 0) in_agent_checkpoint = 1;
			if(strcmp(buffer, "/checkpoint") == 0) in_agent_checkpoint = 0;
			if(strcmp(buffer, "chair_no") == 0) in_agent_chair_no = 1;
			if(strcmp(buffer, "/chair_no") == 0) in_agent_chair_no = 0;
			if(strcmp(buffer, "box_no") == 0) in_agent_box_no = 1;
			if(strcmp(buffer, "/box_no") == 0) in_agent_box_no = 0;
			if(strcmp(buffer, "doctor_no") == 0) in_agent_doctor_no = 1;
			if(strcmp(buffer, "/doctor_no") == 0) in_agent_doctor_no = 0;
			if(strcmp(buffer, "priority") == 0) in_agent_priority = 1;
			if(strcmp(buffer, "/priority") == 0) in_agent_priority = 0;
			if(strcmp(buffer, "x") == 0) in_navmap_x = 1;
			if(strcmp(buffer, "/x") == 0) in_navmap_x = 0;
			if(strcmp(buffer, "y") == 0) in_navmap_y = 1;
			if(strcmp(buffer, "/y") == 0) in_navmap_y = 0;
			if(strcmp(buffer, "exit_no") == 0) in_navmap_exit_no = 1;
			if(strcmp(buffer, "/exit_no") == 0) in_navmap_exit_no = 0;
			if(strcmp(buffer, "height") == 0) in_navmap_height = 1;
			if(strcmp(buffer, "/height") == 0) in_navmap_height = 0;
			if(strcmp(buffer, "collision_x") == 0) in_navmap_collision_x = 1;
			if(strcmp(buffer, "/collision_x") == 0) in_navmap_collision_x = 0;
			if(strcmp(buffer, "collision_y") == 0) in_navmap_collision_y = 1;
			if(strcmp(buffer, "/collision_y") == 0) in_navmap_collision_y = 0;
			if(strcmp(buffer, "exit0_x") == 0) in_navmap_exit0_x = 1;
			if(strcmp(buffer, "/exit0_x") == 0) in_navmap_exit0_x = 0;
			if(strcmp(buffer, "exit0_y") == 0) in_navmap_exit0_y = 1;
			if(strcmp(buffer, "/exit0_y") == 0) in_navmap_exit0_y = 0;
			if(strcmp(buffer, "exit1_x") == 0) in_navmap_exit1_x = 1;
			if(strcmp(buffer, "/exit1_x") == 0) in_navmap_exit1_x = 0;
			if(strcmp(buffer, "exit1_y") == 0) in_navmap_exit1_y = 1;
			if(strcmp(buffer, "/exit1_y") == 0) in_navmap_exit1_y = 0;
			if(strcmp(buffer, "exit2_x") == 0) in_navmap_exit2_x = 1;
			if(strcmp(buffer, "/exit2_x") == 0) in_navmap_exit2_x = 0;
			if(strcmp(buffer, "exit2_y") == 0) in_navmap_exit2_y = 1;
			if(strcmp(buffer, "/exit2_y") == 0) in_navmap_exit2_y = 0;
			if(strcmp(buffer, "exit3_x") == 0) in_navmap_exit3_x = 1;
			if(strcmp(buffer, "/exit3_x") == 0) in_navmap_exit3_x = 0;
			if(strcmp(buffer, "exit3_y") == 0) in_navmap_exit3_y = 1;
			if(strcmp(buffer, "/exit3_y") == 0) in_navmap_exit3_y = 0;
			if(strcmp(buffer, "exit4_x") == 0) in_navmap_exit4_x = 1;
			if(strcmp(buffer, "/exit4_x") == 0) in_navmap_exit4_x = 0;
			if(strcmp(buffer, "exit4_y") == 0) in_navmap_exit4_y = 1;
			if(strcmp(buffer, "/exit4_y") == 0) in_navmap_exit4_y = 0;
			if(strcmp(buffer, "exit5_x") == 0) in_navmap_exit5_x = 1;
			if(strcmp(buffer, "/exit5_x") == 0) in_navmap_exit5_x = 0;
			if(strcmp(buffer, "exit5_y") == 0) in_navmap_exit5_y = 1;
			if(strcmp(buffer, "/exit5_y") == 0) in_navmap_exit5_y = 0;
			if(strcmp(buffer, "exit6_x") == 0) in_navmap_exit6_x = 1;
			if(strcmp(buffer, "/exit6_x") == 0) in_navmap_exit6_x = 0;
			if(strcmp(buffer, "exit6_y") == 0) in_navmap_exit6_y = 1;
			if(strcmp(buffer, "/exit6_y") == 0) in_navmap_exit6_y = 0;
			if(strcmp(buffer, "cant_generados") == 0) in_navmap_cant_generados = 1;
			if(strcmp(buffer, "/cant_generados") == 0) in_navmap_cant_generados = 0;
			if(strcmp(buffer, "id") == 0) in_chair_id = 1;
			if(strcmp(buffer, "/id") == 0) in_chair_id = 0;
			if(strcmp(buffer, "x") == 0) in_chair_x = 1;
			if(strcmp(buffer, "/x") == 0) in_chair_x = 0;
			if(strcmp(buffer, "y") == 0) in_chair_y = 1;
			if(strcmp(buffer, "/y") == 0) in_chair_y = 0;
			if(strcmp(buffer, "state") == 0) in_chair_state = 1;
			if(strcmp(buffer, "/state") == 0) in_chair_state = 0;
			if(strcmp(buffer, "tick") == 0) in_doctor_manager_tick = 1;
			if(strcmp(buffer, "/tick") == 0) in_doctor_manager_tick = 0;
			if(strcmp(buffer, "rear") == 0) in_doctor_manager_rear = 1;
			if(strcmp(buffer, "/rear") == 0) in_doctor_manager_rear = 0;
			if(strcmp(buffer, "size") == 0) in_doctor_manager_size = 1;
			if(strcmp(buffer, "/size") == 0) in_doctor_manager_size = 0;
			if(strcmp(buffer, "doctors_occupied") == 0) in_doctor_manager_doctors_occupied = 1;
			if(strcmp(buffer, "/doctors_occupied") == 0) in_doctor_manager_doctors_occupied = 0;
			if(strcmp(buffer, "free_doctors") == 0) in_doctor_manager_free_doctors = 1;
			if(strcmp(buffer, "/free_doctors") == 0) in_doctor_manager_free_doctors = 0;
			if(strcmp(buffer, "patientQueue") == 0) in_doctor_manager_patientQueue = 1;
			if(strcmp(buffer, "/patientQueue") == 0) in_doctor_manager_patientQueue = 0;
			if(strcmp(buffer, "x") == 0) in_receptionist_x = 1;
			if(strcmp(buffer, "/x") == 0) in_receptionist_x = 0;
			if(strcmp(buffer, "y") == 0) in_receptionist_y = 1;
			if(strcmp(buffer, "/y") == 0) in_receptionist_y = 0;
			if(strcmp(buffer, "patientQueue") == 0) in_receptionist_patientQueue = 1;
			if(strcmp(buffer, "/patientQueue") == 0) in_receptionist_patientQueue = 0;
			if(strcmp(buffer, "front") == 0) in_receptionist_front = 1;
			if(strcmp(buffer, "/front") == 0) in_receptionist_front = 0;
			if(strcmp(buffer, "rear") == 0) in_receptionist_rear = 1;
			if(strcmp(buffer, "/rear") == 0) in_receptionist_rear = 0;
			if(strcmp(buffer, "size") == 0) in_receptionist_size = 1;
			if(strcmp(buffer, "/size") == 0) in_receptionist_size = 0;
			if(strcmp(buffer, "tick") == 0) in_receptionist_tick = 1;
			if(strcmp(buffer, "/tick") == 0) in_receptionist_tick = 0;
			if(strcmp(buffer, "current_patient") == 0) in_receptionist_current_patient = 1;
			if(strcmp(buffer, "/current_patient") == 0) in_receptionist_current_patient = 0;
			if(strcmp(buffer, "attend_patient") == 0) in_receptionist_attend_patient = 1;
			if(strcmp(buffer, "/attend_patient") == 0) in_receptionist_attend_patient = 0;
			if(strcmp(buffer, "estado") == 0) in_receptionist_estado = 1;
			if(strcmp(buffer, "/estado") == 0) in_receptionist_estado = 0;
			if(strcmp(buffer, "chairs_generated") == 0) in_agent_generator_chairs_generated = 1;
			if(strcmp(buffer, "/chairs_generated") == 0) in_agent_generator_chairs_generated = 0;
			if(strcmp(buffer, "boxes_generated") == 0) in_agent_generator_boxes_generated = 1;
			if(strcmp(buffer, "/boxes_generated") == 0) in_agent_generator_boxes_generated = 0;
			if(strcmp(buffer, "doctors_generated") == 0) in_agent_generator_doctors_generated = 1;
			if(strcmp(buffer, "/doctors_generated") == 0) in_agent_generator_doctors_generated = 0;
			if(strcmp(buffer, "id") == 0) in_chair_admin_id = 1;
			if(strcmp(buffer, "/id") == 0) in_chair_admin_id = 0;
			if(strcmp(buffer, "chairArray") == 0) in_chair_admin_chairArray = 1;
			if(strcmp(buffer, "/chairArray") == 0) in_chair_admin_chairArray = 0;
			if(strcmp(buffer, "id") == 0) in_box_id = 1;
			if(strcmp(buffer, "/id") == 0) in_box_id = 0;
			if(strcmp(buffer, "attending") == 0) in_box_attending = 1;
			if(strcmp(buffer, "/attending") == 0) in_box_attending = 0;
			if(strcmp(buffer, "tick") == 0) in_box_tick = 1;
			if(strcmp(buffer, "/tick") == 0) in_box_tick = 0;
			if(strcmp(buffer, "id") == 0) in_doctor_id = 1;
			if(strcmp(buffer, "/id") == 0) in_doctor_id = 0;
			if(strcmp(buffer, "attending") == 0) in_doctor_attending = 1;
			if(strcmp(buffer, "/attending") == 0) in_doctor_attending = 0;
			if(strcmp(buffer, "tick") == 0) in_doctor_tick = 1;
			if(strcmp(buffer, "/tick") == 0) in_doctor_tick = 0;
			if(strcmp(buffer, "front") == 0) in_triage_front = 1;
			if(strcmp(buffer, "/front") == 0) in_triage_front = 0;
			if(strcmp(buffer, "rear") == 0) in_triage_rear = 1;
			if(strcmp(buffer, "/rear") == 0) in_triage_rear = 0;
			if(strcmp(buffer, "size") == 0) in_triage_size = 1;
			if(strcmp(buffer, "/size") == 0) in_triage_size = 0;
			if(strcmp(buffer, "tick") == 0) in_triage_tick = 1;
			if(strcmp(buffer, "/tick") == 0) in_triage_tick = 0;
			if(strcmp(buffer, "boxArray") == 0) in_triage_boxArray = 1;
			if(strcmp(buffer, "/boxArray") == 0) in_triage_boxArray = 0;
			if(strcmp(buffer, "patientQueue") == 0) in_triage_patientQueue = 1;
			if(strcmp(buffer, "/patientQueue") == 0) in_triage_patientQueue = 0;
			
            /* environment variables */
            if(strcmp(buffer, "EMMISION_RATE_EXIT1") == 0) in_env_EMMISION_RATE_EXIT1 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT1") == 0) in_env_EMMISION_RATE_EXIT1 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT2") == 0) in_env_EMMISION_RATE_EXIT2 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT2") == 0) in_env_EMMISION_RATE_EXIT2 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT3") == 0) in_env_EMMISION_RATE_EXIT3 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT3") == 0) in_env_EMMISION_RATE_EXIT3 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT4") == 0) in_env_EMMISION_RATE_EXIT4 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT4") == 0) in_env_EMMISION_RATE_EXIT4 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT5") == 0) in_env_EMMISION_RATE_EXIT5 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT5") == 0) in_env_EMMISION_RATE_EXIT5 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT6") == 0) in_env_EMMISION_RATE_EXIT6 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT6") == 0) in_env_EMMISION_RATE_EXIT6 = 0;
			if(strcmp(buffer, "EMMISION_RATE_EXIT7") == 0) in_env_EMMISION_RATE_EXIT7 = 1;
            if(strcmp(buffer, "/EMMISION_RATE_EXIT7") == 0) in_env_EMMISION_RATE_EXIT7 = 0;
			if(strcmp(buffer, "EXIT1_PROBABILITY") == 0) in_env_EXIT1_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT1_PROBABILITY") == 0) in_env_EXIT1_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT2_PROBABILITY") == 0) in_env_EXIT2_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT2_PROBABILITY") == 0) in_env_EXIT2_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT3_PROBABILITY") == 0) in_env_EXIT3_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT3_PROBABILITY") == 0) in_env_EXIT3_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT4_PROBABILITY") == 0) in_env_EXIT4_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT4_PROBABILITY") == 0) in_env_EXIT4_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT5_PROBABILITY") == 0) in_env_EXIT5_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT5_PROBABILITY") == 0) in_env_EXIT5_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT6_PROBABILITY") == 0) in_env_EXIT6_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT6_PROBABILITY") == 0) in_env_EXIT6_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT7_PROBABILITY") == 0) in_env_EXIT7_PROBABILITY = 1;
            if(strcmp(buffer, "/EXIT7_PROBABILITY") == 0) in_env_EXIT7_PROBABILITY = 0;
			if(strcmp(buffer, "EXIT1_STATE") == 0) in_env_EXIT1_STATE = 1;
            if(strcmp(buffer, "/EXIT1_STATE") == 0) in_env_EXIT1_STATE = 0;
			if(strcmp(buffer, "EXIT2_STATE") == 0) in_env_EXIT2_STATE = 1;
            if(strcmp(buffer, "/EXIT2_STATE") == 0) in_env_EXIT2_STATE = 0;
			if(strcmp(buffer, "EXIT3_STATE") == 0) in_env_EXIT3_STATE = 1;
            if(strcmp(buffer, "/EXIT3_STATE") == 0) in_env_EXIT3_STATE = 0;
			if(strcmp(buffer, "EXIT4_STATE") == 0) in_env_EXIT4_STATE = 1;
            if(strcmp(buffer, "/EXIT4_STATE") == 0) in_env_EXIT4_STATE = 0;
			if(strcmp(buffer, "EXIT5_STATE") == 0) in_env_EXIT5_STATE = 1;
            if(strcmp(buffer, "/EXIT5_STATE") == 0) in_env_EXIT5_STATE = 0;
			if(strcmp(buffer, "EXIT6_STATE") == 0) in_env_EXIT6_STATE = 1;
            if(strcmp(buffer, "/EXIT6_STATE") == 0) in_env_EXIT6_STATE = 0;
			if(strcmp(buffer, "EXIT7_STATE") == 0) in_env_EXIT7_STATE = 1;
            if(strcmp(buffer, "/EXIT7_STATE") == 0) in_env_EXIT7_STATE = 0;
			if(strcmp(buffer, "EXIT1_CELL_COUNT") == 0) in_env_EXIT1_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT1_CELL_COUNT") == 0) in_env_EXIT1_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT2_CELL_COUNT") == 0) in_env_EXIT2_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT2_CELL_COUNT") == 0) in_env_EXIT2_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT3_CELL_COUNT") == 0) in_env_EXIT3_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT3_CELL_COUNT") == 0) in_env_EXIT3_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT4_CELL_COUNT") == 0) in_env_EXIT4_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT4_CELL_COUNT") == 0) in_env_EXIT4_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT5_CELL_COUNT") == 0) in_env_EXIT5_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT5_CELL_COUNT") == 0) in_env_EXIT5_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT6_CELL_COUNT") == 0) in_env_EXIT6_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT6_CELL_COUNT") == 0) in_env_EXIT6_CELL_COUNT = 0;
			if(strcmp(buffer, "EXIT7_CELL_COUNT") == 0) in_env_EXIT7_CELL_COUNT = 1;
            if(strcmp(buffer, "/EXIT7_CELL_COUNT") == 0) in_env_EXIT7_CELL_COUNT = 0;
			if(strcmp(buffer, "TIME_SCALER") == 0) in_env_TIME_SCALER = 1;
            if(strcmp(buffer, "/TIME_SCALER") == 0) in_env_TIME_SCALER = 0;
			if(strcmp(buffer, "STEER_WEIGHT") == 0) in_env_STEER_WEIGHT = 1;
            if(strcmp(buffer, "/STEER_WEIGHT") == 0) in_env_STEER_WEIGHT = 0;
			if(strcmp(buffer, "AVOID_WEIGHT") == 0) in_env_AVOID_WEIGHT = 1;
            if(strcmp(buffer, "/AVOID_WEIGHT") == 0) in_env_AVOID_WEIGHT = 0;
			if(strcmp(buffer, "COLLISION_WEIGHT") == 0) in_env_COLLISION_WEIGHT = 1;
            if(strcmp(buffer, "/COLLISION_WEIGHT") == 0) in_env_COLLISION_WEIGHT = 0;
			if(strcmp(buffer, "GOAL_WEIGHT") == 0) in_env_GOAL_WEIGHT = 1;
            if(strcmp(buffer, "/GOAL_WEIGHT") == 0) in_env_GOAL_WEIGHT = 0;
			

			/* End of tag and reset buffer */
			in_tag = 0;
			i = 0;
		}
		/* If start of tag */
		else if(c == '<')
		{
			/* Place /0 at end of buffer to end numbers */
			buffer[i] = 0;
			/* Flag in tag */
			in_tag = 1;

			if(in_itno) *itno = atoi(buffer);
			if(in_name) strcpy(agentname, buffer);
			else if (in_xagent)
			{
				if(in_agent_id){
                    agent_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_agent_x){
                    agent_x = (float) fgpu_atof(buffer); 
                }
				if(in_agent_y){
                    agent_y = (float) fgpu_atof(buffer); 
                }
				if(in_agent_velx){
                    agent_velx = (float) fgpu_atof(buffer); 
                }
				if(in_agent_vely){
                    agent_vely = (float) fgpu_atof(buffer); 
                }
				if(in_agent_steer_x){
                    agent_steer_x = (float) fgpu_atof(buffer); 
                }
				if(in_agent_steer_y){
                    agent_steer_y = (float) fgpu_atof(buffer); 
                }
				if(in_agent_height){
                    agent_height = (float) fgpu_atof(buffer); 
                }
				if(in_agent_exit_no){
                    agent_exit_no = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_speed){
                    agent_speed = (float) fgpu_atof(buffer); 
                }
				if(in_agent_lod){
                    agent_lod = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_animate){
                    agent_animate = (float) fgpu_atof(buffer); 
                }
				if(in_agent_animate_dir){
                    agent_animate_dir = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_estado){
                    agent_estado = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_tick){
                    agent_tick = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_estado_movimiento){
                    agent_estado_movimiento = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_agent_go_to_x){
                    agent_go_to_x = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_agent_go_to_y){
                    agent_go_to_y = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_agent_checkpoint){
                    agent_checkpoint = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_agent_chair_no){
                    agent_chair_no = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_box_no){
                    agent_box_no = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_agent_doctor_no){
                    agent_doctor_no = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_agent_priority){
                    agent_priority = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_navmap_x){
                    navmap_x = (int) fpgu_strtol(buffer); 
                }
				if(in_navmap_y){
                    navmap_y = (int) fpgu_strtol(buffer); 
                }
				if(in_navmap_exit_no){
                    navmap_exit_no = (int) fpgu_strtol(buffer); 
                }
				if(in_navmap_height){
                    navmap_height = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_collision_x){
                    navmap_collision_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_collision_y){
                    navmap_collision_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit0_x){
                    navmap_exit0_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit0_y){
                    navmap_exit0_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit1_x){
                    navmap_exit1_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit1_y){
                    navmap_exit1_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit2_x){
                    navmap_exit2_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit2_y){
                    navmap_exit2_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit3_x){
                    navmap_exit3_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit3_y){
                    navmap_exit3_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit4_x){
                    navmap_exit4_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit4_y){
                    navmap_exit4_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit5_x){
                    navmap_exit5_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit5_y){
                    navmap_exit5_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit6_x){
                    navmap_exit6_x = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_exit6_y){
                    navmap_exit6_y = (float) fgpu_atof(buffer); 
                }
				if(in_navmap_cant_generados){
                    navmap_cant_generados = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_chair_id){
                    chair_id = (int) fpgu_strtol(buffer); 
                }
				if(in_chair_x){
                    chair_x = (int) fpgu_strtol(buffer); 
                }
				if(in_chair_y){
                    chair_y = (int) fpgu_strtol(buffer); 
                }
				if(in_chair_state){
                    chair_state = (int) fpgu_strtol(buffer); 
                }
				if(in_doctor_manager_tick){
                    doctor_manager_tick = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_doctor_manager_rear){
                    doctor_manager_rear = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_doctor_manager_size){
                    doctor_manager_size = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_doctor_manager_doctors_occupied){
                    readArrayInput<int>(&fpgu_strtol, buffer, doctor_manager_doctors_occupied, 4);    
                }
				if(in_doctor_manager_free_doctors){
                    doctor_manager_free_doctors = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_doctor_manager_patientQueue){
                    readArrayInputVectorType<ivec2, int, 2>(&fpgu_strtol, buffer, doctor_manager_patientQueue, 35);    
                }
				if(in_receptionist_x){
                    receptionist_x = (int) fpgu_strtol(buffer); 
                }
				if(in_receptionist_y){
                    receptionist_y = (int) fpgu_strtol(buffer); 
                }
				if(in_receptionist_patientQueue){
                    readArrayInput<unsigned int>(&fpgu_strtoul, buffer, receptionist_patientQueue, 100);    
                }
				if(in_receptionist_front){
                    receptionist_front = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_receptionist_rear){
                    receptionist_rear = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_receptionist_size){
                    receptionist_size = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_receptionist_tick){
                    receptionist_tick = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_receptionist_current_patient){
                    receptionist_current_patient = (int) fpgu_strtol(buffer); 
                }
				if(in_receptionist_attend_patient){
                    receptionist_attend_patient = (int) fpgu_strtol(buffer); 
                }
				if(in_receptionist_estado){
                    receptionist_estado = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_generator_chairs_generated){
                    agent_generator_chairs_generated = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_generator_boxes_generated){
                    agent_generator_boxes_generated = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_generator_doctors_generated){
                    agent_generator_doctors_generated = (int) fpgu_strtol(buffer); 
                }
				if(in_chair_admin_id){
                    chair_admin_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_chair_admin_chairArray){
                    readArrayInput<unsigned int>(&fpgu_strtoul, buffer, chair_admin_chairArray, 35);    
                }
				if(in_box_id){
                    box_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_box_attending){
                    box_attending = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_box_tick){
                    box_tick = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_doctor_id){
                    doctor_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_doctor_attending){
                    doctor_attending = (int) fpgu_strtol(buffer); 
                }
				if(in_doctor_tick){
                    doctor_tick = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_triage_front){
                    triage_front = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_triage_rear){
                    triage_rear = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_triage_size){
                    triage_size = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_triage_tick){
                    triage_tick = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_triage_boxArray){
                    readArrayInput<unsigned int>(&fpgu_strtoul, buffer, triage_boxArray, 3);    
                }
				if(in_triage_patientQueue){
                    readArrayInput<unsigned int>(&fpgu_strtoul, buffer, triage_patientQueue, 100);    
                }
				
            }
            else if (in_env){
            if(in_env_EMMISION_RATE_EXIT1){
              
                    env_EMMISION_RATE_EXIT1 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT1(&env_EMMISION_RATE_EXIT1);
                  
              }
            if(in_env_EMMISION_RATE_EXIT2){
              
                    env_EMMISION_RATE_EXIT2 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT2(&env_EMMISION_RATE_EXIT2);
                  
              }
            if(in_env_EMMISION_RATE_EXIT3){
              
                    env_EMMISION_RATE_EXIT3 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT3(&env_EMMISION_RATE_EXIT3);
                  
              }
            if(in_env_EMMISION_RATE_EXIT4){
              
                    env_EMMISION_RATE_EXIT4 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT4(&env_EMMISION_RATE_EXIT4);
                  
              }
            if(in_env_EMMISION_RATE_EXIT5){
              
                    env_EMMISION_RATE_EXIT5 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT5(&env_EMMISION_RATE_EXIT5);
                  
              }
            if(in_env_EMMISION_RATE_EXIT6){
              
                    env_EMMISION_RATE_EXIT6 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT6(&env_EMMISION_RATE_EXIT6);
                  
              }
            if(in_env_EMMISION_RATE_EXIT7){
              
                    env_EMMISION_RATE_EXIT7 = (float) fgpu_atof(buffer);
                    
                    set_EMMISION_RATE_EXIT7(&env_EMMISION_RATE_EXIT7);
                  
              }
            if(in_env_EXIT1_PROBABILITY){
              
                    env_EXIT1_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT1_PROBABILITY(&env_EXIT1_PROBABILITY);
                  
              }
            if(in_env_EXIT2_PROBABILITY){
              
                    env_EXIT2_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT2_PROBABILITY(&env_EXIT2_PROBABILITY);
                  
              }
            if(in_env_EXIT3_PROBABILITY){
              
                    env_EXIT3_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT3_PROBABILITY(&env_EXIT3_PROBABILITY);
                  
              }
            if(in_env_EXIT4_PROBABILITY){
              
                    env_EXIT4_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT4_PROBABILITY(&env_EXIT4_PROBABILITY);
                  
              }
            if(in_env_EXIT5_PROBABILITY){
              
                    env_EXIT5_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT5_PROBABILITY(&env_EXIT5_PROBABILITY);
                  
              }
            if(in_env_EXIT6_PROBABILITY){
              
                    env_EXIT6_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT6_PROBABILITY(&env_EXIT6_PROBABILITY);
                  
              }
            if(in_env_EXIT7_PROBABILITY){
              
                    env_EXIT7_PROBABILITY = (int) fpgu_strtol(buffer);
                    
                    set_EXIT7_PROBABILITY(&env_EXIT7_PROBABILITY);
                  
              }
            if(in_env_EXIT1_STATE){
              
                    env_EXIT1_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT1_STATE(&env_EXIT1_STATE);
                  
              }
            if(in_env_EXIT2_STATE){
              
                    env_EXIT2_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT2_STATE(&env_EXIT2_STATE);
                  
              }
            if(in_env_EXIT3_STATE){
              
                    env_EXIT3_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT3_STATE(&env_EXIT3_STATE);
                  
              }
            if(in_env_EXIT4_STATE){
              
                    env_EXIT4_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT4_STATE(&env_EXIT4_STATE);
                  
              }
            if(in_env_EXIT5_STATE){
              
                    env_EXIT5_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT5_STATE(&env_EXIT5_STATE);
                  
              }
            if(in_env_EXIT6_STATE){
              
                    env_EXIT6_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT6_STATE(&env_EXIT6_STATE);
                  
              }
            if(in_env_EXIT7_STATE){
              
                    env_EXIT7_STATE = (int) fpgu_strtol(buffer);
                    
                    set_EXIT7_STATE(&env_EXIT7_STATE);
                  
              }
            if(in_env_EXIT1_CELL_COUNT){
              
                    env_EXIT1_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT1_CELL_COUNT(&env_EXIT1_CELL_COUNT);
                  
              }
            if(in_env_EXIT2_CELL_COUNT){
              
                    env_EXIT2_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT2_CELL_COUNT(&env_EXIT2_CELL_COUNT);
                  
              }
            if(in_env_EXIT3_CELL_COUNT){
              
                    env_EXIT3_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT3_CELL_COUNT(&env_EXIT3_CELL_COUNT);
                  
              }
            if(in_env_EXIT4_CELL_COUNT){
              
                    env_EXIT4_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT4_CELL_COUNT(&env_EXIT4_CELL_COUNT);
                  
              }
            if(in_env_EXIT5_CELL_COUNT){
              
                    env_EXIT5_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT5_CELL_COUNT(&env_EXIT5_CELL_COUNT);
                  
              }
            if(in_env_EXIT6_CELL_COUNT){
              
                    env_EXIT6_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT6_CELL_COUNT(&env_EXIT6_CELL_COUNT);
                  
              }
            if(in_env_EXIT7_CELL_COUNT){
              
                    env_EXIT7_CELL_COUNT = (int) fpgu_strtol(buffer);
                    
                    set_EXIT7_CELL_COUNT(&env_EXIT7_CELL_COUNT);
                  
              }
            if(in_env_TIME_SCALER){
              
                    env_TIME_SCALER = (float) fgpu_atof(buffer);
                    
                    set_TIME_SCALER(&env_TIME_SCALER);
                  
              }
            if(in_env_STEER_WEIGHT){
              
                    env_STEER_WEIGHT = (float) fgpu_atof(buffer);
                    
                    set_STEER_WEIGHT(&env_STEER_WEIGHT);
                  
              }
            if(in_env_AVOID_WEIGHT){
              
                    env_AVOID_WEIGHT = (float) fgpu_atof(buffer);
                    
                    set_AVOID_WEIGHT(&env_AVOID_WEIGHT);
                  
              }
            if(in_env_COLLISION_WEIGHT){
              
                    env_COLLISION_WEIGHT = (float) fgpu_atof(buffer);
                    
                    set_COLLISION_WEIGHT(&env_COLLISION_WEIGHT);
                  
              }
            if(in_env_GOAL_WEIGHT){
              
                    env_GOAL_WEIGHT = (float) fgpu_atof(buffer);
                    
                    set_GOAL_WEIGHT(&env_GOAL_WEIGHT);
                  
              }
            
            }
		/* Reset buffer */
			i = 0;
		}
		/* If in tag put read char into buffer */
		else if(in_tag)
		{
            // Check if we are a comment, when we are in a tag and buffer[0:2] == "!--"
            if(i == 2 && c == '-' && buffer[1] == '-' && buffer[0] == '!'){
                in_comment = 1;
                // Reset the buffer and i.
                buffer[0] = 0;
                i = 0;
            }

            // Store the character and increment the counter
            buffer[i] = c;
            i++;

		}
		/* If in data read char into buffer */
		else
		{
			buffer[i] = c;
			i++;
		}
	}
    // If no bytes were read, raise a warning.
    if(bytesRead == 0){
        fprintf(stdout, "Warning: %s is an empty file\n", inputpath);
        fflush(stdout);
    }

    // If the in_comment flag is still marked, issue a warning.
    if(in_comment){
        fprintf(stdout, "Warning: Un-terminated comment in %s\n", inputpath);
        fflush(stdout);
    }    

	/* Close the file */
	fclose(file);
}

glm::vec3 getMaximumBounds(){
    return agent_maximum;
}

glm::vec3 getMinimumBounds(){
    return agent_minimum;
}


/* Methods to load static networks from disk */
