
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

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_agent_list* h_agents_default, xmachine_memory_agent_list* d_agents_default, int h_xmachine_memory_agent_default_count,xmachine_memory_navmap_list* h_navmaps_static, xmachine_memory_navmap_list* d_navmaps_static, int h_xmachine_memory_navmap_static_count,xmachine_memory_chair_list* h_chairs_defaultChair, xmachine_memory_chair_list* d_chairs_defaultChair, int h_xmachine_memory_chair_defaultChair_count,xmachine_memory_bed_list* h_beds_defaultBed, xmachine_memory_bed_list* d_beds_defaultBed, int h_xmachine_memory_bed_defaultBed_count,xmachine_memory_doctor_manager_list* h_doctor_managers_defaultDoctorManager, xmachine_memory_doctor_manager_list* d_doctor_managers_defaultDoctorManager, int h_xmachine_memory_doctor_manager_defaultDoctorManager_count,xmachine_memory_specialist_manager_list* h_specialist_managers_defaultSpecialistManager, xmachine_memory_specialist_manager_list* d_specialist_managers_defaultSpecialistManager, int h_xmachine_memory_specialist_manager_defaultSpecialistManager_count,xmachine_memory_specialist_list* h_specialists_defaultSpecialist, xmachine_memory_specialist_list* d_specialists_defaultSpecialist, int h_xmachine_memory_specialist_defaultSpecialist_count,xmachine_memory_receptionist_list* h_receptionists_defaultReceptionist, xmachine_memory_receptionist_list* d_receptionists_defaultReceptionist, int h_xmachine_memory_receptionist_defaultReceptionist_count,xmachine_memory_agent_generator_list* h_agent_generators_defaultGenerator, xmachine_memory_agent_generator_list* d_agent_generators_defaultGenerator, int h_xmachine_memory_agent_generator_defaultGenerator_count,xmachine_memory_chair_admin_list* h_chair_admins_defaultAdmin, xmachine_memory_chair_admin_list* d_chair_admins_defaultAdmin, int h_xmachine_memory_chair_admin_defaultAdmin_count,xmachine_memory_uci_list* h_ucis_defaultUci, xmachine_memory_uci_list* d_ucis_defaultUci, int h_xmachine_memory_uci_defaultUci_count,xmachine_memory_box_list* h_boxs_defaultBox, xmachine_memory_box_list* d_boxs_defaultBox, int h_xmachine_memory_box_defaultBox_count,xmachine_memory_doctor_list* h_doctors_defaultDoctor, xmachine_memory_doctor_list* d_doctors_defaultDoctor, int h_xmachine_memory_doctor_defaultDoctor_count,xmachine_memory_triage_list* h_triages_defaultTriage, xmachine_memory_triage_list* d_triages_defaultTriage, int h_xmachine_memory_triage_defaultTriage_count)
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
	cudaStatus = cudaMemcpy( h_beds_defaultBed, d_beds_defaultBed, sizeof(xmachine_memory_bed_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying bed Agent defaultBed State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_doctor_managers_defaultDoctorManager, d_doctor_managers_defaultDoctorManager, sizeof(xmachine_memory_doctor_manager_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying doctor_manager Agent defaultDoctorManager State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_specialist_managers_defaultSpecialistManager, d_specialist_managers_defaultSpecialistManager, sizeof(xmachine_memory_specialist_manager_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying specialist_manager Agent defaultSpecialistManager State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_specialists_defaultSpecialist, d_specialists_defaultSpecialist, sizeof(xmachine_memory_specialist_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying specialist Agent defaultSpecialist State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
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
	cudaStatus = cudaMemcpy( h_ucis_defaultUci, d_ucis_defaultUci, sizeof(xmachine_memory_uci_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying uci Agent defaultUci State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
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
    fputs("\t<SECONDS_PER_TICK>", file);
    sprintf(data, "%d", (*get_SECONDS_PER_TICK()));
    fputs(data, file);
    fputs("</SECONDS_PER_TICK>\n", file);
    fputs("\t<SECONDS_INCUBATING>", file);
    sprintf(data, "%d", (*get_SECONDS_INCUBATING()));
    fputs(data, file);
    fputs("</SECONDS_INCUBATING>\n", file);
    fputs("\t<SECONDS_SICK>", file);
    sprintf(data, "%d", (*get_SECONDS_SICK()));
    fputs(data, file);
    fputs("</SECONDS_SICK>\n", file);
    fputs("\t<CLEANING_PERIOD_SECONDS>", file);
    sprintf(data, "%d", (*get_CLEANING_PERIOD_SECONDS()));
    fputs(data, file);
    fputs("</CLEANING_PERIOD_SECONDS>\n", file);
    fputs("\t<EXIT_X>", file);
    sprintf(data, "%d", (*get_EXIT_X()));
    fputs(data, file);
    fputs("</EXIT_X>\n", file);
    fputs("\t<EXIT_Y>", file);
    sprintf(data, "%d", (*get_EXIT_Y()));
    fputs(data, file);
    fputs("</EXIT_Y>\n", file);
    fputs("\t<PROB_INFECT>", file);
    sprintf(data, "%f", (*get_PROB_INFECT()));
    fputs(data, file);
    fputs("</PROB_INFECT>\n", file);
    fputs("\t<PROB_SPAWN_SICK>", file);
    sprintf(data, "%f", (*get_PROB_SPAWN_SICK()));
    fputs(data, file);
    fputs("</PROB_SPAWN_SICK>\n", file);
    fputs("\t<PROB_INFECT_PERSONAL>", file);
    sprintf(data, "%f", (*get_PROB_INFECT_PERSONAL()));
    fputs(data, file);
    fputs("</PROB_INFECT_PERSONAL>\n", file);
    fputs("\t<PROB_INFECT_CHAIR>", file);
    sprintf(data, "%f", (*get_PROB_INFECT_CHAIR()));
    fputs(data, file);
    fputs("</PROB_INFECT_CHAIR>\n", file);
    fputs("\t<PROB_INFECT_BED>", file);
    sprintf(data, "%f", (*get_PROB_INFECT_BED()));
    fputs(data, file);
    fputs("</PROB_INFECT_BED>\n", file);
    fputs("\t<PROB_VACCINE>", file);
    sprintf(data, "%f", (*get_PROB_VACCINE()));
    fputs(data, file);
    fputs("</PROB_VACCINE>\n", file);
    fputs("\t<PROB_VACCINE_STAFF>", file);
    sprintf(data, "%f", (*get_PROB_VACCINE_STAFF()));
    fputs(data, file);
    fputs("</PROB_VACCINE_STAFF>\n", file);
    fputs("\t<UCI_INFECTION_CHANCE>", file);
    sprintf(data, "%f", (*get_UCI_INFECTION_CHANCE()));
    fputs(data, file);
    fputs("</UCI_INFECTION_CHANCE>\n", file);
    fputs("\t<FIRSTCHAIR_X>", file);
    sprintf(data, "%d", (*get_FIRSTCHAIR_X()));
    fputs(data, file);
    fputs("</FIRSTCHAIR_X>\n", file);
    fputs("\t<FIRSTCHAIR_Y>", file);
    sprintf(data, "%d", (*get_FIRSTCHAIR_Y()));
    fputs(data, file);
    fputs("</FIRSTCHAIR_Y>\n", file);
    fputs("\t<SPACE_BETWEEN>", file);
    sprintf(data, "%d", (*get_SPACE_BETWEEN()));
    fputs(data, file);
    fputs("</SPACE_BETWEEN>\n", file);
    fputs("\t<DOCTOR_SECONDS>", file);
    sprintf(data, "%d", (*get_DOCTOR_SECONDS()));
    fputs(data, file);
    fputs("</DOCTOR_SECONDS>\n", file);
    fputs("\t<FIRSTDOCTOR_X>", file);
    sprintf(data, "%d", (*get_FIRSTDOCTOR_X()));
    fputs(data, file);
    fputs("</FIRSTDOCTOR_X>\n", file);
    fputs("\t<FIRSTDOCTOR_Y>", file);
    sprintf(data, "%d", (*get_FIRSTDOCTOR_Y()));
    fputs(data, file);
    fputs("</FIRSTDOCTOR_Y>\n", file);
    fputs("\t<SPACE_BETWEEN_DOCTORS>", file);
    sprintf(data, "%d", (*get_SPACE_BETWEEN_DOCTORS()));
    fputs(data, file);
    fputs("</SPACE_BETWEEN_DOCTORS>\n", file);
    fputs("\t<BOX_SECONDS>", file);
    sprintf(data, "%d", (*get_BOX_SECONDS()));
    fputs(data, file);
    fputs("</BOX_SECONDS>\n", file);
    fputs("\t<TRIAGE_X>", file);
    sprintf(data, "%d", (*get_TRIAGE_X()));
    fputs(data, file);
    fputs("</TRIAGE_X>\n", file);
    fputs("\t<TRIAGE_Y>", file);
    sprintf(data, "%d", (*get_TRIAGE_Y()));
    fputs(data, file);
    fputs("</TRIAGE_Y>\n", file);
    fputs("\t<UCI_X>", file);
    sprintf(data, "%d", (*get_UCI_X()));
    fputs(data, file);
    fputs("</UCI_X>\n", file);
    fputs("\t<UCI_Y>", file);
    sprintf(data, "%d", (*get_UCI_Y()));
    fputs(data, file);
    fputs("</UCI_Y>\n", file);
    fputs("\t<NUMBER_OF_BEDS>", file);
    sprintf(data, "%d", (*get_NUMBER_OF_BEDS()));
    fputs(data, file);
    fputs("</NUMBER_OF_BEDS>\n", file);
    fputs("\t<PROB_STAY_1>", file);
    sprintf(data, "%f", (*get_PROB_STAY_1()));
    fputs(data, file);
    fputs("</PROB_STAY_1>\n", file);
    fputs("\t<PROB_STAY_2>", file);
    sprintf(data, "%f", (*get_PROB_STAY_2()));
    fputs(data, file);
    fputs("</PROB_STAY_2>\n", file);
    fputs("\t<PROB_STAY_3>", file);
    sprintf(data, "%f", (*get_PROB_STAY_3()));
    fputs(data, file);
    fputs("</PROB_STAY_3>\n", file);
    fputs("\t<PROB_STAY_4>", file);
    sprintf(data, "%f", (*get_PROB_STAY_4()));
    fputs(data, file);
    fputs("</PROB_STAY_4>\n", file);
    fputs("\t<PROB_STAY_5>", file);
    sprintf(data, "%f", (*get_PROB_STAY_5()));
    fputs(data, file);
    fputs("</PROB_STAY_5>\n", file);
    fputs("\t<PROB_STAY_6>", file);
    sprintf(data, "%f", (*get_PROB_STAY_6()));
    fputs(data, file);
    fputs("</PROB_STAY_6>\n", file);
    fputs("\t<PROB_STAY_7>", file);
    sprintf(data, "%f", (*get_PROB_STAY_7()));
    fputs(data, file);
    fputs("</PROB_STAY_7>\n", file);
    fputs("\t<PROB_STAY_8>", file);
    sprintf(data, "%f", (*get_PROB_STAY_8()));
    fputs(data, file);
    fputs("</PROB_STAY_8>\n", file);
    fputs("\t<PROB_STAY_9>", file);
    sprintf(data, "%f", (*get_PROB_STAY_9()));
    fputs(data, file);
    fputs("</PROB_STAY_9>\n", file);
    fputs("\t<PROB_STAY_10>", file);
    sprintf(data, "%f", (*get_PROB_STAY_10()));
    fputs(data, file);
    fputs("</PROB_STAY_10>\n", file);
    fputs("\t<PROB_STAY_11>", file);
    sprintf(data, "%f", (*get_PROB_STAY_11()));
    fputs(data, file);
    fputs("</PROB_STAY_11>\n", file);
    fputs("\t<PROB_STAY_12>", file);
    sprintf(data, "%f", (*get_PROB_STAY_12()));
    fputs(data, file);
    fputs("</PROB_STAY_12>\n", file);
    fputs("\t<PROB_STAY_13>", file);
    sprintf(data, "%f", (*get_PROB_STAY_13()));
    fputs(data, file);
    fputs("</PROB_STAY_13>\n", file);
    fputs("\t<PROB_STAY_14>", file);
    sprintf(data, "%f", (*get_PROB_STAY_14()));
    fputs(data, file);
    fputs("</PROB_STAY_14>\n", file);
    fputs("\t<PROB_STAY_15>", file);
    sprintf(data, "%f", (*get_PROB_STAY_15()));
    fputs(data, file);
    fputs("</PROB_STAY_15>\n", file);
    fputs("\t<PROB_STAY_16>", file);
    sprintf(data, "%f", (*get_PROB_STAY_16()));
    fputs(data, file);
    fputs("</PROB_STAY_16>\n", file);
    fputs("\t<PROB_STAY_17>", file);
    sprintf(data, "%f", (*get_PROB_STAY_17()));
    fputs(data, file);
    fputs("</PROB_STAY_17>\n", file);
    fputs("\t<PROB_STAY_18>", file);
    sprintf(data, "%f", (*get_PROB_STAY_18()));
    fputs(data, file);
    fputs("</PROB_STAY_18>\n", file);
    fputs("\t<STAY_TIME_1>", file);
    sprintf(data, "%f", (*get_STAY_TIME_1()));
    fputs(data, file);
    fputs("</STAY_TIME_1>\n", file);
    fputs("\t<STAY_TIME_2>", file);
    sprintf(data, "%f", (*get_STAY_TIME_2()));
    fputs(data, file);
    fputs("</STAY_TIME_2>\n", file);
    fputs("\t<STAY_TIME_3>", file);
    sprintf(data, "%f", (*get_STAY_TIME_3()));
    fputs(data, file);
    fputs("</STAY_TIME_3>\n", file);
    fputs("\t<STAY_TIME_4>", file);
    sprintf(data, "%f", (*get_STAY_TIME_4()));
    fputs(data, file);
    fputs("</STAY_TIME_4>\n", file);
    fputs("\t<STAY_TIME_5>", file);
    sprintf(data, "%f", (*get_STAY_TIME_5()));
    fputs(data, file);
    fputs("</STAY_TIME_5>\n", file);
    fputs("\t<STAY_TIME_6>", file);
    sprintf(data, "%f", (*get_STAY_TIME_6()));
    fputs(data, file);
    fputs("</STAY_TIME_6>\n", file);
    fputs("\t<STAY_TIME_7>", file);
    sprintf(data, "%f", (*get_STAY_TIME_7()));
    fputs(data, file);
    fputs("</STAY_TIME_7>\n", file);
    fputs("\t<STAY_TIME_8>", file);
    sprintf(data, "%f", (*get_STAY_TIME_8()));
    fputs(data, file);
    fputs("</STAY_TIME_8>\n", file);
    fputs("\t<STAY_TIME_9>", file);
    sprintf(data, "%f", (*get_STAY_TIME_9()));
    fputs(data, file);
    fputs("</STAY_TIME_9>\n", file);
    fputs("\t<STAY_TIME_10>", file);
    sprintf(data, "%f", (*get_STAY_TIME_10()));
    fputs(data, file);
    fputs("</STAY_TIME_10>\n", file);
    fputs("\t<STAY_TIME_11>", file);
    sprintf(data, "%f", (*get_STAY_TIME_11()));
    fputs(data, file);
    fputs("</STAY_TIME_11>\n", file);
    fputs("\t<STAY_TIME_12>", file);
    sprintf(data, "%f", (*get_STAY_TIME_12()));
    fputs(data, file);
    fputs("</STAY_TIME_12>\n", file);
    fputs("\t<STAY_TIME_13>", file);
    sprintf(data, "%f", (*get_STAY_TIME_13()));
    fputs(data, file);
    fputs("</STAY_TIME_13>\n", file);
    fputs("\t<STAY_TIME_14>", file);
    sprintf(data, "%f", (*get_STAY_TIME_14()));
    fputs(data, file);
    fputs("</STAY_TIME_14>\n", file);
    fputs("\t<STAY_TIME_15>", file);
    sprintf(data, "%f", (*get_STAY_TIME_15()));
    fputs(data, file);
    fputs("</STAY_TIME_15>\n", file);
    fputs("\t<STAY_TIME_16>", file);
    sprintf(data, "%f", (*get_STAY_TIME_16()));
    fputs(data, file);
    fputs("</STAY_TIME_16>\n", file);
    fputs("\t<STAY_TIME_17>", file);
    sprintf(data, "%f", (*get_STAY_TIME_17()));
    fputs(data, file);
    fputs("</STAY_TIME_17>\n", file);
    fputs("\t<STAY_TIME_18>", file);
    sprintf(data, "%f", (*get_STAY_TIME_18()));
    fputs(data, file);
    fputs("</STAY_TIME_18>\n", file);
    fputs("\t<CHECKPOINT_1_X>", file);
    sprintf(data, "%d", (*get_CHECKPOINT_1_X()));
    fputs(data, file);
    fputs("</CHECKPOINT_1_X>\n", file);
    fputs("\t<CHECKPOINT_1_Y>", file);
    sprintf(data, "%d", (*get_CHECKPOINT_1_Y()));
    fputs(data, file);
    fputs("</CHECKPOINT_1_Y>\n", file);
    fputs("\t<CHECKPOINT_2_X>", file);
    sprintf(data, "%d", (*get_CHECKPOINT_2_X()));
    fputs(data, file);
    fputs("</CHECKPOINT_2_X>\n", file);
    fputs("\t<CHECKPOINT_2_Y>", file);
    sprintf(data, "%d", (*get_CHECKPOINT_2_Y()));
    fputs(data, file);
    fputs("</CHECKPOINT_2_Y>\n", file);
    fputs("\t<CHECKPOINT_3_X>", file);
    sprintf(data, "%d", (*get_CHECKPOINT_3_X()));
    fputs(data, file);
    fputs("</CHECKPOINT_3_X>\n", file);
    fputs("\t<CHECKPOINT_3_Y>", file);
    sprintf(data, "%d", (*get_CHECKPOINT_3_Y()));
    fputs(data, file);
    fputs("</CHECKPOINT_3_Y>\n", file);
    fputs("\t<CHECKPOINT_4_X>", file);
    sprintf(data, "%d", (*get_CHECKPOINT_4_X()));
    fputs(data, file);
    fputs("</CHECKPOINT_4_X>\n", file);
    fputs("\t<CHECKPOINT_4_Y>", file);
    sprintf(data, "%d", (*get_CHECKPOINT_4_Y()));
    fputs(data, file);
    fputs("</CHECKPOINT_4_Y>\n", file);
    fputs("\t<CHECKPOINT_5_X>", file);
    sprintf(data, "%d", (*get_CHECKPOINT_5_X()));
    fputs(data, file);
    fputs("</CHECKPOINT_5_X>\n", file);
    fputs("\t<CHECKPOINT_5_Y>", file);
    sprintf(data, "%d", (*get_CHECKPOINT_5_Y()));
    fputs(data, file);
    fputs("</CHECKPOINT_5_Y>\n", file);
    fputs("\t<SPECIALIST_SECONDS>", file);
    sprintf(data, "%d", (*get_SPECIALIST_SECONDS()));
    fputs(data, file);
    fputs("</SPECIALIST_SECONDS>\n", file);
    fputs("\t<FIRSTSPECIALIST_X>", file);
    sprintf(data, "%d", (*get_FIRSTSPECIALIST_X()));
    fputs(data, file);
    fputs("</FIRSTSPECIALIST_X>\n", file);
    fputs("\t<FIRSTSPECIALIST_Y>", file);
    sprintf(data, "%d", (*get_FIRSTSPECIALIST_Y()));
    fputs(data, file);
    fputs("</FIRSTSPECIALIST_Y>\n", file);
    fputs("\t<SPACE_BETWEEN_SPECIALISTS>", file);
    sprintf(data, "%d", (*get_SPACE_BETWEEN_SPECIALISTS()));
    fputs(data, file);
    fputs("</SPACE_BETWEEN_SPECIALISTS>\n", file);
    fputs("\t<FIFTHSPECIALIST_X>", file);
    sprintf(data, "%d", (*get_FIFTHSPECIALIST_X()));
    fputs(data, file);
    fputs("</FIFTHSPECIALIST_X>\n", file);
    fputs("\t<FIFTHSPECIALIST_Y>", file);
    sprintf(data, "%d", (*get_FIFTHSPECIALIST_Y()));
    fputs(data, file);
    fputs("</FIFTHSPECIALIST_Y>\n", file);
    fputs("\t<PROB_LEVEL_1>", file);
    sprintf(data, "%f", (*get_PROB_LEVEL_1()));
    fputs(data, file);
    fputs("</PROB_LEVEL_1>\n", file);
    fputs("\t<PROB_LEVEL_2>", file);
    sprintf(data, "%f", (*get_PROB_LEVEL_2()));
    fputs(data, file);
    fputs("</PROB_LEVEL_2>\n", file);
    fputs("\t<PROB_LEVEL_3>", file);
    sprintf(data, "%f", (*get_PROB_LEVEL_3()));
    fputs(data, file);
    fputs("</PROB_LEVEL_3>\n", file);
    fputs("\t<PROB_LEVEL_4>", file);
    sprintf(data, "%f", (*get_PROB_LEVEL_4()));
    fputs(data, file);
    fputs("</PROB_LEVEL_4>\n", file);
    fputs("\t<PROB_LEVEL_5>", file);
    sprintf(data, "%f", (*get_PROB_LEVEL_5()));
    fputs(data, file);
    fputs("</PROB_LEVEL_5>\n", file);
    fputs("\t<PROB_SURGICAL>", file);
    sprintf(data, "%f", (*get_PROB_SURGICAL()));
    fputs(data, file);
    fputs("</PROB_SURGICAL>\n", file);
    fputs("\t<PROB_MEDICAL>", file);
    sprintf(data, "%f", (*get_PROB_MEDICAL()));
    fputs(data, file);
    fputs("</PROB_MEDICAL>\n", file);
    fputs("\t<PROB_PEDIATRICS>", file);
    sprintf(data, "%f", (*get_PROB_PEDIATRICS()));
    fputs(data, file);
    fputs("</PROB_PEDIATRICS>\n", file);
    fputs("\t<PROB_UCI>", file);
    sprintf(data, "%f", (*get_PROB_UCI()));
    fputs(data, file);
    fputs("</PROB_UCI>\n", file);
    fputs("\t<PROB_GYNECOLOGIST>", file);
    sprintf(data, "%f", (*get_PROB_GYNECOLOGIST()));
    fputs(data, file);
    fputs("</PROB_GYNECOLOGIST>\n", file);
    fputs("\t<PROB_GERIATRICS>", file);
    sprintf(data, "%f", (*get_PROB_GERIATRICS()));
    fputs(data, file);
    fputs("</PROB_GERIATRICS>\n", file);
    fputs("\t<PROB_PSYCHIATRY>", file);
    sprintf(data, "%f", (*get_PROB_PSYCHIATRY()));
    fputs(data, file);
    fputs("</PROB_PSYCHIATRY>\n", file);
    fputs("\t<RECEPTION_SECONDS>", file);
    sprintf(data, "%d", (*get_RECEPTION_SECONDS()));
    fputs(data, file);
    fputs("</RECEPTION_SECONDS>\n", file);
    fputs("\t<RECEPTIONIST_X>", file);
    sprintf(data, "%d", (*get_RECEPTIONIST_X()));
    fputs(data, file);
    fputs("</RECEPTIONIST_X>\n", file);
    fputs("\t<RECEPTIONIST_Y>", file);
    sprintf(data, "%d", (*get_RECEPTIONIST_Y()));
    fputs(data, file);
    fputs("</RECEPTIONIST_Y>\n", file);
	fputs("</environment>\n" , file);

	//Write each agent agent to xml
	for (int i=0; i<h_xmachine_memory_agent_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>agent</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%d", h_agents_default->id[i]);
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
        
		fputs("<specialist_no>", file);
        sprintf(data, "%u", h_agents_default->specialist_no[i]);
		fputs(data, file);
		fputs("</specialist_no>\n", file);
        
		fputs("<bed_no>", file);
        sprintf(data, "%u", h_agents_default->bed_no[i]);
		fputs(data, file);
		fputs("</bed_no>\n", file);
        
		fputs("<priority>", file);
        sprintf(data, "%u", h_agents_default->priority[i]);
		fputs(data, file);
		fputs("</priority>\n", file);
        
		fputs("<vaccine>", file);
        sprintf(data, "%u", h_agents_default->vaccine[i]);
		fputs(data, file);
		fputs("</vaccine>\n", file);
        
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
        
		fputs("<tick>", file);
        sprintf(data, "%d", h_chairs_defaultChair->tick[i]);
		fputs(data, file);
		fputs("</tick>\n", file);
        
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
	//Write each bed agent to xml
	for (int i=0; i<h_xmachine_memory_bed_defaultBed_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>bed</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%d", h_beds_defaultBed->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<tick>", file);
        sprintf(data, "%d", h_beds_defaultBed->tick[i]);
		fputs(data, file);
		fputs("</tick>\n", file);
        
		fputs("<state>", file);
        sprintf(data, "%d", h_beds_defaultBed->state[i]);
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
	//Write each specialist_manager agent to xml
	for (int i=0; i<h_xmachine_memory_specialist_manager_defaultSpecialistManager_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>specialist_manager</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_specialist_managers_defaultSpecialistManager->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<tick>", file);
        for (int j=0;j<5;j++){
            fprintf(file, "%u", h_specialist_managers_defaultSpecialistManager->tick[(j*xmachine_memory_specialist_manager_MAX)+i]);
            if(j!=(5-1))
                fprintf(file, ",");
        }
		fputs("</tick>\n", file);
        
		fputs("<free_specialist>", file);
        for (int j=0;j<5;j++){
            fprintf(file, "%u", h_specialist_managers_defaultSpecialistManager->free_specialist[(j*xmachine_memory_specialist_manager_MAX)+i]);
            if(j!=(5-1))
                fprintf(file, ",");
        }
		fputs("</free_specialist>\n", file);
        
		fputs("<rear>", file);
        for (int j=0;j<5;j++){
            fprintf(file, "%u", h_specialist_managers_defaultSpecialistManager->rear[(j*xmachine_memory_specialist_manager_MAX)+i]);
            if(j!=(5-1))
                fprintf(file, ",");
        }
		fputs("</rear>\n", file);
        
		fputs("<size>", file);
        for (int j=0;j<5;j++){
            fprintf(file, "%u", h_specialist_managers_defaultSpecialistManager->size[(j*xmachine_memory_specialist_manager_MAX)+i]);
            if(j!=(5-1))
                fprintf(file, ",");
        }
		fputs("</size>\n", file);
        
		fputs("<surgicalQueue>", file);
        for (int j=0;j<35;j++){
            fprintf(file, "%d, %d", h_specialist_managers_defaultSpecialistManager->surgicalQueue[(j*xmachine_memory_specialist_manager_MAX)+i].x, h_specialist_managers_defaultSpecialistManager->surgicalQueue[(j*xmachine_memory_specialist_manager_MAX)+i].y);
            if(j!=(35-1))
                fprintf(file, "|");
        }
		fputs("</surgicalQueue>\n", file);
        
		fputs("<pediatricsQueue>", file);
        for (int j=0;j<35;j++){
            fprintf(file, "%d, %d", h_specialist_managers_defaultSpecialistManager->pediatricsQueue[(j*xmachine_memory_specialist_manager_MAX)+i].x, h_specialist_managers_defaultSpecialistManager->pediatricsQueue[(j*xmachine_memory_specialist_manager_MAX)+i].y);
            if(j!=(35-1))
                fprintf(file, "|");
        }
		fputs("</pediatricsQueue>\n", file);
        
		fputs("<gynecologistQueue>", file);
        for (int j=0;j<35;j++){
            fprintf(file, "%d, %d", h_specialist_managers_defaultSpecialistManager->gynecologistQueue[(j*xmachine_memory_specialist_manager_MAX)+i].x, h_specialist_managers_defaultSpecialistManager->gynecologistQueue[(j*xmachine_memory_specialist_manager_MAX)+i].y);
            if(j!=(35-1))
                fprintf(file, "|");
        }
		fputs("</gynecologistQueue>\n", file);
        
		fputs("<geriatricsQueue>", file);
        for (int j=0;j<35;j++){
            fprintf(file, "%d, %d", h_specialist_managers_defaultSpecialistManager->geriatricsQueue[(j*xmachine_memory_specialist_manager_MAX)+i].x, h_specialist_managers_defaultSpecialistManager->geriatricsQueue[(j*xmachine_memory_specialist_manager_MAX)+i].y);
            if(j!=(35-1))
                fprintf(file, "|");
        }
		fputs("</geriatricsQueue>\n", file);
        
		fputs("<psychiatristQueue>", file);
        for (int j=0;j<35;j++){
            fprintf(file, "%d, %d", h_specialist_managers_defaultSpecialistManager->psychiatristQueue[(j*xmachine_memory_specialist_manager_MAX)+i].x, h_specialist_managers_defaultSpecialistManager->psychiatristQueue[(j*xmachine_memory_specialist_manager_MAX)+i].y);
            if(j!=(35-1))
                fprintf(file, "|");
        }
		fputs("</psychiatristQueue>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each specialist agent to xml
	for (int i=0; i<h_xmachine_memory_specialist_defaultSpecialist_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>specialist</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_specialists_defaultSpecialist->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<current_patient>", file);
        sprintf(data, "%u", h_specialists_defaultSpecialist->current_patient[i]);
		fputs(data, file);
		fputs("</current_patient>\n", file);
        
		fputs("<tick>", file);
        sprintf(data, "%u", h_specialists_defaultSpecialist->tick[i]);
		fputs(data, file);
		fputs("</tick>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each receptionist agent to xml
	for (int i=0; i<h_xmachine_memory_receptionist_defaultReceptionist_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>receptionist</name>\n", file);
        
		fputs("<patientQueue>", file);
        for (int j=0;j<35;j++){
            fprintf(file, "%u", h_receptionists_defaultReceptionist->patientQueue[(j*xmachine_memory_receptionist_MAX)+i]);
            if(j!=(35-1))
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
        
		fputs("<beds_generated>", file);
        sprintf(data, "%d", h_agent_generators_defaultGenerator->beds_generated[i]);
		fputs(data, file);
		fputs("</beds_generated>\n", file);
        
		fputs("<boxes_generated>", file);
        sprintf(data, "%d", h_agent_generators_defaultGenerator->boxes_generated[i]);
		fputs(data, file);
		fputs("</boxes_generated>\n", file);
        
		fputs("<doctors_generated>", file);
        sprintf(data, "%d", h_agent_generators_defaultGenerator->doctors_generated[i]);
		fputs(data, file);
		fputs("</doctors_generated>\n", file);
        
		fputs("<specialists_generated>", file);
        sprintf(data, "%d", h_agent_generators_defaultGenerator->specialists_generated[i]);
		fputs(data, file);
		fputs("</specialists_generated>\n", file);
        
		fputs("<personal_generated>", file);
        sprintf(data, "%d", h_agent_generators_defaultGenerator->personal_generated[i]);
		fputs(data, file);
		fputs("</personal_generated>\n", file);
        
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
	//Write each uci agent to xml
	for (int i=0; i<h_xmachine_memory_uci_defaultUci_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>uci</name>\n", file);
        
		fputs("<tick>", file);
        sprintf(data, "%u", h_ucis_defaultUci->tick[i]);
		fputs(data, file);
		fputs("</tick>\n", file);
        
		fputs("<bedArray>", file);
        for (int j=0;j<100;j++){
            fprintf(file, "%d, %d", h_ucis_defaultUci->bedArray[(j*xmachine_memory_uci_MAX)+i].x, h_ucis_defaultUci->bedArray[(j*xmachine_memory_uci_MAX)+i].y);
            if(j!=(100-1))
                fprintf(file, "|");
        }
		fputs("</bedArray>\n", file);
        
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
        
		fputs("<current_patient>", file);
        sprintf(data, "%u", h_boxs_defaultBox->current_patient[i]);
		fputs(data, file);
		fputs("</current_patient>\n", file);
        
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
        
		fputs("<current_patient>", file);
        sprintf(data, "%d", h_doctors_defaultDoctor->current_patient[i]);
		fputs(data, file);
		fputs("</current_patient>\n", file);
        
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
        
		fputs("<free_boxes>", file);
        for (int j=0;j<3;j++){
            fprintf(file, "%u", h_triages_defaultTriage->free_boxes[(j*xmachine_memory_triage_MAX)+i]);
            if(j!=(3-1))
                fprintf(file, ",");
        }
		fputs("</free_boxes>\n", file);
        
		fputs("<patientQueue>", file);
        for (int j=0;j<35;j++){
            fprintf(file, "%u", h_triages_defaultTriage->patientQueue[(j*xmachine_memory_triage_MAX)+i]);
            if(j!=(35-1))
                fprintf(file, ",");
        }
		fputs("</patientQueue>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);

}

void readInitialStates(char* inputpath, xmachine_memory_agent_list* h_agents, int* h_xmachine_memory_agent_count,xmachine_memory_navmap_list* h_navmaps, int* h_xmachine_memory_navmap_count,xmachine_memory_chair_list* h_chairs, int* h_xmachine_memory_chair_count,xmachine_memory_bed_list* h_beds, int* h_xmachine_memory_bed_count,xmachine_memory_doctor_manager_list* h_doctor_managers, int* h_xmachine_memory_doctor_manager_count,xmachine_memory_specialist_manager_list* h_specialist_managers, int* h_xmachine_memory_specialist_manager_count,xmachine_memory_specialist_list* h_specialists, int* h_xmachine_memory_specialist_count,xmachine_memory_receptionist_list* h_receptionists, int* h_xmachine_memory_receptionist_count,xmachine_memory_agent_generator_list* h_agent_generators, int* h_xmachine_memory_agent_generator_count,xmachine_memory_chair_admin_list* h_chair_admins, int* h_xmachine_memory_chair_admin_count,xmachine_memory_uci_list* h_ucis, int* h_xmachine_memory_uci_count,xmachine_memory_box_list* h_boxs, int* h_xmachine_memory_box_count,xmachine_memory_doctor_list* h_doctors, int* h_xmachine_memory_doctor_count,xmachine_memory_triage_list* h_triages, int* h_xmachine_memory_triage_count)
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
    int in_agent_specialist_no;
    int in_agent_bed_no;
    int in_agent_priority;
    int in_agent_vaccine;
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
    int in_chair_tick;
    int in_chair_x;
    int in_chair_y;
    int in_chair_state;
    int in_bed_id;
    int in_bed_tick;
    int in_bed_state;
    int in_doctor_manager_tick;
    int in_doctor_manager_rear;
    int in_doctor_manager_size;
    int in_doctor_manager_doctors_occupied;
    int in_doctor_manager_free_doctors;
    int in_doctor_manager_patientQueue;
    int in_specialist_manager_id;
    int in_specialist_manager_tick;
    int in_specialist_manager_free_specialist;
    int in_specialist_manager_rear;
    int in_specialist_manager_size;
    int in_specialist_manager_surgicalQueue;
    int in_specialist_manager_pediatricsQueue;
    int in_specialist_manager_gynecologistQueue;
    int in_specialist_manager_geriatricsQueue;
    int in_specialist_manager_psychiatristQueue;
    int in_specialist_id;
    int in_specialist_current_patient;
    int in_specialist_tick;
    int in_receptionist_patientQueue;
    int in_receptionist_front;
    int in_receptionist_rear;
    int in_receptionist_size;
    int in_receptionist_tick;
    int in_receptionist_current_patient;
    int in_receptionist_attend_patient;
    int in_agent_generator_chairs_generated;
    int in_agent_generator_beds_generated;
    int in_agent_generator_boxes_generated;
    int in_agent_generator_doctors_generated;
    int in_agent_generator_specialists_generated;
    int in_agent_generator_personal_generated;
    int in_chair_admin_id;
    int in_chair_admin_chairArray;
    int in_uci_tick;
    int in_uci_bedArray;
    int in_box_id;
    int in_box_current_patient;
    int in_box_tick;
    int in_doctor_id;
    int in_doctor_current_patient;
    int in_doctor_tick;
    int in_triage_front;
    int in_triage_rear;
    int in_triage_size;
    int in_triage_tick;
    int in_triage_free_boxes;
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
    
    int in_env_SECONDS_PER_TICK;
    
    int in_env_SECONDS_INCUBATING;
    
    int in_env_SECONDS_SICK;
    
    int in_env_CLEANING_PERIOD_SECONDS;
    
    int in_env_EXIT_X;
    
    int in_env_EXIT_Y;
    
    int in_env_PROB_INFECT;
    
    int in_env_PROB_SPAWN_SICK;
    
    int in_env_PROB_INFECT_PERSONAL;
    
    int in_env_PROB_INFECT_CHAIR;
    
    int in_env_PROB_INFECT_BED;
    
    int in_env_PROB_VACCINE;
    
    int in_env_PROB_VACCINE_STAFF;
    
    int in_env_UCI_INFECTION_CHANCE;
    
    int in_env_FIRSTCHAIR_X;
    
    int in_env_FIRSTCHAIR_Y;
    
    int in_env_SPACE_BETWEEN;
    
    int in_env_DOCTOR_SECONDS;
    
    int in_env_FIRSTDOCTOR_X;
    
    int in_env_FIRSTDOCTOR_Y;
    
    int in_env_SPACE_BETWEEN_DOCTORS;
    
    int in_env_BOX_SECONDS;
    
    int in_env_TRIAGE_X;
    
    int in_env_TRIAGE_Y;
    
    int in_env_UCI_X;
    
    int in_env_UCI_Y;
    
    int in_env_NUMBER_OF_BEDS;
    
    int in_env_PROB_STAY_1;
    
    int in_env_PROB_STAY_2;
    
    int in_env_PROB_STAY_3;
    
    int in_env_PROB_STAY_4;
    
    int in_env_PROB_STAY_5;
    
    int in_env_PROB_STAY_6;
    
    int in_env_PROB_STAY_7;
    
    int in_env_PROB_STAY_8;
    
    int in_env_PROB_STAY_9;
    
    int in_env_PROB_STAY_10;
    
    int in_env_PROB_STAY_11;
    
    int in_env_PROB_STAY_12;
    
    int in_env_PROB_STAY_13;
    
    int in_env_PROB_STAY_14;
    
    int in_env_PROB_STAY_15;
    
    int in_env_PROB_STAY_16;
    
    int in_env_PROB_STAY_17;
    
    int in_env_PROB_STAY_18;
    
    int in_env_STAY_TIME_1;
    
    int in_env_STAY_TIME_2;
    
    int in_env_STAY_TIME_3;
    
    int in_env_STAY_TIME_4;
    
    int in_env_STAY_TIME_5;
    
    int in_env_STAY_TIME_6;
    
    int in_env_STAY_TIME_7;
    
    int in_env_STAY_TIME_8;
    
    int in_env_STAY_TIME_9;
    
    int in_env_STAY_TIME_10;
    
    int in_env_STAY_TIME_11;
    
    int in_env_STAY_TIME_12;
    
    int in_env_STAY_TIME_13;
    
    int in_env_STAY_TIME_14;
    
    int in_env_STAY_TIME_15;
    
    int in_env_STAY_TIME_16;
    
    int in_env_STAY_TIME_17;
    
    int in_env_STAY_TIME_18;
    
    int in_env_CHECKPOINT_1_X;
    
    int in_env_CHECKPOINT_1_Y;
    
    int in_env_CHECKPOINT_2_X;
    
    int in_env_CHECKPOINT_2_Y;
    
    int in_env_CHECKPOINT_3_X;
    
    int in_env_CHECKPOINT_3_Y;
    
    int in_env_CHECKPOINT_4_X;
    
    int in_env_CHECKPOINT_4_Y;
    
    int in_env_CHECKPOINT_5_X;
    
    int in_env_CHECKPOINT_5_Y;
    
    int in_env_SPECIALIST_SECONDS;
    
    int in_env_FIRSTSPECIALIST_X;
    
    int in_env_FIRSTSPECIALIST_Y;
    
    int in_env_SPACE_BETWEEN_SPECIALISTS;
    
    int in_env_FIFTHSPECIALIST_X;
    
    int in_env_FIFTHSPECIALIST_Y;
    
    int in_env_PROB_LEVEL_1;
    
    int in_env_PROB_LEVEL_2;
    
    int in_env_PROB_LEVEL_3;
    
    int in_env_PROB_LEVEL_4;
    
    int in_env_PROB_LEVEL_5;
    
    int in_env_PROB_SURGICAL;
    
    int in_env_PROB_MEDICAL;
    
    int in_env_PROB_PEDIATRICS;
    
    int in_env_PROB_UCI;
    
    int in_env_PROB_GYNECOLOGIST;
    
    int in_env_PROB_GERIATRICS;
    
    int in_env_PROB_PSYCHIATRY;
    
    int in_env_RECEPTION_SECONDS;
    
    int in_env_RECEPTIONIST_X;
    
    int in_env_RECEPTIONIST_Y;
    
	/* set agent count to zero */
	*h_xmachine_memory_agent_count = 0;
	*h_xmachine_memory_navmap_count = 0;
	*h_xmachine_memory_chair_count = 0;
	*h_xmachine_memory_bed_count = 0;
	*h_xmachine_memory_doctor_manager_count = 0;
	*h_xmachine_memory_specialist_manager_count = 0;
	*h_xmachine_memory_specialist_count = 0;
	*h_xmachine_memory_receptionist_count = 0;
	*h_xmachine_memory_agent_generator_count = 0;
	*h_xmachine_memory_chair_admin_count = 0;
	*h_xmachine_memory_uci_count = 0;
	*h_xmachine_memory_box_count = 0;
	*h_xmachine_memory_doctor_count = 0;
	*h_xmachine_memory_triage_count = 0;
	
	/* Variables for initial state data */
	int agent_id;
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
	unsigned int agent_specialist_no;
	unsigned int agent_bed_no;
	unsigned int agent_priority;
	unsigned int agent_vaccine;
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
	int chair_tick;
	int chair_x;
	int chair_y;
	int chair_state;
	int bed_id;
	int bed_tick;
	int bed_state;
	unsigned int doctor_manager_tick;
	unsigned int doctor_manager_rear;
	unsigned int doctor_manager_size;
    int doctor_manager_doctors_occupied[4];
	unsigned int doctor_manager_free_doctors;
    ivec2 doctor_manager_patientQueue[35];
	unsigned int specialist_manager_id;
    unsigned int specialist_manager_tick[5];
    unsigned int specialist_manager_free_specialist[5];
    unsigned int specialist_manager_rear[5];
    unsigned int specialist_manager_size[5];
    ivec2 specialist_manager_surgicalQueue[35];
    ivec2 specialist_manager_pediatricsQueue[35];
    ivec2 specialist_manager_gynecologistQueue[35];
    ivec2 specialist_manager_geriatricsQueue[35];
    ivec2 specialist_manager_psychiatristQueue[35];
	unsigned int specialist_id;
	unsigned int specialist_current_patient;
	unsigned int specialist_tick;
    unsigned int receptionist_patientQueue[35];
	unsigned int receptionist_front;
	unsigned int receptionist_rear;
	unsigned int receptionist_size;
	unsigned int receptionist_tick;
	int receptionist_current_patient;
	int receptionist_attend_patient;
	int agent_generator_chairs_generated;
	int agent_generator_beds_generated;
	int agent_generator_boxes_generated;
	int agent_generator_doctors_generated;
	int agent_generator_specialists_generated;
	int agent_generator_personal_generated;
	unsigned int chair_admin_id;
    unsigned int chair_admin_chairArray[35];
	unsigned int uci_tick;
    ivec2 uci_bedArray[100];
	unsigned int box_id;
	unsigned int box_current_patient;
	unsigned int box_tick;
	unsigned int doctor_id;
	int doctor_current_patient;
	unsigned int doctor_tick;
	unsigned int triage_front;
	unsigned int triage_rear;
	unsigned int triage_size;
	unsigned int triage_tick;
    unsigned int triage_free_boxes[3];
    unsigned int triage_patientQueue[35];

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
    int env_SECONDS_PER_TICK;
    int env_SECONDS_INCUBATING;
    int env_SECONDS_SICK;
    int env_CLEANING_PERIOD_SECONDS;
    int env_EXIT_X;
    int env_EXIT_Y;
    float env_PROB_INFECT;
    float env_PROB_SPAWN_SICK;
    float env_PROB_INFECT_PERSONAL;
    float env_PROB_INFECT_CHAIR;
    float env_PROB_INFECT_BED;
    float env_PROB_VACCINE;
    float env_PROB_VACCINE_STAFF;
    float env_UCI_INFECTION_CHANCE;
    int env_FIRSTCHAIR_X;
    int env_FIRSTCHAIR_Y;
    int env_SPACE_BETWEEN;
    int env_DOCTOR_SECONDS;
    int env_FIRSTDOCTOR_X;
    int env_FIRSTDOCTOR_Y;
    int env_SPACE_BETWEEN_DOCTORS;
    int env_BOX_SECONDS;
    int env_TRIAGE_X;
    int env_TRIAGE_Y;
    int env_UCI_X;
    int env_UCI_Y;
    int env_NUMBER_OF_BEDS;
    float env_PROB_STAY_1;
    float env_PROB_STAY_2;
    float env_PROB_STAY_3;
    float env_PROB_STAY_4;
    float env_PROB_STAY_5;
    float env_PROB_STAY_6;
    float env_PROB_STAY_7;
    float env_PROB_STAY_8;
    float env_PROB_STAY_9;
    float env_PROB_STAY_10;
    float env_PROB_STAY_11;
    float env_PROB_STAY_12;
    float env_PROB_STAY_13;
    float env_PROB_STAY_14;
    float env_PROB_STAY_15;
    float env_PROB_STAY_16;
    float env_PROB_STAY_17;
    float env_PROB_STAY_18;
    float env_STAY_TIME_1;
    float env_STAY_TIME_2;
    float env_STAY_TIME_3;
    float env_STAY_TIME_4;
    float env_STAY_TIME_5;
    float env_STAY_TIME_6;
    float env_STAY_TIME_7;
    float env_STAY_TIME_8;
    float env_STAY_TIME_9;
    float env_STAY_TIME_10;
    float env_STAY_TIME_11;
    float env_STAY_TIME_12;
    float env_STAY_TIME_13;
    float env_STAY_TIME_14;
    float env_STAY_TIME_15;
    float env_STAY_TIME_16;
    float env_STAY_TIME_17;
    float env_STAY_TIME_18;
    int env_CHECKPOINT_1_X;
    int env_CHECKPOINT_1_Y;
    int env_CHECKPOINT_2_X;
    int env_CHECKPOINT_2_Y;
    int env_CHECKPOINT_3_X;
    int env_CHECKPOINT_3_Y;
    int env_CHECKPOINT_4_X;
    int env_CHECKPOINT_4_Y;
    int env_CHECKPOINT_5_X;
    int env_CHECKPOINT_5_Y;
    int env_SPECIALIST_SECONDS;
    int env_FIRSTSPECIALIST_X;
    int env_FIRSTSPECIALIST_Y;
    int env_SPACE_BETWEEN_SPECIALISTS;
    int env_FIFTHSPECIALIST_X;
    int env_FIFTHSPECIALIST_Y;
    float env_PROB_LEVEL_1;
    float env_PROB_LEVEL_2;
    float env_PROB_LEVEL_3;
    float env_PROB_LEVEL_4;
    float env_PROB_LEVEL_5;
    float env_PROB_SURGICAL;
    float env_PROB_MEDICAL;
    float env_PROB_PEDIATRICS;
    float env_PROB_UCI;
    float env_PROB_GYNECOLOGIST;
    float env_PROB_GERIATRICS;
    float env_PROB_PSYCHIATRY;
    int env_RECEPTION_SECONDS;
    int env_RECEPTIONIST_X;
    int env_RECEPTIONIST_Y;
    


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
	in_agent_specialist_no = 0;
	in_agent_bed_no = 0;
	in_agent_priority = 0;
	in_agent_vaccine = 0;
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
	in_chair_tick = 0;
	in_chair_x = 0;
	in_chair_y = 0;
	in_chair_state = 0;
	in_bed_id = 0;
	in_bed_tick = 0;
	in_bed_state = 0;
	in_doctor_manager_tick = 0;
	in_doctor_manager_rear = 0;
	in_doctor_manager_size = 0;
	in_doctor_manager_doctors_occupied = 0;
	in_doctor_manager_free_doctors = 0;
	in_doctor_manager_patientQueue = 0;
	in_specialist_manager_id = 0;
	in_specialist_manager_tick = 0;
	in_specialist_manager_free_specialist = 0;
	in_specialist_manager_rear = 0;
	in_specialist_manager_size = 0;
	in_specialist_manager_surgicalQueue = 0;
	in_specialist_manager_pediatricsQueue = 0;
	in_specialist_manager_gynecologistQueue = 0;
	in_specialist_manager_geriatricsQueue = 0;
	in_specialist_manager_psychiatristQueue = 0;
	in_specialist_id = 0;
	in_specialist_current_patient = 0;
	in_specialist_tick = 0;
	in_receptionist_patientQueue = 0;
	in_receptionist_front = 0;
	in_receptionist_rear = 0;
	in_receptionist_size = 0;
	in_receptionist_tick = 0;
	in_receptionist_current_patient = 0;
	in_receptionist_attend_patient = 0;
	in_agent_generator_chairs_generated = 0;
	in_agent_generator_beds_generated = 0;
	in_agent_generator_boxes_generated = 0;
	in_agent_generator_doctors_generated = 0;
	in_agent_generator_specialists_generated = 0;
	in_agent_generator_personal_generated = 0;
	in_chair_admin_id = 0;
	in_chair_admin_chairArray = 0;
	in_uci_tick = 0;
	in_uci_bedArray = 0;
	in_box_id = 0;
	in_box_current_patient = 0;
	in_box_tick = 0;
	in_doctor_id = 0;
	in_doctor_current_patient = 0;
	in_doctor_tick = 0;
	in_triage_front = 0;
	in_triage_rear = 0;
	in_triage_size = 0;
	in_triage_tick = 0;
	in_triage_free_boxes = 0;
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
    in_env_SECONDS_PER_TICK = 0;
    in_env_SECONDS_INCUBATING = 0;
    in_env_SECONDS_SICK = 0;
    in_env_CLEANING_PERIOD_SECONDS = 0;
    in_env_EXIT_X = 0;
    in_env_EXIT_Y = 0;
    in_env_PROB_INFECT = 0;
    in_env_PROB_SPAWN_SICK = 0;
    in_env_PROB_INFECT_PERSONAL = 0;
    in_env_PROB_INFECT_CHAIR = 0;
    in_env_PROB_INFECT_BED = 0;
    in_env_PROB_VACCINE = 0;
    in_env_PROB_VACCINE_STAFF = 0;
    in_env_UCI_INFECTION_CHANCE = 0;
    in_env_FIRSTCHAIR_X = 0;
    in_env_FIRSTCHAIR_Y = 0;
    in_env_SPACE_BETWEEN = 0;
    in_env_DOCTOR_SECONDS = 0;
    in_env_FIRSTDOCTOR_X = 0;
    in_env_FIRSTDOCTOR_Y = 0;
    in_env_SPACE_BETWEEN_DOCTORS = 0;
    in_env_BOX_SECONDS = 0;
    in_env_TRIAGE_X = 0;
    in_env_TRIAGE_Y = 0;
    in_env_UCI_X = 0;
    in_env_UCI_Y = 0;
    in_env_NUMBER_OF_BEDS = 0;
    in_env_PROB_STAY_1 = 0;
    in_env_PROB_STAY_2 = 0;
    in_env_PROB_STAY_3 = 0;
    in_env_PROB_STAY_4 = 0;
    in_env_PROB_STAY_5 = 0;
    in_env_PROB_STAY_6 = 0;
    in_env_PROB_STAY_7 = 0;
    in_env_PROB_STAY_8 = 0;
    in_env_PROB_STAY_9 = 0;
    in_env_PROB_STAY_10 = 0;
    in_env_PROB_STAY_11 = 0;
    in_env_PROB_STAY_12 = 0;
    in_env_PROB_STAY_13 = 0;
    in_env_PROB_STAY_14 = 0;
    in_env_PROB_STAY_15 = 0;
    in_env_PROB_STAY_16 = 0;
    in_env_PROB_STAY_17 = 0;
    in_env_PROB_STAY_18 = 0;
    in_env_STAY_TIME_1 = 0;
    in_env_STAY_TIME_2 = 0;
    in_env_STAY_TIME_3 = 0;
    in_env_STAY_TIME_4 = 0;
    in_env_STAY_TIME_5 = 0;
    in_env_STAY_TIME_6 = 0;
    in_env_STAY_TIME_7 = 0;
    in_env_STAY_TIME_8 = 0;
    in_env_STAY_TIME_9 = 0;
    in_env_STAY_TIME_10 = 0;
    in_env_STAY_TIME_11 = 0;
    in_env_STAY_TIME_12 = 0;
    in_env_STAY_TIME_13 = 0;
    in_env_STAY_TIME_14 = 0;
    in_env_STAY_TIME_15 = 0;
    in_env_STAY_TIME_16 = 0;
    in_env_STAY_TIME_17 = 0;
    in_env_STAY_TIME_18 = 0;
    in_env_CHECKPOINT_1_X = 0;
    in_env_CHECKPOINT_1_Y = 0;
    in_env_CHECKPOINT_2_X = 0;
    in_env_CHECKPOINT_2_Y = 0;
    in_env_CHECKPOINT_3_X = 0;
    in_env_CHECKPOINT_3_Y = 0;
    in_env_CHECKPOINT_4_X = 0;
    in_env_CHECKPOINT_4_Y = 0;
    in_env_CHECKPOINT_5_X = 0;
    in_env_CHECKPOINT_5_Y = 0;
    in_env_SPECIALIST_SECONDS = 0;
    in_env_FIRSTSPECIALIST_X = 0;
    in_env_FIRSTSPECIALIST_Y = 0;
    in_env_SPACE_BETWEEN_SPECIALISTS = 0;
    in_env_FIFTHSPECIALIST_X = 0;
    in_env_FIFTHSPECIALIST_Y = 0;
    in_env_PROB_LEVEL_1 = 0;
    in_env_PROB_LEVEL_2 = 0;
    in_env_PROB_LEVEL_3 = 0;
    in_env_PROB_LEVEL_4 = 0;
    in_env_PROB_LEVEL_5 = 0;
    in_env_PROB_SURGICAL = 0;
    in_env_PROB_MEDICAL = 0;
    in_env_PROB_PEDIATRICS = 0;
    in_env_PROB_UCI = 0;
    in_env_PROB_GYNECOLOGIST = 0;
    in_env_PROB_GERIATRICS = 0;
    in_env_PROB_PSYCHIATRY = 0;
    in_env_RECEPTION_SECONDS = 0;
    in_env_RECEPTIONIST_X = 0;
    in_env_RECEPTIONIST_Y = 0;
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
		h_agents->specialist_no[k] = 0;
		h_agents->bed_no[k] = 0;
		h_agents->priority[k] = 0;
		h_agents->vaccine[k] = 0;
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
		h_chairs->tick[k] = 0;
		h_chairs->x[k] = 0;
		h_chairs->y[k] = 0;
		h_chairs->state[k] = 0;
	}
	
	//set all bed values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_bed_MAX; k++)
	{	
		h_beds->id[k] = 0;
		h_beds->tick[k] = 0;
		h_beds->state[k] = 0;
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
	
	//set all specialist_manager values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_specialist_manager_MAX; k++)
	{	
		h_specialist_managers->id[k] = 0;
        for (i=0;i<5;i++){
            h_specialist_managers->tick[(i*xmachine_memory_specialist_manager_MAX)+k] = 0;
        }
        for (i=0;i<5;i++){
            h_specialist_managers->free_specialist[(i*xmachine_memory_specialist_manager_MAX)+k] = 1;
        }
        for (i=0;i<5;i++){
            h_specialist_managers->rear[(i*xmachine_memory_specialist_manager_MAX)+k] = 0;
        }
        for (i=0;i<5;i++){
            h_specialist_managers->size[(i*xmachine_memory_specialist_manager_MAX)+k] = 0;
        }
        for (i=0;i<35;i++){
            h_specialist_managers->surgicalQueue[(i*xmachine_memory_specialist_manager_MAX)+k] = {-1,-1};
        }
        for (i=0;i<35;i++){
            h_specialist_managers->pediatricsQueue[(i*xmachine_memory_specialist_manager_MAX)+k] = {-1,-1};
        }
        for (i=0;i<35;i++){
            h_specialist_managers->gynecologistQueue[(i*xmachine_memory_specialist_manager_MAX)+k] = {-1,-1};
        }
        for (i=0;i<35;i++){
            h_specialist_managers->geriatricsQueue[(i*xmachine_memory_specialist_manager_MAX)+k] = {-1,-1};
        }
        for (i=0;i<35;i++){
            h_specialist_managers->psychiatristQueue[(i*xmachine_memory_specialist_manager_MAX)+k] = {-1,-1};
        }
	}
	
	//set all specialist values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_specialist_MAX; k++)
	{	
		h_specialists->id[k] = 0;
		h_specialists->current_patient[k] = 0;
		h_specialists->tick[k] = 0;
	}
	
	//set all receptionist values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_receptionist_MAX; k++)
	{	
        for (i=0;i<35;i++){
            h_receptionists->patientQueue[(i*xmachine_memory_receptionist_MAX)+k] = 0;
        }
		h_receptionists->front[k] = 0;
		h_receptionists->rear[k] = 0;
		h_receptionists->size[k] = 0;
		h_receptionists->tick[k] = 0;
		h_receptionists->current_patient[k] = -1;
		h_receptionists->attend_patient[k] = 0;
	}
	
	//set all agent_generator values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_agent_generator_MAX; k++)
	{	
		h_agent_generators->chairs_generated[k] = 0;
		h_agent_generators->beds_generated[k] = 0;
		h_agent_generators->boxes_generated[k] = 0;
		h_agent_generators->doctors_generated[k] = 0;
		h_agent_generators->specialists_generated[k] = 0;
		h_agent_generators->personal_generated[k] = 0;
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
	
	//set all uci values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_uci_MAX; k++)
	{	
		h_ucis->tick[k] = 0;
        for (i=0;i<100;i++){
            h_ucis->bedArray[(i*xmachine_memory_uci_MAX)+k] = {0,0};
        }
	}
	
	//set all box values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_box_MAX; k++)
	{	
		h_boxs->id[k] = 0;
		h_boxs->current_patient[k] = 0;
		h_boxs->tick[k] = 0;
	}
	
	//set all doctor values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_doctor_MAX; k++)
	{	
		h_doctors->id[k] = 0;
		h_doctors->current_patient[k] = 0;
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
            h_triages->free_boxes[(i*xmachine_memory_triage_MAX)+k] = 0;
        }
        for (i=0;i<35;i++){
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
    agent_specialist_no = 0;
    agent_bed_no = 0;
    agent_priority = 0;
    agent_vaccine = 0;
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
    chair_tick = 0;
    chair_x = 0;
    chair_y = 0;
    chair_state = 0;
    bed_id = 0;
    bed_tick = 0;
    bed_state = 0;
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
    specialist_manager_id = 0;
    for (i=0;i<5;i++){
        specialist_manager_tick[i] = 0;
    }
    for (i=0;i<5;i++){
        specialist_manager_free_specialist[i] = 1;
    }
    for (i=0;i<5;i++){
        specialist_manager_rear[i] = 0;
    }
    for (i=0;i<5;i++){
        specialist_manager_size[i] = 0;
    }
    for (i=0;i<35;i++){
        specialist_manager_surgicalQueue[i] = {-1,-1};
    }
    for (i=0;i<35;i++){
        specialist_manager_pediatricsQueue[i] = {-1,-1};
    }
    for (i=0;i<35;i++){
        specialist_manager_gynecologistQueue[i] = {-1,-1};
    }
    for (i=0;i<35;i++){
        specialist_manager_geriatricsQueue[i] = {-1,-1};
    }
    for (i=0;i<35;i++){
        specialist_manager_psychiatristQueue[i] = {-1,-1};
    }
    specialist_id = 0;
    specialist_current_patient = 0;
    specialist_tick = 0;
    for (i=0;i<35;i++){
        receptionist_patientQueue[i] = 0;
    }
    receptionist_front = 0;
    receptionist_rear = 0;
    receptionist_size = 0;
    receptionist_tick = 0;
    receptionist_current_patient = -1;
    receptionist_attend_patient = 0;
    agent_generator_chairs_generated = 0;
    agent_generator_beds_generated = 0;
    agent_generator_boxes_generated = 0;
    agent_generator_doctors_generated = 0;
    agent_generator_specialists_generated = 0;
    agent_generator_personal_generated = 0;
    chair_admin_id = 0;
    for (i=0;i<35;i++){
        chair_admin_chairArray[i] = 0;
    }
    uci_tick = 0;
    for (i=0;i<100;i++){
        uci_bedArray[i] = {0,0};
    }
    box_id = 0;
    box_current_patient = 0;
    box_tick = 0;
    doctor_id = 0;
    doctor_current_patient = 0;
    doctor_tick = 0;
    triage_front = 0;
    triage_rear = 0;
    triage_size = 0;
    triage_tick = 0;
    for (i=0;i<3;i++){
        triage_free_boxes[i] = 0;
    }
    for (i=0;i<35;i++){
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
    env_SECONDS_PER_TICK = 0;
    env_SECONDS_INCUBATING = 0;
    env_SECONDS_SICK = 0;
    env_CLEANING_PERIOD_SECONDS = 0;
    env_EXIT_X = 0;
    env_EXIT_Y = 0;
    env_PROB_INFECT = 0;
    env_PROB_SPAWN_SICK = 0;
    env_PROB_INFECT_PERSONAL = 0;
    env_PROB_INFECT_CHAIR = 0;
    env_PROB_INFECT_BED = 0;
    env_PROB_VACCINE = 0;
    env_PROB_VACCINE_STAFF = 0;
    env_UCI_INFECTION_CHANCE = 0;
    env_FIRSTCHAIR_X = 0;
    env_FIRSTCHAIR_Y = 0;
    env_SPACE_BETWEEN = 0;
    env_DOCTOR_SECONDS = 0;
    env_FIRSTDOCTOR_X = 0;
    env_FIRSTDOCTOR_Y = 0;
    env_SPACE_BETWEEN_DOCTORS = 0;
    env_BOX_SECONDS = 0;
    env_TRIAGE_X = 0;
    env_TRIAGE_Y = 0;
    env_UCI_X = 0;
    env_UCI_Y = 0;
    env_NUMBER_OF_BEDS = 0;
    env_PROB_STAY_1 = 0;
    env_PROB_STAY_2 = 0;
    env_PROB_STAY_3 = 0;
    env_PROB_STAY_4 = 0;
    env_PROB_STAY_5 = 0;
    env_PROB_STAY_6 = 0;
    env_PROB_STAY_7 = 0;
    env_PROB_STAY_8 = 0;
    env_PROB_STAY_9 = 0;
    env_PROB_STAY_10 = 0;
    env_PROB_STAY_11 = 0;
    env_PROB_STAY_12 = 0;
    env_PROB_STAY_13 = 0;
    env_PROB_STAY_14 = 0;
    env_PROB_STAY_15 = 0;
    env_PROB_STAY_16 = 0;
    env_PROB_STAY_17 = 0;
    env_PROB_STAY_18 = 0;
    env_STAY_TIME_1 = 0;
    env_STAY_TIME_2 = 0;
    env_STAY_TIME_3 = 0;
    env_STAY_TIME_4 = 0;
    env_STAY_TIME_5 = 0;
    env_STAY_TIME_6 = 0;
    env_STAY_TIME_7 = 0;
    env_STAY_TIME_8 = 0;
    env_STAY_TIME_9 = 0;
    env_STAY_TIME_10 = 0;
    env_STAY_TIME_11 = 0;
    env_STAY_TIME_12 = 0;
    env_STAY_TIME_13 = 0;
    env_STAY_TIME_14 = 0;
    env_STAY_TIME_15 = 0;
    env_STAY_TIME_16 = 0;
    env_STAY_TIME_17 = 0;
    env_STAY_TIME_18 = 0;
    env_CHECKPOINT_1_X = 0;
    env_CHECKPOINT_1_Y = 0;
    env_CHECKPOINT_2_X = 0;
    env_CHECKPOINT_2_Y = 0;
    env_CHECKPOINT_3_X = 0;
    env_CHECKPOINT_3_Y = 0;
    env_CHECKPOINT_4_X = 0;
    env_CHECKPOINT_4_Y = 0;
    env_CHECKPOINT_5_X = 0;
    env_CHECKPOINT_5_Y = 0;
    env_SPECIALIST_SECONDS = 0;
    env_FIRSTSPECIALIST_X = 0;
    env_FIRSTSPECIALIST_Y = 0;
    env_SPACE_BETWEEN_SPECIALISTS = 0;
    env_FIFTHSPECIALIST_X = 0;
    env_FIFTHSPECIALIST_Y = 0;
    env_PROB_LEVEL_1 = 0;
    env_PROB_LEVEL_2 = 0;
    env_PROB_LEVEL_3 = 0;
    env_PROB_LEVEL_4 = 0;
    env_PROB_LEVEL_5 = 0;
    env_PROB_SURGICAL = 0;
    env_PROB_MEDICAL = 0;
    env_PROB_PEDIATRICS = 0;
    env_PROB_UCI = 0;
    env_PROB_GYNECOLOGIST = 0;
    env_PROB_GERIATRICS = 0;
    env_PROB_PSYCHIATRY = 0;
    env_RECEPTION_SECONDS = 0;
    env_RECEPTIONIST_X = 0;
    env_RECEPTIONIST_Y = 0;
    
    
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
					h_agents->specialist_no[*h_xmachine_memory_agent_count] = agent_specialist_no;
					h_agents->bed_no[*h_xmachine_memory_agent_count] = agent_bed_no;
					h_agents->priority[*h_xmachine_memory_agent_count] = agent_priority;
					h_agents->vaccine[*h_xmachine_memory_agent_count] = agent_vaccine;
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
					h_chairs->tick[*h_xmachine_memory_chair_count] = chair_tick;
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
				else if(strcmp(agentname, "bed") == 0)
				{
					if (*h_xmachine_memory_bed_count > xmachine_memory_bed_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent bed exceeded whilst reading data\n", xmachine_memory_bed_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_beds->id[*h_xmachine_memory_bed_count] = bed_id;
					h_beds->tick[*h_xmachine_memory_bed_count] = bed_tick;
					h_beds->state[*h_xmachine_memory_bed_count] = bed_state;
					(*h_xmachine_memory_bed_count) ++;	
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
				else if(strcmp(agentname, "specialist_manager") == 0)
				{
					if (*h_xmachine_memory_specialist_manager_count > xmachine_memory_specialist_manager_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent specialist_manager exceeded whilst reading data\n", xmachine_memory_specialist_manager_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_specialist_managers->id[*h_xmachine_memory_specialist_manager_count] = specialist_manager_id;
                    for (int k=0;k<5;k++){
                        h_specialist_managers->tick[(k*xmachine_memory_specialist_manager_MAX)+(*h_xmachine_memory_specialist_manager_count)] = specialist_manager_tick[k];
                    }
                    for (int k=0;k<5;k++){
                        h_specialist_managers->free_specialist[(k*xmachine_memory_specialist_manager_MAX)+(*h_xmachine_memory_specialist_manager_count)] = specialist_manager_free_specialist[k];
                    }
                    for (int k=0;k<5;k++){
                        h_specialist_managers->rear[(k*xmachine_memory_specialist_manager_MAX)+(*h_xmachine_memory_specialist_manager_count)] = specialist_manager_rear[k];
                    }
                    for (int k=0;k<5;k++){
                        h_specialist_managers->size[(k*xmachine_memory_specialist_manager_MAX)+(*h_xmachine_memory_specialist_manager_count)] = specialist_manager_size[k];
                    }
                    for (int k=0;k<35;k++){
                        h_specialist_managers->surgicalQueue[(k*xmachine_memory_specialist_manager_MAX)+(*h_xmachine_memory_specialist_manager_count)] = specialist_manager_surgicalQueue[k];
                    }
                    for (int k=0;k<35;k++){
                        h_specialist_managers->pediatricsQueue[(k*xmachine_memory_specialist_manager_MAX)+(*h_xmachine_memory_specialist_manager_count)] = specialist_manager_pediatricsQueue[k];
                    }
                    for (int k=0;k<35;k++){
                        h_specialist_managers->gynecologistQueue[(k*xmachine_memory_specialist_manager_MAX)+(*h_xmachine_memory_specialist_manager_count)] = specialist_manager_gynecologistQueue[k];
                    }
                    for (int k=0;k<35;k++){
                        h_specialist_managers->geriatricsQueue[(k*xmachine_memory_specialist_manager_MAX)+(*h_xmachine_memory_specialist_manager_count)] = specialist_manager_geriatricsQueue[k];
                    }
                    for (int k=0;k<35;k++){
                        h_specialist_managers->psychiatristQueue[(k*xmachine_memory_specialist_manager_MAX)+(*h_xmachine_memory_specialist_manager_count)] = specialist_manager_psychiatristQueue[k];
                    }
					(*h_xmachine_memory_specialist_manager_count) ++;	
				}
				else if(strcmp(agentname, "specialist") == 0)
				{
					if (*h_xmachine_memory_specialist_count > xmachine_memory_specialist_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent specialist exceeded whilst reading data\n", xmachine_memory_specialist_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_specialists->id[*h_xmachine_memory_specialist_count] = specialist_id;
					h_specialists->current_patient[*h_xmachine_memory_specialist_count] = specialist_current_patient;
					h_specialists->tick[*h_xmachine_memory_specialist_count] = specialist_tick;
					(*h_xmachine_memory_specialist_count) ++;	
				}
				else if(strcmp(agentname, "receptionist") == 0)
				{
					if (*h_xmachine_memory_receptionist_count > xmachine_memory_receptionist_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent receptionist exceeded whilst reading data\n", xmachine_memory_receptionist_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
                    for (int k=0;k<35;k++){
                        h_receptionists->patientQueue[(k*xmachine_memory_receptionist_MAX)+(*h_xmachine_memory_receptionist_count)] = receptionist_patientQueue[k];
                    }
					h_receptionists->front[*h_xmachine_memory_receptionist_count] = receptionist_front;
					h_receptionists->rear[*h_xmachine_memory_receptionist_count] = receptionist_rear;
					h_receptionists->size[*h_xmachine_memory_receptionist_count] = receptionist_size;
					h_receptionists->tick[*h_xmachine_memory_receptionist_count] = receptionist_tick;
					h_receptionists->current_patient[*h_xmachine_memory_receptionist_count] = receptionist_current_patient;
					h_receptionists->attend_patient[*h_xmachine_memory_receptionist_count] = receptionist_attend_patient;
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
					h_agent_generators->beds_generated[*h_xmachine_memory_agent_generator_count] = agent_generator_beds_generated;
					h_agent_generators->boxes_generated[*h_xmachine_memory_agent_generator_count] = agent_generator_boxes_generated;
					h_agent_generators->doctors_generated[*h_xmachine_memory_agent_generator_count] = agent_generator_doctors_generated;
					h_agent_generators->specialists_generated[*h_xmachine_memory_agent_generator_count] = agent_generator_specialists_generated;
					h_agent_generators->personal_generated[*h_xmachine_memory_agent_generator_count] = agent_generator_personal_generated;
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
				else if(strcmp(agentname, "uci") == 0)
				{
					if (*h_xmachine_memory_uci_count > xmachine_memory_uci_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent uci exceeded whilst reading data\n", xmachine_memory_uci_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_ucis->tick[*h_xmachine_memory_uci_count] = uci_tick;
                    for (int k=0;k<100;k++){
                        h_ucis->bedArray[(k*xmachine_memory_uci_MAX)+(*h_xmachine_memory_uci_count)] = uci_bedArray[k];
                    }
					(*h_xmachine_memory_uci_count) ++;	
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
					h_boxs->current_patient[*h_xmachine_memory_box_count] = box_current_patient;
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
					h_doctors->current_patient[*h_xmachine_memory_doctor_count] = doctor_current_patient;
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
                        h_triages->free_boxes[(k*xmachine_memory_triage_MAX)+(*h_xmachine_memory_triage_count)] = triage_free_boxes[k];
                    }
                    for (int k=0;k<35;k++){
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
                agent_specialist_no = 0;
                agent_bed_no = 0;
                agent_priority = 0;
                agent_vaccine = 0;
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
                chair_tick = 0;
                chair_x = 0;
                chair_y = 0;
                chair_state = 0;
                bed_id = 0;
                bed_tick = 0;
                bed_state = 0;
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
                specialist_manager_id = 0;
                for (i=0;i<5;i++){
                    specialist_manager_tick[i] = 0;
                }
                for (i=0;i<5;i++){
                    specialist_manager_free_specialist[i] = 1;
                }
                for (i=0;i<5;i++){
                    specialist_manager_rear[i] = 0;
                }
                for (i=0;i<5;i++){
                    specialist_manager_size[i] = 0;
                }
                for (i=0;i<35;i++){
                    specialist_manager_surgicalQueue[i] = {-1,-1};
                }
                for (i=0;i<35;i++){
                    specialist_manager_pediatricsQueue[i] = {-1,-1};
                }
                for (i=0;i<35;i++){
                    specialist_manager_gynecologistQueue[i] = {-1,-1};
                }
                for (i=0;i<35;i++){
                    specialist_manager_geriatricsQueue[i] = {-1,-1};
                }
                for (i=0;i<35;i++){
                    specialist_manager_psychiatristQueue[i] = {-1,-1};
                }
                specialist_id = 0;
                specialist_current_patient = 0;
                specialist_tick = 0;
                for (i=0;i<35;i++){
                    receptionist_patientQueue[i] = 0;
                }
                receptionist_front = 0;
                receptionist_rear = 0;
                receptionist_size = 0;
                receptionist_tick = 0;
                receptionist_current_patient = -1;
                receptionist_attend_patient = 0;
                agent_generator_chairs_generated = 0;
                agent_generator_beds_generated = 0;
                agent_generator_boxes_generated = 0;
                agent_generator_doctors_generated = 0;
                agent_generator_specialists_generated = 0;
                agent_generator_personal_generated = 0;
                chair_admin_id = 0;
                for (i=0;i<35;i++){
                    chair_admin_chairArray[i] = 0;
                }
                uci_tick = 0;
                for (i=0;i<100;i++){
                    uci_bedArray[i] = {0,0};
                }
                box_id = 0;
                box_current_patient = 0;
                box_tick = 0;
                doctor_id = 0;
                doctor_current_patient = 0;
                doctor_tick = 0;
                triage_front = 0;
                triage_rear = 0;
                triage_size = 0;
                triage_tick = 0;
                for (i=0;i<3;i++){
                    triage_free_boxes[i] = 0;
                }
                for (i=0;i<35;i++){
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
			if(strcmp(buffer, "specialist_no") == 0) in_agent_specialist_no = 1;
			if(strcmp(buffer, "/specialist_no") == 0) in_agent_specialist_no = 0;
			if(strcmp(buffer, "bed_no") == 0) in_agent_bed_no = 1;
			if(strcmp(buffer, "/bed_no") == 0) in_agent_bed_no = 0;
			if(strcmp(buffer, "priority") == 0) in_agent_priority = 1;
			if(strcmp(buffer, "/priority") == 0) in_agent_priority = 0;
			if(strcmp(buffer, "vaccine") == 0) in_agent_vaccine = 1;
			if(strcmp(buffer, "/vaccine") == 0) in_agent_vaccine = 0;
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
			if(strcmp(buffer, "tick") == 0) in_chair_tick = 1;
			if(strcmp(buffer, "/tick") == 0) in_chair_tick = 0;
			if(strcmp(buffer, "x") == 0) in_chair_x = 1;
			if(strcmp(buffer, "/x") == 0) in_chair_x = 0;
			if(strcmp(buffer, "y") == 0) in_chair_y = 1;
			if(strcmp(buffer, "/y") == 0) in_chair_y = 0;
			if(strcmp(buffer, "state") == 0) in_chair_state = 1;
			if(strcmp(buffer, "/state") == 0) in_chair_state = 0;
			if(strcmp(buffer, "id") == 0) in_bed_id = 1;
			if(strcmp(buffer, "/id") == 0) in_bed_id = 0;
			if(strcmp(buffer, "tick") == 0) in_bed_tick = 1;
			if(strcmp(buffer, "/tick") == 0) in_bed_tick = 0;
			if(strcmp(buffer, "state") == 0) in_bed_state = 1;
			if(strcmp(buffer, "/state") == 0) in_bed_state = 0;
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
			if(strcmp(buffer, "id") == 0) in_specialist_manager_id = 1;
			if(strcmp(buffer, "/id") == 0) in_specialist_manager_id = 0;
			if(strcmp(buffer, "tick") == 0) in_specialist_manager_tick = 1;
			if(strcmp(buffer, "/tick") == 0) in_specialist_manager_tick = 0;
			if(strcmp(buffer, "free_specialist") == 0) in_specialist_manager_free_specialist = 1;
			if(strcmp(buffer, "/free_specialist") == 0) in_specialist_manager_free_specialist = 0;
			if(strcmp(buffer, "rear") == 0) in_specialist_manager_rear = 1;
			if(strcmp(buffer, "/rear") == 0) in_specialist_manager_rear = 0;
			if(strcmp(buffer, "size") == 0) in_specialist_manager_size = 1;
			if(strcmp(buffer, "/size") == 0) in_specialist_manager_size = 0;
			if(strcmp(buffer, "surgicalQueue") == 0) in_specialist_manager_surgicalQueue = 1;
			if(strcmp(buffer, "/surgicalQueue") == 0) in_specialist_manager_surgicalQueue = 0;
			if(strcmp(buffer, "pediatricsQueue") == 0) in_specialist_manager_pediatricsQueue = 1;
			if(strcmp(buffer, "/pediatricsQueue") == 0) in_specialist_manager_pediatricsQueue = 0;
			if(strcmp(buffer, "gynecologistQueue") == 0) in_specialist_manager_gynecologistQueue = 1;
			if(strcmp(buffer, "/gynecologistQueue") == 0) in_specialist_manager_gynecologistQueue = 0;
			if(strcmp(buffer, "geriatricsQueue") == 0) in_specialist_manager_geriatricsQueue = 1;
			if(strcmp(buffer, "/geriatricsQueue") == 0) in_specialist_manager_geriatricsQueue = 0;
			if(strcmp(buffer, "psychiatristQueue") == 0) in_specialist_manager_psychiatristQueue = 1;
			if(strcmp(buffer, "/psychiatristQueue") == 0) in_specialist_manager_psychiatristQueue = 0;
			if(strcmp(buffer, "id") == 0) in_specialist_id = 1;
			if(strcmp(buffer, "/id") == 0) in_specialist_id = 0;
			if(strcmp(buffer, "current_patient") == 0) in_specialist_current_patient = 1;
			if(strcmp(buffer, "/current_patient") == 0) in_specialist_current_patient = 0;
			if(strcmp(buffer, "tick") == 0) in_specialist_tick = 1;
			if(strcmp(buffer, "/tick") == 0) in_specialist_tick = 0;
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
			if(strcmp(buffer, "chairs_generated") == 0) in_agent_generator_chairs_generated = 1;
			if(strcmp(buffer, "/chairs_generated") == 0) in_agent_generator_chairs_generated = 0;
			if(strcmp(buffer, "beds_generated") == 0) in_agent_generator_beds_generated = 1;
			if(strcmp(buffer, "/beds_generated") == 0) in_agent_generator_beds_generated = 0;
			if(strcmp(buffer, "boxes_generated") == 0) in_agent_generator_boxes_generated = 1;
			if(strcmp(buffer, "/boxes_generated") == 0) in_agent_generator_boxes_generated = 0;
			if(strcmp(buffer, "doctors_generated") == 0) in_agent_generator_doctors_generated = 1;
			if(strcmp(buffer, "/doctors_generated") == 0) in_agent_generator_doctors_generated = 0;
			if(strcmp(buffer, "specialists_generated") == 0) in_agent_generator_specialists_generated = 1;
			if(strcmp(buffer, "/specialists_generated") == 0) in_agent_generator_specialists_generated = 0;
			if(strcmp(buffer, "personal_generated") == 0) in_agent_generator_personal_generated = 1;
			if(strcmp(buffer, "/personal_generated") == 0) in_agent_generator_personal_generated = 0;
			if(strcmp(buffer, "id") == 0) in_chair_admin_id = 1;
			if(strcmp(buffer, "/id") == 0) in_chair_admin_id = 0;
			if(strcmp(buffer, "chairArray") == 0) in_chair_admin_chairArray = 1;
			if(strcmp(buffer, "/chairArray") == 0) in_chair_admin_chairArray = 0;
			if(strcmp(buffer, "tick") == 0) in_uci_tick = 1;
			if(strcmp(buffer, "/tick") == 0) in_uci_tick = 0;
			if(strcmp(buffer, "bedArray") == 0) in_uci_bedArray = 1;
			if(strcmp(buffer, "/bedArray") == 0) in_uci_bedArray = 0;
			if(strcmp(buffer, "id") == 0) in_box_id = 1;
			if(strcmp(buffer, "/id") == 0) in_box_id = 0;
			if(strcmp(buffer, "current_patient") == 0) in_box_current_patient = 1;
			if(strcmp(buffer, "/current_patient") == 0) in_box_current_patient = 0;
			if(strcmp(buffer, "tick") == 0) in_box_tick = 1;
			if(strcmp(buffer, "/tick") == 0) in_box_tick = 0;
			if(strcmp(buffer, "id") == 0) in_doctor_id = 1;
			if(strcmp(buffer, "/id") == 0) in_doctor_id = 0;
			if(strcmp(buffer, "current_patient") == 0) in_doctor_current_patient = 1;
			if(strcmp(buffer, "/current_patient") == 0) in_doctor_current_patient = 0;
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
			if(strcmp(buffer, "free_boxes") == 0) in_triage_free_boxes = 1;
			if(strcmp(buffer, "/free_boxes") == 0) in_triage_free_boxes = 0;
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
			if(strcmp(buffer, "SECONDS_PER_TICK") == 0) in_env_SECONDS_PER_TICK = 1;
            if(strcmp(buffer, "/SECONDS_PER_TICK") == 0) in_env_SECONDS_PER_TICK = 0;
			if(strcmp(buffer, "SECONDS_INCUBATING") == 0) in_env_SECONDS_INCUBATING = 1;
            if(strcmp(buffer, "/SECONDS_INCUBATING") == 0) in_env_SECONDS_INCUBATING = 0;
			if(strcmp(buffer, "SECONDS_SICK") == 0) in_env_SECONDS_SICK = 1;
            if(strcmp(buffer, "/SECONDS_SICK") == 0) in_env_SECONDS_SICK = 0;
			if(strcmp(buffer, "CLEANING_PERIOD_SECONDS") == 0) in_env_CLEANING_PERIOD_SECONDS = 1;
            if(strcmp(buffer, "/CLEANING_PERIOD_SECONDS") == 0) in_env_CLEANING_PERIOD_SECONDS = 0;
			if(strcmp(buffer, "EXIT_X") == 0) in_env_EXIT_X = 1;
            if(strcmp(buffer, "/EXIT_X") == 0) in_env_EXIT_X = 0;
			if(strcmp(buffer, "EXIT_Y") == 0) in_env_EXIT_Y = 1;
            if(strcmp(buffer, "/EXIT_Y") == 0) in_env_EXIT_Y = 0;
			if(strcmp(buffer, "PROB_INFECT") == 0) in_env_PROB_INFECT = 1;
            if(strcmp(buffer, "/PROB_INFECT") == 0) in_env_PROB_INFECT = 0;
			if(strcmp(buffer, "PROB_SPAWN_SICK") == 0) in_env_PROB_SPAWN_SICK = 1;
            if(strcmp(buffer, "/PROB_SPAWN_SICK") == 0) in_env_PROB_SPAWN_SICK = 0;
			if(strcmp(buffer, "PROB_INFECT_PERSONAL") == 0) in_env_PROB_INFECT_PERSONAL = 1;
            if(strcmp(buffer, "/PROB_INFECT_PERSONAL") == 0) in_env_PROB_INFECT_PERSONAL = 0;
			if(strcmp(buffer, "PROB_INFECT_CHAIR") == 0) in_env_PROB_INFECT_CHAIR = 1;
            if(strcmp(buffer, "/PROB_INFECT_CHAIR") == 0) in_env_PROB_INFECT_CHAIR = 0;
			if(strcmp(buffer, "PROB_INFECT_BED") == 0) in_env_PROB_INFECT_BED = 1;
            if(strcmp(buffer, "/PROB_INFECT_BED") == 0) in_env_PROB_INFECT_BED = 0;
			if(strcmp(buffer, "PROB_VACCINE") == 0) in_env_PROB_VACCINE = 1;
            if(strcmp(buffer, "/PROB_VACCINE") == 0) in_env_PROB_VACCINE = 0;
			if(strcmp(buffer, "PROB_VACCINE_STAFF") == 0) in_env_PROB_VACCINE_STAFF = 1;
            if(strcmp(buffer, "/PROB_VACCINE_STAFF") == 0) in_env_PROB_VACCINE_STAFF = 0;
			if(strcmp(buffer, "UCI_INFECTION_CHANCE") == 0) in_env_UCI_INFECTION_CHANCE = 1;
            if(strcmp(buffer, "/UCI_INFECTION_CHANCE") == 0) in_env_UCI_INFECTION_CHANCE = 0;
			if(strcmp(buffer, "FIRSTCHAIR_X") == 0) in_env_FIRSTCHAIR_X = 1;
            if(strcmp(buffer, "/FIRSTCHAIR_X") == 0) in_env_FIRSTCHAIR_X = 0;
			if(strcmp(buffer, "FIRSTCHAIR_Y") == 0) in_env_FIRSTCHAIR_Y = 1;
            if(strcmp(buffer, "/FIRSTCHAIR_Y") == 0) in_env_FIRSTCHAIR_Y = 0;
			if(strcmp(buffer, "SPACE_BETWEEN") == 0) in_env_SPACE_BETWEEN = 1;
            if(strcmp(buffer, "/SPACE_BETWEEN") == 0) in_env_SPACE_BETWEEN = 0;
			if(strcmp(buffer, "DOCTOR_SECONDS") == 0) in_env_DOCTOR_SECONDS = 1;
            if(strcmp(buffer, "/DOCTOR_SECONDS") == 0) in_env_DOCTOR_SECONDS = 0;
			if(strcmp(buffer, "FIRSTDOCTOR_X") == 0) in_env_FIRSTDOCTOR_X = 1;
            if(strcmp(buffer, "/FIRSTDOCTOR_X") == 0) in_env_FIRSTDOCTOR_X = 0;
			if(strcmp(buffer, "FIRSTDOCTOR_Y") == 0) in_env_FIRSTDOCTOR_Y = 1;
            if(strcmp(buffer, "/FIRSTDOCTOR_Y") == 0) in_env_FIRSTDOCTOR_Y = 0;
			if(strcmp(buffer, "SPACE_BETWEEN_DOCTORS") == 0) in_env_SPACE_BETWEEN_DOCTORS = 1;
            if(strcmp(buffer, "/SPACE_BETWEEN_DOCTORS") == 0) in_env_SPACE_BETWEEN_DOCTORS = 0;
			if(strcmp(buffer, "BOX_SECONDS") == 0) in_env_BOX_SECONDS = 1;
            if(strcmp(buffer, "/BOX_SECONDS") == 0) in_env_BOX_SECONDS = 0;
			if(strcmp(buffer, "TRIAGE_X") == 0) in_env_TRIAGE_X = 1;
            if(strcmp(buffer, "/TRIAGE_X") == 0) in_env_TRIAGE_X = 0;
			if(strcmp(buffer, "TRIAGE_Y") == 0) in_env_TRIAGE_Y = 1;
            if(strcmp(buffer, "/TRIAGE_Y") == 0) in_env_TRIAGE_Y = 0;
			if(strcmp(buffer, "UCI_X") == 0) in_env_UCI_X = 1;
            if(strcmp(buffer, "/UCI_X") == 0) in_env_UCI_X = 0;
			if(strcmp(buffer, "UCI_Y") == 0) in_env_UCI_Y = 1;
            if(strcmp(buffer, "/UCI_Y") == 0) in_env_UCI_Y = 0;
			if(strcmp(buffer, "NUMBER_OF_BEDS") == 0) in_env_NUMBER_OF_BEDS = 1;
            if(strcmp(buffer, "/NUMBER_OF_BEDS") == 0) in_env_NUMBER_OF_BEDS = 0;
			if(strcmp(buffer, "PROB_STAY_1") == 0) in_env_PROB_STAY_1 = 1;
            if(strcmp(buffer, "/PROB_STAY_1") == 0) in_env_PROB_STAY_1 = 0;
			if(strcmp(buffer, "PROB_STAY_2") == 0) in_env_PROB_STAY_2 = 1;
            if(strcmp(buffer, "/PROB_STAY_2") == 0) in_env_PROB_STAY_2 = 0;
			if(strcmp(buffer, "PROB_STAY_3") == 0) in_env_PROB_STAY_3 = 1;
            if(strcmp(buffer, "/PROB_STAY_3") == 0) in_env_PROB_STAY_3 = 0;
			if(strcmp(buffer, "PROB_STAY_4") == 0) in_env_PROB_STAY_4 = 1;
            if(strcmp(buffer, "/PROB_STAY_4") == 0) in_env_PROB_STAY_4 = 0;
			if(strcmp(buffer, "PROB_STAY_5") == 0) in_env_PROB_STAY_5 = 1;
            if(strcmp(buffer, "/PROB_STAY_5") == 0) in_env_PROB_STAY_5 = 0;
			if(strcmp(buffer, "PROB_STAY_6") == 0) in_env_PROB_STAY_6 = 1;
            if(strcmp(buffer, "/PROB_STAY_6") == 0) in_env_PROB_STAY_6 = 0;
			if(strcmp(buffer, "PROB_STAY_7") == 0) in_env_PROB_STAY_7 = 1;
            if(strcmp(buffer, "/PROB_STAY_7") == 0) in_env_PROB_STAY_7 = 0;
			if(strcmp(buffer, "PROB_STAY_8") == 0) in_env_PROB_STAY_8 = 1;
            if(strcmp(buffer, "/PROB_STAY_8") == 0) in_env_PROB_STAY_8 = 0;
			if(strcmp(buffer, "PROB_STAY_9") == 0) in_env_PROB_STAY_9 = 1;
            if(strcmp(buffer, "/PROB_STAY_9") == 0) in_env_PROB_STAY_9 = 0;
			if(strcmp(buffer, "PROB_STAY_10") == 0) in_env_PROB_STAY_10 = 1;
            if(strcmp(buffer, "/PROB_STAY_10") == 0) in_env_PROB_STAY_10 = 0;
			if(strcmp(buffer, "PROB_STAY_11") == 0) in_env_PROB_STAY_11 = 1;
            if(strcmp(buffer, "/PROB_STAY_11") == 0) in_env_PROB_STAY_11 = 0;
			if(strcmp(buffer, "PROB_STAY_12") == 0) in_env_PROB_STAY_12 = 1;
            if(strcmp(buffer, "/PROB_STAY_12") == 0) in_env_PROB_STAY_12 = 0;
			if(strcmp(buffer, "PROB_STAY_13") == 0) in_env_PROB_STAY_13 = 1;
            if(strcmp(buffer, "/PROB_STAY_13") == 0) in_env_PROB_STAY_13 = 0;
			if(strcmp(buffer, "PROB_STAY_14") == 0) in_env_PROB_STAY_14 = 1;
            if(strcmp(buffer, "/PROB_STAY_14") == 0) in_env_PROB_STAY_14 = 0;
			if(strcmp(buffer, "PROB_STAY_15") == 0) in_env_PROB_STAY_15 = 1;
            if(strcmp(buffer, "/PROB_STAY_15") == 0) in_env_PROB_STAY_15 = 0;
			if(strcmp(buffer, "PROB_STAY_16") == 0) in_env_PROB_STAY_16 = 1;
            if(strcmp(buffer, "/PROB_STAY_16") == 0) in_env_PROB_STAY_16 = 0;
			if(strcmp(buffer, "PROB_STAY_17") == 0) in_env_PROB_STAY_17 = 1;
            if(strcmp(buffer, "/PROB_STAY_17") == 0) in_env_PROB_STAY_17 = 0;
			if(strcmp(buffer, "PROB_STAY_18") == 0) in_env_PROB_STAY_18 = 1;
            if(strcmp(buffer, "/PROB_STAY_18") == 0) in_env_PROB_STAY_18 = 0;
			if(strcmp(buffer, "STAY_TIME_1") == 0) in_env_STAY_TIME_1 = 1;
            if(strcmp(buffer, "/STAY_TIME_1") == 0) in_env_STAY_TIME_1 = 0;
			if(strcmp(buffer, "STAY_TIME_2") == 0) in_env_STAY_TIME_2 = 1;
            if(strcmp(buffer, "/STAY_TIME_2") == 0) in_env_STAY_TIME_2 = 0;
			if(strcmp(buffer, "STAY_TIME_3") == 0) in_env_STAY_TIME_3 = 1;
            if(strcmp(buffer, "/STAY_TIME_3") == 0) in_env_STAY_TIME_3 = 0;
			if(strcmp(buffer, "STAY_TIME_4") == 0) in_env_STAY_TIME_4 = 1;
            if(strcmp(buffer, "/STAY_TIME_4") == 0) in_env_STAY_TIME_4 = 0;
			if(strcmp(buffer, "STAY_TIME_5") == 0) in_env_STAY_TIME_5 = 1;
            if(strcmp(buffer, "/STAY_TIME_5") == 0) in_env_STAY_TIME_5 = 0;
			if(strcmp(buffer, "STAY_TIME_6") == 0) in_env_STAY_TIME_6 = 1;
            if(strcmp(buffer, "/STAY_TIME_6") == 0) in_env_STAY_TIME_6 = 0;
			if(strcmp(buffer, "STAY_TIME_7") == 0) in_env_STAY_TIME_7 = 1;
            if(strcmp(buffer, "/STAY_TIME_7") == 0) in_env_STAY_TIME_7 = 0;
			if(strcmp(buffer, "STAY_TIME_8") == 0) in_env_STAY_TIME_8 = 1;
            if(strcmp(buffer, "/STAY_TIME_8") == 0) in_env_STAY_TIME_8 = 0;
			if(strcmp(buffer, "STAY_TIME_9") == 0) in_env_STAY_TIME_9 = 1;
            if(strcmp(buffer, "/STAY_TIME_9") == 0) in_env_STAY_TIME_9 = 0;
			if(strcmp(buffer, "STAY_TIME_10") == 0) in_env_STAY_TIME_10 = 1;
            if(strcmp(buffer, "/STAY_TIME_10") == 0) in_env_STAY_TIME_10 = 0;
			if(strcmp(buffer, "STAY_TIME_11") == 0) in_env_STAY_TIME_11 = 1;
            if(strcmp(buffer, "/STAY_TIME_11") == 0) in_env_STAY_TIME_11 = 0;
			if(strcmp(buffer, "STAY_TIME_12") == 0) in_env_STAY_TIME_12 = 1;
            if(strcmp(buffer, "/STAY_TIME_12") == 0) in_env_STAY_TIME_12 = 0;
			if(strcmp(buffer, "STAY_TIME_13") == 0) in_env_STAY_TIME_13 = 1;
            if(strcmp(buffer, "/STAY_TIME_13") == 0) in_env_STAY_TIME_13 = 0;
			if(strcmp(buffer, "STAY_TIME_14") == 0) in_env_STAY_TIME_14 = 1;
            if(strcmp(buffer, "/STAY_TIME_14") == 0) in_env_STAY_TIME_14 = 0;
			if(strcmp(buffer, "STAY_TIME_15") == 0) in_env_STAY_TIME_15 = 1;
            if(strcmp(buffer, "/STAY_TIME_15") == 0) in_env_STAY_TIME_15 = 0;
			if(strcmp(buffer, "STAY_TIME_16") == 0) in_env_STAY_TIME_16 = 1;
            if(strcmp(buffer, "/STAY_TIME_16") == 0) in_env_STAY_TIME_16 = 0;
			if(strcmp(buffer, "STAY_TIME_17") == 0) in_env_STAY_TIME_17 = 1;
            if(strcmp(buffer, "/STAY_TIME_17") == 0) in_env_STAY_TIME_17 = 0;
			if(strcmp(buffer, "STAY_TIME_18") == 0) in_env_STAY_TIME_18 = 1;
            if(strcmp(buffer, "/STAY_TIME_18") == 0) in_env_STAY_TIME_18 = 0;
			if(strcmp(buffer, "CHECKPOINT_1_X") == 0) in_env_CHECKPOINT_1_X = 1;
            if(strcmp(buffer, "/CHECKPOINT_1_X") == 0) in_env_CHECKPOINT_1_X = 0;
			if(strcmp(buffer, "CHECKPOINT_1_Y") == 0) in_env_CHECKPOINT_1_Y = 1;
            if(strcmp(buffer, "/CHECKPOINT_1_Y") == 0) in_env_CHECKPOINT_1_Y = 0;
			if(strcmp(buffer, "CHECKPOINT_2_X") == 0) in_env_CHECKPOINT_2_X = 1;
            if(strcmp(buffer, "/CHECKPOINT_2_X") == 0) in_env_CHECKPOINT_2_X = 0;
			if(strcmp(buffer, "CHECKPOINT_2_Y") == 0) in_env_CHECKPOINT_2_Y = 1;
            if(strcmp(buffer, "/CHECKPOINT_2_Y") == 0) in_env_CHECKPOINT_2_Y = 0;
			if(strcmp(buffer, "CHECKPOINT_3_X") == 0) in_env_CHECKPOINT_3_X = 1;
            if(strcmp(buffer, "/CHECKPOINT_3_X") == 0) in_env_CHECKPOINT_3_X = 0;
			if(strcmp(buffer, "CHECKPOINT_3_Y") == 0) in_env_CHECKPOINT_3_Y = 1;
            if(strcmp(buffer, "/CHECKPOINT_3_Y") == 0) in_env_CHECKPOINT_3_Y = 0;
			if(strcmp(buffer, "CHECKPOINT_4_X") == 0) in_env_CHECKPOINT_4_X = 1;
            if(strcmp(buffer, "/CHECKPOINT_4_X") == 0) in_env_CHECKPOINT_4_X = 0;
			if(strcmp(buffer, "CHECKPOINT_4_Y") == 0) in_env_CHECKPOINT_4_Y = 1;
            if(strcmp(buffer, "/CHECKPOINT_4_Y") == 0) in_env_CHECKPOINT_4_Y = 0;
			if(strcmp(buffer, "CHECKPOINT_5_X") == 0) in_env_CHECKPOINT_5_X = 1;
            if(strcmp(buffer, "/CHECKPOINT_5_X") == 0) in_env_CHECKPOINT_5_X = 0;
			if(strcmp(buffer, "CHECKPOINT_5_Y") == 0) in_env_CHECKPOINT_5_Y = 1;
            if(strcmp(buffer, "/CHECKPOINT_5_Y") == 0) in_env_CHECKPOINT_5_Y = 0;
			if(strcmp(buffer, "SPECIALIST_SECONDS") == 0) in_env_SPECIALIST_SECONDS = 1;
            if(strcmp(buffer, "/SPECIALIST_SECONDS") == 0) in_env_SPECIALIST_SECONDS = 0;
			if(strcmp(buffer, "FIRSTSPECIALIST_X") == 0) in_env_FIRSTSPECIALIST_X = 1;
            if(strcmp(buffer, "/FIRSTSPECIALIST_X") == 0) in_env_FIRSTSPECIALIST_X = 0;
			if(strcmp(buffer, "FIRSTSPECIALIST_Y") == 0) in_env_FIRSTSPECIALIST_Y = 1;
            if(strcmp(buffer, "/FIRSTSPECIALIST_Y") == 0) in_env_FIRSTSPECIALIST_Y = 0;
			if(strcmp(buffer, "SPACE_BETWEEN_SPECIALISTS") == 0) in_env_SPACE_BETWEEN_SPECIALISTS = 1;
            if(strcmp(buffer, "/SPACE_BETWEEN_SPECIALISTS") == 0) in_env_SPACE_BETWEEN_SPECIALISTS = 0;
			if(strcmp(buffer, "FIFTHSPECIALIST_X") == 0) in_env_FIFTHSPECIALIST_X = 1;
            if(strcmp(buffer, "/FIFTHSPECIALIST_X") == 0) in_env_FIFTHSPECIALIST_X = 0;
			if(strcmp(buffer, "FIFTHSPECIALIST_Y") == 0) in_env_FIFTHSPECIALIST_Y = 1;
            if(strcmp(buffer, "/FIFTHSPECIALIST_Y") == 0) in_env_FIFTHSPECIALIST_Y = 0;
			if(strcmp(buffer, "PROB_LEVEL_1") == 0) in_env_PROB_LEVEL_1 = 1;
            if(strcmp(buffer, "/PROB_LEVEL_1") == 0) in_env_PROB_LEVEL_1 = 0;
			if(strcmp(buffer, "PROB_LEVEL_2") == 0) in_env_PROB_LEVEL_2 = 1;
            if(strcmp(buffer, "/PROB_LEVEL_2") == 0) in_env_PROB_LEVEL_2 = 0;
			if(strcmp(buffer, "PROB_LEVEL_3") == 0) in_env_PROB_LEVEL_3 = 1;
            if(strcmp(buffer, "/PROB_LEVEL_3") == 0) in_env_PROB_LEVEL_3 = 0;
			if(strcmp(buffer, "PROB_LEVEL_4") == 0) in_env_PROB_LEVEL_4 = 1;
            if(strcmp(buffer, "/PROB_LEVEL_4") == 0) in_env_PROB_LEVEL_4 = 0;
			if(strcmp(buffer, "PROB_LEVEL_5") == 0) in_env_PROB_LEVEL_5 = 1;
            if(strcmp(buffer, "/PROB_LEVEL_5") == 0) in_env_PROB_LEVEL_5 = 0;
			if(strcmp(buffer, "PROB_SURGICAL") == 0) in_env_PROB_SURGICAL = 1;
            if(strcmp(buffer, "/PROB_SURGICAL") == 0) in_env_PROB_SURGICAL = 0;
			if(strcmp(buffer, "PROB_MEDICAL") == 0) in_env_PROB_MEDICAL = 1;
            if(strcmp(buffer, "/PROB_MEDICAL") == 0) in_env_PROB_MEDICAL = 0;
			if(strcmp(buffer, "PROB_PEDIATRICS") == 0) in_env_PROB_PEDIATRICS = 1;
            if(strcmp(buffer, "/PROB_PEDIATRICS") == 0) in_env_PROB_PEDIATRICS = 0;
			if(strcmp(buffer, "PROB_UCI") == 0) in_env_PROB_UCI = 1;
            if(strcmp(buffer, "/PROB_UCI") == 0) in_env_PROB_UCI = 0;
			if(strcmp(buffer, "PROB_GYNECOLOGIST") == 0) in_env_PROB_GYNECOLOGIST = 1;
            if(strcmp(buffer, "/PROB_GYNECOLOGIST") == 0) in_env_PROB_GYNECOLOGIST = 0;
			if(strcmp(buffer, "PROB_GERIATRICS") == 0) in_env_PROB_GERIATRICS = 1;
            if(strcmp(buffer, "/PROB_GERIATRICS") == 0) in_env_PROB_GERIATRICS = 0;
			if(strcmp(buffer, "PROB_PSYCHIATRY") == 0) in_env_PROB_PSYCHIATRY = 1;
            if(strcmp(buffer, "/PROB_PSYCHIATRY") == 0) in_env_PROB_PSYCHIATRY = 0;
			if(strcmp(buffer, "RECEPTION_SECONDS") == 0) in_env_RECEPTION_SECONDS = 1;
            if(strcmp(buffer, "/RECEPTION_SECONDS") == 0) in_env_RECEPTION_SECONDS = 0;
			if(strcmp(buffer, "RECEPTIONIST_X") == 0) in_env_RECEPTIONIST_X = 1;
            if(strcmp(buffer, "/RECEPTIONIST_X") == 0) in_env_RECEPTIONIST_X = 0;
			if(strcmp(buffer, "RECEPTIONIST_Y") == 0) in_env_RECEPTIONIST_Y = 1;
            if(strcmp(buffer, "/RECEPTIONIST_Y") == 0) in_env_RECEPTIONIST_Y = 0;
			

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
                    agent_id = (int) fpgu_strtol(buffer); 
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
				if(in_agent_specialist_no){
                    agent_specialist_no = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_agent_bed_no){
                    agent_bed_no = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_agent_priority){
                    agent_priority = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_agent_vaccine){
                    agent_vaccine = (unsigned int) fpgu_strtoul(buffer); 
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
				if(in_chair_tick){
                    chair_tick = (int) fpgu_strtol(buffer); 
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
				if(in_bed_id){
                    bed_id = (int) fpgu_strtol(buffer); 
                }
				if(in_bed_tick){
                    bed_tick = (int) fpgu_strtol(buffer); 
                }
				if(in_bed_state){
                    bed_state = (int) fpgu_strtol(buffer); 
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
				if(in_specialist_manager_id){
                    specialist_manager_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_specialist_manager_tick){
                    readArrayInput<unsigned int>(&fpgu_strtoul, buffer, specialist_manager_tick, 5);    
                }
				if(in_specialist_manager_free_specialist){
                    readArrayInput<unsigned int>(&fpgu_strtoul, buffer, specialist_manager_free_specialist, 5);    
                }
				if(in_specialist_manager_rear){
                    readArrayInput<unsigned int>(&fpgu_strtoul, buffer, specialist_manager_rear, 5);    
                }
				if(in_specialist_manager_size){
                    readArrayInput<unsigned int>(&fpgu_strtoul, buffer, specialist_manager_size, 5);    
                }
				if(in_specialist_manager_surgicalQueue){
                    readArrayInputVectorType<ivec2, int, 2>(&fpgu_strtol, buffer, specialist_manager_surgicalQueue, 35);    
                }
				if(in_specialist_manager_pediatricsQueue){
                    readArrayInputVectorType<ivec2, int, 2>(&fpgu_strtol, buffer, specialist_manager_pediatricsQueue, 35);    
                }
				if(in_specialist_manager_gynecologistQueue){
                    readArrayInputVectorType<ivec2, int, 2>(&fpgu_strtol, buffer, specialist_manager_gynecologistQueue, 35);    
                }
				if(in_specialist_manager_geriatricsQueue){
                    readArrayInputVectorType<ivec2, int, 2>(&fpgu_strtol, buffer, specialist_manager_geriatricsQueue, 35);    
                }
				if(in_specialist_manager_psychiatristQueue){
                    readArrayInputVectorType<ivec2, int, 2>(&fpgu_strtol, buffer, specialist_manager_psychiatristQueue, 35);    
                }
				if(in_specialist_id){
                    specialist_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_specialist_current_patient){
                    specialist_current_patient = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_specialist_tick){
                    specialist_tick = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_receptionist_patientQueue){
                    readArrayInput<unsigned int>(&fpgu_strtoul, buffer, receptionist_patientQueue, 35);    
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
				if(in_agent_generator_chairs_generated){
                    agent_generator_chairs_generated = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_generator_beds_generated){
                    agent_generator_beds_generated = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_generator_boxes_generated){
                    agent_generator_boxes_generated = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_generator_doctors_generated){
                    agent_generator_doctors_generated = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_generator_specialists_generated){
                    agent_generator_specialists_generated = (int) fpgu_strtol(buffer); 
                }
				if(in_agent_generator_personal_generated){
                    agent_generator_personal_generated = (int) fpgu_strtol(buffer); 
                }
				if(in_chair_admin_id){
                    chair_admin_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_chair_admin_chairArray){
                    readArrayInput<unsigned int>(&fpgu_strtoul, buffer, chair_admin_chairArray, 35);    
                }
				if(in_uci_tick){
                    uci_tick = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_uci_bedArray){
                    readArrayInputVectorType<ivec2, int, 2>(&fpgu_strtol, buffer, uci_bedArray, 100);    
                }
				if(in_box_id){
                    box_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_box_current_patient){
                    box_current_patient = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_box_tick){
                    box_tick = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_doctor_id){
                    doctor_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_doctor_current_patient){
                    doctor_current_patient = (int) fpgu_strtol(buffer); 
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
				if(in_triage_free_boxes){
                    readArrayInput<unsigned int>(&fpgu_strtoul, buffer, triage_free_boxes, 3);    
                }
				if(in_triage_patientQueue){
                    readArrayInput<unsigned int>(&fpgu_strtoul, buffer, triage_patientQueue, 35);    
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
            if(in_env_SECONDS_PER_TICK){
              
                    env_SECONDS_PER_TICK = (int) fpgu_strtol(buffer);
                    
                    set_SECONDS_PER_TICK(&env_SECONDS_PER_TICK);
                  
              }
            if(in_env_SECONDS_INCUBATING){
              
                    env_SECONDS_INCUBATING = (int) fpgu_strtol(buffer);
                    
                    set_SECONDS_INCUBATING(&env_SECONDS_INCUBATING);
                  
              }
            if(in_env_SECONDS_SICK){
              
                    env_SECONDS_SICK = (int) fpgu_strtol(buffer);
                    
                    set_SECONDS_SICK(&env_SECONDS_SICK);
                  
              }
            if(in_env_CLEANING_PERIOD_SECONDS){
              
                    env_CLEANING_PERIOD_SECONDS = (int) fpgu_strtol(buffer);
                    
                    set_CLEANING_PERIOD_SECONDS(&env_CLEANING_PERIOD_SECONDS);
                  
              }
            if(in_env_EXIT_X){
              
                    env_EXIT_X = (int) fpgu_strtol(buffer);
                    
                    set_EXIT_X(&env_EXIT_X);
                  
              }
            if(in_env_EXIT_Y){
              
                    env_EXIT_Y = (int) fpgu_strtol(buffer);
                    
                    set_EXIT_Y(&env_EXIT_Y);
                  
              }
            if(in_env_PROB_INFECT){
              
                    env_PROB_INFECT = (float) fgpu_atof(buffer);
                    
                    set_PROB_INFECT(&env_PROB_INFECT);
                  
              }
            if(in_env_PROB_SPAWN_SICK){
              
                    env_PROB_SPAWN_SICK = (float) fgpu_atof(buffer);
                    
                    set_PROB_SPAWN_SICK(&env_PROB_SPAWN_SICK);
                  
              }
            if(in_env_PROB_INFECT_PERSONAL){
              
                    env_PROB_INFECT_PERSONAL = (float) fgpu_atof(buffer);
                    
                    set_PROB_INFECT_PERSONAL(&env_PROB_INFECT_PERSONAL);
                  
              }
            if(in_env_PROB_INFECT_CHAIR){
              
                    env_PROB_INFECT_CHAIR = (float) fgpu_atof(buffer);
                    
                    set_PROB_INFECT_CHAIR(&env_PROB_INFECT_CHAIR);
                  
              }
            if(in_env_PROB_INFECT_BED){
              
                    env_PROB_INFECT_BED = (float) fgpu_atof(buffer);
                    
                    set_PROB_INFECT_BED(&env_PROB_INFECT_BED);
                  
              }
            if(in_env_PROB_VACCINE){
              
                    env_PROB_VACCINE = (float) fgpu_atof(buffer);
                    
                    set_PROB_VACCINE(&env_PROB_VACCINE);
                  
              }
            if(in_env_PROB_VACCINE_STAFF){
              
                    env_PROB_VACCINE_STAFF = (float) fgpu_atof(buffer);
                    
                    set_PROB_VACCINE_STAFF(&env_PROB_VACCINE_STAFF);
                  
              }
            if(in_env_UCI_INFECTION_CHANCE){
              
                    env_UCI_INFECTION_CHANCE = (float) fgpu_atof(buffer);
                    
                    set_UCI_INFECTION_CHANCE(&env_UCI_INFECTION_CHANCE);
                  
              }
            if(in_env_FIRSTCHAIR_X){
              
                    env_FIRSTCHAIR_X = (int) fpgu_strtol(buffer);
                    
                    set_FIRSTCHAIR_X(&env_FIRSTCHAIR_X);
                  
              }
            if(in_env_FIRSTCHAIR_Y){
              
                    env_FIRSTCHAIR_Y = (int) fpgu_strtol(buffer);
                    
                    set_FIRSTCHAIR_Y(&env_FIRSTCHAIR_Y);
                  
              }
            if(in_env_SPACE_BETWEEN){
              
                    env_SPACE_BETWEEN = (int) fpgu_strtol(buffer);
                    
                    set_SPACE_BETWEEN(&env_SPACE_BETWEEN);
                  
              }
            if(in_env_DOCTOR_SECONDS){
              
                    env_DOCTOR_SECONDS = (int) fpgu_strtol(buffer);
                    
                    set_DOCTOR_SECONDS(&env_DOCTOR_SECONDS);
                  
              }
            if(in_env_FIRSTDOCTOR_X){
              
                    env_FIRSTDOCTOR_X = (int) fpgu_strtol(buffer);
                    
                    set_FIRSTDOCTOR_X(&env_FIRSTDOCTOR_X);
                  
              }
            if(in_env_FIRSTDOCTOR_Y){
              
                    env_FIRSTDOCTOR_Y = (int) fpgu_strtol(buffer);
                    
                    set_FIRSTDOCTOR_Y(&env_FIRSTDOCTOR_Y);
                  
              }
            if(in_env_SPACE_BETWEEN_DOCTORS){
              
                    env_SPACE_BETWEEN_DOCTORS = (int) fpgu_strtol(buffer);
                    
                    set_SPACE_BETWEEN_DOCTORS(&env_SPACE_BETWEEN_DOCTORS);
                  
              }
            if(in_env_BOX_SECONDS){
              
                    env_BOX_SECONDS = (int) fpgu_strtol(buffer);
                    
                    set_BOX_SECONDS(&env_BOX_SECONDS);
                  
              }
            if(in_env_TRIAGE_X){
              
                    env_TRIAGE_X = (int) fpgu_strtol(buffer);
                    
                    set_TRIAGE_X(&env_TRIAGE_X);
                  
              }
            if(in_env_TRIAGE_Y){
              
                    env_TRIAGE_Y = (int) fpgu_strtol(buffer);
                    
                    set_TRIAGE_Y(&env_TRIAGE_Y);
                  
              }
            if(in_env_UCI_X){
              
                    env_UCI_X = (int) fpgu_strtol(buffer);
                    
                    set_UCI_X(&env_UCI_X);
                  
              }
            if(in_env_UCI_Y){
              
                    env_UCI_Y = (int) fpgu_strtol(buffer);
                    
                    set_UCI_Y(&env_UCI_Y);
                  
              }
            if(in_env_NUMBER_OF_BEDS){
              
                    env_NUMBER_OF_BEDS = (int) fpgu_strtol(buffer);
                    
                    set_NUMBER_OF_BEDS(&env_NUMBER_OF_BEDS);
                  
              }
            if(in_env_PROB_STAY_1){
              
                    env_PROB_STAY_1 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_1(&env_PROB_STAY_1);
                  
              }
            if(in_env_PROB_STAY_2){
              
                    env_PROB_STAY_2 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_2(&env_PROB_STAY_2);
                  
              }
            if(in_env_PROB_STAY_3){
              
                    env_PROB_STAY_3 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_3(&env_PROB_STAY_3);
                  
              }
            if(in_env_PROB_STAY_4){
              
                    env_PROB_STAY_4 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_4(&env_PROB_STAY_4);
                  
              }
            if(in_env_PROB_STAY_5){
              
                    env_PROB_STAY_5 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_5(&env_PROB_STAY_5);
                  
              }
            if(in_env_PROB_STAY_6){
              
                    env_PROB_STAY_6 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_6(&env_PROB_STAY_6);
                  
              }
            if(in_env_PROB_STAY_7){
              
                    env_PROB_STAY_7 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_7(&env_PROB_STAY_7);
                  
              }
            if(in_env_PROB_STAY_8){
              
                    env_PROB_STAY_8 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_8(&env_PROB_STAY_8);
                  
              }
            if(in_env_PROB_STAY_9){
              
                    env_PROB_STAY_9 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_9(&env_PROB_STAY_9);
                  
              }
            if(in_env_PROB_STAY_10){
              
                    env_PROB_STAY_10 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_10(&env_PROB_STAY_10);
                  
              }
            if(in_env_PROB_STAY_11){
              
                    env_PROB_STAY_11 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_11(&env_PROB_STAY_11);
                  
              }
            if(in_env_PROB_STAY_12){
              
                    env_PROB_STAY_12 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_12(&env_PROB_STAY_12);
                  
              }
            if(in_env_PROB_STAY_13){
              
                    env_PROB_STAY_13 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_13(&env_PROB_STAY_13);
                  
              }
            if(in_env_PROB_STAY_14){
              
                    env_PROB_STAY_14 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_14(&env_PROB_STAY_14);
                  
              }
            if(in_env_PROB_STAY_15){
              
                    env_PROB_STAY_15 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_15(&env_PROB_STAY_15);
                  
              }
            if(in_env_PROB_STAY_16){
              
                    env_PROB_STAY_16 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_16(&env_PROB_STAY_16);
                  
              }
            if(in_env_PROB_STAY_17){
              
                    env_PROB_STAY_17 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_17(&env_PROB_STAY_17);
                  
              }
            if(in_env_PROB_STAY_18){
              
                    env_PROB_STAY_18 = (float) fgpu_atof(buffer);
                    
                    set_PROB_STAY_18(&env_PROB_STAY_18);
                  
              }
            if(in_env_STAY_TIME_1){
              
                    env_STAY_TIME_1 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_1(&env_STAY_TIME_1);
                  
              }
            if(in_env_STAY_TIME_2){
              
                    env_STAY_TIME_2 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_2(&env_STAY_TIME_2);
                  
              }
            if(in_env_STAY_TIME_3){
              
                    env_STAY_TIME_3 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_3(&env_STAY_TIME_3);
                  
              }
            if(in_env_STAY_TIME_4){
              
                    env_STAY_TIME_4 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_4(&env_STAY_TIME_4);
                  
              }
            if(in_env_STAY_TIME_5){
              
                    env_STAY_TIME_5 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_5(&env_STAY_TIME_5);
                  
              }
            if(in_env_STAY_TIME_6){
              
                    env_STAY_TIME_6 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_6(&env_STAY_TIME_6);
                  
              }
            if(in_env_STAY_TIME_7){
              
                    env_STAY_TIME_7 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_7(&env_STAY_TIME_7);
                  
              }
            if(in_env_STAY_TIME_8){
              
                    env_STAY_TIME_8 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_8(&env_STAY_TIME_8);
                  
              }
            if(in_env_STAY_TIME_9){
              
                    env_STAY_TIME_9 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_9(&env_STAY_TIME_9);
                  
              }
            if(in_env_STAY_TIME_10){
              
                    env_STAY_TIME_10 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_10(&env_STAY_TIME_10);
                  
              }
            if(in_env_STAY_TIME_11){
              
                    env_STAY_TIME_11 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_11(&env_STAY_TIME_11);
                  
              }
            if(in_env_STAY_TIME_12){
              
                    env_STAY_TIME_12 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_12(&env_STAY_TIME_12);
                  
              }
            if(in_env_STAY_TIME_13){
              
                    env_STAY_TIME_13 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_13(&env_STAY_TIME_13);
                  
              }
            if(in_env_STAY_TIME_14){
              
                    env_STAY_TIME_14 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_14(&env_STAY_TIME_14);
                  
              }
            if(in_env_STAY_TIME_15){
              
                    env_STAY_TIME_15 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_15(&env_STAY_TIME_15);
                  
              }
            if(in_env_STAY_TIME_16){
              
                    env_STAY_TIME_16 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_16(&env_STAY_TIME_16);
                  
              }
            if(in_env_STAY_TIME_17){
              
                    env_STAY_TIME_17 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_17(&env_STAY_TIME_17);
                  
              }
            if(in_env_STAY_TIME_18){
              
                    env_STAY_TIME_18 = (float) fgpu_atof(buffer);
                    
                    set_STAY_TIME_18(&env_STAY_TIME_18);
                  
              }
            if(in_env_CHECKPOINT_1_X){
              
                    env_CHECKPOINT_1_X = (int) fpgu_strtol(buffer);
                    
                    set_CHECKPOINT_1_X(&env_CHECKPOINT_1_X);
                  
              }
            if(in_env_CHECKPOINT_1_Y){
              
                    env_CHECKPOINT_1_Y = (int) fpgu_strtol(buffer);
                    
                    set_CHECKPOINT_1_Y(&env_CHECKPOINT_1_Y);
                  
              }
            if(in_env_CHECKPOINT_2_X){
              
                    env_CHECKPOINT_2_X = (int) fpgu_strtol(buffer);
                    
                    set_CHECKPOINT_2_X(&env_CHECKPOINT_2_X);
                  
              }
            if(in_env_CHECKPOINT_2_Y){
              
                    env_CHECKPOINT_2_Y = (int) fpgu_strtol(buffer);
                    
                    set_CHECKPOINT_2_Y(&env_CHECKPOINT_2_Y);
                  
              }
            if(in_env_CHECKPOINT_3_X){
              
                    env_CHECKPOINT_3_X = (int) fpgu_strtol(buffer);
                    
                    set_CHECKPOINT_3_X(&env_CHECKPOINT_3_X);
                  
              }
            if(in_env_CHECKPOINT_3_Y){
              
                    env_CHECKPOINT_3_Y = (int) fpgu_strtol(buffer);
                    
                    set_CHECKPOINT_3_Y(&env_CHECKPOINT_3_Y);
                  
              }
            if(in_env_CHECKPOINT_4_X){
              
                    env_CHECKPOINT_4_X = (int) fpgu_strtol(buffer);
                    
                    set_CHECKPOINT_4_X(&env_CHECKPOINT_4_X);
                  
              }
            if(in_env_CHECKPOINT_4_Y){
              
                    env_CHECKPOINT_4_Y = (int) fpgu_strtol(buffer);
                    
                    set_CHECKPOINT_4_Y(&env_CHECKPOINT_4_Y);
                  
              }
            if(in_env_CHECKPOINT_5_X){
              
                    env_CHECKPOINT_5_X = (int) fpgu_strtol(buffer);
                    
                    set_CHECKPOINT_5_X(&env_CHECKPOINT_5_X);
                  
              }
            if(in_env_CHECKPOINT_5_Y){
              
                    env_CHECKPOINT_5_Y = (int) fpgu_strtol(buffer);
                    
                    set_CHECKPOINT_5_Y(&env_CHECKPOINT_5_Y);
                  
              }
            if(in_env_SPECIALIST_SECONDS){
              
                    env_SPECIALIST_SECONDS = (int) fpgu_strtol(buffer);
                    
                    set_SPECIALIST_SECONDS(&env_SPECIALIST_SECONDS);
                  
              }
            if(in_env_FIRSTSPECIALIST_X){
              
                    env_FIRSTSPECIALIST_X = (int) fpgu_strtol(buffer);
                    
                    set_FIRSTSPECIALIST_X(&env_FIRSTSPECIALIST_X);
                  
              }
            if(in_env_FIRSTSPECIALIST_Y){
              
                    env_FIRSTSPECIALIST_Y = (int) fpgu_strtol(buffer);
                    
                    set_FIRSTSPECIALIST_Y(&env_FIRSTSPECIALIST_Y);
                  
              }
            if(in_env_SPACE_BETWEEN_SPECIALISTS){
              
                    env_SPACE_BETWEEN_SPECIALISTS = (int) fpgu_strtol(buffer);
                    
                    set_SPACE_BETWEEN_SPECIALISTS(&env_SPACE_BETWEEN_SPECIALISTS);
                  
              }
            if(in_env_FIFTHSPECIALIST_X){
              
                    env_FIFTHSPECIALIST_X = (int) fpgu_strtol(buffer);
                    
                    set_FIFTHSPECIALIST_X(&env_FIFTHSPECIALIST_X);
                  
              }
            if(in_env_FIFTHSPECIALIST_Y){
              
                    env_FIFTHSPECIALIST_Y = (int) fpgu_strtol(buffer);
                    
                    set_FIFTHSPECIALIST_Y(&env_FIFTHSPECIALIST_Y);
                  
              }
            if(in_env_PROB_LEVEL_1){
              
                    env_PROB_LEVEL_1 = (float) fgpu_atof(buffer);
                    
                    set_PROB_LEVEL_1(&env_PROB_LEVEL_1);
                  
              }
            if(in_env_PROB_LEVEL_2){
              
                    env_PROB_LEVEL_2 = (float) fgpu_atof(buffer);
                    
                    set_PROB_LEVEL_2(&env_PROB_LEVEL_2);
                  
              }
            if(in_env_PROB_LEVEL_3){
              
                    env_PROB_LEVEL_3 = (float) fgpu_atof(buffer);
                    
                    set_PROB_LEVEL_3(&env_PROB_LEVEL_3);
                  
              }
            if(in_env_PROB_LEVEL_4){
              
                    env_PROB_LEVEL_4 = (float) fgpu_atof(buffer);
                    
                    set_PROB_LEVEL_4(&env_PROB_LEVEL_4);
                  
              }
            if(in_env_PROB_LEVEL_5){
              
                    env_PROB_LEVEL_5 = (float) fgpu_atof(buffer);
                    
                    set_PROB_LEVEL_5(&env_PROB_LEVEL_5);
                  
              }
            if(in_env_PROB_SURGICAL){
              
                    env_PROB_SURGICAL = (float) fgpu_atof(buffer);
                    
                    set_PROB_SURGICAL(&env_PROB_SURGICAL);
                  
              }
            if(in_env_PROB_MEDICAL){
              
                    env_PROB_MEDICAL = (float) fgpu_atof(buffer);
                    
                    set_PROB_MEDICAL(&env_PROB_MEDICAL);
                  
              }
            if(in_env_PROB_PEDIATRICS){
              
                    env_PROB_PEDIATRICS = (float) fgpu_atof(buffer);
                    
                    set_PROB_PEDIATRICS(&env_PROB_PEDIATRICS);
                  
              }
            if(in_env_PROB_UCI){
              
                    env_PROB_UCI = (float) fgpu_atof(buffer);
                    
                    set_PROB_UCI(&env_PROB_UCI);
                  
              }
            if(in_env_PROB_GYNECOLOGIST){
              
                    env_PROB_GYNECOLOGIST = (float) fgpu_atof(buffer);
                    
                    set_PROB_GYNECOLOGIST(&env_PROB_GYNECOLOGIST);
                  
              }
            if(in_env_PROB_GERIATRICS){
              
                    env_PROB_GERIATRICS = (float) fgpu_atof(buffer);
                    
                    set_PROB_GERIATRICS(&env_PROB_GERIATRICS);
                  
              }
            if(in_env_PROB_PSYCHIATRY){
              
                    env_PROB_PSYCHIATRY = (float) fgpu_atof(buffer);
                    
                    set_PROB_PSYCHIATRY(&env_PROB_PSYCHIATRY);
                  
              }
            if(in_env_RECEPTION_SECONDS){
              
                    env_RECEPTION_SECONDS = (int) fpgu_strtol(buffer);
                    
                    set_RECEPTION_SECONDS(&env_RECEPTION_SECONDS);
                  
              }
            if(in_env_RECEPTIONIST_X){
              
                    env_RECEPTIONIST_X = (int) fpgu_strtol(buffer);
                    
                    set_RECEPTIONIST_X(&env_RECEPTIONIST_X);
                  
              }
            if(in_env_RECEPTIONIST_Y){
              
                    env_RECEPTIONIST_Y = (int) fpgu_strtol(buffer);
                    
                    set_RECEPTIONIST_Y(&env_RECEPTIONIST_Y);
                  
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
