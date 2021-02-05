
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

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_keratinocyte_list* h_keratinocytes_default, xmachine_memory_keratinocyte_list* d_keratinocytes_default, int h_xmachine_memory_keratinocyte_default_count,xmachine_memory_keratinocyte_list* h_keratinocytes_resolve, xmachine_memory_keratinocyte_list* d_keratinocytes_resolve, int h_xmachine_memory_keratinocyte_resolve_count)
{
    PROFILE_SCOPED_RANGE("saveIterationData");
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_keratinocytes_default, d_keratinocytes_default, sizeof(xmachine_memory_keratinocyte_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying keratinocyte Agent default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	cudaStatus = cudaMemcpy( h_keratinocytes_resolve, d_keratinocytes_resolve, sizeof(xmachine_memory_keratinocyte_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying keratinocyte Agent resolve State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
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
    
    fputs("\t<calcium_level>", file);
    sprintf(data, "%f", (*get_calcium_level()));
    fputs(data, file);
    fputs("</calcium_level>\n", file);
    fputs("\t<CYCLE_LENGTH>", file);
    for (int j=0;j<5;j++){
        fprintf(file, "%d", get_CYCLE_LENGTH()[j]);
        if(j!=(5-1))
            fprintf(file, ",");
    }
    fputs("</CYCLE_LENGTH>\n", file);
    fputs("\t<SUBSTRATE_FORCE>", file);
    for (int j=0;j<5;j++){
        fprintf(file, "%f", get_SUBSTRATE_FORCE()[j]);
        if(j!=(5-1))
            fprintf(file, ",");
    }
    fputs("</SUBSTRATE_FORCE>\n", file);
    fputs("\t<DOWNWARD_FORCE>", file);
    for (int j=0;j<5;j++){
        fprintf(file, "%f", get_DOWNWARD_FORCE()[j]);
        if(j!=(5-1))
            fprintf(file, ",");
    }
    fputs("</DOWNWARD_FORCE>\n", file);
    fputs("\t<FORCE_MATRIX>", file);
    for (int j=0;j<25;j++){
        fprintf(file, "%f", get_FORCE_MATRIX()[j]);
        if(j!=(25-1))
            fprintf(file, ",");
    }
    fputs("</FORCE_MATRIX>\n", file);
    fputs("\t<FORCE_REP>", file);
    sprintf(data, "%f", (*get_FORCE_REP()));
    fputs(data, file);
    fputs("</FORCE_REP>\n", file);
    fputs("\t<FORCE_DAMPENER>", file);
    sprintf(data, "%f", (*get_FORCE_DAMPENER()));
    fputs(data, file);
    fputs("</FORCE_DAMPENER>\n", file);
    fputs("\t<BASEMENT_MAX_Z>", file);
    sprintf(data, "%d", (*get_BASEMENT_MAX_Z()));
    fputs(data, file);
    fputs("</BASEMENT_MAX_Z>\n", file);
	fputs("</environment>\n" , file);

	//Write each keratinocyte agent to xml
	for (int i=0; i<h_xmachine_memory_keratinocyte_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>keratinocyte</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%d", h_keratinocytes_default->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<type>", file);
        sprintf(data, "%d", h_keratinocytes_default->type[i]);
		fputs(data, file);
		fputs("</type>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_keratinocytes_default->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_keratinocytes_default->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z>", file);
        sprintf(data, "%f", h_keratinocytes_default->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
        
		fputs("<force_x>", file);
        sprintf(data, "%f", h_keratinocytes_default->force_x[i]);
		fputs(data, file);
		fputs("</force_x>\n", file);
        
		fputs("<force_y>", file);
        sprintf(data, "%f", h_keratinocytes_default->force_y[i]);
		fputs(data, file);
		fputs("</force_y>\n", file);
        
		fputs("<force_z>", file);
        sprintf(data, "%f", h_keratinocytes_default->force_z[i]);
		fputs(data, file);
		fputs("</force_z>\n", file);
        
		fputs("<num_xy_bonds>", file);
        sprintf(data, "%d", h_keratinocytes_default->num_xy_bonds[i]);
		fputs(data, file);
		fputs("</num_xy_bonds>\n", file);
        
		fputs("<num_z_bonds>", file);
        sprintf(data, "%d", h_keratinocytes_default->num_z_bonds[i]);
		fputs(data, file);
		fputs("</num_z_bonds>\n", file);
        
		fputs("<num_stem_bonds>", file);
        sprintf(data, "%d", h_keratinocytes_default->num_stem_bonds[i]);
		fputs(data, file);
		fputs("</num_stem_bonds>\n", file);
        
		fputs("<cycle>", file);
        sprintf(data, "%d", h_keratinocytes_default->cycle[i]);
		fputs(data, file);
		fputs("</cycle>\n", file);
        
		fputs("<diff_noise_factor>", file);
        sprintf(data, "%f", h_keratinocytes_default->diff_noise_factor[i]);
		fputs(data, file);
		fputs("</diff_noise_factor>\n", file);
        
		fputs("<dead_ticks>", file);
        sprintf(data, "%d", h_keratinocytes_default->dead_ticks[i]);
		fputs(data, file);
		fputs("</dead_ticks>\n", file);
        
		fputs("<contact_inhibited_ticks>", file);
        sprintf(data, "%d", h_keratinocytes_default->contact_inhibited_ticks[i]);
		fputs(data, file);
		fputs("</contact_inhibited_ticks>\n", file);
        
		fputs("<motility>", file);
        sprintf(data, "%f", h_keratinocytes_default->motility[i]);
		fputs(data, file);
		fputs("</motility>\n", file);
        
		fputs("<dir>", file);
        sprintf(data, "%f", h_keratinocytes_default->dir[i]);
		fputs(data, file);
		fputs("</dir>\n", file);
        
		fputs("<movement>", file);
        sprintf(data, "%f", h_keratinocytes_default->movement[i]);
		fputs(data, file);
		fputs("</movement>\n", file);
        
		fputs("</xagent>\n", file);
	}
	//Write each keratinocyte agent to xml
	for (int i=0; i<h_xmachine_memory_keratinocyte_resolve_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>keratinocyte</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%d", h_keratinocytes_resolve->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<type>", file);
        sprintf(data, "%d", h_keratinocytes_resolve->type[i]);
		fputs(data, file);
		fputs("</type>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
        
		fputs("<force_x>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->force_x[i]);
		fputs(data, file);
		fputs("</force_x>\n", file);
        
		fputs("<force_y>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->force_y[i]);
		fputs(data, file);
		fputs("</force_y>\n", file);
        
		fputs("<force_z>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->force_z[i]);
		fputs(data, file);
		fputs("</force_z>\n", file);
        
		fputs("<num_xy_bonds>", file);
        sprintf(data, "%d", h_keratinocytes_resolve->num_xy_bonds[i]);
		fputs(data, file);
		fputs("</num_xy_bonds>\n", file);
        
		fputs("<num_z_bonds>", file);
        sprintf(data, "%d", h_keratinocytes_resolve->num_z_bonds[i]);
		fputs(data, file);
		fputs("</num_z_bonds>\n", file);
        
		fputs("<num_stem_bonds>", file);
        sprintf(data, "%d", h_keratinocytes_resolve->num_stem_bonds[i]);
		fputs(data, file);
		fputs("</num_stem_bonds>\n", file);
        
		fputs("<cycle>", file);
        sprintf(data, "%d", h_keratinocytes_resolve->cycle[i]);
		fputs(data, file);
		fputs("</cycle>\n", file);
        
		fputs("<diff_noise_factor>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->diff_noise_factor[i]);
		fputs(data, file);
		fputs("</diff_noise_factor>\n", file);
        
		fputs("<dead_ticks>", file);
        sprintf(data, "%d", h_keratinocytes_resolve->dead_ticks[i]);
		fputs(data, file);
		fputs("</dead_ticks>\n", file);
        
		fputs("<contact_inhibited_ticks>", file);
        sprintf(data, "%d", h_keratinocytes_resolve->contact_inhibited_ticks[i]);
		fputs(data, file);
		fputs("</contact_inhibited_ticks>\n", file);
        
		fputs("<motility>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->motility[i]);
		fputs(data, file);
		fputs("</motility>\n", file);
        
		fputs("<dir>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->dir[i]);
		fputs(data, file);
		fputs("</dir>\n", file);
        
		fputs("<movement>", file);
        sprintf(data, "%f", h_keratinocytes_resolve->movement[i]);
		fputs(data, file);
		fputs("</movement>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);

}

void readInitialStates(char* inputpath, xmachine_memory_keratinocyte_list* h_keratinocytes, int* h_xmachine_memory_keratinocyte_count)
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
    int in_keratinocyte_id;
    int in_keratinocyte_type;
    int in_keratinocyte_x;
    int in_keratinocyte_y;
    int in_keratinocyte_z;
    int in_keratinocyte_force_x;
    int in_keratinocyte_force_y;
    int in_keratinocyte_force_z;
    int in_keratinocyte_num_xy_bonds;
    int in_keratinocyte_num_z_bonds;
    int in_keratinocyte_num_stem_bonds;
    int in_keratinocyte_cycle;
    int in_keratinocyte_diff_noise_factor;
    int in_keratinocyte_dead_ticks;
    int in_keratinocyte_contact_inhibited_ticks;
    int in_keratinocyte_motility;
    int in_keratinocyte_dir;
    int in_keratinocyte_movement;
    
    /* tags for environment global variables */
    int in_env;
    int in_env_calcium_level;
    
    int in_env_CYCLE_LENGTH;
    
    int in_env_SUBSTRATE_FORCE;
    
    int in_env_DOWNWARD_FORCE;
    
    int in_env_FORCE_MATRIX;
    
    int in_env_FORCE_REP;
    
    int in_env_FORCE_DAMPENER;
    
    int in_env_BASEMENT_MAX_Z;
    
	/* set agent count to zero */
	*h_xmachine_memory_keratinocyte_count = 0;
	
	/* Variables for initial state data */
	int keratinocyte_id;
	int keratinocyte_type;
	float keratinocyte_x;
	float keratinocyte_y;
	float keratinocyte_z;
	float keratinocyte_force_x;
	float keratinocyte_force_y;
	float keratinocyte_force_z;
	int keratinocyte_num_xy_bonds;
	int keratinocyte_num_z_bonds;
	int keratinocyte_num_stem_bonds;
	int keratinocyte_cycle;
	float keratinocyte_diff_noise_factor;
	int keratinocyte_dead_ticks;
	int keratinocyte_contact_inhibited_ticks;
	float keratinocyte_motility;
	float keratinocyte_dir;
	float keratinocyte_movement;

    /* Variables for environment variables */
    float env_calcium_level;
    int env_CYCLE_LENGTH[5];
    float env_SUBSTRATE_FORCE[5];
    float env_DOWNWARD_FORCE[5];
    float env_FORCE_MATRIX[25];
    float env_FORCE_REP;
    float env_FORCE_DAMPENER;
    int env_BASEMENT_MAX_Z;
    


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
	in_keratinocyte_id = 0;
	in_keratinocyte_type = 0;
	in_keratinocyte_x = 0;
	in_keratinocyte_y = 0;
	in_keratinocyte_z = 0;
	in_keratinocyte_force_x = 0;
	in_keratinocyte_force_y = 0;
	in_keratinocyte_force_z = 0;
	in_keratinocyte_num_xy_bonds = 0;
	in_keratinocyte_num_z_bonds = 0;
	in_keratinocyte_num_stem_bonds = 0;
	in_keratinocyte_cycle = 0;
	in_keratinocyte_diff_noise_factor = 0;
	in_keratinocyte_dead_ticks = 0;
	in_keratinocyte_contact_inhibited_ticks = 0;
	in_keratinocyte_motility = 0;
	in_keratinocyte_dir = 0;
	in_keratinocyte_movement = 0;
    in_env_calcium_level = 0;
    in_env_CYCLE_LENGTH = 0;
    in_env_SUBSTRATE_FORCE = 0;
    in_env_DOWNWARD_FORCE = 0;
    in_env_FORCE_MATRIX = 0;
    in_env_FORCE_REP = 0;
    in_env_FORCE_DAMPENER = 0;
    in_env_BASEMENT_MAX_Z = 0;
	//set all keratinocyte values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_keratinocyte_MAX; k++)
	{	
		h_keratinocytes->id[k] = 0;
		h_keratinocytes->type[k] = 0;
		h_keratinocytes->x[k] = 0;
		h_keratinocytes->y[k] = 0;
		h_keratinocytes->z[k] = 0;
		h_keratinocytes->force_x[k] = 0;
		h_keratinocytes->force_y[k] = 0;
		h_keratinocytes->force_z[k] = 0;
		h_keratinocytes->num_xy_bonds[k] = 0;
		h_keratinocytes->num_z_bonds[k] = 0;
		h_keratinocytes->num_stem_bonds[k] = 0;
		h_keratinocytes->cycle[k] = 0;
		h_keratinocytes->diff_noise_factor[k] = 0;
		h_keratinocytes->dead_ticks[k] = 0;
		h_keratinocytes->contact_inhibited_ticks[k] = 0;
		h_keratinocytes->motility[k] = 0;
		h_keratinocytes->dir[k] = 0;
		h_keratinocytes->movement[k] = 0;
	}
	

	/* Default variables for memory */
    keratinocyte_id = 0;
    keratinocyte_type = 0;
    keratinocyte_x = 0;
    keratinocyte_y = 0;
    keratinocyte_z = 0;
    keratinocyte_force_x = 0;
    keratinocyte_force_y = 0;
    keratinocyte_force_z = 0;
    keratinocyte_num_xy_bonds = 0;
    keratinocyte_num_z_bonds = 0;
    keratinocyte_num_stem_bonds = 0;
    keratinocyte_cycle = 0;
    keratinocyte_diff_noise_factor = 0;
    keratinocyte_dead_ticks = 0;
    keratinocyte_contact_inhibited_ticks = 0;
    keratinocyte_motility = 0;
    keratinocyte_dir = 0;
    keratinocyte_movement = 0;

    /* Default variables for environment variables */
    env_calcium_level = 0;
    
    for (i=0;i<5;i++){
        env_CYCLE_LENGTH[i] = 0;
    }
    
    for (i=0;i<5;i++){
        env_SUBSTRATE_FORCE[i] = 0;
    }
    
    for (i=0;i<5;i++){
        env_DOWNWARD_FORCE[i] = 0;
    }
    
    for (i=0;i<25;i++){
        env_FORCE_MATRIX[i] = 0;
    }
    env_FORCE_REP = 0;
    env_FORCE_DAMPENER = 0;
    env_BASEMENT_MAX_Z = 0;
    
    
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
				if(strcmp(agentname, "keratinocyte") == 0)
				{
					if (*h_xmachine_memory_keratinocyte_count > xmachine_memory_keratinocyte_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent keratinocyte exceeded whilst reading data\n", xmachine_memory_keratinocyte_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_keratinocytes->id[*h_xmachine_memory_keratinocyte_count] = keratinocyte_id;
					h_keratinocytes->type[*h_xmachine_memory_keratinocyte_count] = keratinocyte_type;
					h_keratinocytes->x[*h_xmachine_memory_keratinocyte_count] = keratinocyte_x;//Check maximum x value
                    if(agent_maximum.x < keratinocyte_x)
                        agent_maximum.x = (float)keratinocyte_x;
                    //Check minimum x value
                    if(agent_minimum.x > keratinocyte_x)
                        agent_minimum.x = (float)keratinocyte_x;
                    
					h_keratinocytes->y[*h_xmachine_memory_keratinocyte_count] = keratinocyte_y;//Check maximum y value
                    if(agent_maximum.y < keratinocyte_y)
                        agent_maximum.y = (float)keratinocyte_y;
                    //Check minimum y value
                    if(agent_minimum.y > keratinocyte_y)
                        agent_minimum.y = (float)keratinocyte_y;
                    
					h_keratinocytes->z[*h_xmachine_memory_keratinocyte_count] = keratinocyte_z;//Check maximum z value
                    if(agent_maximum.z < keratinocyte_z)
                        agent_maximum.z = (float)keratinocyte_z;
                    //Check minimum z value
                    if(agent_minimum.z > keratinocyte_z)
                        agent_minimum.z = (float)keratinocyte_z;
                    
					h_keratinocytes->force_x[*h_xmachine_memory_keratinocyte_count] = keratinocyte_force_x;
					h_keratinocytes->force_y[*h_xmachine_memory_keratinocyte_count] = keratinocyte_force_y;
					h_keratinocytes->force_z[*h_xmachine_memory_keratinocyte_count] = keratinocyte_force_z;
					h_keratinocytes->num_xy_bonds[*h_xmachine_memory_keratinocyte_count] = keratinocyte_num_xy_bonds;
					h_keratinocytes->num_z_bonds[*h_xmachine_memory_keratinocyte_count] = keratinocyte_num_z_bonds;
					h_keratinocytes->num_stem_bonds[*h_xmachine_memory_keratinocyte_count] = keratinocyte_num_stem_bonds;
					h_keratinocytes->cycle[*h_xmachine_memory_keratinocyte_count] = keratinocyte_cycle;
					h_keratinocytes->diff_noise_factor[*h_xmachine_memory_keratinocyte_count] = keratinocyte_diff_noise_factor;
					h_keratinocytes->dead_ticks[*h_xmachine_memory_keratinocyte_count] = keratinocyte_dead_ticks;
					h_keratinocytes->contact_inhibited_ticks[*h_xmachine_memory_keratinocyte_count] = keratinocyte_contact_inhibited_ticks;
					h_keratinocytes->motility[*h_xmachine_memory_keratinocyte_count] = keratinocyte_motility;
					h_keratinocytes->dir[*h_xmachine_memory_keratinocyte_count] = keratinocyte_dir;
					h_keratinocytes->movement[*h_xmachine_memory_keratinocyte_count] = keratinocyte_movement;
					(*h_xmachine_memory_keratinocyte_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}



				/* Reset xagent variables */
                keratinocyte_id = 0;
                keratinocyte_type = 0;
                keratinocyte_x = 0;
                keratinocyte_y = 0;
                keratinocyte_z = 0;
                keratinocyte_force_x = 0;
                keratinocyte_force_y = 0;
                keratinocyte_force_z = 0;
                keratinocyte_num_xy_bonds = 0;
                keratinocyte_num_z_bonds = 0;
                keratinocyte_num_stem_bonds = 0;
                keratinocyte_cycle = 0;
                keratinocyte_diff_noise_factor = 0;
                keratinocyte_dead_ticks = 0;
                keratinocyte_contact_inhibited_ticks = 0;
                keratinocyte_motility = 0;
                keratinocyte_dir = 0;
                keratinocyte_movement = 0;
                
                in_xagent = 0;
			}
			if(strcmp(buffer, "id") == 0) in_keratinocyte_id = 1;
			if(strcmp(buffer, "/id") == 0) in_keratinocyte_id = 0;
			if(strcmp(buffer, "type") == 0) in_keratinocyte_type = 1;
			if(strcmp(buffer, "/type") == 0) in_keratinocyte_type = 0;
			if(strcmp(buffer, "x") == 0) in_keratinocyte_x = 1;
			if(strcmp(buffer, "/x") == 0) in_keratinocyte_x = 0;
			if(strcmp(buffer, "y") == 0) in_keratinocyte_y = 1;
			if(strcmp(buffer, "/y") == 0) in_keratinocyte_y = 0;
			if(strcmp(buffer, "z") == 0) in_keratinocyte_z = 1;
			if(strcmp(buffer, "/z") == 0) in_keratinocyte_z = 0;
			if(strcmp(buffer, "force_x") == 0) in_keratinocyte_force_x = 1;
			if(strcmp(buffer, "/force_x") == 0) in_keratinocyte_force_x = 0;
			if(strcmp(buffer, "force_y") == 0) in_keratinocyte_force_y = 1;
			if(strcmp(buffer, "/force_y") == 0) in_keratinocyte_force_y = 0;
			if(strcmp(buffer, "force_z") == 0) in_keratinocyte_force_z = 1;
			if(strcmp(buffer, "/force_z") == 0) in_keratinocyte_force_z = 0;
			if(strcmp(buffer, "num_xy_bonds") == 0) in_keratinocyte_num_xy_bonds = 1;
			if(strcmp(buffer, "/num_xy_bonds") == 0) in_keratinocyte_num_xy_bonds = 0;
			if(strcmp(buffer, "num_z_bonds") == 0) in_keratinocyte_num_z_bonds = 1;
			if(strcmp(buffer, "/num_z_bonds") == 0) in_keratinocyte_num_z_bonds = 0;
			if(strcmp(buffer, "num_stem_bonds") == 0) in_keratinocyte_num_stem_bonds = 1;
			if(strcmp(buffer, "/num_stem_bonds") == 0) in_keratinocyte_num_stem_bonds = 0;
			if(strcmp(buffer, "cycle") == 0) in_keratinocyte_cycle = 1;
			if(strcmp(buffer, "/cycle") == 0) in_keratinocyte_cycle = 0;
			if(strcmp(buffer, "diff_noise_factor") == 0) in_keratinocyte_diff_noise_factor = 1;
			if(strcmp(buffer, "/diff_noise_factor") == 0) in_keratinocyte_diff_noise_factor = 0;
			if(strcmp(buffer, "dead_ticks") == 0) in_keratinocyte_dead_ticks = 1;
			if(strcmp(buffer, "/dead_ticks") == 0) in_keratinocyte_dead_ticks = 0;
			if(strcmp(buffer, "contact_inhibited_ticks") == 0) in_keratinocyte_contact_inhibited_ticks = 1;
			if(strcmp(buffer, "/contact_inhibited_ticks") == 0) in_keratinocyte_contact_inhibited_ticks = 0;
			if(strcmp(buffer, "motility") == 0) in_keratinocyte_motility = 1;
			if(strcmp(buffer, "/motility") == 0) in_keratinocyte_motility = 0;
			if(strcmp(buffer, "dir") == 0) in_keratinocyte_dir = 1;
			if(strcmp(buffer, "/dir") == 0) in_keratinocyte_dir = 0;
			if(strcmp(buffer, "movement") == 0) in_keratinocyte_movement = 1;
			if(strcmp(buffer, "/movement") == 0) in_keratinocyte_movement = 0;
			
            /* environment variables */
            if(strcmp(buffer, "calcium_level") == 0) in_env_calcium_level = 1;
            if(strcmp(buffer, "/calcium_level") == 0) in_env_calcium_level = 0;
			if(strcmp(buffer, "CYCLE_LENGTH") == 0) in_env_CYCLE_LENGTH = 1;
            if(strcmp(buffer, "/CYCLE_LENGTH") == 0) in_env_CYCLE_LENGTH = 0;
			if(strcmp(buffer, "SUBSTRATE_FORCE") == 0) in_env_SUBSTRATE_FORCE = 1;
            if(strcmp(buffer, "/SUBSTRATE_FORCE") == 0) in_env_SUBSTRATE_FORCE = 0;
			if(strcmp(buffer, "DOWNWARD_FORCE") == 0) in_env_DOWNWARD_FORCE = 1;
            if(strcmp(buffer, "/DOWNWARD_FORCE") == 0) in_env_DOWNWARD_FORCE = 0;
			if(strcmp(buffer, "FORCE_MATRIX") == 0) in_env_FORCE_MATRIX = 1;
            if(strcmp(buffer, "/FORCE_MATRIX") == 0) in_env_FORCE_MATRIX = 0;
			if(strcmp(buffer, "FORCE_REP") == 0) in_env_FORCE_REP = 1;
            if(strcmp(buffer, "/FORCE_REP") == 0) in_env_FORCE_REP = 0;
			if(strcmp(buffer, "FORCE_DAMPENER") == 0) in_env_FORCE_DAMPENER = 1;
            if(strcmp(buffer, "/FORCE_DAMPENER") == 0) in_env_FORCE_DAMPENER = 0;
			if(strcmp(buffer, "BASEMENT_MAX_Z") == 0) in_env_BASEMENT_MAX_Z = 1;
            if(strcmp(buffer, "/BASEMENT_MAX_Z") == 0) in_env_BASEMENT_MAX_Z = 0;
			

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
				if(in_keratinocyte_id){
                    keratinocyte_id = (int) fpgu_strtol(buffer); 
                }
				if(in_keratinocyte_type){
                    keratinocyte_type = (int) fpgu_strtol(buffer); 
                }
				if(in_keratinocyte_x){
                    keratinocyte_x = (float) fgpu_atof(buffer); 
                }
				if(in_keratinocyte_y){
                    keratinocyte_y = (float) fgpu_atof(buffer); 
                }
				if(in_keratinocyte_z){
                    keratinocyte_z = (float) fgpu_atof(buffer); 
                }
				if(in_keratinocyte_force_x){
                    keratinocyte_force_x = (float) fgpu_atof(buffer); 
                }
				if(in_keratinocyte_force_y){
                    keratinocyte_force_y = (float) fgpu_atof(buffer); 
                }
				if(in_keratinocyte_force_z){
                    keratinocyte_force_z = (float) fgpu_atof(buffer); 
                }
				if(in_keratinocyte_num_xy_bonds){
                    keratinocyte_num_xy_bonds = (int) fpgu_strtol(buffer); 
                }
				if(in_keratinocyte_num_z_bonds){
                    keratinocyte_num_z_bonds = (int) fpgu_strtol(buffer); 
                }
				if(in_keratinocyte_num_stem_bonds){
                    keratinocyte_num_stem_bonds = (int) fpgu_strtol(buffer); 
                }
				if(in_keratinocyte_cycle){
                    keratinocyte_cycle = (int) fpgu_strtol(buffer); 
                }
				if(in_keratinocyte_diff_noise_factor){
                    keratinocyte_diff_noise_factor = (float) fgpu_atof(buffer); 
                }
				if(in_keratinocyte_dead_ticks){
                    keratinocyte_dead_ticks = (int) fpgu_strtol(buffer); 
                }
				if(in_keratinocyte_contact_inhibited_ticks){
                    keratinocyte_contact_inhibited_ticks = (int) fpgu_strtol(buffer); 
                }
				if(in_keratinocyte_motility){
                    keratinocyte_motility = (float) fgpu_atof(buffer); 
                }
				if(in_keratinocyte_dir){
                    keratinocyte_dir = (float) fgpu_atof(buffer); 
                }
				if(in_keratinocyte_movement){
                    keratinocyte_movement = (float) fgpu_atof(buffer); 
                }
				
            }
            else if (in_env){
            if(in_env_calcium_level){
              
                    env_calcium_level = (float) fgpu_atof(buffer);
                    
                    set_calcium_level(&env_calcium_level);
                  
              }
            if(in_env_CYCLE_LENGTH){
              readArrayInput<int>(&fpgu_strtol, buffer, env_CYCLE_LENGTH, 5);
                    set_CYCLE_LENGTH(env_CYCLE_LENGTH);
                  
              }
            if(in_env_SUBSTRATE_FORCE){
              readArrayInput<float>(&fgpu_atof, buffer, env_SUBSTRATE_FORCE, 5);
                    set_SUBSTRATE_FORCE(env_SUBSTRATE_FORCE);
                  
              }
            if(in_env_DOWNWARD_FORCE){
              readArrayInput<float>(&fgpu_atof, buffer, env_DOWNWARD_FORCE, 5);
                    set_DOWNWARD_FORCE(env_DOWNWARD_FORCE);
                  
              }
            if(in_env_FORCE_MATRIX){
              readArrayInput<float>(&fgpu_atof, buffer, env_FORCE_MATRIX, 25);
                    set_FORCE_MATRIX(env_FORCE_MATRIX);
                  
              }
            if(in_env_FORCE_REP){
              
                    env_FORCE_REP = (float) fgpu_atof(buffer);
                    
                    set_FORCE_REP(&env_FORCE_REP);
                  
              }
            if(in_env_FORCE_DAMPENER){
              
                    env_FORCE_DAMPENER = (float) fgpu_atof(buffer);
                    
                    set_FORCE_DAMPENER(&env_FORCE_DAMPENER);
                  
              }
            if(in_env_BASEMENT_MAX_Z){
              
                    env_BASEMENT_MAX_Z = (int) fpgu_strtol(buffer);
                    
                    set_BASEMENT_MAX_Z(&env_BASEMENT_MAX_Z);
                  
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
