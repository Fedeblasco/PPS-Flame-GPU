
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

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_crystal_list* h_crystals_default, xmachine_memory_crystal_list* d_crystals_default, int h_xmachine_memory_crystal_default_count)
{
    PROFILE_SCOPED_RANGE("saveIterationData");
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_crystals_default, d_crystals_default, sizeof(xmachine_memory_crystal_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying crystal Agent default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
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
    
    fputs("\t<I_agg>", file);
    sprintf(data, "%f", (*get_I_agg()));
    fputs(data, file);
    fputs("</I_agg>\n", file);
    fputs("\t<G0>", file);
    sprintf(data, "%f", (*get_G0()));
    fputs(data, file);
    fputs("</G0>\n", file);
    fputs("\t<dT>", file);
    sprintf(data, "%f", (*get_dT()));
    fputs(data, file);
    fputs("</dT>\n", file);
    fputs("\t<aggNo>", file);
    sprintf(data, "%d", (*get_aggNo()));
    fputs(data, file);
    fputs("</aggNo>\n", file);
	fputs("</environment>\n" , file);

	//Write each crystal agent to xml
	for (int i=0; i<h_xmachine_memory_crystal_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>crystal</name>\n", file);
        
		fputs("<rank>", file);
        sprintf(data, "%f", h_crystals_default->rank[i]);
		fputs(data, file);
		fputs("</rank>\n", file);
        
		fputs("<l>", file);
        sprintf(data, "%f", h_crystals_default->l[i]);
		fputs(data, file);
		fputs("</l>\n", file);
        
		fputs("<bin>", file);
        sprintf(data, "%d", h_crystals_default->bin[i]);
		fputs(data, file);
		fputs("</bin>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);

}

void readInitialStates(char* inputpath, xmachine_memory_crystal_list* h_crystals, int* h_xmachine_memory_crystal_count)
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
    int in_crystal_rank;
    int in_crystal_l;
    int in_crystal_bin;
    
    /* tags for environment global variables */
    int in_env;
    int in_env_I_agg;
    
    int in_env_G0;
    
    int in_env_dT;
    
    int in_env_aggNo;
    
	/* set agent count to zero */
	*h_xmachine_memory_crystal_count = 0;
	
	/* Variables for initial state data */
	float crystal_rank;
	float crystal_l;
	int crystal_bin;

    /* Variables for environment variables */
    float env_I_agg;
    float env_G0;
    float env_dT;
    int env_aggNo;
    


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
	in_crystal_rank = 0;
	in_crystal_l = 0;
	in_crystal_bin = 0;
    in_env_I_agg = 0;
    in_env_G0 = 0;
    in_env_dT = 0;
    in_env_aggNo = 0;
	//set all crystal values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_crystal_MAX; k++)
	{	
		h_crystals->rank[k] = 0;
		h_crystals->l[k] = 0;
		h_crystals->bin[k] = 0;
	}
	

	/* Default variables for memory */
    crystal_rank = 0;
    crystal_l = 0;
    crystal_bin = 0;

    /* Default variables for environment variables */
    env_I_agg = 0;
    env_G0 = 0;
    env_dT = 0;
    env_aggNo = 0;
    
    
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
				if(strcmp(agentname, "crystal") == 0)
				{
					if (*h_xmachine_memory_crystal_count > xmachine_memory_crystal_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent crystal exceeded whilst reading data\n", xmachine_memory_crystal_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_crystals->rank[*h_xmachine_memory_crystal_count] = crystal_rank;
					h_crystals->l[*h_xmachine_memory_crystal_count] = crystal_l;
					h_crystals->bin[*h_xmachine_memory_crystal_count] = crystal_bin;
					(*h_xmachine_memory_crystal_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}



				/* Reset xagent variables */
                crystal_rank = 0;
                crystal_l = 0;
                crystal_bin = 0;
                
                in_xagent = 0;
			}
			if(strcmp(buffer, "rank") == 0) in_crystal_rank = 1;
			if(strcmp(buffer, "/rank") == 0) in_crystal_rank = 0;
			if(strcmp(buffer, "l") == 0) in_crystal_l = 1;
			if(strcmp(buffer, "/l") == 0) in_crystal_l = 0;
			if(strcmp(buffer, "bin") == 0) in_crystal_bin = 1;
			if(strcmp(buffer, "/bin") == 0) in_crystal_bin = 0;
			
            /* environment variables */
            if(strcmp(buffer, "I_agg") == 0) in_env_I_agg = 1;
            if(strcmp(buffer, "/I_agg") == 0) in_env_I_agg = 0;
			if(strcmp(buffer, "G0") == 0) in_env_G0 = 1;
            if(strcmp(buffer, "/G0") == 0) in_env_G0 = 0;
			if(strcmp(buffer, "dT") == 0) in_env_dT = 1;
            if(strcmp(buffer, "/dT") == 0) in_env_dT = 0;
			if(strcmp(buffer, "aggNo") == 0) in_env_aggNo = 1;
            if(strcmp(buffer, "/aggNo") == 0) in_env_aggNo = 0;
			

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
				if(in_crystal_rank){
                    crystal_rank = (float) fgpu_atof(buffer); 
                }
				if(in_crystal_l){
                    crystal_l = (float) fgpu_atof(buffer); 
                }
				if(in_crystal_bin){
                    crystal_bin = (int) fpgu_strtol(buffer); 
                }
				
            }
            else if (in_env){
            if(in_env_I_agg){
              
                    env_I_agg = (float) fgpu_atof(buffer);
                    
                    set_I_agg(&env_I_agg);
                  
              }
            if(in_env_G0){
              
                    env_G0 = (float) fgpu_atof(buffer);
                    
                    set_G0(&env_G0);
                  
              }
            if(in_env_dT){
              
                    env_dT = (float) fgpu_atof(buffer);
                    
                    set_dT(&env_dT);
                  
              }
            if(in_env_aggNo){
              
                    env_aggNo = (int) fpgu_strtol(buffer);
                    
                    set_aggNo(&env_aggNo);
                  
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
