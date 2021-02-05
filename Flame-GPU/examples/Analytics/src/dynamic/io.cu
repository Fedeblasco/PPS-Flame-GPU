
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

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Circle_list* h_Circles_default, xmachine_memory_Circle_list* d_Circles_default, int h_xmachine_memory_Circle_default_count)
{
    PROFILE_SCOPED_RANGE("saveIterationData");
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_Circles_default, d_Circles_default, sizeof(xmachine_memory_Circle_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying Circle Agent default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
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
    
    fputs("\t<KR>", file);
    sprintf(data, "%f", (*get_KR()));
    fputs(data, file);
    fputs("</KR>\n", file);
    fputs("\t<KA>", file);
    sprintf(data, "%f", (*get_KA()));
    fputs(data, file);
    fputs("</KA>\n", file);
	fputs("</environment>\n" , file);

	//Write each Circle agent to xml
	for (int i=0; i<h_xmachine_memory_Circle_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Circle</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%d", h_Circles_default->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_Circles_default->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_Circles_default->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z>", file);
        sprintf(data, "%f", h_Circles_default->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
        
		fputs("<fx>", file);
        sprintf(data, "%f", h_Circles_default->fx[i]);
		fputs(data, file);
		fputs("</fx>\n", file);
        
		fputs("<fy>", file);
        sprintf(data, "%f", h_Circles_default->fy[i]);
		fputs(data, file);
		fputs("</fy>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);

}

void readInitialStates(char* inputpath, xmachine_memory_Circle_list* h_Circles, int* h_xmachine_memory_Circle_count)
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
    int in_Circle_id;
    int in_Circle_x;
    int in_Circle_y;
    int in_Circle_z;
    int in_Circle_fx;
    int in_Circle_fy;
    
    /* tags for environment global variables */
    int in_env;
    int in_env_KR;
    
    int in_env_KA;
    
	/* set agent count to zero */
	*h_xmachine_memory_Circle_count = 0;
	
	/* Variables for initial state data */
	int Circle_id;
	float Circle_x;
	float Circle_y;
	float Circle_z;
	float Circle_fx;
	float Circle_fy;

    /* Variables for environment variables */
    float env_KR;
    float env_KA;
    


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
	in_Circle_id = 0;
	in_Circle_x = 0;
	in_Circle_y = 0;
	in_Circle_z = 0;
	in_Circle_fx = 0;
	in_Circle_fy = 0;
    in_env_KR = 0;
    in_env_KA = 0;
	//set all Circle values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Circle_MAX; k++)
	{	
		h_Circles->id[k] = 0;
		h_Circles->x[k] = 0;
		h_Circles->y[k] = 0;
		h_Circles->z[k] = 0;
		h_Circles->fx[k] = 0;
		h_Circles->fy[k] = 0;
	}
	

	/* Default variables for memory */
    Circle_id = 0;
    Circle_x = 0;
    Circle_y = 0;
    Circle_z = 0;
    Circle_fx = 0;
    Circle_fy = 0;

    /* Default variables for environment variables */
    env_KR = 0;
    env_KA = 0;
    
    
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
				if(strcmp(agentname, "Circle") == 0)
				{
					if (*h_xmachine_memory_Circle_count > xmachine_memory_Circle_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent Circle exceeded whilst reading data\n", xmachine_memory_Circle_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_Circles->id[*h_xmachine_memory_Circle_count] = Circle_id;
					h_Circles->x[*h_xmachine_memory_Circle_count] = Circle_x;//Check maximum x value
                    if(agent_maximum.x < Circle_x)
                        agent_maximum.x = (float)Circle_x;
                    //Check minimum x value
                    if(agent_minimum.x > Circle_x)
                        agent_minimum.x = (float)Circle_x;
                    
					h_Circles->y[*h_xmachine_memory_Circle_count] = Circle_y;//Check maximum y value
                    if(agent_maximum.y < Circle_y)
                        agent_maximum.y = (float)Circle_y;
                    //Check minimum y value
                    if(agent_minimum.y > Circle_y)
                        agent_minimum.y = (float)Circle_y;
                    
					h_Circles->z[*h_xmachine_memory_Circle_count] = Circle_z;//Check maximum z value
                    if(agent_maximum.z < Circle_z)
                        agent_maximum.z = (float)Circle_z;
                    //Check minimum z value
                    if(agent_minimum.z > Circle_z)
                        agent_minimum.z = (float)Circle_z;
                    
					h_Circles->fx[*h_xmachine_memory_Circle_count] = Circle_fx;
					h_Circles->fy[*h_xmachine_memory_Circle_count] = Circle_fy;
					(*h_xmachine_memory_Circle_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}



				/* Reset xagent variables */
                Circle_id = 0;
                Circle_x = 0;
                Circle_y = 0;
                Circle_z = 0;
                Circle_fx = 0;
                Circle_fy = 0;
                
                in_xagent = 0;
			}
			if(strcmp(buffer, "id") == 0) in_Circle_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Circle_id = 0;
			if(strcmp(buffer, "x") == 0) in_Circle_x = 1;
			if(strcmp(buffer, "/x") == 0) in_Circle_x = 0;
			if(strcmp(buffer, "y") == 0) in_Circle_y = 1;
			if(strcmp(buffer, "/y") == 0) in_Circle_y = 0;
			if(strcmp(buffer, "z") == 0) in_Circle_z = 1;
			if(strcmp(buffer, "/z") == 0) in_Circle_z = 0;
			if(strcmp(buffer, "fx") == 0) in_Circle_fx = 1;
			if(strcmp(buffer, "/fx") == 0) in_Circle_fx = 0;
			if(strcmp(buffer, "fy") == 0) in_Circle_fy = 1;
			if(strcmp(buffer, "/fy") == 0) in_Circle_fy = 0;
			
            /* environment variables */
            if(strcmp(buffer, "KR") == 0) in_env_KR = 1;
            if(strcmp(buffer, "/KR") == 0) in_env_KR = 0;
			if(strcmp(buffer, "KA") == 0) in_env_KA = 1;
            if(strcmp(buffer, "/KA") == 0) in_env_KA = 0;
			

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
				if(in_Circle_id){
                    Circle_id = (int) fpgu_strtol(buffer); 
                }
				if(in_Circle_x){
                    Circle_x = (float) fgpu_atof(buffer); 
                }
				if(in_Circle_y){
                    Circle_y = (float) fgpu_atof(buffer); 
                }
				if(in_Circle_z){
                    Circle_z = (float) fgpu_atof(buffer); 
                }
				if(in_Circle_fx){
                    Circle_fx = (float) fgpu_atof(buffer); 
                }
				if(in_Circle_fy){
                    Circle_fy = (float) fgpu_atof(buffer); 
                }
				
            }
            else if (in_env){
            if(in_env_KR){
              
                    env_KR = (float) fgpu_atof(buffer);
                    
                    set_KR(&env_KR);
                  
              }
            if(in_env_KA){
              
                    env_KA = (float) fgpu_atof(buffer);
                    
                    set_KA(&env_KA);
                  
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
