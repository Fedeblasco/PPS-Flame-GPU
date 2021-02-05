
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


#if defined __NVCC__
   // Disable the "statement is unreachable" message
   #pragma diag_suppress code_is_unreachable
   // Disable the "dynamic initialization in unreachable code" message
   #pragma diag_suppress initialization_not_reachable
#endif 
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"


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

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_Agent_list* h_Agents_default, xmachine_memory_Agent_list* d_Agents_default, int h_xmachine_memory_Agent_default_count)
{
    PROFILE_SCOPED_RANGE("saveIterationData");
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_Agents_default, d_Agents_default, sizeof(xmachine_memory_Agent_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr,"Error Copying Agent Agent default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
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
    
    fputs("\t<SEED>", file);
    sprintf(data, "%u", (*get_SEED()));
    fputs(data, file);
    fputs("</SEED>\n", file);
    fputs("\t<INIT_POPULATION>", file);
    sprintf(data, "%u", (*get_INIT_POPULATION()));
    fputs(data, file);
    fputs("</INIT_POPULATION>\n", file);
    fputs("\t<PARAM_MIN_SPEED>", file);
    sprintf(data, "%f", (*get_PARAM_MIN_SPEED()));
    fputs(data, file);
    fputs("</PARAM_MIN_SPEED>\n", file);
    fputs("\t<PARAM_MAX_SPEED>", file);
    sprintf(data, "%f", (*get_PARAM_MAX_SPEED()));
    fputs(data, file);
    fputs("</PARAM_MAX_SPEED>\n", file);
	fputs("</environment>\n" , file);

	//Write each Agent agent to xml
	for (int i=0; i<h_xmachine_memory_Agent_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>Agent</name>\n", file);
        
		fputs("<id>", file);
        sprintf(data, "%u", h_Agents_default->id[i]);
		fputs(data, file);
		fputs("</id>\n", file);
        
		fputs("<currentEdge>", file);
        sprintf(data, "%u", h_Agents_default->currentEdge[i]);
		fputs(data, file);
		fputs("</currentEdge>\n", file);
        
		fputs("<nextEdge>", file);
        sprintf(data, "%u", h_Agents_default->nextEdge[i]);
		fputs(data, file);
		fputs("</nextEdge>\n", file);
        
		fputs("<nextEdgeRemainingCapacity>", file);
        sprintf(data, "%u", h_Agents_default->nextEdgeRemainingCapacity[i]);
		fputs(data, file);
		fputs("</nextEdgeRemainingCapacity>\n", file);
        
		fputs("<hasIntent>", file);
        sprintf(data, "%d", h_Agents_default->hasIntent[i]);
		fputs(data, file);
		fputs("</hasIntent>\n", file);
        
		fputs("<position>", file);
        sprintf(data, "%f", h_Agents_default->position[i]);
		fputs(data, file);
		fputs("</position>\n", file);
        
		fputs("<distanceTravelled>", file);
        sprintf(data, "%f", h_Agents_default->distanceTravelled[i]);
		fputs(data, file);
		fputs("</distanceTravelled>\n", file);
        
		fputs("<blockedIterationCount>", file);
        sprintf(data, "%u", h_Agents_default->blockedIterationCount[i]);
		fputs(data, file);
		fputs("</blockedIterationCount>\n", file);
        
		fputs("<speed>", file);
        sprintf(data, "%f", h_Agents_default->speed[i]);
		fputs(data, file);
		fputs("</speed>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_Agents_default->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_Agents_default->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<z>", file);
        sprintf(data, "%f", h_Agents_default->z[i]);
		fputs(data, file);
		fputs("</z>\n", file);
        
		fputs("<colour>", file);
        sprintf(data, "%f", h_Agents_default->colour[i]);
		fputs(data, file);
		fputs("</colour>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);

}

void initEnvVars()
{
PROFILE_SCOPED_RANGE("initEnvVars");

    unsigned int t_SEED = (unsigned int)0;
    set_SEED(&t_SEED);
    unsigned int t_INIT_POPULATION = (unsigned int)1;
    set_INIT_POPULATION(&t_INIT_POPULATION);
    float t_PARAM_MIN_SPEED = (float)1;
    set_PARAM_MIN_SPEED(&t_PARAM_MIN_SPEED);
    float t_PARAM_MAX_SPEED = (float)1;
    set_PARAM_MAX_SPEED(&t_PARAM_MAX_SPEED);
}

void readInitialStates(char* inputpath, xmachine_memory_Agent_list* h_Agents, int* h_xmachine_memory_Agent_count)
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
    int in_Agent_id;
    int in_Agent_currentEdge;
    int in_Agent_nextEdge;
    int in_Agent_nextEdgeRemainingCapacity;
    int in_Agent_hasIntent;
    int in_Agent_position;
    int in_Agent_distanceTravelled;
    int in_Agent_blockedIterationCount;
    int in_Agent_speed;
    int in_Agent_x;
    int in_Agent_y;
    int in_Agent_z;
    int in_Agent_colour;
    
    /* tags for environment global variables */
    int in_env;
    int in_env_SEED;
    
    int in_env_INIT_POPULATION;
    
    int in_env_PARAM_MIN_SPEED;
    
    int in_env_PARAM_MAX_SPEED;
    
	/* set agent count to zero */
	*h_xmachine_memory_Agent_count = 0;
	
	/* Variables for initial state data */
	unsigned int Agent_id;
	unsigned int Agent_currentEdge;
	unsigned int Agent_nextEdge;
	unsigned int Agent_nextEdgeRemainingCapacity;
	bool Agent_hasIntent;
	float Agent_position;
	float Agent_distanceTravelled;
	unsigned int Agent_blockedIterationCount;
	float Agent_speed;
	float Agent_x;
	float Agent_y;
	float Agent_z;
	float Agent_colour;

    /* Variables for environment variables */
    unsigned int env_SEED;
    unsigned int env_INIT_POPULATION;
    float env_PARAM_MIN_SPEED;
    float env_PARAM_MAX_SPEED;
    


	/* Initialise variables */
    initEnvVars();
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
	in_Agent_id = 0;
	in_Agent_currentEdge = 0;
	in_Agent_nextEdge = 0;
	in_Agent_nextEdgeRemainingCapacity = 0;
	in_Agent_hasIntent = 0;
	in_Agent_position = 0;
	in_Agent_distanceTravelled = 0;
	in_Agent_blockedIterationCount = 0;
	in_Agent_speed = 0;
	in_Agent_x = 0;
	in_Agent_y = 0;
	in_Agent_z = 0;
	in_Agent_colour = 0;
    in_env_SEED = 0;
    in_env_INIT_POPULATION = 0;
    in_env_PARAM_MIN_SPEED = 0;
    in_env_PARAM_MAX_SPEED = 0;
	//set all Agent values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_Agent_MAX; k++)
	{	
		h_Agents->id[k] = 0;
		h_Agents->currentEdge[k] = 0;
		h_Agents->nextEdge[k] = 0;
		h_Agents->nextEdgeRemainingCapacity[k] = 0;
		h_Agents->hasIntent[k] = 0;
		h_Agents->position[k] = 0;
		h_Agents->distanceTravelled[k] = 0;
		h_Agents->blockedIterationCount[k] = 0;
		h_Agents->speed[k] = 0;
		h_Agents->x[k] = 0;
		h_Agents->y[k] = 0;
		h_Agents->z[k] = 0;
		h_Agents->colour[k] = 0.0;
	}
	

	/* Default variables for memory */
    Agent_id = 0;
    Agent_currentEdge = 0;
    Agent_nextEdge = 0;
    Agent_nextEdgeRemainingCapacity = 0;
    Agent_hasIntent = 0;
    Agent_position = 0;
    Agent_distanceTravelled = 0;
    Agent_blockedIterationCount = 0;
    Agent_speed = 0;
    Agent_x = 0;
    Agent_y = 0;
    Agent_z = 0;
    Agent_colour = 0.0;

    /* Default variables for environment variables */
    env_SEED = 0;
    env_INIT_POPULATION = 1;
    env_PARAM_MIN_SPEED = 1;
    env_PARAM_MAX_SPEED = 1;
    
    
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
				if(strcmp(agentname, "Agent") == 0)
				{
					if (*h_xmachine_memory_Agent_count > xmachine_memory_Agent_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent Agent exceeded whilst reading data\n", xmachine_memory_Agent_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(EXIT_FAILURE);
					}
                    
					h_Agents->id[*h_xmachine_memory_Agent_count] = Agent_id;
					h_Agents->currentEdge[*h_xmachine_memory_Agent_count] = Agent_currentEdge;
					h_Agents->nextEdge[*h_xmachine_memory_Agent_count] = Agent_nextEdge;
					h_Agents->nextEdgeRemainingCapacity[*h_xmachine_memory_Agent_count] = Agent_nextEdgeRemainingCapacity;
					h_Agents->hasIntent[*h_xmachine_memory_Agent_count] = Agent_hasIntent;
					h_Agents->position[*h_xmachine_memory_Agent_count] = Agent_position;
					h_Agents->distanceTravelled[*h_xmachine_memory_Agent_count] = Agent_distanceTravelled;
					h_Agents->blockedIterationCount[*h_xmachine_memory_Agent_count] = Agent_blockedIterationCount;
					h_Agents->speed[*h_xmachine_memory_Agent_count] = Agent_speed;
					h_Agents->x[*h_xmachine_memory_Agent_count] = Agent_x;//Check maximum x value
                    if(agent_maximum.x < Agent_x)
                        agent_maximum.x = (float)Agent_x;
                    //Check minimum x value
                    if(agent_minimum.x > Agent_x)
                        agent_minimum.x = (float)Agent_x;
                    
					h_Agents->y[*h_xmachine_memory_Agent_count] = Agent_y;//Check maximum y value
                    if(agent_maximum.y < Agent_y)
                        agent_maximum.y = (float)Agent_y;
                    //Check minimum y value
                    if(agent_minimum.y > Agent_y)
                        agent_minimum.y = (float)Agent_y;
                    
					h_Agents->z[*h_xmachine_memory_Agent_count] = Agent_z;//Check maximum z value
                    if(agent_maximum.z < Agent_z)
                        agent_maximum.z = (float)Agent_z;
                    //Check minimum z value
                    if(agent_minimum.z > Agent_z)
                        agent_minimum.z = (float)Agent_z;
                    
					h_Agents->colour[*h_xmachine_memory_Agent_count] = Agent_colour;
					(*h_xmachine_memory_Agent_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}



				/* Reset xagent variables */
                Agent_id = 0;
                Agent_currentEdge = 0;
                Agent_nextEdge = 0;
                Agent_nextEdgeRemainingCapacity = 0;
                Agent_hasIntent = 0;
                Agent_position = 0;
                Agent_distanceTravelled = 0;
                Agent_blockedIterationCount = 0;
                Agent_speed = 0;
                Agent_x = 0;
                Agent_y = 0;
                Agent_z = 0;
                Agent_colour = 0.0;
                
                in_xagent = 0;
			}
			if(strcmp(buffer, "id") == 0) in_Agent_id = 1;
			if(strcmp(buffer, "/id") == 0) in_Agent_id = 0;
			if(strcmp(buffer, "currentEdge") == 0) in_Agent_currentEdge = 1;
			if(strcmp(buffer, "/currentEdge") == 0) in_Agent_currentEdge = 0;
			if(strcmp(buffer, "nextEdge") == 0) in_Agent_nextEdge = 1;
			if(strcmp(buffer, "/nextEdge") == 0) in_Agent_nextEdge = 0;
			if(strcmp(buffer, "nextEdgeRemainingCapacity") == 0) in_Agent_nextEdgeRemainingCapacity = 1;
			if(strcmp(buffer, "/nextEdgeRemainingCapacity") == 0) in_Agent_nextEdgeRemainingCapacity = 0;
			if(strcmp(buffer, "hasIntent") == 0) in_Agent_hasIntent = 1;
			if(strcmp(buffer, "/hasIntent") == 0) in_Agent_hasIntent = 0;
			if(strcmp(buffer, "position") == 0) in_Agent_position = 1;
			if(strcmp(buffer, "/position") == 0) in_Agent_position = 0;
			if(strcmp(buffer, "distanceTravelled") == 0) in_Agent_distanceTravelled = 1;
			if(strcmp(buffer, "/distanceTravelled") == 0) in_Agent_distanceTravelled = 0;
			if(strcmp(buffer, "blockedIterationCount") == 0) in_Agent_blockedIterationCount = 1;
			if(strcmp(buffer, "/blockedIterationCount") == 0) in_Agent_blockedIterationCount = 0;
			if(strcmp(buffer, "speed") == 0) in_Agent_speed = 1;
			if(strcmp(buffer, "/speed") == 0) in_Agent_speed = 0;
			if(strcmp(buffer, "x") == 0) in_Agent_x = 1;
			if(strcmp(buffer, "/x") == 0) in_Agent_x = 0;
			if(strcmp(buffer, "y") == 0) in_Agent_y = 1;
			if(strcmp(buffer, "/y") == 0) in_Agent_y = 0;
			if(strcmp(buffer, "z") == 0) in_Agent_z = 1;
			if(strcmp(buffer, "/z") == 0) in_Agent_z = 0;
			if(strcmp(buffer, "colour") == 0) in_Agent_colour = 1;
			if(strcmp(buffer, "/colour") == 0) in_Agent_colour = 0;
			
            /* environment variables */
            if(strcmp(buffer, "SEED") == 0) in_env_SEED = 1;
            if(strcmp(buffer, "/SEED") == 0) in_env_SEED = 0;
			if(strcmp(buffer, "INIT_POPULATION") == 0) in_env_INIT_POPULATION = 1;
            if(strcmp(buffer, "/INIT_POPULATION") == 0) in_env_INIT_POPULATION = 0;
			if(strcmp(buffer, "PARAM_MIN_SPEED") == 0) in_env_PARAM_MIN_SPEED = 1;
            if(strcmp(buffer, "/PARAM_MIN_SPEED") == 0) in_env_PARAM_MIN_SPEED = 0;
			if(strcmp(buffer, "PARAM_MAX_SPEED") == 0) in_env_PARAM_MAX_SPEED = 1;
            if(strcmp(buffer, "/PARAM_MAX_SPEED") == 0) in_env_PARAM_MAX_SPEED = 0;
			

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
				if(in_Agent_id){
                    Agent_id = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Agent_currentEdge){
                    Agent_currentEdge = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Agent_nextEdge){
                    Agent_nextEdge = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Agent_nextEdgeRemainingCapacity){
                    Agent_nextEdgeRemainingCapacity = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Agent_hasIntent){
                    Agent_hasIntent = (bool) fpgu_strtol(buffer); 
                }
				if(in_Agent_position){
                    Agent_position = (float) fgpu_atof(buffer); 
                }
				if(in_Agent_distanceTravelled){
                    Agent_distanceTravelled = (float) fgpu_atof(buffer); 
                }
				if(in_Agent_blockedIterationCount){
                    Agent_blockedIterationCount = (unsigned int) fpgu_strtoul(buffer); 
                }
				if(in_Agent_speed){
                    Agent_speed = (float) fgpu_atof(buffer); 
                }
				if(in_Agent_x){
                    Agent_x = (float) fgpu_atof(buffer); 
                }
				if(in_Agent_y){
                    Agent_y = (float) fgpu_atof(buffer); 
                }
				if(in_Agent_z){
                    Agent_z = (float) fgpu_atof(buffer); 
                }
				if(in_Agent_colour){
                    Agent_colour = (float) fgpu_atof(buffer); 
                }
				
            }
            else if (in_env){
            if(in_env_SEED){
              
                    env_SEED = (unsigned int) fpgu_strtoul(buffer);
                    
                    set_SEED(&env_SEED);
                  
              }
            if(in_env_INIT_POPULATION){
              
                    env_INIT_POPULATION = (unsigned int) fpgu_strtoul(buffer);
                    
                    set_INIT_POPULATION(&env_INIT_POPULATION);
                  
              }
            if(in_env_PARAM_MIN_SPEED){
              
                    env_PARAM_MIN_SPEED = (float) fgpu_atof(buffer);
                    
                    set_PARAM_MIN_SPEED(&env_PARAM_MIN_SPEED);
                  
              }
            if(in_env_PARAM_MAX_SPEED){
              
                    env_PARAM_MAX_SPEED = (float) fgpu_atof(buffer);
                    
                    set_PARAM_MAX_SPEED(&env_PARAM_MAX_SPEED);
                  
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


/*
 * bool checkForDuplicates_staticGraph_network(staticGraph_memory_network* graph)
 * Checks a static graph for duplicate entries, which are not allowed.
 * @param graph pointer to static graph
 * @return boolean indicator of success
 */
bool checkForDuplicates_staticGraph_network(staticGraph_memory_network* graph){
    // Check for duplicate entries, by parsing all edges which are now sorted.
    if(graph->edge.count > 1){
        unsigned int prevSource = graph->edge.source[0];
        unsigned int prevDest = graph->edge.destination[0];
        for(unsigned int i = 1; i < graph->edge.count; i++){
            // If 2 sequential edges are the same, there is an error.
            if(prevSource == graph->edge.source[i] && prevDest == graph->edge.destination[i]){
                return true;
            }
            prevSource = graph->edge.source[i];
            prevDest = graph->edge.destination[i];
        }
    }
    return false;
}

/*
 * void coo_to_csr_staticGraph_network(staticGraph_memory_network* coo, staticGraph_memory_network* csr)
 * Converts a COO (unsorted) graph into the Compressed Sparse Row (CSR) representation.
 * @param coo graph in unsorted order
 * @param csr graph sorted and stored as CSR
 */
 void coo_to_csr_staticGraph_network(staticGraph_memory_network* coo, staticGraph_memory_network* csr){
    // Copy counts across to the CSR data structure using the new indices.
    csr->vertex.count = coo->vertex.count;
    csr->edge.count = coo->edge.count;

    // Initialise the csr first edge pointers to 0.
    std::fill(csr->vertex.first_edge_index, csr->vertex.first_edge_index + coo->vertex.count, 0);

    // For each edge, increment the pointer for the source vertex.
    for(unsigned int i = 0; i < coo->edge.count; i++){
        csr->vertex.first_edge_index[coo->edge.source[i]]++;
    }

    // Inclusive prefix sum across these values to get the final value for each vertex
    unsigned int total = 0;
    for(unsigned int i = 0; i < coo->vertex.count; i++){
        unsigned int old_value = csr->vertex.first_edge_index[i];
        csr->vertex.first_edge_index[i] = total;
        total += old_value;
    }
    // Populate the |V| + 1 value
    csr->vertex.first_edge_index[coo->vertex.count] = coo->edge.count;


    // Sort vertices by id. 
    // Create a vector of pairs 
    
    std::vector<std::pair<unsigned int,unsigned int>> vertex_indices (coo->vertex.count);
    // Populate the pairs.
    for(unsigned int i = 0; i < coo->vertex.count; i++){
        vertex_indices.at(i).first = i;
        vertex_indices.at(i).second = coo->vertex.id[i] ;
    }
    // sort the vector of indices based on the value of the COO vertex ids.
    std::sort(vertex_indices.begin(), vertex_indices.end(), [](const std::pair<unsigned int,unsigned int> &left, const std::pair<unsigned int,unsigned int> &right) {
        return left.second < right.second;
    });

    
    // Scatter vertices data from coo to csr order
    for(unsigned int coo_index = 0; coo_index < coo->vertex.count; coo_index++){
        unsigned int csr_index = vertex_indices.at(coo_index).first;
        csr->vertex.id[csr_index] = coo->vertex.id[coo_index];
        csr->vertex.x[csr_index] = coo->vertex.x[coo_index];
        csr->vertex.y[csr_index] = coo->vertex.y[coo_index];
        csr->vertex.z[csr_index] = coo->vertex.z[coo_index];
        
    }

    // Scatter values to complete the csr data
    for(unsigned int coo_index = 0; coo_index < coo->edge.count; coo_index++){
        unsigned int source_vertex = coo->edge.source[coo_index];
        unsigned int csr_index = csr->vertex.first_edge_index[source_vertex];
        csr->edge.id[csr_index] = coo->edge.id[coo_index];
        csr->edge.source[csr_index] = coo->edge.source[coo_index];
        csr->edge.destination[csr_index] = coo->edge.destination[coo_index];
        csr->edge.length[csr_index] = coo->edge.length[coo_index];
        csr->edge.capacity[csr_index] = coo->edge.capacity[coo_index];
        
        csr->vertex.first_edge_index[source_vertex]++;
    }

    // Fill in any gaps in the CSR
    unsigned int previous_value = 0;
    for (unsigned int i = 0 ; i <= csr->vertex.count; i++){
        unsigned int old_value = csr->vertex.first_edge_index[i];
        csr->vertex.first_edge_index[i] = previous_value;
        previous_value = old_value;
    }
}



/* void load_staticGraph_network_from_json(const char* file, staticGraph_memory_network* h_staticGraph_memory_network)
 * Load a static graph from a JSON file on disk.
 * @param file input filename
 * @param h_staticGraph_memory_network pointer to graph.
 */
void load_staticGraph_network_from_json(const char* file, staticGraph_memory_network* h_staticGraph_memory_network){
    PROFILE_SCOPED_RANGE("loadGraphFromJSON");
    // Build the path to the file from the working directory by joining the input directory path and the specified file name from XML
    std::string pathToFile(getOutputDir(), strlen(getOutputDir()));
    pathToFile.append("network.json");

    FILE *filePointer = fopen(pathToFile.c_str(), "rb");
    // Ensure the File exists
    if (filePointer == nullptr){
        fprintf(stderr, "FATAL ERROR: network file %s could not be opened.\n", pathToFile.c_str());
        exit(EXIT_FAILURE);
    }

    // Print the file being loaded
    fprintf(stdout, "Loading staticGraph network from json file %s\n", pathToFile.c_str());

    // Get the length of the file
    fseek(filePointer, 0, SEEK_END);
    long filesize = ftell(filePointer);
    fseek(filePointer, 0, SEEK_SET);

    // Allocate and load the file into memory
    char *string = (char*)malloc(filesize + 1);
    if(string == nullptr){
        fprintf(stderr, "FATAL ERROR: Could not allocate memory to parse %s\n", pathToFile.c_str());
        fclose(filePointer);
        exit(EXIT_FAILURE);
    }
    fread(string, filesize, 1, filePointer);
    fclose(filePointer);
    // terminate the string
    string[filesize] = 0;

    // Use rapidJson to parse the loaded data.
    rapidjson::Document document;
    document.Parse(string);

    

    // Check Json was valid and contained the required values.
    if (document.IsObject()){
        // Get value references to the relevant json objects
        const rapidjson::Value& vertices = document["vertices"];
        const rapidjson::Value& edges = document["edges"];

        // Get the number of edges and vertices
        unsigned int vertex_count = (vertices.IsArray()) ? vertices.Size() : 0;
        unsigned int edge_count = (edges.IsArray()) ? edges.Size() : 0;

        // If either dimensions is greater than the maximum allowed elements then we must error and exit.
        if(vertex_count > staticGraph_network_vertex_bufferSize || edge_count > staticGraph_network_edge_bufferSize){
            fprintf(
                stderr,
                "FATAL ERROR: Static Graph network (%u vertices, %u edges) exceeds buffer dimensions (%u vertices, %u edges)",
                vertex_count,
                edge_count,
                staticGraph_network_vertex_bufferSize,
                staticGraph_network_edge_bufferSize 
            );
            exit(EXIT_FAILURE);
        }

        // Allocate a local COO object to load data into from disk.
        staticGraph_memory_network* coo = (staticGraph_memory_network *) malloc(sizeof(staticGraph_memory_network));

        // Ensure it allocated.
        if(coo == nullptr){
            fprintf(stderr, "FATAL ERROR: Could not allocate memory for staticGraph network while loading from disk\n");
            exit(EXIT_FAILURE);
        }

        // Store the counts in the COO graph
        coo->edge.count = edge_count;
        coo->vertex.count = vertex_count;

        // For each vertex element in the file, load the relevant values into the COO memory, otherwise set defaults.
        for (rapidjson::SizeType i = 0; i < coo->vertex.count; i++){
            // Set default values for variables.
            coo->vertex.id[i] = 0;
            coo->vertex.x[i] = 1.0f;
            coo->vertex.y[i] = 1.0f;
            coo->vertex.z[i] = 1.0f;
            

            // Attempt to read the correct value from JSON
            if (vertices[i].HasMember("id") && vertices[i]["id"].Is<unsigned int>()){
                coo->vertex.id[i] = vertices[i]["id"].Get<unsigned int>();
            }
            if (vertices[i].HasMember("x") && vertices[i]["x"].Is<float>()){
                coo->vertex.x[i] = vertices[i]["x"].Get<float>();
            }
            if (vertices[i].HasMember("y") && vertices[i]["y"].Is<float>()){
                coo->vertex.y[i] = vertices[i]["y"].Get<float>();
            }
            if (vertices[i].HasMember("z") && vertices[i]["z"].Is<float>()){
                coo->vertex.z[i] = vertices[i]["z"].Get<float>();
            }
            
        }

        // For each edge element in the file, load the relevant values into memory, otherwise set defaults.
        for (rapidjson::SizeType i = 0; i < coo->edge.count; i++){
            // Set default values for variables.
            coo->edge.id[i] = 0;
            coo->edge.source[i] = 0;
            coo->edge.destination[i] = 0;
            coo->edge.length[i] = 1;
            coo->edge.capacity[i] = 1;
            

            // Attempt to read the correct value from JSON
            if (edges[i].HasMember("id") && edges[i]["id"].Is<unsigned int>()){
                coo->edge.id[i] = edges[i]["id"].Get<unsigned int>();
            }
            if (edges[i].HasMember("source") && edges[i]["source"].Is<unsigned int>()){
                coo->edge.source[i] = edges[i]["source"].Get<unsigned int>();
            }
            if (edges[i].HasMember("destination") && edges[i]["destination"].Is<unsigned int>()){
                coo->edge.destination[i] = edges[i]["destination"].Get<unsigned int>();
            }
            if (edges[i].HasMember("length") && edges[i]["length"].Is<float>()){
                coo->edge.length[i] = edges[i]["length"].Get<float>();
            }
            if (edges[i].HasMember("capacity") && edges[i]["capacity"].Is<unsigned int>()){
                coo->edge.capacity[i] = edges[i]["capacity"].Get<unsigned int>();
            }
            
        }

        // Construct the CSR representation from COO
        coo_to_csr_staticGraph_network(coo, h_staticGraph_memory_network);

        // Check for duplicate edges (undefined behaviour)
        bool has_duplicates = checkForDuplicates_staticGraph_network( h_staticGraph_memory_network);
        if(has_duplicates){
            printf("FATAL ERROR: Duplicate edge found in staticGraph network\n");
            free(coo);
            exit(EXIT_FAILURE);
        }

        // Free the COO representation
        free(coo);
        coo = nullptr;

    } else {
        // Otherwise it is not an object and we have failed.
        printf("FATAL ERROR: Network file %s is not a valid JSON file\n", pathToFile.c_str());
        exit(EXIT_FAILURE);
    }

    fprintf(stdout, "Loaded %u vertices, %u edges\n", h_staticGraph_memory_network->vertex.count, h_staticGraph_memory_network->edge.count);

}
