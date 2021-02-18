/*
 * Copyright 2011 University of Sheffield.
 * Author: Dr Paul Richmond 
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

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include "header.h"
#include "CustomVisualisation.h"
#include "receptionist.c"

#define SCALE_FACTOR 0.03125

#define I_SCALER (SCALE_FACTOR*0.35f)
#define MESSAGE_RADIUS d_message_pedestrian_location_radius
#define MIN_DISTANCE 0.0001f

//#define NUM_EXITS 7

#define PI 3.1415f
#define RADIANS(x) (PI / 180.0f) * x

//Probabilidades usadas para manejar la cantidad de enfermos
#define probabilidad_estornudar 1.0
#define probabilidad_contagio 1.0
#define probabilidad_generar_enfermo 0.5
//Cantidad de ticks enfermo y portador
#define ticks_portador 50000
#define ticks_enfermo 50000
//Cantidad de personas a generar
#define cant_personas 2

#define ir_a_x 150
#define ir_a_y 80


//This function creates all the agents requiered to run the hospital
__FLAME_GPU_INIT_FUNC__ void inicializarMapa(){
	printf("Inicializando todo\n");

	// Allocating memory in CPU to save the agent
	xmachine_memory_receptionist * h_receptionist = h_allocate_agent_receptionist();
	// Copying the agent from the CPU to GPU
	h_add_agent_receptionist_defaultReceptionist(h_receptionist);
	// Freeing the previously allocated memory
	h_free_agent_receptionist(&h_receptionist);

}

/**
 * output_location FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.
 */
__FLAME_GPU_FUNC__ int output_pedestrian_location(xmachine_memory_agent* agent, xmachine_message_pedestrian_location_list* pedestrian_location_messages){

	add_pedestrian_location_message(pedestrian_location_messages, agent->x, agent->y, 0.0,agent->estado);
  
    return 0;
}

__FLAME_GPU_FUNC__ int output_pedestrian_state(xmachine_memory_agent* agent, xmachine_message_pedestrian_state_list* pedestrian_state_messages){

	add_pedestrian_state_message(pedestrian_state_messages, agent->x, agent->y, 0.0,agent->estado);
  
    return 0;
}

/**
 * output_navmap_cells FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_navmap. This represents a single agent instance and can be modified directly.
 * @param navmap_cell_messages Pointer to output message list of type xmachine_message_navmap_cell_list. Must be passed as an argument to the add_navmap_cell_message function ??.
 */
__FLAME_GPU_FUNC__ int output_navmap_cells(xmachine_memory_navmap* agent, xmachine_message_navmap_cell_list* navmap_cell_messages){
    
	add_navmap_cell_message<DISCRETE_2D>(navmap_cell_messages, 
		agent->x, agent->y, 
		agent->exit_no, 
		agent->height,
		agent->collision_x, agent->collision_y);
       
    return 0;
}



/**
 * move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int avoid_pedestrians(xmachine_memory_agent* agent, xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix, RNG_rand48* rand48){

	glm::vec2 agent_pos = glm::vec2(agent->x, agent->y);
	glm::vec2 agent_vel = glm::vec2(agent->velx, agent->vely);

	glm::vec2 navigate_velocity = glm::vec2(0.0f, 0.0f);
	glm::vec2 avoid_velocity = glm::vec2(0.0f, 0.0f);

	xmachine_message_pedestrian_location* current_message = get_first_pedestrian_location_message(pedestrian_location_messages, partition_matrix, agent->x, agent->y, 0.0);
	while (current_message)
	{	
		glm::vec2 message_pos = glm::vec2(current_message->x, current_message->y);
		float separation = length(agent_pos - message_pos);
		//Si la distancia entre uno y el otro es menor a la distancia minima
		if ((separation < MESSAGE_RADIUS)&&(separation>MIN_DISTANCE)){
			glm::vec2 to_agent = normalize(agent_pos - message_pos);
			float ang = acosf(dot(agent_vel, to_agent));
			float perception = 45.0f;

			//STEER
			if ((ang < RADIANS(perception)) || (ang > 3.14159265f-RADIANS(perception))){
				glm::vec2 s_velocity = to_agent;
				s_velocity *= powf(I_SCALER/separation, 1.25f)*STEER_WEIGHT;
				navigate_velocity += s_velocity;
			}

			//AVOID
			glm::vec2 a_velocity = to_agent;
			a_velocity *= powf(I_SCALER/separation, 2.00f)*AVOID_WEIGHT;
			avoid_velocity += a_velocity;

			//Si estoy sano, y me cruce con un paciente que esta enfermo o es portador, cambio mi estado a portador
			/*if(agent->estado==0){
				if(current_message->estado==1 || current_message->estado==2){
					float temp = rnd<DISCRETE_2D>(rand48);//Valor de 0 a 1
					if(temp<probabilidad_estornudar*probabilidad_contagio){//Si el random es mas chico que la probabilidad de contagiarme, me contagio
						agent->estado = 1;
						int prueba1 = floor(((current_message->x+ENV_MAX)/ENV_WIDTH)*d_message_navmap_cell_width);
						int prueba2 = floor(((current_message->y+ENV_MAX)/ENV_WIDTH)*d_message_navmap_cell_width);
						printf("Me contagie y el que me contagió está en la posición %d, %d", prueba1, prueba2);
					}
				}	
			}*/				
		}
		 current_message = get_next_pedestrian_location_message(current_message, pedestrian_location_messages, partition_matrix);
	}

	//maximum velocity rule
	glm::vec2 steer_velocity = navigate_velocity + avoid_velocity;

	agent->steer_x = steer_velocity.x;
	agent->steer_y = steer_velocity.y;

    return 0;
}

__FLAME_GPU_FUNC__ int infect_pedestrians(xmachine_memory_agent* agent, xmachine_message_pedestrian_state_list* pedestrian_state_messages, xmachine_message_pedestrian_state_PBM* partition_matrix, RNG_rand48* rand48){

    xmachine_message_pedestrian_state* current_message = get_first_pedestrian_state_message(pedestrian_state_messages, partition_matrix, agent->x, agent->y, 0.0);
	
	glm::vec2 agent_pos = glm::vec2(agent->x, agent->y);
	
	while (current_message)
	{	
		glm::vec2 agent_pos = glm::vec2(agent->x, agent->y);
		glm::vec2 message_pos = glm::vec2(current_message->x, current_message->y);
		float separation = length(agent_pos - message_pos);
		//Si la distancia entre uno y el otro es menor a la distancia minima
		if ((separation < MESSAGE_RADIUS)&&(separation>MIN_DISTANCE)){

			//Si estoy sano, y me cruce con un paciente que esta enfermo o es portador, cambio mi estado a portador
			if(agent->estado==0){
				if(current_message->estado==1 || current_message->estado==2){
					float temp = rnd<DISCRETE_2D>(rand48);//Valor de 0 a 1
					if(temp<probabilidad_estornudar*probabilidad_contagio){//Si el random es mas chico que la probabilidad de contagiarme, me contagio
						agent->estado = 1;
						int prueba1 = floor(((current_message->x+ENV_MAX)/ENV_WIDTH)*d_message_navmap_cell_width);
						int prueba2 = floor(((current_message->y+ENV_MAX)/ENV_WIDTH)*d_message_navmap_cell_width);
						printf("Me contagie y el que me contagió está en la posición %d, %d", prueba1, prueba2);
					}
				}	
			}			
		}
		 current_message = get_next_pedestrian_state_message(current_message, pedestrian_state_messages, partition_matrix);
	}

	return 0;
}

__FLAME_GPU_FUNC__ int mover_a_destino(xmachine_memory_agent* agent, int ir_x, int ir_y){

	int x = floor(((agent->x+ENV_MAX)/ENV_WIDTH)*d_message_navmap_cell_width);
	int y = floor(((agent->y+ENV_MAX)/ENV_WIDTH)*d_message_navmap_cell_width);

	int dest_x = ir_x-x;
	int dest_y = ir_y-y;
	
	if(dest_x!=0 || dest_y!=0){
		glm::vec2 agent_pos = glm::vec2(agent->x, agent->y);
		glm::vec2 agent_vel = glm::vec2(agent->velx, agent->vely);
		
		//printf("%d, %d\n",dest_x, dest_y);
		
		glm::vec2 agent_steer = glm::vec2(dest_x, dest_y);

		float current_speed = length(agent_vel)+0.025f;//(powf(length(agent_vel), 1.75f)*0.01f)+0.025f;

		//apply more steer if speed is greater
		agent_vel += current_speed*agent_steer;
		float speed = length(agent_vel);
		//limit speed
		if (speed >= agent->speed){
			agent_vel = normalize(agent_vel)*agent->speed;
			speed = agent->speed;
		}

		//update position
		agent_pos += agent_vel*TIME_SCALER;

		
		//animation
		agent->animate += (agent->animate_dir * powf(speed,2.0f)*TIME_SCALER*100.0f);
		if (agent->animate >= 1)
			agent->animate_dir = -1;
		if (agent->animate <= 0)
			agent->animate_dir = 1;
		//lod
		agent->lod = 1;

		//update
		agent->x = agent_pos.x;
		agent->y = agent_pos.y;
		agent->velx = agent_vel.x;
		agent->vely = agent_vel.y;

		//bound by wrapping
		if (agent->x < -1.0f)
			agent->x+=2.0f;
		if (agent->x > 1.0f)
			agent->x-=2.0f;
		if (agent->y < -1.0f)
			agent->y+=2.0f;
		if (agent->y > 1.0f)
			agent->y-=2.0f;

		return 1;
	}
	return 0;
}

/**
 * move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_agent* agent, xmachine_message_check_in_list* checkInMessageList){

	//glm::vec2 agent_steer = glm::vec2(agent->steer_x, agent->steer_y);

    //Como el movimiento se hace cada un tick, se incrementa aca el tick del paciente
	if(agent->estado==1){//Si el agente es portador
		agent->tick++;
		if(agent->tick>=ticks_portador){//Si paso los ticks de portador, me enfermo
			agent->tick=0;
			agent->estado=2;
		}
	}
	if(agent->estado==2){//Si el agente esta enfermo
		agent->tick++;
		if(agent->tick>=ticks_enfermo){//Si paso los ticks de enfermo, me curo
			agent->tick=0;
			agent->estado=0;
			//printf("Me cure");
		}
	}

	switch(agent->estado_movimiento){
		case 0:
			if(mover_a_destino(agent,ir_a_x,ir_a_y) == 0){
				printf("Ya llegue, soy %u\n",agent->id);
				agent->estado_movimiento = 1;
				add_check_in_message(checkInMessageList, agent->id);
			}
			break;
		case 1:
			mover_a_destino(agent,130,90);
			break;
	}
	
	//add_check_in_message(checkInMessageList, agent->id);

	//Por cada tick, informo mi posición
	//printf("%d, ",x);
	//printf("%d\n",y);
	
	//Por cada tick, informo mi estado
	//printf("%d\n",agent->estado);
	
	return 0;
}

/**
 * generate_pedestrians FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_navmap. This represents a single agent instance and can be modified directly.
 * @param agent_agents Pointer to agent list of type xmachine_memory_agent_list. This must be passed as an argument to the add_agent_agent function to add a new agent.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int generate_pedestrians(xmachine_memory_navmap* agent, xmachine_memory_agent_list* agent_agents, RNG_rand48* rand48){

	if (agent->exit_no > 0)
	{
		float random = rnd<DISCRETE_2D>(rand48);
		bool emit_agent = false;
		
		
		if ((agent->exit_no == 1)&&((random < EMMISION_RATE_EXIT1*TIME_SCALER)))
			emit_agent = true;

		if (agent->cant_generados<cant_personas){
			if (emit_agent){
				float x = ((agent->x+0.5f)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
				float y = ((agent->y+0.5f)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
				
				//int exit = getNewExitLocation(rand48);
				float animate = rnd<DISCRETE_2D>(rand48);
				float speed = (rnd<DISCRETE_2D>(rand48))*0.5f + 1.0f;
				
				//Hago el random e imprimo
				float rand = rnd<DISCRETE_2D>(rand48);//Valor de 0 a 1
				/*int estado;
				if(rand<=probabilidad_generar_enfermo){
					estado=2;
					//printf("Enfermo");
				}else{
					estado=0;
					//printf("Sano");
				}*/
				
				//add_agent_agent(agent_agents, agent->cant_generados, x, y, 0.0f, 0.0f, 0.0f, 0.0f, agent->height, 0/*exit*/, speed, 1, animate, 1, estado, 0, 0);
				
				if(agent->cant_generados==0){
					add_agent_agent(agent_agents, agent->cant_generados+1, x, y, 0.0f, 0.0f, 0.0f, 0.0f, agent->height, 0/*exit*/, speed, 1, animate, 1, 0, 0, 0);
				}else{
					add_agent_agent(agent_agents, agent->cant_generados+1, x, y, 0.0f, 0.0f, 0.0f, 0.0f, agent->height, 0/*exit*/, speed, 1, animate, 1, 2, 0, 0);
				}
				
				//printf("%d\n",agent->cant_generados);

				//Imprimo que se creo un paciente
				//printf("Creado\n");
				agent->cant_generados++;
				
				if(agent->cant_generados==cant_personas){
					//printf("Termine de generar personas");
				}
			}
		}
	}


    return 0;
}

__FLAME_GPU_FUNC__ int prueba(xmachine_memory_medic* agent){
	//printf("Hola soy %d\n",agent->x);
	return 0;
}

__FLAME_GPU_FUNC__ int generate_medics(xmachine_memory_navmap* agent, xmachine_memory_medic_list* agent_medics, RNG_rand48* rand48){

	/*float random = rnd<DISCRETE_2D>(rand48);
	bool emit_agent = false;
		
		
	if ((agent->exit_no == 1)&&((random < EMMISION_RATE_EXIT1*TIME_SCALER)))
		emit_agent = true;
	
	if(emit_agent){
		if(agent->cant_generados == 0){
			add_medic_agent(agent_medics, 0);
		}
		if(agent->cant_generados == 1){
			add_medic_agent(agent_medics, 1);
		}
		agent->cant_generados++;
	}*/
	return 0;

}

#endif //_FLAMEGPU_FUNCTIONS
