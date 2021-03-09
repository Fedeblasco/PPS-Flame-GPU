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
#include "parameters.h"
#include "queue.c"
#include "priority_queue.c"
#include "CustomVisualisation.h"

#include "specialist.c"
#include "receptionist.c"
#include "chair_admin.c"
#include "doctor_manager.c"
#include "doctor.c"
#include "chair.c"
#include "agent_generator.c"
#include "triage.c"
#include "box.c"

//This function creates all the agents requiered to run the hospital
__FLAME_GPU_INIT_FUNC__ void inicializarMapa(){
	printf("Inicializando todo\n");
	
	//Agregado del recepcionista
	// Allocating memory in CPU to save the agent
	xmachine_memory_receptionist * h_receptionist = h_allocate_agent_receptionist();
	// Copying the agent from the CPU to GPU
	h_add_agent_receptionist_defaultReceptionist(h_receptionist);
	// Freeing the previously allocated memory
	h_free_agent_receptionist(&h_receptionist);

	//Agregado del administrador de sillas
	// Allocating memory in CPU to save the agent
	xmachine_memory_chair_admin * h_chair_admin = h_allocate_agent_chair_admin();
	// Copying the agent from the CPU to GPU
	h_add_agent_chair_admin_defaultAdmin(h_chair_admin);
	// Freeing the previously allocated memory
	h_free_agent_chair_admin(&h_chair_admin);

	//Agregado del generador de agentes
	// Allocating memory in CPU to save the agent
	xmachine_memory_agent_generator * h_agent_generator = h_allocate_agent_agent_generator();
	// Copying the agent from the CPU to GPU
	h_add_agent_agent_generator_defaultGenerator(h_agent_generator);
	// Freeing the previously allocated memory
	h_free_agent_agent_generator(&h_agent_generator);

	//Agregado del triage
	// Allocating memory in CPU to save the agent
	xmachine_memory_triage * h_triage = h_allocate_agent_triage();
	// Copying the agent from the CPU to GPU
	h_add_agent_triage_defaultTriage(h_triage);
	// Freeing the previously allocated memory
	h_free_agent_triage(&h_triage);

	//Agregado del manejador de doctores
	// Allocating memory in CPU to save the agent
	xmachine_memory_doctor_manager * h_doctor_manager = h_allocate_agent_doctor_manager();
	// Copying the agent from the CPU to GPU
	h_add_agent_doctor_manager_defaultDoctorManager(h_doctor_manager);
	// Freeing the previously allocated memory
	h_free_agent_doctor_manager(&h_doctor_manager);

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

__FLAME_GPU_FUNC__ int output_chair_petition(xmachine_memory_agent* agent, xmachine_message_chair_petition_list* chairPetitionMessages){

	add_chair_petition_message(chairPetitionMessages, agent->id);
	if(agent->id == 32){
		printf("Soy 32 y quiero una silla\n");
	}
	agent->estado_movimiento++;

	return 0;
} 

__FLAME_GPU_FUNC__ int output_chair_contact(xmachine_memory_agent* agent, xmachine_message_chair_contact_list* chairContactMessages){
	
	add_chair_contact_message(chairContactMessages, agent->id, agent->chair_no, agent->estado);

	return 0;
}

__FLAME_GPU_FUNC__ int output_triage_petition(xmachine_memory_agent* agent, xmachine_message_triage_petition_list* triagePetitionMessages){
	
	add_triage_petition_message(triagePetitionMessages, agent->id);
	agent->estado_movimiento++;

	return 0;
}

__FLAME_GPU_FUNC__ int output_doctor_petition(xmachine_memory_agent* agent, xmachine_message_doctor_petition_list* doctorPetitionMessages){
	
	add_doctor_petition_message(doctorPetitionMessages, agent->id, agent->doctor_no, agent->priority);
	agent->estado_movimiento++;

	return 0;
}

__FLAME_GPU_FUNC__ int output_doctor_reached(xmachine_memory_agent* agent, xmachine_message_doctor_reached_list* doctorReachedMessages){
	
	add_doctor_reached_message(doctorReachedMessages, agent->id, agent->doctor_no);
	agent->estado_movimiento++;

	return 0;
}

__FLAME_GPU_FUNC__ int output_specialist_petition(xmachine_memory_agent* agent, xmachine_message_specialist_petition_list* specialistPetitionMessages){
	
	add_specialist_petition_message(specialistPetitionMessages, agent->id, agent->doctor_no);
	agent->estado_movimiento++;

	return 0;
}

__FLAME_GPU_FUNC__ int output_box_petition(xmachine_memory_agent* agent, xmachine_message_box_petition_list* boxPetitionMessages){
	
	add_box_petition_message(boxPetitionMessages, agent->id, agent->box_no);
	//printf("Soy %d y mande el mensaje %d\n",agent->id, agent->box_no);
	agent->estado_movimiento++;

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
		//Si la distancia entre uno y el otro es mayor a la distancia minima y menor al radio definido
		if ((separation < MESSAGE_RADIUS)&&(separation>MIN_DISTANCE)){

			//Si estoy sano, y me cruce con un paciente que esta enfermo o es portador, cambio mi estado a portador
			if(agent->estado==0){
				if(current_message->estado==1 || current_message->estado==2){
					float temp = rnd<DISCRETE_2D>(rand48);//Valor de 0 a 1
					if(temp<probabilidad_estornudar*probabilidad_contagio){//Si el random es mas chico que la probabilidad de contagiarme, me contagio
						agent->estado = 1;
						//int prueba1 = floor(((current_message->x+ENV_MAX)/ENV_WIDTH)*d_message_navmap_cell_width);
						//int prueba2 = floor(((current_message->y+ENV_MAX)/ENV_WIDTH)*d_message_navmap_cell_width);
						//printf("Me contagie y el que me contagió está en la posición %d, %d", prueba1, prueba2);
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

__FLAME_GPU_FUNC__ int go_to_doctor(xmachine_memory_agent* agent){
	
	switch(agent->checkpoint){
		case 0:
			//El que va al cuarto doctor se maneja distinto para que esquive la esquina
			if(agent->go_to_y == 41){
				if(mover_a_destino(agent,agent->go_to_x+20,59) == 0){
					agent->checkpoint++;
				}
			}else{
				if(mover_a_destino(agent,agent->go_to_x+20,agent->go_to_y) == 0){
					agent->checkpoint = 2;
				}
			}
			break;
		//Utilizado solo por el que va al cuarto doctor
		case 1:
			if(mover_a_destino(agent,agent->go_to_x+20,agent->go_to_y) == 0){
					agent->checkpoint++;
			}
			break;
		case 2:
			if(mover_a_destino(agent,agent->go_to_x,agent->go_to_y) == 0){
				return 1;
			}
			break;
	}

	return 0;

}

__FLAME_GPU_FUNC__ int go_to_exit(xmachine_memory_agent* agent){

	switch(agent->checkpoint){
		case 0:
			if(mover_a_destino(agent,agent->go_to_x+20,agent->go_to_y) == 0){
				//printf("Llegue al primer checkpoint\n");
				agent->checkpoint++;
			}
			break;
		case 1:
			if(mover_a_destino(agent,88,96) == 0){
				//printf("Llegue al segundo checkpoint\n");
				agent->checkpoint++;
			}
			break;
		case 2:
			if(mover_a_destino(agent,150,102) == 0){
				return 1;
			}
			break;
	}

	return 0;

}

/**
 * move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.

 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_agent* agent, xmachine_message_check_in_list* checkInMessages){

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
			if(mover_a_destino(agent,88,96) == 0){
				agent->estado_movimiento++;
			}
			break; 
		case 3:
			if(mover_a_destino(agent,agent->go_to_x,agent->go_to_y) == 0){
				add_check_in_message(checkInMessages, agent->id);
				agent->estado_movimiento++;
			}
			break;
		case 5:
			if(mover_a_destino(agent,88,96) == 0){
				agent->estado_movimiento++;
			}
			break;
		case 6:
			if(mover_a_destino(agent,130,90) == 0){
				agent->estado_movimiento++;
			}
			break;
		case 7:
			if(mover_a_destino(agent,140,80) == 0){
				agent->estado_movimiento ++;
				add_check_in_message(checkInMessages, agent->id);
			}
			break;
		case 9:
			if(mover_a_destino(agent,130,90) == 0){
				agent->estado_movimiento++;
			}
			break;
		case 10:
			if(mover_a_destino(agent,88,96) == 0){
				agent->estado_movimiento++;
			}
			break;
		case 13:
			if(mover_a_destino(agent,agent->go_to_x,agent->go_to_y) == 0){
				agent->estado_movimiento++;
			}
			break;
		case 16:
			if(mover_a_destino(agent,88,96) == 0){
				agent->estado_movimiento++;
			}
			break;
		case 17:
			if(mover_a_destino(agent,106,96) == 0){
				agent->estado_movimiento++;
			}
			break;
		case 18:
			if(mover_a_destino(agent,106,78) == 0){
				agent->estado_movimiento++;
			}
			break;
		case 22:
			if(mover_a_destino(agent,106,96) == 0){
				agent->estado_movimiento++;
			}
			break;
		case 23:
			if(mover_a_destino(agent,88,96) == 0){
				agent->estado_movimiento++;
			}
			break;
		case 26:
			if(mover_a_destino(agent,agent->go_to_x,agent->go_to_y) == 0){
				agent->estado_movimiento++;
			}
			break;
		case 30:
			if(go_to_doctor(agent)){
				printf("Llegue al doctor\n");
				agent->checkpoint=0;
				agent->estado_movimiento++;
			}
			break;
		case 40:
			if(go_to_exit(agent)){
				return 1;
			}
			break;
	}
	 
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
				float rand = rnd<CONTINUOUS>(rand48);//Valor de 0 a 1
				int estado;
				if(rand<=probabilidad_generar_enfermo){
					estado=2;
					//printf("Enfermo");
				}else{
					estado=0;
					//printf("Sano");
				}
				
				add_agent_agent(agent_agents, agent->cant_generados+1, x, y, 0.0f, 0.0f, 0.0f, 0.0f, agent->height, 0/*exit*/, speed, 1, animate, 1, estado, 0, 0, 0, 0, 0, 0, 0, 0, 0);
				
				/*if(agent->cant_generados==0){
					add_agent_agent(agent_agents, agent->cant_generados+1, x, y, 0.0f, 0.0f, 0.0f, 0.0f, agent->height, 0, speed, 1, animate, 1, 0, 0, 0, 0, 0);
				}else{
					add_agent_agent(agent_agents, agent->cant_generados+1, x, y, 0.0f, 0.0f, 0.0f, 0.0f, agent->height, 0, speed, 1, animate, 1, 2, 0, 0, 0, 0);
				}*/
				
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

/*-------------------------------------------------------------Recepción de mensajes-------------------------------------------------------------*/

__FLAME_GPU_FUNC__ int receive_check_in_response(xmachine_memory_agent* agent, xmachine_message_check_in_response_list* avisarPacienteMessages, xmachine_message_chair_petition_list* chairPetitionMessages){

	xmachine_message_check_in_response* current_message = get_first_check_in_response_message(avisarPacienteMessages);
	while(current_message){
		if(current_message->id == agent->id){
			//printf("Soy %d y recibi este mensaje %d\n",agent->id,current_message->id);
			if(agent->chair_no != -1){
				add_chair_petition_message(chairPetitionMessages, agent->id);
				agent->chair_no = -1;
			}
			agent->estado_movimiento++;
		}
		current_message = get_next_check_in_response_message(current_message, avisarPacienteMessages);
	}

	return 0;
}

__FLAME_GPU_FUNC__ int receive_chair_response(xmachine_memory_agent* agent, xmachine_message_chair_response_list* chairResponseMessages){

	xmachine_message_chair_response* current_message = get_first_chair_response_message(chairResponseMessages);
	while(current_message){
		if((current_message->id == agent->id) && (current_message->chair_no != -1)){
			agent->go_to_x = firstChair_x + (space_between * ((current_message->chair_no+7)%7));
			agent->go_to_y = firstChair_y - (space_between * int(current_message->chair_no/7));
			agent->estado_movimiento++;
			agent->chair_no = current_message->chair_no;
			//printf("Soy %d y me voy a sentar en la silla %d, posX %d, posY %d\n\n",agent->id,current_message->chair_no,agent->go_to_x,agent->go_to_y);
		}else{
			//printf("Me muero");
			return 1;
		}
		current_message = get_next_chair_response_message(current_message, chairResponseMessages);
	}

	return 0;
}

__FLAME_GPU_FUNC__ int receive_chair_state(xmachine_memory_agent* agent, xmachine_message_chair_state_list* chairStateMessages, RNG_rand48* rand48){
	
	xmachine_message_chair_state* current_message = get_first_chair_state_message(chairStateMessages);
	while(current_message){
		if(current_message->id == agent->id){
			if(current_message->state == 1 && agent->estado == 0){
               agent->estado = 1;
            }
		}
		current_message = get_next_chair_state_message(current_message, chairStateMessages);
	}

	return 0;
}

__FLAME_GPU_FUNC__ int receive_triage_response(xmachine_memory_agent* agent, xmachine_message_triage_response_list* triageResponseMessages, xmachine_message_chair_petition_list* chairPetitionMessages){
	
	xmachine_message_triage_response* current_message = get_first_triage_response_message(triageResponseMessages);
	while(current_message){
		if(agent->id == current_message->id){
			agent->estado_movimiento++;
			agent->box_no = current_message->box_no;
			add_chair_petition_message(chairPetitionMessages, agent->id);
			agent->chair_no = -1;
			//printf("Tengo que ir al box %d, soy %d\n",current_message->box_no,agent->id);
		}
		current_message = get_next_triage_response_message(current_message, triageResponseMessages);
	}

	return 0;
}

__FLAME_GPU_FUNC__ int receive_doctor_response(xmachine_memory_agent* agent, xmachine_message_doctor_response_list* doctorResponseMessages, xmachine_message_chair_petition_list* chairPetitionMessages){
	
	xmachine_message_doctor_response* current_message = get_first_doctor_response_message(doctorResponseMessages);
	while(current_message){
		if(agent->id == current_message->id){
			if(current_message->doctor_no !=-1){
				agent->go_to_x = firstDoctor_x;
				agent->go_to_y = firstDoctor_y - (space_between_doctors * current_message->doctor_no);;
				agent->estado_movimiento++;
				agent->chair_no = -1;
				agent->doctor_no = current_message->doctor_no;
				//printf("Tengo que ir al doctor %d, soy %d\n",current_message->doctor_no,agent->id);
			}else{
				//printf("No hay doctores disponibles, soy %d\n",agent->id);
			}
			//printf("Enviando un mensaje, soy %d\n",agent->id);
			//add_chair_petition_message(chairPetitionMessages, agent->id);
		}
		current_message = get_next_doctor_response_message(current_message, doctorResponseMessages);
	}

	return 0;
}

__FLAME_GPU_FUNC__ int receive_attention_terminated(xmachine_memory_agent* agent, xmachine_message_attention_terminated_list* attentionTerminatedMessages, xmachine_message_free_doctor_list* freeDoctorMessages){
	
	xmachine_message_attention_terminated* current_message = get_first_attention_terminated_message(attentionTerminatedMessages);
	while(current_message){
		if(agent->id == current_message->id){
			add_free_doctor_message(freeDoctorMessages, agent->doctor_no);
			agent->estado_movimiento = 40;
		}
		current_message = get_next_attention_terminated_message(current_message, attentionTerminatedMessages);
	}

	return 0;
}

__FLAME_GPU_FUNC__ int receive_box_response(xmachine_memory_agent* agent, xmachine_message_box_response_list* boxResponseMessages){
	
	xmachine_message_box_response* current_message = get_first_box_response_message(boxResponseMessages);
	while(current_message){
		if(agent->id == current_message->id){
			agent->estado_movimiento++;
			agent->priority = current_message->priority;
			agent->doctor_no = current_message->doctor_no;
			printf("Mi prioridad es %d, tengo que ir a %d, soy %d\n",current_message->priority,current_message->doctor_no,agent->id);
		}
		current_message = get_next_box_response_message(current_message, boxResponseMessages);
	}

	return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
