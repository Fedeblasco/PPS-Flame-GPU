//Archivo con las funciones del recepcionista	

//Función que chequea los pacientes que llegan y los atiende
__FLAME_GPU_FUNC__ int receptionServer(xmachine_memory_receptionist* agent, xmachine_message_check_in_list* checkInMessages, xmachine_message_check_in_response_list* patientMessages){
	
	xmachine_message_check_in* current_message = get_first_check_in_message(checkInMessages);
	while(current_message){
        //Si llega el paciente que tengo que atender, prendo el flag de atención
        if(current_message->id == agent->current_patient){
            agent->attend_patient = 1;
        }else{
            enqueue(agent->patientQueue, current_message->id,&agent->size, &agent->rear);
        }
        current_message = get_next_check_in_message(current_message, checkInMessages);	
	}

    //Si tengo algun paciente esperando y no estoy procesando a nadie
    if((!isEmpty(&agent->size)) && (agent->current_patient == -1)){
        unsigned int patient = dequeue(agent->patientQueue, &agent->size, &agent->front);
        add_check_in_response_message(patientMessages, patient);
        agent->current_patient = patient;
        //printf("Enviando mensaje 1 a %d\n",agent->current_patient);
        /*agent->tick++;
        if(agent->tick >= espera){
            unsigned int prueba = dequeue(agent);
            add_check_in_response_message(patientMessages, prueba);
            agent->tick = 0;
        }*/
    }else if(agent->attend_patient == 1){
        agent->tick++;
        if(agent->tick*MINUTES_PER_TICK >= min_espera_recepcionista){
            //printf("Enviando mensaje 2 a %d\n",agent->current_patient);
            add_check_in_response_message(patientMessages, agent->current_patient);
            agent->tick = 0;
            agent->current_patient = -1;
            agent->attend_patient = 0;
        }
    }
	
	return 0;
}

__FLAME_GPU_FUNC__ int infect_receptionist(xmachine_memory_receptionist* agent, xmachine_message_pedestrian_state_list* pedestrian_state_messages, xmachine_message_pedestrian_state_PBM* partition_matrix, RNG_rand48* rand48){
    /*float x = ((140)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
    float y = ((80)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
    printf(" nuevo %f %f\n",x,y);*/
	
    xmachine_message_pedestrian_state* current_message = get_first_pedestrian_state_message(pedestrian_state_messages, partition_matrix, 0.093750, -0.375000, 0.0);
	
	while (current_message){	
        glm::vec2 agent_pos = glm::vec2(0.093750, -0.375000);
		glm::vec2 message_pos = glm::vec2(current_message->x, current_message->y);
		float separation = length(agent_pos - message_pos);
		
        //Si la distancia entre uno y el otro es mayor a la distancia minima y menor al radio definido
		if((separation < MESSAGE_RADIUS)&&(separation>MIN_DISTANCE)){
			//Si estoy sano, y me cruce con un paciente que esta enfermo o es portador, cambio mi estado a portador
            if(agent->estado==0){
				if(current_message->estado==1 || current_message->estado==2){
                    float temp = rnd<DISCRETE_2D>(rand48);//Valor de 0 a 1
					if(temp<probabilidad_estornudar*probabilidad_contagio_personal){//Si el random es mas chico que la probabilidad de contagiarme, me contagio
						agent->estado = 1;
						int prueba1 = floor(((current_message->x+ENV_MAX)/ENV_WIDTH)*d_message_navmap_cell_width);
						int prueba2 = floor(((current_message->y+ENV_MAX)/ENV_WIDTH)*d_message_navmap_cell_width);
						//printf("Me contagie y el que me contagió está en la posición %d, %d", prueba1, prueba2);
					}
				}	
			}			
		}
		current_message = get_next_pedestrian_state_message(current_message, pedestrian_state_messages, partition_matrix);
	}

	return 0;
}