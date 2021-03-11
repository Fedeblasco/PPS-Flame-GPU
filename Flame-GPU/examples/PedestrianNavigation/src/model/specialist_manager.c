__FLAME_GPU_FUNC__ int receive_specialist_petitions(xmachine_memory_specialist_manager* agent, xmachine_message_specialist_petition_list* specialistPetitionMessages, xmachine_message_specialist_response_list* specialistResponseMessages){
	
    ivec2 * ptr[5];
    ptr[0] = agent->surgicalQueue;
    ptr[1] = agent->pediatricsQueue;
    ptr[2] = agent->gynecologistQueue;
    ptr[3] = agent->geriatricsQueue;
    ptr[4] = agent->psychiatristQueue;
    
    //Chequeo todos los mensajes que recibo y encolo los que necesite
    xmachine_message_specialist_petition* current_message = get_first_specialist_petition_message(specialistPetitionMessages);
    while(current_message){
        int index = current_message->specialist_no-1;
        priorityEnqueue(ptr[index],current_message->id,current_message->priority,&agent->size[index],&agent->rear[index]);
        //printQueue(ptr[index]);
        current_message = get_next_specialist_petition_message(current_message, specialistPetitionMessages);
	}

    //Manejo de colas para cada uno de los especialistas
    for(int i = 0; i<5; i++){
        if(!isPriorityEmpty(&agent->size[i])){
            if(agent->free_specialist[i] > 0){
                agent->free_specialist[i]--;
                ivec2 paciente = priorityDequeue(ptr[i],&agent->size[i],&agent->rear[i]);
                add_specialist_response_message(specialistResponseMessages,paciente.x,1);
                //printf("El paciente %d va al especialista 52\n",paciente.x);
                //printQueue(ptr[i]);
            }
            if(agent->tick[i] == TICKS_PER_MINUTE){
                for(int j=0;j<agent->rear[i];j++){
                    ptr[i][j].y--;
                    if(ptr[i][j].y < 0){
                        //printf("El paciente %d va al especialista -1\n",ptr[i][j].x);
                        add_specialist_response_message(specialistResponseMessages,ptr[i][j].x,-1);
                        priorityDequeue(ptr[i],&agent->size[i],&agent->rear[i]);
                    }
                }
                agent->tick[i]=0;
            }
            //printQueue(ptr[i]);
            agent->tick[i]++;
        }
    }

	return 0;
}

__FLAME_GPU_FUNC__ int receive_free_specialists(xmachine_memory_specialist_manager* agent){
    
    return 0;
}