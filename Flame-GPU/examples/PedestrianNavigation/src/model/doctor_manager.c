//Archivo con las funciones del doctor_manager

/*--------------------------------- Atención de pacientes ---------------------------------*/

__FLAME_GPU_FUNC__ int receive_doctor_petitions(xmachine_memory_doctor_manager* agent, xmachine_message_doctor_petition_list* doctorPetitionMessages, xmachine_message_doctor_response_list* doctorResponseMessages){
	
	//Chequeo todos los mensajes que recibo y encolo los que necesite
    xmachine_message_doctor_petition* current_message = get_first_doctor_petition_message(doctorPetitionMessages);
    while(current_message){
        //printf("Me pide %d, el médico %d, con la priorirdad %d\n",current_message->id,current_message->doctor_no,current_message->priority);
        priorityEnqueue(agent->patientQueue,current_message->id,current_message->priority,&agent->size,&agent->rear);
        //probando(agent->patientQueue);
        current_message = get_next_doctor_petition_message(current_message, doctorPetitionMessages);	
	}
    if(!isPriorityEmpty(&agent->size)){
        if(agent->free_doctors > 0){
            for (int i = 0; i<4; i++){
                if(agent->doctors_occupied[i]==0){
                    agent->doctors_occupied[i]=1;
                    agent->free_doctors--;
                    ivec2 paciente = priorityDequeue(agent->patientQueue,&agent->size,&agent->rear);
                    add_doctor_response_message(doctorResponseMessages,paciente.x,i);
                    //printf("El paciente %d va al medico %d\n",paciente.x,i);
                    //probando(agent->patientQueue);
                    break;
                }
            }
        }
        if(agent->tick == TICKS_PER_MINUTE){
            for(int i=0;i<agent->rear;i++){
                agent->patientQueue[i].y--;
                if(agent->patientQueue[i].y < 0){
                    add_doctor_response_message(doctorResponseMessages,agent->patientQueue[i].x,-1);
                    priorityDequeue(agent->patientQueue,&agent->size,&agent->rear);
                }
            }
            agent->tick=0;
        }
        agent->tick++;
    }
	return 0;
}

__FLAME_GPU_FUNC__ int receive_free_doctors(xmachine_memory_doctor_manager* agent, xmachine_message_free_doctor_list* freeDoctorMessages){
    xmachine_message_free_doctor* current_message = get_first_free_doctor_message(freeDoctorMessages);
    while(current_message){
        printf("Liberando el doctor %d\n",current_message->doctor_no);
        agent->free_doctors++;
        agent->doctors_occupied[current_message->doctor_no]=0;
        current_message = get_next_free_doctor_message(current_message, freeDoctorMessages);	
	}
    return 0;
}