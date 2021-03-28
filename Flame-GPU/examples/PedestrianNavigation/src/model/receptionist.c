//Archivo con las funciones del recepcionista	

//Función que chequea los pacientes que llegan y los atiende
__FLAME_GPU_FUNC__ int reception_server(xmachine_memory_receptionist* agent, xmachine_message_check_in_list* checkInMessages, xmachine_message_check_in_response_list* patientMessages){
	
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
    }else if(agent->attend_patient == 1){
        agent->tick++;
        if(agent->tick * SECONDS_PER_TICK >= RECEPTION_SECONDS){
            //printf("Enviando mensaje 2 a %d\n",agent->current_patient);
            add_check_in_response_message(patientMessages, agent->current_patient);
            agent->tick = 0;
            agent->current_patient = -1;
            agent->attend_patient = 0;
        }
    }
	
	return 0;
}