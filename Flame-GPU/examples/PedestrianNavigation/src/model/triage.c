//Archivo con las funciones del triage

/*--------------------------------- Atención de pacientes ---------------------------------*/

__FLAME_GPU_FUNC__ int receive_triage_petitions(xmachine_memory_triage* agent, xmachine_message_triage_petition_list* triagePetitionMessages, xmachine_message_triage_response_list* triageResponseMessages){
	
	//Chequeo todos los mensajes que recibo y encolo los que necesite
    xmachine_message_triage_petition* current_message = get_first_triage_petition_message(triagePetitionMessages);
    while(current_message){
        enqueue(agent->patientQueue, current_message->id,&agent->size, &agent->rear);
        current_message = get_next_triage_petition_message(current_message, triagePetitionMessages);	
	}

    if(!isEmpty(&agent->size)){
        for (int i = 0; i<3; i++){
            //printf("La posicion %d tiene el valor %d\n",i,agent->free_boxes[i]);
            if(agent->free_boxes[i] == 0){
                agent->free_boxes[i] = dequeue(agent->patientQueue, &agent->size, &agent->front);
                add_triage_response_message(triageResponseMessages,agent->free_boxes[i],i);
                break;
            }
        }
    }

	return 0;
}

__FLAME_GPU_FUNC__ int receive_free_box(xmachine_memory_triage* agent, xmachine_message_free_box_list* freeBoxMessages){
    xmachine_message_free_box* current_message = get_first_free_box_message(freeBoxMessages);
    while(current_message){
        agent->free_boxes[current_message->box_no] = 0;
        //printf("Liberando el box %d, su valor despues es %d\n",current_message->box_no, agent->free_boxes[current_message->box_no]);
        current_message = get_next_free_box_message(current_message, freeBoxMessages);	
	}
    return 0;
}