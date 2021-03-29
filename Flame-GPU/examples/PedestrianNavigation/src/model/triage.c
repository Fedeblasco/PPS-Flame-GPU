//Archivo con las funciones del triage

/*--------------------------------- Atención de pacientes ---------------------------------*/

__FLAME_GPU_FUNC__ int receive_triage_petitions(xmachine_memory_triage* agent, xmachine_message_triage_petition_list* triagePetitionMessages, xmachine_message_triage_response_list* triageResponseMessages){
	
	//Chequeo todos los mensajes que recibo y encolo los que necesite
    xmachine_message_triage_petition* current_message = get_first_triage_petition_message(triagePetitionMessages);
	int enqueue_message = 1;
    while(current_message){
        for(int i=0;i<3;i++){
            if(current_message->id == agent->free_boxes[i]){
                agent->free_boxes[i] = 0; 
                //printf("Libero la posicion %d, quedo en el valor %d\n",i,agent->free_boxes[i]);
                enqueue_message = 0;
            }
        }
        if(enqueue_message){
            //printf("Encolando el mensaje %d\n",current_message->id);
            enqueue(agent->patientQueue, current_message->id,&agent->size, &agent->rear);
        }
        current_message = get_next_triage_petition_message(current_message, triagePetitionMessages);	
	}

    if(!isEmpty(&agent->size)){
        for (int i = 0; i<3; i++){
            if(agent->free_boxes[i] == 0){
                agent->free_boxes[i] = dequeue(agent->patientQueue, &agent->size, &agent->front);
                add_triage_response_message(triageResponseMessages,agent->free_boxes[i],i);
                break;
            }
        }
    }

	return 0;
}