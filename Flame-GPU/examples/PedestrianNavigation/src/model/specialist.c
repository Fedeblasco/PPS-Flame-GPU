__FLAME_GPU_FUNC__ int receive_specialist_petitions(xmachine_memory_specialist* agent, xmachine_message_specialist_petition_list* specialistPetitionMessages, xmachine_message_specialist_response_list* specialistResponseMessages){
	
	//Chequeo todos los mensajes que recibo y encolo los que necesite
    xmachine_message_specialist_petition* current_message = get_first_specialist_petition_message(specialistPetitionMessages);
    while(current_message){
        if(current_message->specialist_no == agent->id){
            printf("Recibi un mensaje de %d, con prioridad %d, soy %d\n",current_message->id,current_message->priority,current_message->specialist_no);
            priorityEnqueue(agent->patientQueue,current_message->id,current_message->priority,&agent->size,&agent->rear);
            current_message = get_next_specialist_petition_message(current_message, specialistPetitionMessages);
        }	
	}
	return 0;
}

__FLAME_GPU_FUNC__ int specialist_server(xmachine_memory_specialist* agent, xmachine_message_specialist_reached_list* specialistReachedMessages, xmachine_message_attention_terminated_list* attentionTerminatedMessages){
    
    return 0;
}