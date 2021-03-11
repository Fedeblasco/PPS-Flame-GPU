__FLAME_GPU_FUNC__ int receive_specialist_reached(xmachine_memory_specialist* agent, xmachine_message_specialist_reached_list* specialistReachedMessages, xmachine_message_attention_terminated_list* attentionTerminatedMessages){
    
    xmachine_message_specialist_reached* current_message = get_first_specialist_reached_message(specialistReachedMessages);
    while(current_message){
        if(current_message->specialist_no == agent->id){
            printf("Llego la persona %d al especialista %d\n",current_message->id,current_message->specialist_no);
            agent->current_patient = current_message->id;
        }
        current_message = get_next_specialist_reached_message(current_message, specialistReachedMessages);
	}

    if(agent->current_patient != 0){
        agent->tick++;
        if(agent->tick*TICKS_PER_MINUTE == 300){
            printf("Bueno ahi termine de atenderlo a %d\n",agent->current_patient);
            add_attention_terminated_message(attentionTerminatedMessages,agent->current_patient);
            agent->tick = 0;
            agent->current_patient = 0;
        }
    }

    return 0;
}