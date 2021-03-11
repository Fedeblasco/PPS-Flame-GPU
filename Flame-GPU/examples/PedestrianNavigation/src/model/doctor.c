__FLAME_GPU_FUNC__ int doctor_server(xmachine_memory_doctor* agent, xmachine_message_doctor_reached_list* doctorReachedMessages, xmachine_message_attention_terminated_list* attentionTerminatedMessages){
    xmachine_message_doctor_reached* current_message = get_first_doctor_reached_message(doctorReachedMessages);
    while(current_message){
        if(current_message->doctor_no == agent->id){
            //printf("Me pide %d, ir a %d, soy el médico %d\n",current_message->id,current_message->doctor_no,agent->id);
            agent->current_patient = current_message->id;
            agent->tick=0;
        }
        current_message = get_next_doctor_reached_message(current_message, doctorReachedMessages);	
	}

    if(agent->current_patient!=0){
        if(agent->tick * MINUTES_PER_TICK == 600){
            //printf("Soy el doctor %d y le aviso a %d que se vaya\n",agent->id,agent->current_patient);
            add_attention_terminated_message(attentionTerminatedMessages,agent->current_patient);
            agent->current_patient = 0;
        }
        agent->tick++;
    }
    return 0;
}