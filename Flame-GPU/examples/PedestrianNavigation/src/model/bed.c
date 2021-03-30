__FLAME_GPU_FUNC__ int output_bed_state(xmachine_memory_bed* agent, xmachine_message_bed_contact_list* bedContactMessages, xmachine_message_bed_state_list* bedStateMessages, RNG_rand48* rand48){
    
    xmachine_message_bed_contact* current_message = get_first_bed_contact_message(bedContactMessages);
	while(current_message){
        if(current_message->bed_no == agent->id){
			//printf("Soy la cama %d y me llegó el mensaje %d de %d con estado %d\n",agent->id,current_message->bed_no,current_message->id,current_message->state);
            if((current_message->state == 1) || (current_message->state == 2)){
                float rand = rnd<CONTINUOUS>(rand48);
                if(rand <= PROB_INFECT_BED){
                    agent->state = 1;
                }
            }
            add_bed_state_message(bedStateMessages,current_message->id,agent->state);
		}
		current_message = get_next_bed_contact_message(current_message, bedContactMessages);
	}

    agent->tick++;
    if(agent->tick * SECONDS_PER_TICK>=CLEANING_PERIOD_SECONDS){
        //printf("Soy la cama %d y me limpio\n",agent->id);
        agent->state = 0;
        agent->tick = 0;
    }
    
    return 0; 
}