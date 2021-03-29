__FLAME_GPU_FUNC__ int output_chair_state(xmachine_memory_chair* agent, xmachine_message_chair_contact_list* chairContactMessages, xmachine_message_chair_state_list* chairStateMessages, RNG_rand48* rand48){
    
    xmachine_message_chair_contact* current_message = get_first_chair_contact_message(chairContactMessages);
	while(current_message){
		if(current_message->chair_no == agent->id){
			if((current_message->state == 1) || (current_message->state == 2)){
                float rand = rnd<CONTINUOUS>(rand48);
                if(rand <= PROB_INFECT_CHAIR){
                    agent->state = 1;
                }
            }
		}
        add_chair_state_message(chairStateMessages,current_message->id,agent->state);
		current_message = get_next_chair_contact_message(current_message, chairContactMessages);
	}

    agent->tick++;
    if(agent->tick * SECONDS_PER_TICK>=CLEANING_PERIOD_SECONDS){
        //printf("Soy la silla %d y me limpio\n",agent->id);
        agent->state = 0;
        agent->tick = 0;
    }
    
    return 0; 
}