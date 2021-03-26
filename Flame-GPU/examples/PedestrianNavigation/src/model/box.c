__FLAME_GPU_FUNC__ int determine_priority(RNG_rand48* rand48){
	float random = rnd<DISCRETE_2D>(rand48);
	float acc = 0;
	float prob_level[] = {PROB_LEVEL_1,PROB_LEVEL_2,PROB_LEVEL_3,PROB_LEVEL_4,PROB_LEVEL_5};
	int level_time[] = {0,15,60,120,240};
	for(int i = 0; i<5; i++){
		if(acc < random <= (acc+prob_level[i])){
			return level_time[i];
		}
		acc+=prob_level[i];
	}
	
	return 0;
}

__FLAME_GPU_FUNC__ int determine_room(RNG_rand48* rand48){
	float random = rnd<DISCRETE_2D>(rand48);
	float acc = 0;
	float prob_esp[] = {PROB_MEDICAL,PROB_SURGICAL,PROB_PEDIATRICS,PROB_GYNECOLOGIST,PROB_GERIATRICS,PROB_PSYCHIATRY,PROB_UCI};
	for(int i = 0; i<7; i++){
		float acctemp = acc+prob_esp[i];
		if((acc < random) && (random <= acctemp)){
			return i;
		}
		acc+=prob_esp[i];
	}
	
	return 0;
}

__FLAME_GPU_FUNC__ int box_server(xmachine_memory_box* agent, xmachine_message_box_petition_list* boxPetitionMessages){
	
	xmachine_message_box_petition* current_message = get_first_box_petition_message(boxPetitionMessages);
	while(current_message){
        if(current_message->box_no == agent->id){
			agent->attending = current_message->id;
			agent->tick = 0;
		}
        current_message = get_next_box_petition_message(current_message, boxPetitionMessages);	
	}
	
	return 0;
}

__FLAME_GPU_FUNC__ int attend_box_patient(xmachine_memory_box* agent, xmachine_message_box_response_list* boxResponseMessages, RNG_rand48* rand48){
	
	agent->tick++;
	if(agent->tick * MINUTES_PER_TICK == 15){
		int room = determine_room(rand48);
		int priority = determine_priority(rand48);
		//printf("Soy el box %d y le estoy mandando al paciente %d que vaya a %d con prioridad %d\n",agent->id,agent->attending,room,priority);
		add_box_response_message(boxResponseMessages, agent->attending, room, priority);
		agent->attending = 0;
	}
	
	return 0;
}