//Archivo con las funciones del administrador de sillas

__FLAME_GPU_FUNC__ int attend_chair_petitions(xmachine_memory_chair_admin* agent, xmachine_message_chair_petition_list* chairPetitionMessages, xmachine_message_chair_response_list* chairResponseMessages){
	
	xmachine_message_chair_petition* current_message = get_first_chair_petition_message(chairPetitionMessages);
	while(current_message){
		int index = -1;
		for(int i=0;i<35;i++){
			if(agent->chairArray[i] == 0){
				index = i;
				break;
			}
		}

		if(index != -1){
			agent->chairArray[index] = 1;//Marco que esta ocupada
		}
		add_chair_response_message(chairResponseMessages, current_message->id, index);

        current_message = get_next_chair_petition_message(current_message, chairPetitionMessages);	
	}

	return 0;
}


