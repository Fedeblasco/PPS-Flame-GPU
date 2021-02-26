//Archivo con las funciones del administrador de sillas

__FLAME_GPU_FUNC__ int attend_chair_petitions(xmachine_memory_chair_admin* agent, xmachine_message_chair_petition_list* chairPetitionMessages, xmachine_message_chair_response_list* chairResponseMessages){
	
	xmachine_message_chair_petition* current_message = get_first_chair_petition_message(chairPetitionMessages);
	while(current_message){
		int index = -1;
		for(int i=0;i<35;i++){
			if((agent->chairArray[i] == 0) && (index == -1)){//Si una silla esta disponible y todavía no elegi ninguna, me guardo su número
				index = i;
			}
			if(agent->chairArray[i] == current_message->id){//Si recibo un mensaje de la persona que esta sentada, libero el asiento
				printf("Libero la sillita %d\n", i);
				agent->chairArray[i] = 0;
				return 0;
			}
		}

		if(index != -1){
			agent->chairArray[index] = current_message->id;//Marco que esta ocupada
		}
		
		add_chair_response_message(chairResponseMessages, current_message->id, index);

        current_message = get_next_chair_petition_message(current_message, chairPetitionMessages);	
	}

	return 0;
}


