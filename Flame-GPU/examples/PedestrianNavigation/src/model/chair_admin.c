//Archivo con las funciones del administrador de sillas

__FLAME_GPU_FUNC__ int attend_chair_petitions(xmachine_memory_chair_admin* agent, xmachine_message_chair_petition_list* chairPetitionMessages, xmachine_message_chair_response_list* chairResponseMessages, RNG_rand48* rand48){
	
	xmachine_message_chair_petition* current_message = get_first_chair_petition_message(chairPetitionMessages);
	int send_message = 1;
	int corte_de_control = 0;
	while((current_message)&&(corte_de_control<2)){
		
		//printf("Hola, me llego el mensaje %d",current_message->id);
		int index = -1;  //Variable utilizada para el cálculo de índice random
		int index2 = -1; //Variable utilizada para el cálculo de índice lineal
		//printf("Recibi un mensaje de la persona %d\n",current_message->id);
		corte_de_control++;

		for(int i=0;i<35;i++){
			
			if(index == -1){
				int random = (rnd<DISCRETE_2D>(rand48))*35;
				if((agent->chairArray[random] == 0) && (random < 35)){ //Se pone el random < 35 por la remota posibilidad devuelva 35
					index = random;
				}
				
				if((agent->chairArray[i] == 0) && (index2 == -1)){//Si una silla esta disponible y todavía no elegi ninguna linealmente, me guardo su número
					index2 = i;
				}
			}
			if(agent->chairArray[i] == current_message->id){//Si recibo un mensaje de la persona que esta sentada, libero el asiento
				agent->chairArray[i] = 0;
				//printf("Liberando silla %d, de la persona %d\n\n",i,current_message->id);
				send_message = 0;
			}
		}
		if(send_message){
			if(index != -1){
				agent->chairArray[index] = current_message->id;//Marco que esta ocupada
				add_chair_response_message(chairResponseMessages, current_message->id, index);//Envío la silla, si mando -1 es que no tiene sillas disponibles
				//printf("Sentate en la posición random %d\n",index);
			}else{
				if(index2 != -1){
					agent->chairArray[index2] = current_message->id;//Marco que esta ocupada
					//printf("Sentate en la posición lineal %d\n",index2);
				}
				add_chair_response_message(chairResponseMessages, current_message->id, index2);//Envío la silla, si mando -1 es que no tiene sillas disponibles
			}
		}

        current_message = get_next_chair_petition_message(current_message, chairPetitionMessages);	
	}
	if(corte_de_control == 2){
		printf("Corte de control para %d\n",current_message->id);
	}

	return 0;
}

__FLAME_GPU_FUNC__ int receive_free_chair(xmachine_memory_chair_admin* agent, xmachine_message_free_chair_list* freeChairMessages){
	
	xmachine_message_free_chair* current_message = get_first_free_chair_message(freeChairMessages);
	while(current_message){
		agent->chairArray[current_message->chair_no] = 0;
		//printf("Libere la silla número %d\n",current_message->chair_no);
		current_message = get_next_free_chair_message(current_message, freeChairMessages);
	}
	return 0;
}