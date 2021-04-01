//Archivo con las funciones del administrador de sillas

__FLAME_GPU_FUNC__ int determine_stay_time(RNG_rand48* rand48){
	float random = rnd<CONTINUOUS>(rand48);
	float acc = 0;
    float acctemp = 0;
	float prob[] = {PROB_STAY_1, PROB_STAY_2, PROB_STAY_3, PROB_STAY_4, PROB_STAY_5, PROB_STAY_6, PROB_STAY_7, PROB_STAY_8, PROB_STAY_9,
                         PROB_STAY_10, PROB_STAY_11, PROB_STAY_12, PROB_STAY_13, PROB_STAY_14, PROB_STAY_15, PROB_STAY_16, PROB_STAY_17, PROB_STAY_18};
    float stay_time[] = {STAY_TIME_1, STAY_TIME_2, STAY_TIME_3, STAY_TIME_4, STAY_TIME_5, STAY_TIME_6, STAY_TIME_7, STAY_TIME_8, STAY_TIME_9,
                         STAY_TIME_10, STAY_TIME_11, STAY_TIME_12, STAY_TIME_13, STAY_TIME_14, STAY_TIME_15, STAY_TIME_16, STAY_TIME_17, STAY_TIME_18};
	for(int i = 0; i<18; i++){
		acctemp = acc+prob[i];
		if((acc < random) && (random <= acctemp)){
			return stay_time[i]*24*60;//Devuelvo el tiempo en minutos que debe estar el paciente internado
		}
		acc+=prob[i];
	}
	
	return 0;
}

__FLAME_GPU_FUNC__ int attend_bed_petitions(xmachine_memory_uci* agent, xmachine_message_bed_petition_list* bedPetitionMessages, xmachine_message_bed_response_list* bedResponseMessages, RNG_rand48* rand48){
	
    xmachine_message_bed_petition* current_message = get_first_bed_petition_message(bedPetitionMessages);
	while(current_message){
        
        int index = -1;  //Variable utilizada para el cálculo de índice random
		int index2 = -1; //Variable utilizada para el cálculo de índice lineal

		for(int i=0;i<NUMBER_OF_BEDS;i++){
			
			if(index == -1){
				int random = (rnd<CONTINUOUS>(rand48))*NUMBER_OF_BEDS;
				if((agent->bedArray[random].x == 0) && (random < NUMBER_OF_BEDS)){ //Se pone el random < NUMBER_OF_BEDS por la remota posibilidad devuelva NUMBER_OF_BEDS, que está fuera de rango
					index = random;
				}
				
				if((agent->bedArray[i].x == 0) && (index2 == -1)){//Si una cama está disponible y todavía no elegi ninguna linealmente, me guardo su número
					index2 = i;
				}
			}
		}
        if(index != -1){
            agent->bedArray[index].x = current_message->id;//Marco que esta ocupada
            agent->bedArray[index].y = determine_stay_time(rand48);
            add_bed_response_message(bedResponseMessages, current_message->id, index);
        }else{
            if(index2 != -1){
                agent->bedArray[index2].x = current_message->id;//Marco que esta ocupada
                agent->bedArray[index2].y = determine_stay_time(rand48);
            }
            add_bed_response_message(bedResponseMessages, current_message->id, index2);
        }

        current_message = get_next_bed_petition_message(current_message, bedPetitionMessages);	
	}

    //Manejo de pacientes y su tiempo hospitalizados

    agent->tick++;
    if(agent->tick * SECONDS_PER_TICK >= 60){
        for(int i=0;i<NUMBER_OF_BEDS;i++){
            //Si la cama está ocupada
            if(agent->bedArray[i].x != 0){
                agent->bedArray[i].y--;
                //Si se le acabo el tiempo al paciente, le envío un mensaje para que se retire
                if(agent->bedArray[i].y <= 0){
                    //printf("Enviandole a %d que se retire\n",agent->bedArray[i].x);
                    add_bed_response_message(bedResponseMessages, agent->bedArray[i].x, -1);
                    agent->bedArray[i].x = 0;
                    agent->bedArray[i].y = 0;
                }
            }
        }
        agent->tick = 0;
    }

	return 0;
}