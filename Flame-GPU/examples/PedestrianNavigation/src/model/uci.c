//Archivo con las funciones del administrador de sillas

__FLAME_GPU_FUNC__ int determine_stay_time(RNG_rand48* rand48){
	float random = rnd<CONTINUOUS>(rand48);
	float acc = 0;
    float acctemp = 0;
	float prob[] = {0.004748328,0.088623115,0.017333166,0.032968386,0.013086353,0.100335789,0.066380434,0.000017555,0.007899849,
                    0.100224175,0.084432757,0.117953925,0.053206605,0.026187069,0.122177398,0.033379033,0.037753818,0.092694777};
    float stay_time[] = {2.6,3.0,3.3,3.7,4.2,4.4,4.7,4.9,6.3,6.4,6.5,6.7,7.4,8.0,9.2,9.3,10.2,29.7};
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
                }
            }
        }
        agent->tick = 0;
    }

	return 0;
}