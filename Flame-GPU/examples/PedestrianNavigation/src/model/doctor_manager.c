//Archivo con las funciones del doctor_manager

/*--------------------------------- Atención de pacientes ---------------------------------*/

__FLAME_GPU_FUNC__ int receive_doctor_petitions(xmachine_memory_doctor_manager* agent, xmachine_message_doctor_petition_list* doctorPetitionMessages, xmachine_message_doctor_response_list* doctorResponseMessages){
	
	//Chequeo todos los mensajes que recibo y encolo los que necesite
    xmachine_message_doctor_petition* current_message = get_first_doctor_petition_message(doctorPetitionMessages);
    while(current_message){
        printf("Me pide %d, el médico %d, con la priorirdad %d\n",current_message->id,current_message->doctor_no,current_message->priority);
        current_message = get_next_doctor_petition_message(current_message, doctorPetitionMessages);	
	}

    /*if(!isDoctorEmpty(agent)){
        for (int i = 0; i<4; i++){
            if(agent->doctorArray[i] == 0){
                agent->doctorArray[i] = doctorDequeue(agent);
                add_doctor_response_message(doctorResponseMessages,agent->doctorArray[i],i);
                break;
            }
        }
    }*/

	return 0;
}