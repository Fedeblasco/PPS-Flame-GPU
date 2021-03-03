//Archivo con las funciones del doctor_manager

/*---------------------------------IMPLEMENTACIÓN DE LA COLA---------------------------------*/

// Inicializa todas las variables necesarias para el manejo de la cola
__FLAME_GPU_FUNC__ int createDoctorQueue(xmachine_memory_doctor_manager* agent){ 
    agent->front = 0;
    agent->size = 0;
    agent->rear = capacity - 1; 
    return 0; 
} 
  
// Devuelve !=0 si la cola esta llena
__FLAME_GPU_FUNC__ int isDoctorFull(xmachine_memory_doctor_manager* agent){ 
    return (agent->size == capacity); 
}
  
// Devuelve !=0 si la cola está vacía
__FLAME_GPU_FUNC__ int isDoctorEmpty(xmachine_memory_doctor_manager* agent){ 
    return (agent->size == 0); 
} 
  
// Función para encolar un valor
__FLAME_GPU_FUNC__ int doctorEnqueue(xmachine_memory_doctor_manager* agent, unsigned int value){ 
    if (isDoctorFull(agent)) 
        return 1; 
    agent->patientQueue[agent->rear] = value;
    agent->rear = (agent->rear + 1) % capacity;
    agent->size = agent->size + 1; 
    
    return 0;
} 

// Función para desencolar un valor
__FLAME_GPU_FUNC__ unsigned int doctorDequeue(xmachine_memory_doctor_manager* agent) 
{ 
    if (isDoctorEmpty(agent)) 
        return 0; 
    int item = agent->patientQueue[agent->front];
    agent->front = (agent->front + 1) % capacity; 
    agent->size = agent->size - 1;
    return item;
} 

/*--------------------------------- Atención de pacientes ---------------------------------*/

__FLAME_GPU_FUNC__ int receive_doctor_petitions(xmachine_memory_doctor_manager* agent, xmachine_message_doctor_petition_list* doctorPetitionMessages, xmachine_message_doctor_response_list* doctorResponseMessages){
	
	//Chequeo todos los mensajes que recibo y encolo los que necesite
    xmachine_message_doctor_petition* current_message = get_first_doctor_petition_message(doctorPetitionMessages);
	int enqueue_message = 1;
    while(current_message){
        for(int i=0;i<4;i++){
            if(current_message->id == agent->doctorArray[i]){
                agent->doctorArray[i] = 0; 
                //printf("Libero la posicion %d, quedo en el valor %d\n",i,agent->doctorArray[i]);
                enqueue_message = 0;
            }
        }
        if(enqueue_message){
            //printf("Encolando el mensaje %d\n",current_message->id);
            doctorEnqueue(agent, current_message->id);
        }
        current_message = get_next_doctor_petition_message(current_message, doctorPetitionMessages);	
	}

    if(!isDoctorEmpty(agent)){
        for (int i = 0; i<4; i++){
            if(agent->doctorArray[i] == 0){
                agent->doctorArray[i] = doctorDequeue(agent);
                add_doctor_response_message(doctorResponseMessages,agent->doctorArray[i],i);
                break;
            }
        }
    }   

	return 0;
}