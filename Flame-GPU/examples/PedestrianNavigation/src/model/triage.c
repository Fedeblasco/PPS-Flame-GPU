//Archivo con las funciones del triage

/*---------------------------------IMPLEMENTACIÓN DE LA COLA---------------------------------*/

// Inicializa todas las variables necesarias para el manejo de la cola
__FLAME_GPU_FUNC__ int createTriageQueue(xmachine_memory_triage* agent){ 
    agent->front = 0;
    agent->size = 0;
    agent->rear = capacity - 1; 
    return 0; 
} 
  
// Devuelve !=0 si la cola esta llena
__FLAME_GPU_FUNC__ int isTriageFull(xmachine_memory_triage* agent){ 
    return (agent->size == capacity); 
}
  
// Devuelve !=0 si la cola está vacía
__FLAME_GPU_FUNC__ int isTriageEmpty(xmachine_memory_triage* agent){ 
    return (agent->size == 0); 
} 
  
// Función para encolar un valor
__FLAME_GPU_FUNC__ int triageEnqueue(xmachine_memory_triage* agent, unsigned int value){ 
    if (isTriageFull(agent)) 
        return 1; 
    agent->patientQueue[agent->rear] = value;
    agent->rear = (agent->rear + 1) % capacity;
    agent->size = agent->size + 1; 
    
    return 0;
} 

// Función para desencolar un valor
__FLAME_GPU_FUNC__ unsigned int triageDequeue(xmachine_memory_triage* agent) 
{ 
    if (isTriageEmpty(agent)) 
        return 0; 
    int item = agent->patientQueue[agent->front];
    agent->front = (agent->front + 1) % capacity; 
    agent->size = agent->size - 1;
    return item;
} 

/*--------------------------------- Atención de pacientes ---------------------------------*/

__FLAME_GPU_FUNC__ int receive_triage_petitions(xmachine_memory_triage* agent, xmachine_message_triage_petition_list* triagePetitionMessages, xmachine_message_triage_response_list* triageResponseMessages, RNG_rand48* rand48){
	
	//Chequeo todos los mensajes que recibo y encolo los que necesite
    xmachine_message_triage_petition* current_message = get_first_triage_petition_message(triagePetitionMessages);
	while(current_message){
        printf("Encolando el mensaje %d\n",current_message->id);
        triageEnqueue(agent, current_message->id);
        current_message = get_next_triage_petition_message(current_message, triagePetitionMessages);	
	}

    if(!isTriageEmpty(agent)){
        for (int i = 0; i<3; i++){
            if(agent->boxArray[i] == 0){
                agent->boxArray[i] = triageDequeue(agent);
                add_triage_response_message(triageResponseMessages,agent->boxArray[i],i);
                break;
            }
        }
    }

	return 0;
}