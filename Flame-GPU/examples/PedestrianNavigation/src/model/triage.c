//Archivo con las funciones del triage

/*---------------------------------IMPLEMENTACIÓN DE LA COLA---------------------------------*/

/*// Inicializa todas las variables necesarias para el manejo de la cola
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
} */

/*--------------------------------- Atención de pacientes ---------------------------------*/

__FLAME_GPU_FUNC__ int receive_triage_petitions(xmachine_memory_triage* agent, xmachine_message_triage_petition_list* triagePetitionMessages, xmachine_message_triage_response_list* triageResponseMessages){
	
	//Chequeo todos los mensajes que recibo y encolo los que necesite
    xmachine_message_triage_petition* current_message = get_first_triage_petition_message(triagePetitionMessages);
	int enqueue_message = 1;
    while(current_message){
        for(int i=0;i<3;i++){
            if(current_message->id == agent->boxArray[i]){
                agent->boxArray[i] = 0; 
                //printf("Libero la posicion %d, quedo en el valor %d\n",i,agent->boxArray[i]);
                enqueue_message = 0;
            }
        }
        if(enqueue_message){
            //printf("Encolando el mensaje %d\n",current_message->id);
            enqueue(agent->patientQueue, current_message->id,&agent->size, &agent->rear);
        }
        current_message = get_next_triage_petition_message(current_message, triagePetitionMessages);	
	}

    if(!isEmpty(&agent->size)){
        for (int i = 0; i<3; i++){
            if(agent->boxArray[i] == 0){
                agent->boxArray[i] = dequeue(agent->patientQueue, &agent->size, &agent->front);
                add_triage_response_message(triageResponseMessages,agent->boxArray[i],i);
                break;
            }
        }
    }

	return 0;
}