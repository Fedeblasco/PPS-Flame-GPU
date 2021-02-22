//Archivo con las funciones del recepcionista	

#define capacity 2000
#define espera 60

/*---------------------------------IMPLEMENTACIÓN DE LA COLA---------------------------------*/

// Inicializa todas las variables necesarias para el manejo de la cola
__FLAME_GPU_FUNC__ int createQueue(xmachine_memory_receptionist* agent){ 
    agent->front = 0;
    agent->size = 0;
    agent->rear = capacity - 1; 
    return 0; 
} 
  
// Devuelve !=0 si la cola esta llena
__FLAME_GPU_FUNC__ int isFull(xmachine_memory_receptionist* agent){ 
    return (agent->size == capacity); 
}
  
// Devuelve !=0 si la cola está vacía
__FLAME_GPU_FUNC__ int isEmpty(xmachine_memory_receptionist* agent){ 
    return (agent->size == 0); 
} 
  
// Función para encolar un valor
__FLAME_GPU_FUNC__ int enqueue(xmachine_memory_receptionist* agent, unsigned int value){ 
    if (isFull(agent)) 
        return 1; 
    agent->colaPacientes[agent->rear] = value;
    agent->rear = (agent->rear + 1) % capacity;
    agent->size = agent->size + 1; 
    
    return 0;
} 

// Función para desencolar un valor
__FLAME_GPU_FUNC__ unsigned int dequeue(xmachine_memory_receptionist* agent) 
{ 
    if (isEmpty(agent)) 
        return 0; 
    int item = agent->colaPacientes[agent->front];
    agent->front = (agent->front + 1) % capacity; 
    agent->size = agent->size - 1;
    return item;
} 

//Función que chequea los pacientes que llegan y los atiende
__FLAME_GPU_FUNC__ int receptionServer(xmachine_memory_receptionist* agent, xmachine_message_check_in_list* checkInMessages, xmachine_message_avisar_paciente_list* patientMessages){
	
	xmachine_message_check_in* current_message = get_first_check_in_message(checkInMessages);
	while(current_message && current_message->id!=0){
        if(current_message->id > agent->last){
            enqueue(agent, current_message->id);
            if(current_message->estado >= 1){
                agent->estado = 1;
                //printf("Uy me enferme");
            }
            agent->last = current_message->id;
        }
        current_message = get_next_check_in_message(current_message, checkInMessages);	
	}
    if(!isEmpty(agent)){
        agent->tick++;
        if(agent->tick >= espera){
            unsigned int prueba = dequeue(agent);
            add_avisar_paciente_message(patientMessages, prueba);
            agent->tick = 0;
        }
    }
	
	return 0;
}

/* 
// Función que devuelve el principio de la cola 
int front(xmachine_memory_receptionist* agent) 
{ 
    if (isEmpty(queue)) 
        return INT_MIN; 
    return queue->array[queue->front]; 
} 
  
// Function to get rear of queue 
int rear(struct Queue* queue) 
{ 
    if (isEmpty(queue)) 
        return INT_MIN; 
    return queue->array[queue->rear]; 
} */

/*  
// Driver program to test above functions./ 
int main() 
{ 
    struct Queue* queue = createQueue(1000); 
  
    enqueue(queue, 10); 
    enqueue(queue, 20); 
    enqueue(queue, 30); 
    enqueue(queue, 40); 
  
    printf("%d dequeued from queue\n\n", 
           dequeue(queue)); 
  
    printf("Front item is %d\n", front(queue)); 
    printf("Rear item is %d\n", rear(queue)); 
  
    return 0; 
}
*/