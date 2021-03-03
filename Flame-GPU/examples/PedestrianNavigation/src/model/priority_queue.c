/*---------------------------------IMPLEMENTACIÓN DE LA COLA---------------------------------*/
  
// Devuelve !=0 si la cola esta llena
__FLAME_GPU_FUNC__ int isPriorityFull(unsigned int * size){ 
    return (*size == capacity); 
}
  
// Devuelve !=0 si la cola está vacía
__FLAME_GPU_FUNC__ int isPriorityEmpty(unsigned int * size){ 
    return (*size == 0); 
} 
  
// Función para encolar un valor
__FLAME_GPU_FUNC__ int priorityEnqueue(ivec2 patientQueue[], int patient, int priority, unsigned int * size, unsigned int * rear){ 
    if (isFull(size)) 
        return 1; 
    //patientQueue[*rear] = value;
    *rear = (*rear + 1) % capacity;
    *size = *size + 1; 
    
    return 0;
} 

// Función para desencolar un valor
__FLAME_GPU_FUNC__ ivec2 priorityDequeue(ivec2 patientQueue[], unsigned int * size, unsigned int * front) 
{ 
    /*if (isEmpty(size)){
        return 0;
    }*/
    /*ivec2 item;
    item.x = patientQueue[*front].x;
    item.y = patientQueue[*front].y;
    *front = (*front + 1) % capacity; 
    *size = *size - 1;
    return item;*/
}