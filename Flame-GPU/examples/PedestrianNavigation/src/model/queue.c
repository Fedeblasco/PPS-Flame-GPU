/*---------------------------------IMPLEMENTACIÓN DE LA COLA---------------------------------*/
  
// Devuelve !=0 si la cola esta llena
__FLAME_GPU_FUNC__ int isFull(unsigned int * size){ 
    return (*size == capacity); 
}
  
// Devuelve !=0 si la cola está vacía
__FLAME_GPU_FUNC__ int isEmpty(unsigned int * size){ 
    return (*size == 0); 
} 
  
// Función para encolar un valor
__FLAME_GPU_FUNC__ int enqueue(unsigned int patientQueue[], unsigned int value, unsigned int * size, unsigned int * rear){ 
    if (isFull(size)) 
        return 1; 
    patientQueue[*rear] = value;
    *rear = (*rear + 1) % capacity;
    *size = *size + 1; 
    
    return 0;
} 

// Función para desencolar un valor
__FLAME_GPU_FUNC__ unsigned int dequeue(unsigned int patientQueue[], unsigned int * size, unsigned int * front) 
{ 
    if (isEmpty(size)) 
        return 0; 
    int item = patientQueue[*front];
    *front = (*front + 1) % capacity; 
    *size = *size - 1;
    return item;
}