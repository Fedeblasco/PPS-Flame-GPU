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
    if (isPriorityFull(size)) 
        return 1; 
    int index = 0;
    int index2 = -1;
    //Me guardo la última posicion vacía, además de la posición donde debería encolarme
    while(patientQueue[index].x != -1){
        //La primer posición que sea más grande que yo me la guardo
        if((patientQueue[index].y > priority) && (index2 == -1)){
            index2 = index;
        }
        index++;
    }
    //Si llegue a una posición vacía, me encolo en esa posición
    if(index2 == -1){
        patientQueue[index].x = patient;
        patientQueue[index].y = priority;
    }else{
        //Corro todos los valores una posicion para la derecha
        for(int i=index;i>=index2;i--){
            patientQueue[i].x = patientQueue[i-1].x;
            patientQueue[i].y = patientQueue[i-1].y;
        }
        //Me encolo donde corresponde
        patientQueue[index2].x = patient;
        patientQueue[index2].y = priority;
    }
    *rear = (*rear + 1) % capacity;
    *size = *size + 1; 
    
    return 0;
} 

// Función para desencolar un valor
__FLAME_GPU_FUNC__ ivec2 priorityDequeue(ivec2 patientQueue[], unsigned int * size, unsigned int * rear) 
{ 
    ivec2 item = {0,0};
    if (isPriorityEmpty(size)){
        return item;
    }
    item.x = patientQueue[0].x;
    item.y = patientQueue[0].y;
    for(int i=0;i < *rear; i++){
        patientQueue[i].x = patientQueue[i+1].x;
        patientQueue[i].y = patientQueue[i+1].y;
    }
    *rear = (*rear - 1) % capacity;
    *size = *size - 1;
    return item;
}

__FLAME_GPU_FUNC__ int printQueue(ivec2 patientQueue[]){ 
    for(int i=0;i<35;i++){
        printf("Posicion %d, paciente %d, prioridad %d\n",i,patientQueue[i].x,patientQueue[i].y);
    }
    return 0;
}