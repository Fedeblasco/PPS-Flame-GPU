# Cosas a probar si funcionan/son implementables en Flame-GPU

1. Ir hacia un lugar en específico con el pedestrian 
   * Bien, por lo visto en la función force_flow se determina la posición en x e
     y a la que tiene que ir el pedestrian, y después hay unas cosas muy
     complicadas para que no se choquen entre los personajes y las paredes que
     son horribles pero bueno es lo que hay.
   * Cada posición en donde estás parado sabe decirle al pedestrian para donde
     tiene que ir para llegar a la salida, por esto es super complicado
     asignarle otro valor.
   * Posible solución: tomar a los doctores y asientos como si fuesen salidas y
     así hacer que vayan a ese lugar. Es muy probable que no pueda usar el
     pedestrian navigation.
2. Crear una cola con prioridades
   * Aparentemente Flame no permite variables globales por lo que la cola la va a tener que tener el doctor
   * Por lo visto se puede hacer una cola de mensajes para el doctor, y mandarle un vector con dos posiciones, mi id y el timeout (con el tipo de dato
     vector, uvec2)
   * Los arreglos se pueden definir en la memoria del agente de la siguiente manera:
         <gpu:variable>
            <type>float</type>
            <name>nums</name>
            <arrayLength>64<arrayLength>
         </gpu:variable>
   * StableMarriage es un ejemplo que tiene todo lo de los id, y como mandar
     mensajes entre agentes por id
   * En simulation.cu de StableMarriage, linea 1444 se puede ver como se accede
     a los vectores como preferred_woman
     * preferred_woman[(j * xmachine_memory_Man_MAX) + i]
   * Para hacer el mensaje para una persona en particular se usa su id, y se
     manda un mensaje con partitioningNONE, esta persona chequea si coincide su
     id, y en caso de que coincida responde en consecuencia
   * Ver StableMarriage, el functions.c y el xml a lo último para ver como estan
     hechos los mensajes, pero parece simple
3. Finalizar la simulación en un punto
   * Aparentemente se puede hacer con la consola, se le manda la cantidad de ticks que se quieren ejecutar y listo
   * Ojo con la salida, porque constantemente quiere escribir a disco y se muere pobre
   * Bien, se puede sacar la escritura a xml desde el archivo main.cu dentro de la carpeta Examples/PedestrianNavigation/src/dynamic, se comentan las lineas y listo
   * Parece ser que tarda 8 veces menos al ejecutarse por consola, con visualización tarda aproximadamente 2:40 min, por consola 21 segundos
   * Ejecutando sin ningun tipo de impresión por consola, el tiempo que tarda es de 6.5 segundos, que es increible
   * Parece ser que la mejor manera va a ser hacer un script que envíe la cantidad de iteraciones que se quieren hacer del modelo por parámetro
   * También se podría hacer un script de python para que escriba el código y lo compile
4. Modificar la cantidad de iteraciones
   * Seguramente lo tenga que hacer desde afuera con python, pero es muy simple de hacer
   * Se le dice la cantidad de ticks que querés que corra cuando lo haces
     ejecutar por consola
5. Generar un id único para el agente
   * Parte Generating new agent IDs en la documentación de Flame
   * Por lo visto se pone una variable de tipo unsigned int llamada id y genera solo las funciones para que ande
   * Se puede modificar el valor de donde van a empezar los id