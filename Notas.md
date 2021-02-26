# Curiosidades de la simulación

1. La función getNewExitLocation cambia el color de los pedestrian por algún motivo
   1. Si los pedestrian no tienen una salida elegida, se convierten de color azul
   2. Parece que en el archivo PedestrianPopulation.h se definen los colores a
      usar en el pedestrian, habría que seguir investigando pero está por ahi la
      cosa

2. En IO.cu tengo como leer un archivo desde flame, capaz sirve para algo
3. En la función generate pedestrians, 0.5f significa un desplazamiento de 1
   posición. Le suma 0.5f porque en los navmap las posiciones arrancan de 0 en
   vez de 1
   1. Tomando los valores x e y de la función generate pedestrians se le puede
      hacer lo de la función floor y ver el tamaño real del piso
4. Para generar personas, tiene que ser una vez por tick, sino no te deja
5. Si se usa defaultValue en arreglos, modifica solo el primer valor del arreglo (pos 0)
6. No hay persistencia de datos al cambiar de estado en los agentes (o por lo menos no hay persistencia en los arreglos)
7. Al final se optó por ponerles condiciones a cada una de las funciones y que se ejecuten si cumplen estas condiciones
8. Se hace un doble chequeo del radio del mensaje enviado, para no tener problemas al generarlo y poder variar el radio más facil

# Cosas de Flame-GPU

1. Throughout a simulation, agent data is persistent however message information (and in particular message lists) is persistent only over the lifecycle of a single iteration. (Pag 3)
2. El procesador XSLT permite generar todos los archivos necesarios para la simulación, por ejemplo el functions.c
3. Las funciones init, step y exit se ejecutan en CPU así que en principio es como utilizar C de toda la vida (se puede utilizar para setear las constantes globales, está bastante piola)
4. Hay que tener cuidado con el buffer size de cada agente
5. The current state is defined within the currentState element and is used to filter the agent function by only applying it to agents in the specified state. (Pág 16)
6. The reallocate element is used as an optional flag to indicate the possibility that an agent performing the agent function may die as a result (and hence require removing from the agent population). By default this value is assumed true however if a value of false is specified then the processes for removing dead agents will not be executed even if an agent indicates it has died. (Pág 16)
7. **Agent function message outputs** Importantisimo, final página 16 y principio de la 17, explica como hacer mensajes opcionales para no tener problemas
8. An xagentOutput does not require a type (as is the case with a message output) and any agent function outputting an agent is assumed to be optional. I.e. each agent performing the function may output either one or zero agents. (Pág 17)
9. Ver condiciones en funciones, puede ser útil
10. Within a given layer, the order of execution of layer functions should not be assumed to be sequential (although in the current version of the software it is, future versions will execute functions within the same layer in parallel). (Pág 19)
11. A return value of anything other than 0 indicates that the agent has died and should be removed from the simulation (unless the agent function definition had specifically set the reallocate element value to false in which case any agent deaths will be ignored). (Pág 24)
12. Agents are only permitted to output at most a single message per agent function and repeated calls to an add message function will result in previous message information simply being overwritten. (Pág 24)
13. Firstly it is essential that message loop complete naturally. I.e. the get_next_*name*_message function must be called without breaking from the while loop until the end of the message list is reached. (Pág 25)
14. Por lo visto se pueden usar mensajes discretos agregandole el template parameter <continuous> para indicar que es un agente continuo (Pág 27)
15. En la página 28 habla de que los mensajes de tipo graph edge permiten identificar un agente y que este vea solamente los mensajes que le mandan a él, en vez de iterar por todos los mensajes, habría que probarlo.
16. Generating new agent IDs (Pág 31)
17. If a template parameter value is not specified then the simulation will assume a DISCRETE_2D value which will work in either case but is more compu-tationally expensive. (Pág 32)
18. The initialisation function declaration should be preceded with a __FLAME_GPU_INIT_FUNC__ macro definition, should have no arguments and should return void. (Pág 32)
19. Las step y exit functions se definen de la misma manera que la init, no devuelven nada y necesitan de su header específico
20. Getting and Setting Simulation Constants (Global Variables) (Página 33) Ver para poder editar las constantes globales.
21. Analytics functions sirve para la salida, se pueden calcular muchas cosas utilizando estas funciones en las host hooks (Pág 36)
22. Generating a functions file template (Pág 39)
23. Ver parameter exploration, para la calibración del sistema una vez esté funcionando. (Pág 47)