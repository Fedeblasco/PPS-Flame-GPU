# Curiosidades de la simulación

1. La función getNewExitLocation cambia el color de los pedestrian por algún motivo
   1. Si los pedestrian no tienen una salida elegida, se convierten de color azul
   2. Parece que en el archivo PedestrianPopulation.h se definen los colores a
      usar en el pedestrian, habría que seguir investigando pero está por ahi la
      cosa

2. En IO.cu tengo como leer un archivo desde flame, capaz sirve para algo