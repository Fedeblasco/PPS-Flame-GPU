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