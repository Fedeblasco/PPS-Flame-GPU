#define SCALE_FACTOR 0.03125

#define I_SCALER (SCALE_FACTOR*0.35f)
#define MESSAGE_RADIUS d_message_pedestrian_location_radius
#define MIN_DISTANCE 0.0001f

//#define NUM_EXITS 7  

#define PI 3.1415f
#define RADIANS(x) (PI / 180.0f) * x

//Probabilidades usadas para manejar la cantidad de enfermos
#define probabilidad_estornudar 1.0
#define probabilidad_contagio 1.0
#define probabilidad_generar_enfermo 0.2
//Cantidad de ticks enfermo y portador 
#define ticks_portador 10000
#define ticks_enfermo 10000
//Cantidad de personas a generar
#define cant_personas 2
 
//Manejo de las sillas
#ifndef firstChair_x
#define firstChair_x 50
#endif
#ifndef firstChair_y
#define firstChair_y 85
#endif
#ifndef space_between
#define space_between 3
#endif

#define capacity 2000
#define espera 60
#define probabilidad_contagio_personal 1.0

#define probabilidad_contagiar_silla 1.0