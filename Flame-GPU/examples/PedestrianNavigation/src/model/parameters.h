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
#define cant_personas 600
 
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

//Constante de tiempo
#ifndef MINUTES_PER_TICK
#define MINUTES_PER_TICK 1
#endif

//Probabilidades de prioridad del triage
#ifndef prob_level_1
#define prob_level_1 0.0418719
#endif
#ifndef prob_level_2
#define prob_level_2 0.0862069
#endif
#ifndef prob_level_3
#define prob_level_3 0.6305419
#endif
#ifndef prob_level_4
#define prob_level_4 0.2266010
#endif
#ifndef prob_level_5
#define prob_level_5 0.0147783
#endif

//Probabilidades de especialidad
#ifndef prob_esp_quirurgicas
#define prob_esp_quirurgicas 0.292939087//Modificado del original para que sume 1
#endif
#ifndef prob_esp_medicas
#define prob_esp_medicas 0.444567551
#endif
#ifndef prob_pediatria
#define prob_pediatria 0.051335318
#endif
#ifndef prob_cuid_intensivos
#define prob_cuid_intensivos 0.070724557
#endif
#ifndef prob_ginecologia
#define prob_ginecologia 0.075895021
#endif
#ifndef prob_geriatria
#define prob_geriatria 0.01892759
#endif
#ifndef prob_psiquiatria
#define prob_psiquiatria 0.045610876
#endif

#define capacity 100
#define espera 60
#define probabilidad_contagio_personal 1.0

#define probabilidad_contagiar_silla 1.0