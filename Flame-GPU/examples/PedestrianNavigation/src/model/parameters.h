#define SCALE_FACTOR 0.03125

#define I_SCALER (SCALE_FACTOR*0.35f)
#define MESSAGE_RADIUS d_message_pedestrian_location_radius
#define MIN_DISTANCE 0.0001f

//#define NUM_EXITS 7  

//Cantidad de ticks enfermo y portador 
#define ticks_portador 500
#define ticks_enfermo 500

//Cantidad de personas a generar
#define cant_personas 600

//Capacidad de las colas
#define capacity 35

#define PI 3.1415f
#define RADIANS(x) (PI / 180.0f) * x

//Constante de tiempo
#ifndef MINUTES_PER_TICK
#define MINUTES_PER_TICK 1
#endif

#ifndef TICKS_PER_MINUTE
#define TICKS_PER_MINUTE 1
#endif

//Posición de la salida
/*#define EXIT_X 150
#define EXIT_Y 102*/

//Probabilidades usadas para manejar la cantidad de enfermos
/*#define probabilidad_estornudar 0.1
#define probabilidad_contagio 0.1
#define probabilidad_generar_enfermo 0.2
#define probabilidad_contagio_personal 1.0
#define probabilidad_contagiar_silla 1.0*/
 
//Manejo de las sillas
/*#ifndef firstChair_x
#define firstChair_x 50
#endif
#ifndef firstChair_y
#define firstChair_y 85
#endif
#ifndef space_between
#define space_between 3
#endif*/

//Manejo de los doctores
/*#ifndef firstDoctor_x
#define firstDoctor_x 15
#endif
#ifndef firstDoctor_y
#define firstDoctor_y 95
#endif
#ifndef space_between_doctors
#define space_between_doctors 18
#endif*/

//Manejo de los especialistas
/*#ifndef firstSpecialist_x
#define firstSpecialist_x 58
#endif
#ifndef firstSpecialist_y
#define firstSpecialist_y 40
#endif
#ifndef space_between_specialists
#define space_between_specialists 27
#endif*/

/*#ifndef fifthSpecialist_x
#define fifthSpecialist_x 15
#endif
#ifndef fifthSpecialist_y
#define fifthSpecialist_y 16
#endif*/

//Probabilidades de prioridad del triage
/*#ifndef prob_level_1
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
#endif*/

//Probabilidades de especialidad
/*#ifndef prob_esp_quirurgicas
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
#endif*/

// Manejo del recepcionista
/*#define min_espera_recepcionista 1
  #define receptionist_x 1
  #define receptionist_y 1
*/

// Checkpoints
/*  #define CHECKPOINT_1_X 87
  #define CHECKPOINT_1_Y 60
  #define CHECKPOINT_2_X 71
  #define CHECKPOINT_2_Y 64
  #define CHECKPOINT_3_X 59
  #define CHECKPOINT_3_Y 64
  #define CHECKPOINT_4_X 26
  #define CHECKPOINT_4_Y 40
  #define CHECKPOINT_5_X 26
  #define CHECKPOINT_5_Y 11

//UCI
  #define UCI_X 66
  #define UCI_Y 11*/