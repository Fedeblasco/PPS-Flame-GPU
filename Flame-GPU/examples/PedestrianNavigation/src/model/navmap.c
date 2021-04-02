/**
 * generate_pedestrians FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_navmap. This represents a single agent instance and can be modified directly.
 * @param agent_agents Pointer to agent list of type xmachine_memory_agent_list. This must be passed as an argument to the add_agent_agent function to add a new agent.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int generate_pedestrians(xmachine_memory_navmap* agent, xmachine_memory_agent_list* agent_agents, RNG_rand48* rand48){

	if (agent->exit_no > 0)
	{
		float random = rnd<CONTINUOUS>(rand48);
		bool emit_agent = false;

		if ((agent->exit_no == 1)&&((random < EMMISION_RATE_EXIT1*TIME_SCALER)))
			emit_agent = true;
			
		if (agent->cant_generados<cant_personas){ 
			if (emit_agent){
				float x = ((EXIT_X+0.5f)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
				float y = ((EXIT_Y+0.5f)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
				//int exit = getNewExitLocation(rand48);
				float animate = rnd<CONTINUOUS>(rand48);
				float speed = (rnd<CONTINUOUS>(rand48))*0.5f + 1.0f;
				
				//Hago el random e imprimo
				float rand = rnd<CONTINUOUS>(rand48);//Valor de 0 a 1
				int estado;
				if(rand<=PROB_SPAWN_SICK){
					estado=2;
					//printf("Enfermo");
				}else{
					estado=0;
					//printf("Sano");
				}
				
				int vaccine = 0;
				float random = rnd<CONTINUOUS>(rand48);
				if(random <= PROB_VACCINE){
					vaccine = 1;
				}
				add_agent_agent(agent_agents, agent->cant_generados+1, x, y, 0.0f, 0.0f, 0.0f, 0.0f, agent->height, 0/*exit*/, speed, 1, animate, 1, estado, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, vaccine);
				
				
				//printf("%d\n",agent->cant_generados);

				//Imprimo que se creo un paciente
				//printf("Creado\n");
				agent->cant_generados++;
				
				/*if(agent->cant_generados==cant_personas){
					printf("Termine de generar personas");
				}*/
			}
		}
	}


    return 0;
}