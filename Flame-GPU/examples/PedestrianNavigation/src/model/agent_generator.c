
//Manejo de las sillas
#include "parameters.h"

__FLAME_GPU_FUNC__ int generate_chairs(xmachine_memory_agent_generator* agent, xmachine_memory_chair_list* chair_agents){
    
    int x = firstChair_x + (space_between * ((agent->chairs_generated+7)%7));
    int y = firstChair_y - (space_between * int(agent->chairs_generated/7));
    
    //printf("Silla %d, x:%d, y:%d\n", agent->chairs_generated, x, y); 
    
    add_chair_agent(chair_agents, agent->chairs_generated, x, y, 0);

    agent->chairs_generated++;
    return 0;
}