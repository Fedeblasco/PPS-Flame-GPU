//Manejo de las sillas

__FLAME_GPU_FUNC__ int generate_chairs(xmachine_memory_agent_generator* agent, xmachine_memory_chair_list* chair_agents){
    
    int x = firstChair_x + (space_between * ((agent->chairs_generated+7)%7));
    int y = firstChair_y - (space_between * int(agent->chairs_generated/7));
    
    //printf("Silla %d, x:%d, y:%d\n", agent->chairs_generated, x, y); 
    
    add_chair_agent(chair_agents, agent->chairs_generated, x, y, 0);

    agent->chairs_generated++;
    return 0;
}

__FLAME_GPU_FUNC__ int generate_boxes(xmachine_memory_agent_generator* agent, xmachine_memory_box_list* box_agents){
    
    printf("Box generado\n"); 
    
    add_box_agent(box_agents, agent->boxes_generated,0,0);

    agent->boxes_generated++;
    return 0;
}

__FLAME_GPU_FUNC__ int generate_doctors(xmachine_memory_agent_generator* agent, xmachine_memory_doctor_list* doctor_agents){
    
    printf("Doctor generado\n"); 
    
    add_doctor_agent(doctor_agents, agent->doctors_generated,0,0);

    agent->doctors_generated++;
    return 0;
}

__FLAME_GPU_FUNC__ int generate_specialists(xmachine_memory_agent_generator* agent, xmachine_memory_specialist_list* specialist_agents){
    
    printf("Especialista generado\n"); 
    
    add_specialist_agent(specialist_agents, agent->specialists_generated+1,0,0);

    agent->specialists_generated++;
    return 0;
}

__FLAME_GPU_FUNC__ int generate_real_doctors(xmachine_memory_agent_generator* agent, xmachine_memory_agent_list* agent_agents){
    
    printf("Doctor real generado\n"); 
    
    float x = ((firstDoctor_x-1.0f)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
    float y = ((firstDoctor_y-(space_between_doctors*agent->real_doctors_generated))/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
    
    add_agent_agent(agent_agents, -(agent->real_doctors_generated+1), x, y, 0.0f, 0.0f, 0.0f, 0.0f, 1, 0/*exit*/, 0, 1, 0, 1, 0, 0, 37, 0, 0, 0, 0, 0, 0, 0, 0);

    agent->real_doctors_generated++;
    return 0;
}