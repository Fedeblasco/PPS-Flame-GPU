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

__FLAME_GPU_FUNC__ int generate_personal(xmachine_memory_agent_generator* agent, xmachine_memory_agent_list* agent_agents){
    
    printf("Personal generado\n");
    float x = 0;
    float y = 0;
    if(agent->personal_generated<4){
        x = ((firstDoctor_x-1.0f)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
        y = ((firstDoctor_y-(space_between_doctors*agent->personal_generated))/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
    }else{ 
        if(agent->personal_generated<8){
            x = ((firstSpecialist_x + (space_between_specialists*(agent->personal_generated-4)))/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
            y = ((firstSpecialist_y-1.0f)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
        }else{ 
            if(agent->personal_generated==8){//Creación del recepcionista
                x = ((receptionist_x)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
                y = ((receptionist_y)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
            }
            if(agent->personal_generated==9){
                x = ((106)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
                y = ((77)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
            }
            if(agent->personal_generated==10){
                x = ((fifthSpecialist_x-1.0f)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
                y = ((fifthSpecialist_y)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
            }
        }
    }
    
    add_agent_agent(agent_agents, -(agent->personal_generated+1), x, y, 0.0f, 0.0f, 0.0f, 0.0f, 1, 0/*exit*/, 0, 1, 0, 1, 0, 0, 37, 0, 0, 0, 0, 0, 0, 0, 0);

    agent->personal_generated++;
    return 0;
}