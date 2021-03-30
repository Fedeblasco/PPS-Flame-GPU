//Manejo de las sillas

__FLAME_GPU_FUNC__ int generate_chairs(xmachine_memory_agent_generator* agent, xmachine_memory_chair_list* chair_agents){
    
    int x = FIRSTCHAIR_X + (SPACE_BETWEEN * ((agent->chairs_generated+7)%7));
    int y = FIRSTCHAIR_Y - (SPACE_BETWEEN * int(agent->chairs_generated/7));
    
    //printf("Silla %d, x:%d, y:%d\n", agent->chairs_generated, x, y); 
    
    add_chair_agent(chair_agents, agent->chairs_generated, 0, x, y, 0);

    agent->chairs_generated++;
    return 0;
}

__FLAME_GPU_FUNC__ int generate_boxes(xmachine_memory_agent_generator* agent, xmachine_memory_box_list* box_agents){
    
    printf("Box generado\n"); 
    
    add_box_agent(box_agents, agent->boxes_generated,0,0);

    agent->boxes_generated++;
    return 0;
}

__FLAME_GPU_FUNC__ int generate_beds(xmachine_memory_agent_generator* agent, xmachine_memory_bed_list* bed_agents){
    
    printf("Cama generada\n"); 
    
    add_bed_agent(bed_agents, agent->beds_generated,0,0);

    agent->beds_generated++;
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

__FLAME_GPU_FUNC__ int generate_personal(xmachine_memory_agent_generator* agent, xmachine_memory_agent_list* agent_agents, RNG_rand48* rand48){
    
    printf("Personal generado\n");
    float x = 0;
    float y = 0;
    if(agent->personal_generated<4){//Creación de doctores
        x = ((FIRSTDOCTOR_X-1.0f)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
        y = ((FIRSTDOCTOR_Y-(SPACE_BETWEEN_DOCTORS*agent->personal_generated))/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
    }else{ 
        if(agent->personal_generated<8){//Creación de especialistas
            x = ((FIRSTSPECIALIST_X + (SPACE_BETWEEN_SPECIALISTS*(agent->personal_generated-4)))/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
            y = ((FIRSTSPECIALIST_Y-1.0f)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
        }else{ 
            if(agent->personal_generated==8){//Creación del recepcionista
                x = ((RECEPTIONIST_X)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
                y = ((RECEPTIONIST_Y)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
            }
            if(agent->personal_generated==9){//Creación del triage
                x = ((TRIAGE_X)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
                y = ((TRIAGE_Y)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
            }
            if(agent->personal_generated==10){//Creación del quinto especialista
                x = ((FIFTHSPECIALIST_X-1.0f)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
                y = ((FIFTHSPECIALIST_Y)/(d_message_navmap_cell_width/ENV_WIDTH))-ENV_MAX;
            }
        }
    }
    int vaccine = 0;
    float random = rnd<CONTINUOUS>(rand48);
    if(random <= PROB_VACCINE_STAFF){
        vaccine = 1;
    }
    add_agent_agent(agent_agents, -(agent->personal_generated+1), x, y, 0.0f, 0.0f, 0.0f, 0.0f, 1, 0/*exit*/, 0, 1, 0, 1, 0, 0, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, vaccine);

    agent->personal_generated++;
    return 0;
}