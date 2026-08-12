#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:45:23 2026

@author: yor5
"""

# ==============================================================================
# CONFIGURACIÓN MAESTRA GLOBAL
# Todos los algoritmos leen de aquí para garantizar una comparación 100% justa
# ==============================================================================

TIMESTEPS_MAX = 2_000_000 
POBLACION_B = 4        

CONFIG_EXPERIMENTOS = {  
    
    "Hopper-v5": {        
        "semillas": [1042, 2854, 3910, 4721, 5603, 6198, 7433, 8256, 9107, 9845],                

    }, 
    
    # "HalfCheetah-v5": {        
    #     "semillas": [1042, 2854, 3910, 4721, 5603, 6198, 7433, 8256, 9107, 9845],                

    # }, 
    
    # "Swimmer-v5": {        
    #     "semillas": [1042, 2854, 3910, 4721, 5603, 6198, 7433, 8256, 9107, 9845],                

    # }, 
    
    # "Walker2d-v5": {        
    #     "semillas": [1042, 2854, 3910, 4721, 5603, 6198, 7433, 8256, 9107, 9845],                

    # }, 
    
    # "Ant-v5": {        
    #     "semillas": [1042, 2854, 3910, 4721, 5603, 6198, 7433, 8256, 9107, 9845],                

    # }, 
    
    # "Humanoid-v5": {        
    #     "semillas": [1042, 2854, 3910, 4721, 5603, 6198, 7433, 8256, 9107, 9845],                

    # }, 
    

    
    # "Pong-v5": {        
     
    #     "semillas": [7433],        
        
    #     "pb2_params": {
    #         "perturbation_interval": 50_000,            
    #         "quantile_fraction": 0.25,
    #     },

    # },       


}