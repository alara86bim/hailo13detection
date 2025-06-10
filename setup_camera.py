#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import yaml
import subprocess
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("camera_setup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Camera_Setup")

# Configuración predeterminada para ArduCam 64MP Ultra High
DEFAULT_ARDUCAM_CONFIG = {
    "camera_type": "ArduCam 64MP Ultra High",
    "sensor_id": 0,
    "resolution": {
        "width": 9152,
        "height": 6944,
        "downscaled_width": 1920,
        "downscaled_height": 1080
    },
    "fps": 30,
    "exposure": {
        "mode": "auto",
        "value": 100,
        "min": 1,
        "max": 10000
    },
    "white_balance": "auto",
    "focus": {
        "mode": "auto",
        "value": 0
    },
    "gain": {
        "mode": "auto",
        "value": 1.0
    },
    "format": "MJPEG",
    "i2c_address": "0x10"
}

def check_camera_connected():
    """Comprueba si la cámara está conectada."""
    logger.info("Comprobando conexión de la cámara...")
    
    try:
        # En Linux/Raspberry Pi podríamos comprobar con v4l2-ctl
        if sys.platform.startswith('linux'):
            result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True, 
                                   check=False)
            
            if "ArduCam" in result.stdout:
                logger.info("Cámara ArduCam detectada.")
                return True
            else:
                # Simplemente verificamos si hay alguna cámara
                devices = [line for line in result.stdout.split('\n') if '/dev/video' in line]
                if devices:
                    logger.info(f"Cámara detectada: {devices[0].strip()}")
                    return True
                else:
                    logger.warning("No se detectó ninguna cámara.")
                    return False
        else:
            # Para Windows u otros sistemas, simplemente asumimos que está conectada
            logger.info(f"Plataforma {sys.platform}: asumiendo que la cámara está conectada.")
            return True
            
    except Exception as e:
        logger.error(f"Error comprobando la cámara: {e}")
        logger.info("Asumiendo que la cámara está conectada.")
        return True

def create_camera_config(config_path):
    """Crea el archivo de configuración para la cámara ArduCam."""
    logger.info(f"Creando configuración de cámara en {config_path}")
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Guardar configuración
    with open(config_path, 'w') as f:
        json.dump(DEFAULT_ARDUCAM_CONFIG, f, indent=4)
    
    logger.info(f"Configuración de cámara guardada en {config_path}")

def main():
    """Función principal para configurar la cámara."""
    logger.info("Iniciando configuración de la cámara ArduCam 64MP")
    
    # Comprobar si la cámara está conectada
    if not check_camera_connected():
        logger.warning("No se detectó la cámara. Verifique la conexión y vuelva a intentarlo.")
        logger.warning("Continuando con la configuración de todos modos...")
    
    # Cargar configuración del proyecto
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
            
        camera_config_path = config['camera']['arducam_config']
        logger.info(f"Usando ruta de configuración de cámara: {camera_config_path}")
        
    except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
        logger.error(f"Error cargando configuración: {e}")
        logger.info("Usando configuración predeterminada.")
        camera_config_path = "camera_config/arducam_64mp.json"
    
    # Crear configuración de cámara
    create_camera_config(camera_config_path)
    
    # En un entorno real, aquí iría el código para inicializar la cámara
    # con la biblioteca específica de ArduCam
    logger.info("Nota: En un entorno de producción, aquí se inicializaría la cámara con la API de ArduCam.")
    logger.info("Para ello, se necesitaría instalar y configurar los controladores específicos.")
    
    logger.info("Configuración de cámara completada.")
    logger.info("Para personalizar parámetros, edite el archivo de configuración en: " + camera_config_path)

if __name__ == "__main__":
    main() 