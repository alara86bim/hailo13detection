#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import sys
import subprocess

def print_header(title):
    """Imprime un título con formato."""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60 + "\n")

def toggle_simulation_mode():
    """Cambia entre modo simulación y modo real en el archivo de configuración."""
    config_file = "config.yaml"
    
    # Verificar si existe el archivo de configuración
    if not os.path.exists(config_file):
        # Si no existe, crear uno a partir del ejemplo
        example_file = "config.yaml.example"
        if os.path.exists(example_file):
            print(f"No se encontró {config_file}, creando a partir de {example_file}")
            with open(example_file, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            print("No se encontró archivo de configuración ni ejemplo.")
            print("Creando configuración básica...")
            config_data = {
                "hailo": {
                    "device_id": 0,
                    "power_mode": "PERFORMANCE",
                    "force_simulation_mode": True
                }
            }
    else:
        # Cargar configuración existente
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
    
    # Asegurarse de que existe la sección hailo
    if "hailo" not in config_data:
        config_data["hailo"] = {}
    
    # Verificar modo actual
    current_mode = config_data["hailo"].get("force_simulation_mode", False)
    
    # Cambiar modo
    new_mode = not current_mode
    config_data["hailo"]["force_simulation_mode"] = new_mode
    
    # Guardar configuración
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    # Mostrar información
    mode_str = "SIMULACIÓN" if new_mode else "REAL (hardware Hailo)"
    print_header(f"MODO {mode_str} ACTIVADO")
    
    if new_mode:
        print("El sistema ahora funcionará en modo simulación.")
        print("No se intentará conectar con el hardware Hailo.")
    else:
        print("El sistema ahora intentará usar el hardware Hailo.")
        print("Si el hardware no está disponible, puede fallar.")
    
    print("\nConfiguración guardada en:", config_file)
    return new_mode

def show_status():
    """Muestra el estado actual del modo simulación."""
    config_file = "config.yaml"
    
    # Verificar si existe el archivo de configuración
    if not os.path.exists(config_file):
        print("No se encontró archivo de configuración.")
        return
    
    # Cargar configuración
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Verificar modo actual
    current_mode = config_data.get("hailo", {}).get("force_simulation_mode", False)
    
    # Mostrar información
    mode_str = "SIMULACIÓN" if current_mode else "REAL (hardware Hailo)"
    print_header(f"ESTADO ACTUAL: MODO {mode_str}")
    
    # Verificar si hay un dispositivo Hailo conectado
    print("Verificando dispositivo Hailo...")
    try:
        result = subprocess.run(
            ["ls", "-la", "/dev/hailo*"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0 and result.stdout:
            print("\n✅ Dispositivo Hailo detectado:")
            print(result.stdout)
        else:
            print("\n❌ No se detectó ningún dispositivo Hailo.")
    except:
        print("\n❌ No se pudo verificar el dispositivo Hailo.")

def main():
    """Función principal."""
    print_header("CAMBIAR MODO DE OPERACIÓN")
    
    print("Este script permite cambiar entre modo simulación y modo real.")
    print("En modo simulación, el sistema no intenta conectar con el hardware Hailo.")
    print("En modo real, el sistema intenta usar el hardware Hailo si está disponible.")
    
    # Procesar argumentos
    if len(sys.argv) > 1:
        if sys.argv[1] == "--status":
            show_status()
            return
        elif sys.argv[1] == "--simulation":
            if not toggle_simulation_mode():
                # Si estaba en modo real y queremos simulación, activar simulación
                toggle_simulation_mode()
            return
        elif sys.argv[1] == "--real":
            if toggle_simulation_mode():
                # Si estaba en modo simulación y queremos real, desactivar simulación
                toggle_simulation_mode()
            return
        elif sys.argv[1] == "--help":
            print("\nOpciones disponibles:")
            print("  --status      - Mostrar estado actual")
            print("  --simulation  - Activar modo simulación")
            print("  --real        - Activar modo real (hardware Hailo)")
            print("  --toggle      - Cambiar entre modos")
            print("  --help        - Mostrar esta ayuda")
            return
    
    # Por defecto, alternar modo
    toggle_simulation_mode()

if __name__ == "__main__":
    main() 