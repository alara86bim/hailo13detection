#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import platform
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HailoCheck")

def print_section(title):
    """Imprime un título de sección en formato destacado."""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60 + "\n")

def check_system_info():
    """Verifica información del sistema operativo y kernel."""
    print_section("INFORMACIÓN DEL SISTEMA")
    
    # Sistema operativo
    os_info = platform.platform()
    print(f"Sistema Operativo: {os_info}")
    
    # Kernel
    kernel_version = platform.release()
    print(f"Versión del Kernel: {kernel_version}")
    
    # Comprobar si es una versión de kernel problemática
    if kernel_version.startswith("6.12"):
        print("\n⚠️  ADVERTENCIA: Kernel 6.12 detectado")
        print("   Esta versión puede tener problemas de compatibilidad con los controladores Hailo.")
        print("   Si experimenta problemas, considere usar una versión de kernel diferente o")
        print("   actualizar los controladores de Hailo.\n")

def check_hailo_sdk():
    """Verifica si el SDK de Hailo está instalado correctamente."""
    print_section("VERIFICACIÓN DEL SDK DE HAILO")
    
    try:
        import hailo
        print("✅ Módulo 'hailo' encontrado e importado correctamente.")
        
        # Verificar versión
        if hasattr(hailo, "__version__"):
            print(f"Versión del SDK de Hailo: {hailo.__version__}")
        
        # Verificar disponibilidad de clases principales
        if hasattr(hailo, "Device"):
            print("✅ Clase 'Device' disponible.")
        else:
            print("❌ Clase 'Device' no encontrada. La instalación puede estar incompleta.")
        
        # Verificar otras clases importantes
        for cls_name in ["InferVStreams", "ConfigureParams", "InputVStreamParams", "OutputVStreamParams"]:
            if hasattr(hailo, cls_name):
                print(f"✅ Clase '{cls_name}' disponible.")
            else:
                print(f"❌ Clase '{cls_name}' no encontrada.")
    
    except ImportError:
        print("❌ No se pudo importar el módulo 'hailo'.")
        print("\nSolución posible:")
        print("1. Instale el SDK de Hailo siguiendo las instrucciones en:")
        print("   https://hailo.ai/developer-zone/")
        print("2. O active el entorno virtual donde está instalado:")
        print("   source /path/to/venv/bin/activate")

def check_hailo_device():
    """Intenta detectar y conectarse a un dispositivo Hailo."""
    print_section("DETECCIÓN DE DISPOSITIVO HAILO")
    
    # Verificar si hay dispositivos Hailo reconocidos por el sistema
    try:
        hailo_devices = subprocess.run(
            ["ls", "-la", "/dev/hailo*"], 
            capture_output=True, 
            text=True
        )
        
        if hailo_devices.returncode == 0 and hailo_devices.stdout:
            print("Dispositivos Hailo detectados:")
            print(hailo_devices.stdout)
        else:
            print("❌ No se encontraron dispositivos Hailo en /dev/")
            print("\nPosibles causas:")
            print("1. El dispositivo no está conectado físicamente")
            print("2. Los controladores no están instalados correctamente")
            print("3. Los módulos del kernel no están cargados")
            
            # Comprobar módulos del kernel
            print("\nVerificando módulos del kernel para Hailo:")
            hailo_modules = subprocess.run(
                ["lsmod | grep hailo"], 
                shell=True,
                capture_output=True, 
                text=True
            )
            
            if hailo_modules.stdout:
                print("Módulos Hailo cargados:")
                print(hailo_modules.stdout)
            else:
                print("❌ No se encontraron módulos Hailo cargados.")
                print("\nIntente cargar los módulos con:")
                print("   sudo modprobe hailo_pci")
                print("O reinstale los controladores:")
                print("   sudo apt-get install --reinstall hailo-driver")
    
    except Exception as e:
        print(f"Error verificando dispositivos: {e}")
    
    # Intentar conectarse a un dispositivo Hailo
    try:
        import hailo
        print("\nIntentando conectar con dispositivo Hailo...")
        
        device = hailo.Device()
        device_info = device.control.get_info()
        
        print("✅ Conexión exitosa al dispositivo Hailo")
        print(f"  - Descripción: {device_info.description}")
        print(f"  - ID: {device_info.id}")
        print(f"  - Número de serie: {device_info.serial_number}")
        print(f"  - Versión de firmware: {device_info.firmware_version}")
        
        # Liberar el dispositivo
        del device
        
    except ImportError:
        print("❌ No se pudo importar el módulo 'hailo' para probar la conexión.")
    except Exception as e:
        print(f"❌ Error al conectar con el dispositivo: {e}")
        
        # Verificar permisos
        if "Permission denied" in str(e):
            print("\nError de permisos. Intente:")
            print("   sudo chmod 777 /dev/hailo*")
            print("O ejecute este script como root")

def check_hailo_models():
    """Verifica los modelos Hailo disponibles."""
    print_section("VERIFICACIÓN DE MODELOS HAILO")
    
    # Verificar carpeta models en el directorio actual
    if os.path.exists("models"):
        print("✅ Carpeta 'models' encontrada.")
        
        # Buscar archivos .hef
        hef_files = []
        for root, dirs, files in os.walk("models"):
            for file in files:
                if file.endswith(".hef"):
                    hef_files.append(os.path.join(root, file))
        
        if hef_files:
            print(f"Se encontraron {len(hef_files)} modelos Hailo (.hef):")
            for hef_file in hef_files:
                print(f"  - {hef_file}")
        else:
            print("❌ No se encontraron archivos de modelo Hailo (.hef).")
            print("   Debe descargar y colocar modelos .hef en la carpeta 'models'.")
    else:
        print("❌ Carpeta 'models' no encontrada en el directorio actual.")
        print("   Cree una carpeta 'models' y coloque sus archivos .hef allí.")

def suggest_fixes():
    """Sugiere soluciones para problemas comunes con Hailo."""
    print_section("SOLUCIONES RECOMENDADAS")
    
    print("Si tiene problemas con Hailo en el kernel 6.12, considere estas opciones:")
    print("\n1. USAR MODO SIMULACIÓN")
    print("   Edite config.yaml y añada lo siguiente en la sección 'hailo':")
    print("     force_simulation_mode: true")
    print("\n2. INSTALAR CONTROLADORES COMPATIBLES")
    print("   Descargue los controladores específicos para su kernel desde:")
    print("   https://hailo.ai/developer-zone/documentation/hailort/latest/")
    print("\n3. CAMBIAR A UN KERNEL COMPATIBLE")
    print("   Considere usar una versión anterior del kernel (5.15 o 6.1)")
    print("   que se sabe que funciona bien con Hailo.")
    print("\n4. VERIFICAR CONEXIÓN FÍSICA")
    print("   Asegúrese de que el dispositivo Hailo esté correctamente conectado")
    print("   y reciba suficiente energía.")

def main():
    """Función principal."""
    print("\n" + "#" * 70)
    print("#  DIAGNÓSTICO DE HAILO AI ACCELERATOR  #".center(70, "#"))
    print("#" * 70 + "\n")
    
    check_system_info()
    check_hailo_sdk()
    check_hailo_device()
    check_hailo_models()
    suggest_fixes()
    
    print("\n" + "#" * 70)
    print("#  DIAGNÓSTICO COMPLETADO  #".center(70, "#"))
    print("#" * 70 + "\n")

if __name__ == "__main__":
    main() 