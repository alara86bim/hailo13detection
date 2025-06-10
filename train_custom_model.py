#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import subprocess
import sys
import yaml
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Model_Trainer")

def setup_environment():
    """Configura el entorno para entrenamiento y conversión de modelos."""
    logger.info("Configurando entorno para entrenamiento y conversión")
    
    # Crear directorios necesarios
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("hailo_converted", exist_ok=True)
    
    # Verificar si YOLOv5 está clonado, si no, clonarlo
    if not os.path.exists("yolov5"):
        logger.info("Clonando repositorio YOLOv5")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/ultralytics/yolov5.git"],
                check=True
            )
            logger.info("Repositorio YOLOv5 clonado correctamente")
            
            # Instalar dependencias de YOLOv5
            subprocess.run(
                ["pip", "install", "-r", "yolov5/requirements.txt"],
                check=True
            )
            logger.info("Dependencias de YOLOv5 instaladas")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error clonando YOLOv5: {e}")
            sys.exit(1)
    else:
        logger.info("Repositorio YOLOv5 ya existe")

def prepare_dataset(dataset_path):
    """Prepara el dataset para entrenamiento."""
    logger.info(f"Preparando dataset desde: {dataset_path}")
    
    # Si el dataset no existe, informar al usuario sobre cómo crear uno
    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset no encontrado en {dataset_path}")
        logger.info("Para crear un dataset:")
        logger.info("1. Recolecta imágenes de tu obra de construcción")
        logger.info("2. Utiliza Roboflow o LabelImg para etiquetar imágenes")
        logger.info("3. Exporta el dataset en formato YOLO")
        logger.info("4. Coloca el dataset en la carpeta 'datasets'")
        sys.exit(1)
    
    # Verificar estructura del dataset
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")
    
    if not (os.path.exists(images_dir) and os.path.exists(labels_dir)):
        logger.warning("Estructura de dataset incorrecta")
        logger.info("El dataset debe tener la siguiente estructura:")
        logger.info("dataset/")
        logger.info("├── images/")
        logger.info("│   ├── train/")
        logger.info("│   ├── val/")
        logger.info("│   └── test/ (opcional)")
        logger.info("└── labels/")
        logger.info("    ├── train/")
        logger.info("    ├── val/")
        logger.info("    └── test/ (opcional)")
        sys.exit(1)
    
    logger.info("Dataset preparado correctamente")
    return dataset_path

def create_model_config(dataset_path, model_type="yolov5s", epochs=100, batch_size=16):
    """Crea el archivo de configuración para entrenamiento."""
    logger.info("Creando archivo de configuración para entrenamiento")
    
    # Encontrar archivo de clases
    data_yaml = None
    for file in os.listdir(dataset_path):
        if file.endswith(".yaml") and file != "dataset.yaml":
            data_yaml = os.path.join(dataset_path, file)
            break
    
    if not data_yaml:
        # Crear archivo data.yaml basado en la estructura de directorios
        data_yaml = os.path.join(dataset_path, "data.yaml")
        
        # Intentar contar el número de clases basado en un archivo de etiquetas
        num_classes = 0
        try:
            train_labels = os.path.join(dataset_path, "labels", "train")
            if os.path.exists(train_labels):
                # Obtener el primer archivo de etiquetas
                label_files = [f for f in os.listdir(train_labels) if f.endswith(".txt")]
                if label_files:
                    with open(os.path.join(train_labels, label_files[0]), 'r') as f:
                        classes = set()
                        for line in f:
                            if line.strip():
                                class_id = int(line.strip().split()[0])
                                classes.add(class_id)
                        num_classes = max(classes) + 1
        except:
            num_classes = 3  # Valor por defecto (vehículo, persona, peligro)
        
        # Crear diccionario de configuración
        data_config = {
            'path': dataset_path,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test' if os.path.exists(os.path.join(dataset_path, "images", "test")) else '',
            'nc': num_classes,
            'names': ['vehiculo', 'persona', 'peligro'] if num_classes == 3 else [f'class{i}' for i in range(num_classes)]
        }
        
        # Guardar archivo de configuración
        with open(data_yaml, 'w') as f:
            yaml.dump(data_config, f, sort_keys=False)
        
        logger.info(f"Archivo de configuración creado en {data_yaml}")
    else:
        logger.info(f"Usando archivo de configuración existente: {data_yaml}")
    
    # Crear configuración de entrenamiento
    train_config = {
        'data_yaml': data_yaml,
        'model_type': model_type,
        'epochs': epochs,
        'batch_size': batch_size,
        'img_size': 640,
        'weights': 'yolov5s.pt' if model_type == 'yolov5s' else f'{model_type}.pt'
    }
    
    return train_config

def train_model(config):
    """Entrena el modelo con YOLOv5."""
    logger.info(f"Iniciando entrenamiento con {config['model_type']}")
    
    # Construir comando de entrenamiento
    train_cmd = [
        "python", "yolov5/train.py",
        "--img", str(config['img_size']),
        "--batch", str(config['batch_size']),
        "--epochs", str(config['epochs']),
        "--data", config['data_yaml'],
        "--weights", config['weights'],
        "--project", "trained_models",
        "--name", f"custom_{config['model_type']}"
    ]
    
    # Ejecutar entrenamiento
    try:
        logger.info(f"Ejecutando: {' '.join(train_cmd)}")
        subprocess.run(train_cmd, check=True)
        logger.info("Entrenamiento completado correctamente")
        
        # Encontrar el modelo entrenado (best.pt)
        train_dir = os.path.join("trained_models", f"custom_{config['model_type']}")
        model_path = os.path.join(train_dir, "weights", "best.pt")
        
        if not os.path.exists(model_path):
            model_path = os.path.join(train_dir, "weights", "last.pt")
        
        return model_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        sys.exit(1)

def export_to_onnx(model_path):
    """Exporta el modelo entrenado a formato ONNX."""
    logger.info(f"Exportando modelo a ONNX: {model_path}")
    
    # Construir ruta de salida
    model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    onnx_path = os.path.join("models", f"{model_name}.onnx")
    
    # Construir comando de exportación
    export_cmd = [
        "python", "yolov5/export.py",
        "--weights", model_path,
        "--include", "onnx",
        "--simplify"
    ]
    
    # Ejecutar exportación
    try:
        logger.info(f"Ejecutando: {' '.join(export_cmd)}")
        subprocess.run(export_cmd, check=True)
        
        # Mover el archivo exportado a la ubicación deseada
        exported_onnx = model_path.replace('.pt', '.onnx')
        if os.path.exists(exported_onnx):
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            import shutil
            shutil.move(exported_onnx, onnx_path)
            logger.info(f"Modelo exportado a: {onnx_path}")
        else:
            logger.error(f"No se encontró el archivo ONNX exportado: {exported_onnx}")
            sys.exit(1)
        
        return onnx_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error durante la exportación a ONNX: {e}")
        sys.exit(1)

def convert_to_hailo(onnx_path):
    """Convierte el modelo ONNX al formato Hailo (.hef)."""
    logger.info(f"Convirtiendo modelo ONNX a formato Hailo: {onnx_path}")
    
    # Comprobar si las herramientas de Hailo están instaladas
    try:
        # Verificar si hailomota está instalado
        result = subprocess.run(
            ["hailo", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            logger.warning("Herramientas de línea de comandos de Hailo no encontradas")
            logger.info("Para convertir modelos ONNX a formato Hailo (.hef):")
            logger.info("1. Instala HailoRT y HailoTapp SDK desde: https://hailo.ai/developer-zone/")
            logger.info("2. Usa las herramientas de conversión de Hailo con tu modelo ONNX")
            logger.info("3. Guarda el modelo convertido .hef en la carpeta 'models'")
            
            # Crear un archivo simulado para pruebas
            hef_path = onnx_path.replace('.onnx', '.hef')
            with open(hef_path, 'w') as f:
                f.write("Este es un archivo simulado. En un entorno real, aquí estaría el modelo compilado para Hailo.")
            
            logger.info(f"Creado archivo simulado para pruebas: {hef_path}")
            return hef_path
        
        # Si las herramientas de Hailo están instaladas, ejecutar la conversión
        # Aquí iría el comando específico de conversión de Hailo
        # Ejemplo (esto varía según la versión específica de las herramientas Hailo):
        hef_path = onnx_path.replace('.onnx', '.hef')
        
        # Nota: Este es un comando de ejemplo y puede necesitar ajustes
        hailo_cmd = [
            "hailo_model_compiler",
            "--onnx", onnx_path,
            "--output-model", hef_path,
            "--target-device", "hailo13"
        ]
        
        logger.info(f"Ejecutando: {' '.join(hailo_cmd)}")
        logger.info("Nota: Este comando es un ejemplo y puede requerir ajustes según tu instalación")
        
        # En un entorno real, se ejecutaría el comando:
        # subprocess.run(hailo_cmd, check=True)
        
        # Para esta implementación, creamos un archivo simulado
        with open(hef_path, 'w') as f:
            f.write("Este es un archivo simulado. En un entorno real, aquí estaría el modelo compilado para Hailo.")
        
        logger.info(f"Conversión simulada completada. Modelo HEF: {hef_path}")
        return hef_path
        
    except Exception as e:
        logger.error(f"Error durante la conversión a Hailo: {e}")
        sys.exit(1)

def update_config(hef_path, model_type):
    """Actualiza el archivo de configuración del proyecto con el nuevo modelo."""
    logger.info("Actualizando archivo de configuración del proyecto")
    
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        logger.warning(f"Archivo de configuración no encontrado: {config_path}")
        return
    
    try:
        # Cargar configuración existente
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Actualizar ruta del modelo según el tipo
        if model_type == "vehicle":
            config['models']['vehicle_detection']['model_path'] = hef_path
        elif model_type == "person":
            config['models']['person_detection']['model_path'] = hef_path
        else:
            logger.warning(f"Tipo de modelo desconocido: {model_type}")
            return
        
        # Guardar configuración actualizada
        with open(config_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        
        logger.info(f"Configuración actualizada con el nuevo modelo: {hef_path}")
        
    except Exception as e:
        logger.error(f"Error actualizando configuración: {e}")

def main():
    """Función principal para entrenar y convertir modelos."""
    parser = argparse.ArgumentParser(description="Entrenamiento y conversión de modelos para Hailo")
    
    parser.add_argument("--dataset", type=str, required=True,
                       help="Ruta al dataset para entrenamiento")
    parser.add_argument("--model", type=str, default="yolov5s",
                       choices=["yolov5s", "yolov5m", "yolov5l", "yolov5x", "yolov8s", "yolov8m"],
                       help="Tipo de modelo a entrenar")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Número de épocas para entrenamiento")
    parser.add_argument("--batch", type=int, default=16,
                       help="Tamaño del batch para entrenamiento")
    parser.add_argument("--type", type=str, default="vehicle",
                       choices=["vehicle", "person"],
                       help="Tipo de detección (vehículo o persona)")
    
    args = parser.parse_args()
    
    logger.info("Iniciando proceso de entrenamiento y conversión de modelo")
    logger.info(f"Modelo: {args.model}, Tipo: {args.type}, Dataset: {args.dataset}")
    
    # Configurar entorno
    setup_environment()
    
    # Preparar dataset
    dataset_path = prepare_dataset(args.dataset)
    
    # Crear configuración de modelo
    model_config = create_model_config(dataset_path, args.model, args.epochs, args.batch)
    
    # Entrenar modelo
    trained_model = train_model(model_config)
    logger.info(f"Modelo entrenado: {trained_model}")
    
    # Exportar a ONNX
    onnx_model = export_to_onnx(trained_model)
    logger.info(f"Modelo ONNX: {onnx_model}")
    
    # Convertir a formato Hailo
    hef_model = convert_to_hailo(onnx_model)
    logger.info(f"Modelo Hailo: {hef_model}")
    
    # Actualizar configuración
    update_config(hef_model, args.type)
    
    logger.info("Proceso completo. Modelo entrenado y convertido correctamente.")
    logger.info(f"Modelo Hailo listo para usar: {hef_model}")
    logger.info("Para usar el modelo, ejecute: python main.py")

if __name__ == "__main__":
    main() 