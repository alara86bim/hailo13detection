# Sistema de Detección para Obra de Construcción

Sistema de monitoreo en tiempo real para detectar:
- Vehículos entrando/saliendo de la obra
- Comportamiento humano durante descarga de materiales
- Prevención de accidentes

## Requisitos de Hardware
- Raspberry Pi 5
- Hailo-13 AI Accelerator HAT
- Cámara ArduCam 64MP Ultra High

## Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Descargar los modelos pre-entrenados:
```bash
python download_models.py
```

3. Configurar la cámara:
```bash
python setup_camera.py
```

## Ejecución
```bash
python main.py
```

## Configuración
Editar el archivo `config.yaml` para ajustar parámetros de detección. 