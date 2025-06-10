# Guía de Conversión de Modelos para Hailo 13

Esta guía explica cómo preparar y convertir modelos de detección para su uso con el acelerador Hailo 13 en Raspberry Pi 5.

## Proceso General

1. **Entrenamiento del modelo** → **Exportación a ONNX** → **Conversión a formato Hailo (.hef)**

## 1. Entrenamiento con Datasets Personalizados

### Preparación del Dataset

1. Recolecta imágenes de tu obra de construcción
   - Asegúrate de capturar diferentes ángulos, condiciones de luz, etc.
   - Incluye imágenes de vehículos entrando/saliendo y personas en zonas de riesgo

2. Etiquetado de imágenes
   - Usa [Roboflow](https://roboflow.com/) (recomendado) o [LabelImg](https://github.com/tzutalin/labelImg)
   - Etiqueta las siguientes clases:
     - Vehículos (camiones, excavadoras, etc.)
     - Personas
     - Situaciones de riesgo (opcional)

3. Exporta el dataset en formato YOLO
   - Estructura de carpetas:
     ```
     dataset/
     ├── images/
     │   ├── train/
     │   ├── val/
     │   └── test/ (opcional)
     └── labels/
         ├── train/
         ├── val/
         └── test/ (opcional)
     ```

## 2. Entrenamiento del Modelo

Usa nuestro script `train_custom_model.py` para entrenar y convertir automáticamente:

```bash
python train_custom_model.py --dataset path/to/dataset --model yolov5s --type vehicle --epochs 100
```

### Parámetros:
- `--dataset`: Ruta al dataset de entrenamiento
- `--model`: Tipo de modelo (yolov5s, yolov5m, yolov5l, yolov8s, etc.)
- `--type`: Tipo de detección (vehicle o person)
- `--epochs`: Número de épocas para entrenar
- `--batch`: Tamaño del batch (default: 16)

## 3. Conversión Manual a Formato Hailo

Si prefieres realizar la conversión manualmente:

1. **Instala las herramientas de Hailo**
   - Descarga HailoRT SDK y Hailo Model Tools desde [Hailo Developer Zone](https://hailo.ai/developer-zone/)
   - Sigue las instrucciones de instalación oficiales

2. **Exporta tu modelo a ONNX**
   ```bash
   # Para YOLOv5
   python yolov5/export.py --weights runs/train/exp/weights/best.pt --include onnx --simplify
   
   # Para YOLOv8
   yolo export model=runs/detect/train/weights/best.pt format=onnx
   ```

3. **Convierte ONNX a formato Hailo**
   ```bash
   # Comando genérico (ajustar según tu instalación)
   hailo_model_compiler --onnx model.onnx --output-model model.hef --target-device hailo13
   ```

4. **Optimización para Hailo 13**
   - Usa Hailo Dataflow Compiler (si está disponible) para optimizar el rendimiento
   - Ajusta parámetros de cuantización según necesidad

## 4. Integración con la Aplicación

1. Coloca el archivo `.hef` en la carpeta `models/`
2. Actualiza el archivo `config.yaml` con la ruta al nuevo modelo
3. Ejecuta la aplicación:
   ```bash
   python main.py
   ```

## Recursos Adicionales

- [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) - Modelos pre-entrenados
- [Hailo Application Examples](https://github.com/hailo-ai/hailo_model_zoo/tree/master/docs/PUBLIC_MODELS.md) - Ejemplos de aplicaciones
- [Hailo Documentation](https://hailo.ai/developer-zone/documentation/) - Documentación oficial

## Notas Importantes

- La velocidad de inferencia depende del tamaño y complejidad del modelo
- YOLOv5s y YOLOv8s ofrecen buen equilibrio entre precisión y velocidad
- Para detección en tiempo real, prioriza modelos pequeños y rápidos
- Asegúrate de que las herramientas de Hailo estén actualizadas a la última versión 