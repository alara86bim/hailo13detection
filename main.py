#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import cv2
import numpy as np
import yaml
import logging
import subprocess
import threading
import queue
from datetime import datetime
from pathlib import Path
import json
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Construction_Monitor")

# Mapeo de IDs de clase a nombres descriptivos de vehículos
VEHICLE_CLASSES = {
    1: "automóvil",
    2: "camión",
    3: "autobús",
    5: "camioneta",
    7: "camión grande"
}

class ConstructionMonitor:
    def __init__(self, config_path="config.yaml"):
        """Inicializa el monitor de construcción con la configuración especificada."""
        logger.info("Inicializando sistema de monitoreo de construcción")
        
        # Cargar configuración
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Crear directorios si no existen
        os.makedirs(self.config['alerts']['events_folder'], exist_ok=True)
        os.makedirs("camera_config", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Crear directorio para eventos por tipo
        os.makedirs(os.path.join(self.config['alerts']['events_folder'], "vehiculos_entrada"), exist_ok=True)
        os.makedirs(os.path.join(self.config['alerts']['events_folder'], "vehiculos_salida"), exist_ok=True)
        os.makedirs(os.path.join(self.config['alerts']['events_folder'], "situaciones_riesgo"), exist_ok=True)
        
        # Variables para el manejo de frames
        self.frame_queue = queue.Queue(maxsize=10)
        self.camera_process = None
        self.stop_capture = False
        
        # Inicializar cámara
        self.init_camera()
        
        # Inicializar Hailo
        self.init_hailo()
        
        # Inicializar zonas
        self.init_zones()
        
        # Estado de detección
        self.detected_vehicles = {}  # Para seguimiento de vehículos
        self.detected_persons = {}   # Para seguimiento de personas
        self.alerts_active = {}      # Alertas activas
        self.vehicle_history = {}    # Historial de vehículos para determinar dirección
        self.event_counter = 0       # Contador de eventos para registro

        # Registro de eventos
        self.events_log_path = os.path.join(self.config['alerts']['events_folder'], "events_log.json")
        self.events_log = self._load_events_log()
        
        logger.info("Sistema inicializado correctamente")
    
    def _load_events_log(self):
        """Carga el registro de eventos o crea uno nuevo si no existe."""
        if os.path.exists(self.events_log_path):
            try:
                with open(self.events_log_path, 'r') as f:
                    return json.load(f)
            except:
                logger.warning("Error al cargar el registro de eventos. Creando nuevo registro.")
        
        return {
            "events": [],
            "statistics": {
                "vehiculos_entrada": 0,
                "vehiculos_salida": 0,
                "situaciones_riesgo": 0
            }
        }
    
    def _save_events_log(self):
        """Guarda el registro de eventos en disco."""
        with open(self.events_log_path, 'w') as f:
            json.dump(self.events_log, f, indent=4)
    
    def _camera_capture_thread(self):
        """Hilo para capturar frames desde libcamera-vid."""
        try:
            # Crear el comando para libcamera-vid
            width = self.config['camera']['width']
            height = self.config['camera']['height']
            fps = self.config['camera']['fps']
            
            cmd = [
                'libcamera-vid',
                '--width', str(width),
                '--height', str(height),
                '--framerate', str(fps),
                '--codec', 'yuv420',
                '--timeout', '0',  # Sin límite de tiempo
                '--output', '-'    # Salida a stdout
            ]
            
            logger.info(f"Iniciando captura con comando: {' '.join(cmd)}")
            
            # Iniciar el proceso
            self.camera_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10*1024*1024  # Buffer de 10MB
            )
            
            # Calcular tamaño del frame en bytes (formato YUV420)
            frame_size = width * height * 3 // 2  # YUV420 usa 1.5 bytes por pixel
            
            # Leer continuamente de la salida
            while not self.stop_capture:
                # Leer un frame completo
                raw_data = self.camera_process.stdout.read(frame_size)
                if len(raw_data) < frame_size:
                    logger.error("Error leyendo frame completo de libcamera-vid")
                    break
                
                # Convertir de YUV420 a BGR para OpenCV
                yuv = np.frombuffer(raw_data, dtype=np.uint8).reshape((height * 3 // 2, width))
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                
                # Poner en la cola, descartando frames antiguos si está llena
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                try:
                    self.frame_queue.put_nowait(bgr)
                except queue.Full:
                    pass
                
        except Exception as e:
            logger.error(f"Error en hilo de captura: {e}")
        finally:
            # Limpiar al terminar
            if self.camera_process:
                self.camera_process.terminate()
                self.camera_process = None
            logger.info("Hilo de captura de cámara terminado")
    
    def init_camera(self):
        """Inicializa la cámara usando libcamera-vid."""
        logger.info("Inicializando cámara con libcamera-vid")
        
        # Comprobar si se debe usar ArduCam con libcamera
        if self.config['camera']['use_arducam']:
            try:
                # Verificar que libcamera-vid esté disponible
                result = subprocess.run(['which', 'libcamera-vid'], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE, 
                                       text=True)
                
                if result.returncode != 0:
                    logger.error("libcamera-vid no encontrado. Verifica la instalación de libcamera")
                    logger.info("Cambiando a OpenCV estándar")
                    self._init_opencv_camera()
                    return
                
                # Iniciar el hilo de captura de cámara
                self.stop_capture = False
                self.capture_thread = threading.Thread(target=self._camera_capture_thread)
                self.capture_thread.daemon = True
                self.capture_thread.start()
                
                # Esperar un momento para que se inicialice
                time.sleep(2)
                
                # Verificar si el hilo está funcionando
                if not self.capture_thread.is_alive():
                    logger.error("El hilo de captura de cámara no se inició correctamente")
                    self._init_opencv_camera()
                else:
                    logger.info("Cámara iniciada correctamente con libcamera-vid")
                
            except Exception as e:
                logger.error(f"Error inicializando cámara con libcamera: {e}")
                logger.info("Cambiando a OpenCV estándar")
                self._init_opencv_camera()
        else:
            # Usar OpenCV estándar
            self._init_opencv_camera()
    
    def _init_opencv_camera(self):
        """Inicializa la cámara usando OpenCV (fallback)."""
        logger.info("Inicializando cámara con OpenCV")
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
        self.camera.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
        
        # Verificar que la cámara se haya inicializado correctamente
        if not self.camera.isOpened():
            logger.error("No se pudo inicializar la cámara con OpenCV")
            raise RuntimeError("Error inicializando cámara")
        
        logger.info("Cámara inicializada correctamente con OpenCV")
    
    def init_hailo(self):
        """Inicializa el acelerador Hailo AI y carga los modelos."""
        logger.info("Inicializando Hailo AI Accelerator")
        
        # Variable para indicar si estamos en modo simulación
        self.simulation_mode = False
        
        try:
            # Importar la biblioteca Hailo
            import hailo
            
            try:
                # Intentar inicializar dispositivo Hailo
                self.hailo_device = hailo.Device(self.config['hailo']['device_id'])
                
                # Configurar modo de energía
                power_mode = getattr(hailo.PowerMode, self.config['hailo']['power_mode'].upper())
                self.hailo_device.control.set_power_mode(power_mode)
                
                logger.info(f"Hailo inicializado: {self.hailo_device.control.get_info().description}")
            except Exception as e:
                logger.warning(f"No se pudo inicializar el dispositivo Hailo: {e}")
                logger.warning("Cambiando a modo de simulación")
                self.simulation_mode = True
            
        except ImportError:
            logger.warning("No se pudo importar el módulo Hailo. Cambiando a modo de simulación")
            self.simulation_mode = True
        except AttributeError:
            logger.warning("API de Hailo no disponible. Cambiando a modo de simulación")
            self.simulation_mode = True
        except Exception as e:
            logger.warning(f"Error inicializando Hailo: {e}. Cambiando a modo de simulación")
            self.simulation_mode = True
        
        # Cargar modelos de detección (reales o simulados)
        self.vehicle_model = self._load_hailo_model(self.config['models']['vehicle_detection'])
        self.person_model = self._load_hailo_model(self.config['models']['person_detection'])
        
        # Cargar modelo de pose si está habilitado
        if self.config['models'].get('pose_detection', {}).get('enable', False):
            try:
                self.pose_model = self._load_hailo_model(self.config['models']['pose_detection'])
                logger.info("Modelo de poses cargado correctamente")
            except Exception as e:
                logger.error(f"Error cargando modelo de poses: {e}")
                # Continuar sin modelo de poses
        
        if self.simulation_mode:
            logger.info("Sistema funcionando en MODO SIMULACIÓN - Detecciones simuladas")
        else:
            logger.info("Sistema funcionando con acelerador Hailo - Detecciones reales")
    
    def _load_hailo_model(self, model_config):
        """Carga un modelo en el dispositivo Hailo."""
        # Implementación simplificada - en la práctica necesitarías usar la API de Hailo
        # para cargar el modelo .hef y prepararlo para inferencia
        logger.info(f"Cargando modelo: {model_config['model_path']}")
        
        # Aquí iría el código real de carga del modelo Hailo
        # Por ahora, devolvemos un objeto dummy para simular
        class DummyModel:
            def __init__(self, config):
                self.config = config
                self.name = Path(config['model_path']).stem
            
            def infer(self, frame):
                # Simulación de detección - en la implementación real
                # aquí se usaría el dispositivo Hailo para inferencia
                return []
        
        return DummyModel(model_config)
    
    def init_zones(self):
        """Inicializa las zonas de monitoreo."""
        logger.info("Configurando zonas de monitoreo")
        
        # Zonas de entrada/salida
        self.entry_exit_zones = []
        for zone in self.config['zones']['entry_exit']:
            name = zone['name']
            points = np.array(zone['points'])
            self.entry_exit_zones.append({
                'name': name,
                'points': points,
                'polygon': np.array(points, dtype=np.int32),
                'direction': 'entrada' if 'entrada' in name.lower() else 'salida'
            })
        
        # Zonas de peligro
        self.danger_zones = []
        for zone in self.config['zones']['danger_zones']:
            name = zone['name']
            points = np.array(zone['points'])
            self.danger_zones.append({
                'name': name,
                'points': points,
                'polygon': np.array(points, dtype=np.int32)
            })
        
        logger.info(f"Configuradas {len(self.entry_exit_zones)} zonas de entrada/salida y {len(self.danger_zones)} zonas de peligro")
    
    def _get_frame(self):
        """Obtiene el frame actual de la cámara."""
        # Si estamos usando libcamera-vid
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            try:
                # Obtener el frame más reciente de la cola
                return True, self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                logger.warning("Timeout esperando frame de libcamera-vid")
                return False, None
        # Si estamos usando OpenCV
        elif hasattr(self, 'camera'):
            return self.camera.read()
        else:
            logger.error("No hay método de captura disponible")
            return False, None
    
    def process_frame(self, frame):
        """Procesa un frame para detección de vehículos y personas."""
        # Normalizar dimensiones del frame para zonas
        height, width = frame.shape[:2]
        
        # Detectar vehículos
        vehicle_detections = self._detect_objects(frame, self.vehicle_model)
        
        # Detectar personas
        person_detections = self._detect_objects(frame, self.person_model)
        
        # Detectar poses (si está habilitado)
        pose_detections = self._detect_poses(frame)
        
        # Verificar entradas/salidas de vehículos
        self._check_vehicle_entry_exit(frame, vehicle_detections, width, height)
        
        # Verificar comportamiento peligroso
        self._check_danger_situations(frame, vehicle_detections, person_detections, width, height)
        
        # Verificar posturas peligrosas
        if pose_detections:
            pose_alerts = self._analyze_pose_safety(pose_detections, vehicle_detections, width, height)
            self._handle_pose_alerts(frame, pose_alerts)
        
        # Dibujar zonas y detecciones en el frame
        self._draw_zones(frame, width, height)
        self._draw_detections(frame, vehicle_detections, person_detections)
        
        # Dibujar poses si están disponibles
        if pose_detections:
            self._draw_poses(frame, pose_detections)
        
        return frame
    
    def _detect_objects(self, frame, model):
        """Detecta objetos usando el modelo Hailo especificado."""
        # En una implementación real, aquí se haría la inferencia con Hailo
        # Para simular, devolvemos algunas detecciones de prueba
        
        # Simular algunas detecciones
        if model.name == 'yolov5s_vehicle':
            # Simular algunos vehículos (bbox, confianza, clase)
            return [
                ([100, 100, 200, 200], 0.85, 2),  # Un camión simulado
                ([400, 300, 600, 500], 0.92, 7)   # Un camión grande simulado
            ]
        elif model.name == 'yolov5s_person':
            # Simular algunas personas
            return [
                ([250, 250, 300, 400], 0.78, 0),  # Una persona simulada
                ([520, 380, 580, 480], 0.81, 0)   # Otra persona simulada
            ]
        elif model.name == 'posenet_resnet_v1_50':
            # Simular detección de poses de personas
            # Formato: (bbox, confianza, lista_keypoints)
            # Cada keypoint es (x, y, confianza)
            keypoints1 = [
                (270, 260, 0.9),  # nariz
                (275, 255, 0.8),  # ojo_izq
                (265, 255, 0.8),  # ojo_der
                (280, 260, 0.7),  # oreja_izq
                (260, 260, 0.7),  # oreja_der
                (290, 300, 0.9),  # hombro_izq
                (250, 300, 0.9),  # hombro_der
                (300, 350, 0.8),  # codo_izq
                (240, 350, 0.8),  # codo_der
                (310, 380, 0.7),  # muñeca_izq
                (230, 380, 0.7),  # muñeca_der
                (285, 370, 0.9),  # cadera_izq
                (265, 370, 0.9),  # cadera_der
                (290, 420, 0.8),  # rodilla_izq
                (260, 420, 0.8),  # rodilla_der
                (295, 450, 0.7),  # tobillo_izq
                (255, 450, 0.7)   # tobillo_der
            ]
            
            keypoints2 = [
                (550, 390, 0.9),  # nariz
                (555, 385, 0.8),  # ojo_izq
                (545, 385, 0.8),  # ojo_der
                (560, 390, 0.7),  # oreja_izq
                (540, 390, 0.7),  # oreja_der
                (560, 410, 0.9),  # hombro_izq - hombros más bajos (agachado)
                (540, 410, 0.9),  # hombro_der
                (570, 380, 0.8),  # codo_izq - brazos levantados
                (530, 380, 0.8),  # codo_der
                (580, 360, 0.7),  # muñeca_izq - manos arriba
                (520, 360, 0.7),  # muñeca_der
                (560, 430, 0.9),  # cadera_izq
                (540, 430, 0.9),  # cadera_der
                (565, 460, 0.8),  # rodilla_izq
                (535, 460, 0.8),  # rodilla_der
                (570, 480, 0.7),  # tobillo_izq
                (530, 480, 0.7)   # tobillo_der
            ]
            
            return [
                ([250, 250, 300, 450], 0.85, keypoints1),  # Persona en postura normal
                ([520, 380, 580, 480], 0.83, keypoints2)   # Persona con brazos elevados
            ]
        
        return []
    
    def _detect_poses(self, frame):
        """Detecta poses de personas usando PoseNet."""
        # Verificar si la detección de poses está habilitada
        if not self.config['models'].get('pose_detection', {}).get('enable', False):
            return []
        
        try:
            # Usar el modelo de poses ya cargado si existe
            if hasattr(self, 'pose_model'):
                return self._detect_objects(frame, self.pose_model)
            else:
                # Intentar cargar modelo de poses
                pose_model = self._load_hailo_model(self.config['models']['pose_detection'])
                return self._detect_objects(frame, pose_model)
        except Exception as e:
            logger.error(f"Error en detección de poses: {e}")
            return []
    
    def _analyze_pose_safety(self, pose_detections, vehicle_detections, width, height):
        """Analiza las posturas de las personas para detectar situaciones peligrosas."""
        if not pose_detections or not vehicle_detections:
            return []
        
        # Obtener configuración de posturas peligrosas
        dangerous_poses_config = self.config['alerts'].get('dangerous_poses', [])
        if not dangerous_poses_config:
            return []
        
        danger_alerts = []
        
        # Para cada persona detectada con pose
        for i, (person_bbox, confidence, keypoints) in enumerate(pose_detections):
            person_id = f"pose_{i}"
            
            p_x1, p_y1, p_x2, p_y2 = person_bbox
            person_center = ((p_x1 + p_x2) / 2, (p_y1 + p_y2) / 2)
            
            # Verificar cada vehículo
            for j, (v_bbox, v_confidence, v_class_id) in enumerate(vehicle_detections):
                v_x1, v_y1, v_x2, v_y2 = v_bbox
                vehicle_center = ((v_x1 + v_x2) / 2, (v_y1 + v_y2) / 2)
                
                # Calcular distancia entre persona y vehículo
                distance = np.sqrt(
                    (person_center[0] - vehicle_center[0])**2 + 
                    (person_center[1] - vehicle_center[1])**2
                )
                
                # Solo analizar si están relativamente cerca
                if distance < 300:  # Umbral arbitrario, ajustar según necesidad
                    vehicle_desc = self._get_vehicle_description(v_class_id)
                    
                    # Verificar cada tipo de postura peligrosa
                    for pose_config in dangerous_poses_config:
                        pose_name = pose_config['name']
                        pose_desc = pose_config['description']
                        
                        # Verificar criterios de keypoints
                        is_dangerous = False
                        for kp_criterion in pose_config.get('keypoints_criteria', []):
                            if len(kp_criterion) == 3:
                                kp1_idx, kp2_idx, ratio = kp_criterion
                                
                                # Verificar índices válidos
                                if (0 <= kp1_idx < len(keypoints) and 
                                    0 <= kp2_idx < len(keypoints)):
                                    
                                    kp1 = keypoints[kp1_idx]
                                    kp2 = keypoints[kp2_idx]
                                    
                                    # Verificar confianza de keypoints
                                    if kp1[2] > 0.5 and kp2[2] > 0.5:
                                        # Calcular relación espacial (simplificado para ejemplo)
                                        # Por ejemplo, comparar altura relativa (coordenada y)
                                        actual_ratio = (kp2[1] - kp1[1]) / max(1, abs(kp2[0] - kp1[0]))
                                        
                                        # Si la relación actual cumple con el criterio
                                        if abs(actual_ratio) > abs(ratio):
                                            is_dangerous = True
                                            break
                        
                        if is_dangerous:
                            alert_id = f"pose_danger_{person_id}_{j}_{pose_name}"
                            
                            # Evitar alertas duplicadas
                            if alert_id not in self.alerts_active:
                                event_msg = f"¡ALERTA! {pose_desc} cerca de {vehicle_desc}"
                                danger_alerts.append({
                                    'alert_id': alert_id,
                                    'person_id': person_id,
                                    'vehicle_id': j,
                                    'message': event_msg,
                                    'pose_name': pose_name,
                                    'keypoints': keypoints,
                                    'person_bbox': person_bbox,
                                    'vehicle_bbox': v_bbox,
                                    'distance': distance
                                })
        
        return danger_alerts
    
    def _get_vehicle_description(self, class_id):
        """Obtiene una descripción del vehículo basada en su clase."""
        return VEHICLE_CLASSES.get(class_id, "vehículo desconocido")
    
    def _calculate_vehicle_direction(self, vehicle_id, current_position):
        """Determina si un vehículo está entrando o saliendo basado en su movimiento."""
        if vehicle_id not in self.vehicle_history:
            self.vehicle_history[vehicle_id] = {
                'positions': [current_position],
                'timestamp': time.time()
            }
            return "indeterminado"
        
        # Obtener posición anterior
        prev_position = self.vehicle_history[vehicle_id]['positions'][-1]
        
        # Actualizar historial
        self.vehicle_history[vehicle_id]['positions'].append(current_position)
        self.vehicle_history[vehicle_id]['timestamp'] = time.time()
        
        # Calcular dirección
        dx = current_position[0] - prev_position[0]
        
        # Simplificación: asumimos que x positivo es hacia la derecha (entrada)
        # y x negativo es hacia la izquierda (salida)
        # Esto debe ajustarse según la configuración específica de la cámara
        if dx > 5:
            return "entrada"
        elif dx < -5:
            return "salida"
        else:
            return "indeterminado"
    
    def _check_vehicle_entry_exit(self, frame, vehicle_detections, width, height):
        """Verifica si hay vehículos entrando o saliendo de las zonas definidas."""
        # Implementación mejorada
        for i, (bbox, confidence, class_id) in enumerate(vehicle_detections):
            vehicle_id = f"vehicle_{i}"  # En una implementación real, usar tracking
            
            # Convertir bbox a formato [x, y, w, h]
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Obtener descripción del vehículo
            vehicle_desc = self._get_vehicle_description(class_id)
            
            # Determinar dirección de movimiento
            direction = self._calculate_vehicle_direction(vehicle_id, (center_x, center_y))
            
            # Verificar si está en alguna zona de entrada/salida
            for zone in self.entry_exit_zones:
                zone_points = np.array(zone['points']) * np.array([width, height])
                if cv2.pointPolygonTest(zone_points.astype(np.int32), (center_x, center_y), False) >= 0:
                    # Si la dirección no está determinada, usar la de la zona
                    if direction == "indeterminado":
                        direction = zone['direction']
                    
                    # Vehículo en zona de entrada/salida
                    event_msg = f"{vehicle_desc.capitalize()} en {direction} por {zone['name']}"
                    
                    # Registrar evento solo si es nuevo o ha pasado suficiente tiempo
                    event_key = f"{vehicle_id}_{zone['name']}"
                    current_time = time.time()
                    
                    if (event_key not in self.detected_vehicles or 
                        current_time - self.detected_vehicles.get(event_key, 0) > 10):
                        
                        logger.info(event_msg)
                        self.detected_vehicles[event_key] = current_time
                        
                        # Actualizar contador de estadísticas
                        if direction == "entrada":
                            self.events_log["statistics"]["vehiculos_entrada"] += 1
                        elif direction == "salida":
                            self.events_log["statistics"]["vehiculos_salida"] += 1
                        
                        # Registrar evento
                        event_data = {
                            "id": self.event_counter,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "type": f"vehiculo_{direction}",
                            "description": event_msg,
                            "vehicle_type": vehicle_desc,
                            "confidence": float(confidence),
                            "zone_name": zone['name']
                        }
                        
                        self.events_log["events"].append(event_data)
                        self.event_counter += 1
                        self._save_events_log()
                        
                        # Guardar imagen del evento si está configurado
                        if self.config['alerts']['save_event_images']:
                            self._save_event_image(
                                frame, 
                                event_msg, 
                                f"vehiculos_{direction}",
                                event_data
                            )
    
    def _check_danger_situations(self, frame, vehicle_detections, person_detections, width, height):
        """Verifica situaciones de peligro entre vehículos y personas."""
        # Implementación mejorada - en la realidad se usaría un algoritmo más sofisticado
        
        # Verificar personas en zonas de peligro
        for i, (bbox, confidence, class_id) in enumerate(person_detections):
            person_id = f"person_{i}"  # En una implementación real, usar tracking
            
            # Convertir bbox a formato [x, y, w, h]
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Verificar si está en alguna zona de peligro
            for zone in self.danger_zones:
                zone_points = np.array(zone['points']) * np.array([width, height])
                if cv2.pointPolygonTest(zone_points.astype(np.int32), (center_x, center_y), False) >= 0:
                    # Persona en zona de peligro
                    
                    # Verificar si hay vehículos cerca
                    for j, (v_bbox, v_confidence, v_class_id) in enumerate(vehicle_detections):
                        v_x1, v_y1, v_x2, v_y2 = v_bbox
                        v_center_x = (v_x1 + v_x2) / 2
                        v_center_y = (v_y1 + v_y2) / 2
                        
                        # Calcular distancia entre persona y vehículo
                        distance = np.sqrt((center_x - v_center_x)**2 + (center_y - v_center_y)**2)
                        
                        # Obtener descripción del vehículo
                        vehicle_desc = self._get_vehicle_description(v_class_id)
                        
                        # Si están muy cerca, generar alerta
                        if distance < 150:  # Umbral arbitrario, ajustar según necesidad
                            alert_id = f"danger_{person_id}_{j}"
                            if alert_id not in self.alerts_active:
                                event_msg = f"¡ALERTA! Persona cerca de {vehicle_desc} en {zone['name']}"
                                logger.warning(event_msg)
                                
                                # Activar alerta visual
                                self.alerts_active[alert_id] = time.time()
                                
                                # Actualizar estadísticas
                                self.events_log["statistics"]["situaciones_riesgo"] += 1
                                
                                # Registrar evento
                                event_data = {
                                    "id": self.event_counter,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "type": "situacion_riesgo",
                                    "description": event_msg,
                                    "vehicle_type": vehicle_desc,
                                    "distance": float(distance),
                                    "zone": zone['name'],
                                    "person_id": person_id,
                                    "vehicle_id": j
                                }
                                
                                self.events_log["events"].append(event_data)
                                self.event_counter += 1
                                self._save_events_log()
                                
                                # Guardar imagen del evento
                                if self.config['alerts']['save_event_images']:
                                    self._save_event_image(
                                        frame, 
                                        event_msg, 
                                        "situaciones_riesgo", 
                                        event_data
                                    )
    
    def _draw_zones(self, frame, width, height):
        """Dibuja las zonas configuradas en el frame."""
        # Dibujar zonas de entrada/salida
        for zone in self.entry_exit_zones:
            points = zone['points'] * np.array([width, height])
            
            # Color basado en dirección (verde para entrada, azul para salida)
            color = (0, 255, 0) if zone['direction'] == 'entrada' else (255, 0, 0)
            
            cv2.polylines(frame, [points.astype(np.int32)], True, color, 2)
            
            # Añadir etiqueta
            label_x = int(np.mean(points[:, 0]))
            label_y = int(np.mean(points[:, 1]))
            cv2.putText(frame, zone['name'], (label_x, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Dibujar zonas de peligro
        for zone in self.danger_zones:
            points = zone['points'] * np.array([width, height])
            cv2.polylines(frame, [points.astype(np.int32)], True, (0, 0, 255), 2)
            
            # Añadir etiqueta
            label_x = int(np.mean(points[:, 0]))
            label_y = int(np.mean(points[:, 1]))
            cv2.putText(frame, zone['name'], (label_x, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def _draw_detections(self, frame, vehicle_detections, person_detections):
        """Dibuja las detecciones en el frame."""
        # Dibujar vehículos
        for bbox, confidence, class_id in vehicle_detections:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            
            # Obtener descripción del vehículo
            vehicle_desc = self._get_vehicle_description(class_id)
            
            label = f"{vehicle_desc.capitalize()}: {confidence:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Dibujar personas
        for bbox, confidence, class_id in person_detections:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            label = f"Persona: {confidence:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Dibujar alertas activas
        current_time = time.time()
        alerts_to_remove = []
        
        for alert_id, alert_time in self.alerts_active.items():
            # Mantener alertas por 5 segundos
            if current_time - alert_time > 5.0:
                alerts_to_remove.append(alert_id)
                continue
            
            # Parpadeo de alerta
            if int((current_time - alert_time) * 2) % 2 == 0:
                cv2.putText(frame, "¡ALERTA DE PELIGRO!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Eliminar alertas expiradas
        for alert_id in alerts_to_remove:
            del self.alerts_active[alert_id]
    
    def _save_event_image(self, frame, event_msg, subfolder, event_data):
        """Guarda una imagen de un evento detectado con metadatos."""
        # Obtener fecha y hora actual
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        
        # Crear estructura de carpetas organizada por fecha
        date_folder = os.path.join(self.config['alerts']['events_folder'], date_str)
        type_folder = os.path.join(date_folder, subfolder)
        
        # Crear carpetas si no existen
        os.makedirs(type_folder, exist_ok=True)
        
        # Determinar nombre de archivo base
        if "vehicle_type" in event_data:
            # Para vehículos, incluir tipo en nombre de archivo
            vehicle_type = event_data["vehicle_type"].replace(" ", "_")
            base_filename = f"{time_str}_{vehicle_type}_{self.event_counter}"
        else:
            # Para otros eventos
            base_filename = f"{time_str}_evento_{self.event_counter}"
        
        # Ruta completa para imagen y metadatos
        image_filename = os.path.join(type_folder, f"{base_filename}.jpg")
        metadata_filename = os.path.join(type_folder, f"{base_filename}.json")
        
        # Añadir texto con el evento
        img_with_text = frame.copy()
        
        # Añadir recuadro semitransparente para el texto
        overlay = img_with_text.copy()
        cv2.rectangle(overlay, (5, 5), (600, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img_with_text, 0.4, 0, img_with_text)
        
        # Añadir fecha/hora y mensaje
        time_info = now.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img_with_text, time_info, (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img_with_text, event_msg, (10, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Guardar imagen
        cv2.imwrite(image_filename, img_with_text)
        logger.info(f"Imagen del evento guardada: {image_filename}")
        
        # Mejorar metadatos con información adicional
        enhanced_data = event_data.copy()
        enhanced_data.update({
            "timestamp_iso": now.isoformat(),
            "date": date_str,
            "time": time_str,
            "image_path": image_filename,
            "event_folder": type_folder
        })
        
        # Guardar metadatos junto con la imagen
        with open(metadata_filename, 'w') as f:
            json.dump(enhanced_data, f, indent=4)
        
        # Actualizar el registro diario
        self._update_daily_register(date_str, enhanced_data)
        
        return image_filename
    
    def _update_daily_register(self, date_str, event_data):
        """Actualiza el registro diario de eventos."""
        # Ruta al registro diario
        register_path = os.path.join(
            self.config['alerts']['events_folder'], 
            date_str, 
            "registro_diario.json"
        )
        
        # Cargar registro existente o crear uno nuevo
        if os.path.exists(register_path):
            try:
                with open(register_path, 'r') as f:
                    daily_register = json.load(f)
            except:
                daily_register = {"fecha": date_str, "eventos": []}
        else:
            daily_register = {"fecha": date_str, "eventos": []}
        
        # Añadir evento al registro
        if "eventos" not in daily_register:
            daily_register["eventos"] = []
        
        # Crear resumen del evento
        event_summary = {
            "id": event_data["id"],
            "hora": event_data.get("time", ""),
            "tipo": event_data.get("type", ""),
            "descripcion": event_data.get("description", ""),
            "imagen": os.path.basename(event_data.get("image_path", ""))
        }
        
        # Añadir resumen al registro
        daily_register["eventos"].append(event_summary)
        
        # Actualizar estadísticas
        if "estadisticas" not in daily_register:
            daily_register["estadisticas"] = {
                "vehiculos_entrada": 0,
                "vehiculos_salida": 0,
                "situaciones_riesgo": 0
            }
        
        # Incrementar contador según tipo de evento
        event_type = event_data.get("type", "")
        if event_type == "vehiculo_entrada":
            daily_register["estadisticas"]["vehiculos_entrada"] += 1
        elif event_type == "vehiculo_salida":
            daily_register["estadisticas"]["vehiculos_salida"] += 1
        elif event_type in ["situacion_riesgo", "postura_peligrosa"]:
            daily_register["estadisticas"]["situaciones_riesgo"] += 1
        
        # Guardar registro actualizado
        os.makedirs(os.path.dirname(register_path), exist_ok=True)
        with open(register_path, 'w') as f:
            json.dump(daily_register, f, indent=4)
    
    def _handle_pose_alerts(self, frame, pose_alerts):
        """Maneja las alertas de posturas peligrosas."""
        current_time = time.time()
        
        for alert in pose_alerts:
            alert_id = alert['alert_id']
            event_msg = alert['message']
            
            # Activar alerta visual
            self.alerts_active[alert_id] = current_time
            
            # Registrar en log
            logger.warning(event_msg)
            
            # Actualizar estadísticas
            self.events_log["statistics"]["situaciones_riesgo"] += 1
            
            # Registrar evento
            event_data = {
                "id": self.event_counter,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "postura_peligrosa",
                "description": event_msg,
                "pose_name": alert['pose_name'],
                "distance": float(alert['distance']),
                "person_bbox": alert['person_bbox'],
                "vehicle_bbox": alert['vehicle_bbox']
            }
            
            self.events_log["events"].append(event_data)
            self.event_counter += 1
            self._save_events_log()
            
            # Guardar imagen del evento
            if self.config['alerts']['save_event_images']:
                self._save_event_image(
                    frame, 
                    event_msg, 
                    "situaciones_riesgo", 
                    event_data
                )
    
    def _draw_poses(self, frame, pose_detections):
        """Dibuja las poses detectadas en el frame."""
        # Conexiones entre keypoints para dibujar el esqueleto
        skeleton_connections = [
            (0, 1), (0, 2),  # nariz a ojos
            (1, 3), (2, 4),  # ojos a orejas
            (0, 5), (0, 6),  # nariz a hombros
            (5, 7), (6, 8),  # hombros a codos
            (7, 9), (8, 10), # codos a muñecas
            (5, 11), (6, 12), # hombros a caderas
            (11, 13), (12, 14), # caderas a rodillas
            (13, 15), (14, 16)  # rodillas a tobillos
        ]
        
        # Para cada persona detectada con pose
        for person_bbox, confidence, keypoints in pose_detections:
            # Dibujar bbox de la persona
            x1, y1, x2, y2 = [int(coord) for coord in person_bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            
            # Dibujar keypoints
            for i, (x, y, kp_conf) in enumerate(keypoints):
                if kp_conf > 0.5:  # Solo dibujar keypoints con suficiente confianza
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            # Dibujar conexiones del esqueleto
            for connection in skeleton_connections:
                idx1, idx2 = connection
                if (idx1 < len(keypoints) and idx2 < len(keypoints) and
                    keypoints[idx1][2] > 0.5 and keypoints[idx2][2] > 0.5):
                    
                    pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                    pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
    
    def run(self):
        """Ejecuta el bucle principal de monitoreo."""
        logger.info("Iniciando monitoreo en tiempo real")
        
        try:
            while True:
                # Capturar frame
                ret, frame = self._get_frame()
                if not ret:
                    logger.error("Error al capturar frame de la cámara")
                    time.sleep(0.1)  # Pequeña pausa para evitar CPU al 100%
                    continue
                
                # Procesar frame
                processed_frame = self.process_frame(frame)
                
                # Mostrar resultado
                cv2.imshow("Construction Site Monitoring", processed_frame)
                
                # Salir si se presiona 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            logger.info("Monitoreo detenido por el usuario")
        finally:
            # Cerrar limpiamente
            self._cleanup()
    
    def _cleanup(self):
        """Limpia los recursos y genera informe final del día."""
        logger.info("Limpiando recursos...")
        
        # Detener captura
        self.stop_capture = True
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=3.0)
        
        # Terminar proceso de libcamera si existe
        if self.camera_process:
            self.camera_process.terminate()
            self.camera_process = None
        
        # Liberar recursos de OpenCV si los hay
        if hasattr(self, 'camera'):
            self.camera.release()
        
        # Cerrar ventanas
        cv2.destroyAllWindows()
        
        # Generar informe del día actual
        current_date = datetime.now().strftime("%Y-%m-%d")
        report_path = self.generate_daily_report(current_date)
        if report_path:
            logger.info(f"Informe final generado: {report_path}")
        
        logger.info("Sistema finalizado correctamente")
    
    def generate_daily_report(self, date_str=None):
        """Genera un informe HTML con los eventos del día especificado."""
        if date_str is None:
            # Usar fecha actual si no se especifica
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Ruta al registro diario
        register_path = os.path.join(
            self.config['alerts']['events_folder'],
            date_str,
            "registro_diario.json"
        )
        
        # Verificar si existe el registro para esa fecha
        if not os.path.exists(register_path):
            logger.warning(f"No existe registro para la fecha {date_str}")
            return None
        
        # Cargar registro diario
        with open(register_path, 'r') as f:
            daily_register = json.load(f)
        
        # Ruta para el informe HTML
        report_path = os.path.join(
            self.config['alerts']['events_folder'],
            date_str,
            "informe_diario.html"
        )
        
        # Generar HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Informe de Eventos - {date_str}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .event-entry {{ margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; }}
                .stats {{ background-color: #eef; padding: 10px; border-radius: 5px; }}
                .vehicle-entry {{ background-color: #efe; }}
                .vehicle-exit {{ background-color: #fee; }}
                .danger {{ background-color: #fdd; }}
                img {{ max-width: 300px; max-height: 200px; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>Informe de Eventos - {date_str}</h1>
            
            <div class="stats">
                <h2>Estadísticas del Día</h2>
                <table>
                    <tr>
                        <th>Vehículos entrando</th>
                        <th>Vehículos saliendo</th>
                        <th>Situaciones de riesgo</th>
                    </tr>
                    <tr>
                        <td>{daily_register.get('estadisticas', {}).get('vehiculos_entrada', 0)}</td>
                        <td>{daily_register.get('estadisticas', {}).get('vehiculos_salida', 0)}</td>
                        <td>{daily_register.get('estadisticas', {}).get('situaciones_riesgo', 0)}</td>
                    </tr>
                </table>
            </div>
            
            <h2>Eventos Registrados</h2>
        """
        
        # Añadir eventos
        eventos = daily_register.get('eventos', [])
        if eventos:
            for evento in eventos:
                tipo = evento.get('tipo', '')
                clase_css = 'event-entry'
                if 'entrada' in tipo:
                    clase_css += ' vehicle-entry'
                elif 'salida' in tipo:
                    clase_css += ' vehicle-exit'
                elif 'riesgo' in tipo or 'peligro' in tipo:
                    clase_css += ' danger'
                
                # Ruta de la imagen
                img_name = evento.get('imagen', '')
                img_path = ""
                if img_name:
                    # Determinar subfolder basado en tipo
                    subfolder = 'vehiculos_entrada'
                    if 'salida' in tipo:
                        subfolder = 'vehiculos_salida'
                    elif 'riesgo' in tipo or 'peligro' in tipo:
                        subfolder = 'situaciones_riesgo'
                    
                    img_path = f"{date_str}/{subfolder}/{img_name}"
                
                html_content += f"""
                <div class="{clase_css}">
                    <h3>Evento #{evento.get('id', 'N/A')} - {evento.get('hora', 'Sin hora')}</h3>
                    <p><strong>Tipo:</strong> {tipo}</p>
                    <p><strong>Descripción:</strong> {evento.get('descripcion', 'Sin descripción')}</p>
                    {f'<img src="../{img_path}" alt="Imagen del evento" />' if img_path else ''}
                </div>
                """
        else:
            html_content += "<p>No hay eventos registrados para esta fecha.</p>"
        
        # Cerrar HTML
        html_content += """
        </body>
        </html>
        """
        
        # Guardar informe
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Informe diario generado: {report_path}")
        return report_path
        
    def list_event_dates(self):
        """Lista las fechas que tienen eventos registrados."""
        events_folder = self.config['alerts']['events_folder']
        dates = []
        
        # Verificar carpetas de fecha
        if os.path.exists(events_folder):
            for item in os.listdir(events_folder):
                item_path = os.path.join(events_folder, item)
                if os.path.isdir(item_path) and item[0].isdigit():
                    # Verificar si tiene formato de fecha
                    try:
                        datetime.strptime(item, "%Y-%m-%d")
                        dates.append(item)
                    except ValueError:
                        continue
        
        return sorted(dates)

if __name__ == "__main__":
    # Comprobar argumentos de línea de comandos
    if len(sys.argv) > 1:
        # Procesar argumentos
        if sys.argv[1] == "--report":
            # Inicializar el monitor sin iniciar la cámara
            monitor = ConstructionMonitor()
            
            # Determinar fecha para el informe
            date_str = None
            if len(sys.argv) > 2:
                date_str = sys.argv[2]
            
            # Si se pidió listar fechas disponibles
            if date_str == "list":
                dates = monitor.list_event_dates()
                if dates:
                    print("Fechas con eventos registrados:")
                    for date in dates:
                        print(f"  - {date}")
                else:
                    print("No hay fechas con eventos registrados.")
            else:
                # Generar informe para la fecha especificada o actual
                report_path = monitor.generate_daily_report(date_str)
                if report_path:
                    print(f"Informe generado: {report_path}")
                else:
                    print(f"No se pudo generar el informe para la fecha {'especificada' if date_str else 'actual'}")
            
            sys.exit(0)
        elif sys.argv[1] == "--help":
            print("Uso:")
            print("  python main.py                    - Iniciar el sistema de monitoreo")
            print("  python main.py --report           - Generar informe del día actual")
            print("  python main.py --report YYYY-MM-DD - Generar informe para una fecha específica")
            print("  python main.py --report list      - Listar fechas con eventos registrados")
            print("  python main.py --help             - Mostrar esta ayuda")
            sys.exit(0)
    
    # Iniciar el sistema de monitoreo normal
    monitor = ConstructionMonitor()
    monitor.run()