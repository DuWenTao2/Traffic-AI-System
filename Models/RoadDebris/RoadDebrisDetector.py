# Vision Patrol - Road Debris Detection Module
import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
from ultralytics import YOLO
import logging

class RoadDebrisDetector:
    def __init__(self, stream_id="default", model_path=None, conf_threshold=0.2, cooldown=5, frame_skip=2, violation_manager=None, **kwargs):
        self.stream_id = stream_id
        self.violation_manager = violation_manager
        
        # 状态管理变量
        self.detection_enabled = False
        self.last_detection_time = 0
        self.frame_counter = 0
        
        # 抛洒物检测历史，用于避免重复检测
        self.debris_history = []
        self.history_max_length = 5
        
        # 1. 设置代码默认参数
        self._set_default_parameters()
        
        # 2. 从配置文件加载参数
        self._load_config_parameters()
        
        # 3. 应用启动参数（优先级最高）
        if model_path is not None:
            self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.cooldown = cooldown
        self.frame_skip = frame_skip
        
        # 应用额外的启动参数
        self.set_parameters(**kwargs)
        
        # 加载道路抛洒物检测模型
        try:
            self.model = YOLO(self.model_path)
            print(f"[{self.stream_id}] Road debris detector initialized with model: {self.model_path}")
            print(f"[{self.stream_id}] Detection confidence threshold: {self.conf_threshold}")
            print(f"[{self.stream_id}] Detection cooldown period: {self.cooldown} seconds")
            print(f"[{self.stream_id}] Road debris detection is DISABLED by default. Press 'x' to toggle.")
            
            # 直接从模型提取类别名称
            if hasattr(self.model, 'names') and self.model.names:
                # 直接使用模型中的类别名称
                self.debris_classes = list(self.model.names.values()) if isinstance(self.model.names, dict) else self.model.names
                print(f"[{self.stream_id}] Detected classes: {self.debris_classes}")
            else:
                # 回退到默认抛洒物类别
                self.debris_classes = ['0']
                print(f"[{self.stream_id}] Using default debris classes: {self.debris_classes}")
        except Exception as e:
            print(f"[{self.stream_id}] Error loading road debris detection model: {str(e)}")
            self.model = None
            self.debris_classes = ['Debris']
    
    def _set_default_parameters(self):
        """设置代码默认参数"""
        # 模型参数
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "..", "..", "weights", "Road_Debris_Detection", "best.pt")
        self.model_path = os.path.abspath(self.model_path)
        
        # 检测参数
        self.conf_threshold = 0.3
        self.cooldown = 30
        self.frame_skip = 3
        
        # 抛洒物检测参数
        self.min_area = 500  # 最小检测面积
        
        # 历史记录参数
        self.debris_history_max_length = 5
        self.debris_distance_threshold = 100  # 重复检测距离阈值
    
    def _load_config_parameters(self):
        """从配置文件加载参数"""
        # 配置文件路径
        config_path = os.path.join(os.path.dirname(__file__), "..", "Config", "RoadDebrisDetectionConfig.json")
        config_path = os.path.abspath(config_path)
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 加载参数
                if 'parameters' in config:
                    params = config['parameters']
                    # 跳过注释键（以#开头的键）
                    loaded_params = {}
                    for key, value in params.items():
                        if not key.startswith('#') and hasattr(self, key):
                            setattr(self, key, value)
                            loaded_params[key] = value
                    print(f"[{self.stream_id}] Road debris detection parameters loaded from config file: {loaded_params}")
            except Exception as e:
                print(f"[{self.stream_id}] Error loading config file: {str(e)}")
        else:
            print(f"[{self.stream_id}] Config file not found at {config_path}, using default parameters")
    
    def set_parameters(self, **kwargs):
        """设置检测参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        print(f"[{self.stream_id}] Road debris detection parameters updated")
    
    def get_parameters(self):
        """获取当前参数"""
        return {
            'model_path': self.model_path,
            'conf_threshold': self.conf_threshold,
            'cooldown': self.cooldown,
            'frame_skip': self.frame_skip,
            'min_area': self.min_area,
            'debris_history_max_length': self.debris_history_max_length,
            'debris_distance_threshold': self.debris_distance_threshold
        }
    
    def detect_debris(self, frame, tracked_objects=None):
        """检测道路抛洒物"""
        if frame is None or self.model is None:
            return frame
        
        # 创建副本以避免修改原始帧
        processed_frame = frame.copy()
        
        # 获取当前时间
        current_time = time.time()
        
        # 显示检测状态
        self._draw_status(processed_frame)
        
        # 如果检测被禁用，直接返回
        if not self.detection_enabled:
            return processed_frame
        
        # 增加帧计数器
        self.frame_counter += 1
        
        # 基于frame_skip跳过检测以提高效率
        if self.frame_counter % self.frame_skip != 0:
            return processed_frame
        
        # 检查冷却时间
        if current_time - self.last_detection_time < self.cooldown:
            return processed_frame
        
        try:
            # 运行抛洒物检测
            results = self.model(processed_frame, verbose=False)[0]
            
            # 处理检测结果
            debris_detected = []
            for box in results.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                class_name = results.names[cls]
                
                # 检查是否是抛洒物且置信度足够
                if class_name in self.debris_classes and conf >= self.conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    box_area = (x2 - x1) * (y2 - y1)
                    
                    # 检查最小面积
                    if box_area >= self.min_area:
                        # 检查是否是重复检测
                        if not self._is_duplicate_detection(box_center, current_time):
                            debris_detected.append({
                                'class_name': class_name,
                                'confidence': conf,
                                'box': (x1, y1, x2, y2),
                                'center': box_center,
                                'area': box_area
                            })
            
            # 处理检测到的抛洒物
            if debris_detected:
                # 按置信度排序
                debris_detected.sort(key=lambda x: x['confidence'], reverse=True)
                
                # 处理最高置信度的检测结果
                self._handle_debris_detection(processed_frame, debris_detected[0], current_time)
                
                # 绘制检测框
                for debris in debris_detected:
                    self._draw_debris_box(processed_frame, debris)
            
        except Exception as e:
            print(f"[{self.stream_id}] Error in road debris detection: {str(e)}")
        
        return processed_frame
    
    def _is_duplicate_detection(self, center, current_time):
        """检查是否是重复检测"""
        # 清理过期的历史记录
        self.debris_history = [(c, t) for c, t in self.debris_history if current_time - t < self.cooldown]
        
        # 检查是否在历史记录中有相近的检测
        for hist_center, hist_time in self.debris_history:
            distance = np.sqrt((center[0] - hist_center[0])**2 + (center[1] - hist_center[1])**2)
            if distance < self.debris_distance_threshold:
                return True
        
        return False
    
    def _handle_debris_detection(self, frame, debris, current_time):
        """处理抛洒物检测结果"""
        # 记录检测时间
        self.last_detection_time = current_time
        
        # 添加到历史记录
        self.debris_history.append((debris['center'], current_time))
        
        # 限制历史记录长度
        if len(self.debris_history) > self.debris_history_max_length:
            self.debris_history = self.debris_history[-self.debris_history_max_length:]
        
        # 保存截图
        self._save_debris_snapshot(frame, debris)
        
        # 记录违规
        self._log_debris_violation(debris, current_time)
    
    def _save_debris_snapshot(self, frame, debris):
        """保存抛洒物检测截图"""
        try:
            x1, y1, x2, y2 = debris['box']
            
            # 计算扩展框以获取更多上下文
            h, w = frame.shape[:2]
            margin = int(max(x2-x1, y2-y1) * 0.5)
            
            # 确保扩展框在帧边界内
            ex1 = max(0, x1 - margin)
            ey1 = max(0, y1 - margin)
            ex2 = min(w-1, x2 + margin)
            ey2 = min(h-1, y2 + margin)
            
            # 提取区域
            if ex1 < ex2 and ey1 < ey2:
                debris_closeup = frame[ey1:ey2, ex1:ex2]
                
                # 使用违规管理器保存截图
                if self.violation_manager:
                    # 确保snapshots目录存在
                    snapshots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Violation_Proc", "violations", "snapshots", "road_debris")
                    snapshots_dir = os.path.abspath(snapshots_dir)
                    os.makedirs(snapshots_dir, exist_ok=True)
                    
                    # 创建带时间戳的文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(snapshots_dir, f"road_debris_{self.stream_id}_{timestamp}.jpg")
                    
                    # 保存截图
                    cv2.imwrite(filename, debris_closeup)
                    print(f"[{self.stream_id}] Road debris snapshot saved: {os.path.basename(filename)}")
        except Exception as e:
            print(f"[{self.stream_id}] Error saving debris snapshot: {str(e)}")
    
    def _log_debris_violation(self, debris, current_time):
        """记录抛洒物违规"""
        try:
            # 构建违规信息
            violation_info = {
                'type': 'road_debris',
                'confidence': debris['confidence'],
                'location': debris['center'],
                'timestamp': current_time,
                'stream_id': self.stream_id
            }
            
            # 打印违规信息
            print(f"[{self.stream_id}] 🚨 ROAD DEBRIS DETECTED 🚨 (confidence: {debris['confidence']:.2f})")
            print(f"[{self.stream_id}] Location: {debris['center']}")
            
            # 如果有违规管理器，使用它记录违规
            if self.violation_manager:
                # 这里可以扩展为使用violation_manager的方法
                pass
                
        except Exception as e:
            print(f"[{self.stream_id}] Error logging debris violation: {str(e)}")
    
    def _draw_status(self, frame):
        """绘制检测状态"""
        status_text = "ROAD DEBRIS: " + ("ENABLED" if self.detection_enabled else "DISABLED")
        color = (0, 255, 0) if self.detection_enabled else (0, 0, 255)
        cv2.putText(frame, status_text, (20, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_debris_box(self, frame, debris):
        """绘制抛洒物检测框"""
        x1, y1, x2, y2 = debris['box']
        # 绘制红色矩形框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # 绘制标签
        label = f"Debris: {debris['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def toggle_detection(self):
        """切换检测状态"""
        self.detection_enabled = not self.detection_enabled
        status = "ENABLED" if self.detection_enabled else "DISABLED"
        print(f"[{self.stream_id}] Road debris detection {status}")
        return self.detection_enabled
