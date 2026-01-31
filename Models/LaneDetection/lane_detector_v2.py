import cv2
import numpy as np
import json
import os

from preprocessing import Preprocessor
from edge_detection import EdgeDetectionPipeline
from lane_fitting import LaneFittingPipeline
from lane_tracker import LaneTracker


class LaneDetectorV2:
    """
    车道线检测器 V2.0
    整合预处理、边缘检测、拟合、跟踪全流程
    """
    
    # 车道线类型定义
    LANE_TYPE_EMERGENCY = "emergency"
    LANE_TYPE_LEFT = "left"
    LANE_TYPE_RIGHT = "right"
    LANE_TYPE_MIDDLE = "middle"
    
    def __init__(self, stream_id="default", debug=False, config_path=None):
        """
        初始化车道线检测器
        
        参数:
            stream_id: 流ID
            debug: 是否开启调试模式
            config_path: 配置文件路径
        """
        self.stream_id = stream_id
        self.debug = debug
        self.detection_enabled = True
        self.version = "2.0"
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化各模块
        self.preprocessor = Preprocessor(self.config.get('preprocessing', {}))
        self.edge_pipeline = EdgeDetectionPipeline(self.config.get('edge_detection', {}))
        self.fitting_pipeline = LaneFittingPipeline(self.config.get('fitting', {}))
        self.tracker = LaneTracker(self.config.get('tracking', {}))
        
        # 消失点（用于动态ROI）
        self.vanishing_point = None
        
        print(f"[{self.stream_id}] LaneDetector V{self.version} initialized")
    
    def _load_config(self, config_path=None):
        """加载配置文件"""
        if config_path is None:
            # 默认配置文件路径
            config_path = os.path.join(
                os.path.dirname(__file__), 
                "..", "Config", "lane_detection_config.json"
            )
            config_path = os.path.abspath(config_path)
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"[{self.stream_id}] Config loaded from {config_path}")
                return config
            except Exception as e:
                print(f"[{self.stream_id}] Error loading config: {e}")
        
        # 返回默认配置
        return self._get_default_config()
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            "preprocessing": {
                "enable_color_enhance": True,
                "enable_shadow_removal": True,
                "clahe_clip_limit": 2.0,
                "clahe_grid_size": [8, 8],
                "shadow_kernel_size": 15,
                "shadow_k": 0.5
            },
            "edge_detection": {
                "sigma_normal": 0.33,
                "sigma_low": 0.50,
                "sigma_high": 0.20,
                "hough_rho": 1,
                "hough_theta": 0.0174533,
                "hough_threshold": 80,
                "hough_min_line_length": 80,
                "hough_max_line_gap": 50,
                "enable_color_filter": True,
                "enable_dynamic_roi": True,
                "camera_offset": 0,
                "center_margin": 0.1
            },
            "fitting": {
                "ransac_threshold": 5.0,
                "ransac_max_iterations": 100,
                "curve_threshold": 0.7,
                "parallel_threshold": 15,
                "width_ratio_range": [0.7, 1.4],
                "vanishing_tolerance": 50
            },
            "tracking": {
                "process_noise": 0.01,
                "measurement_noise": 0.1,
                "max_missed_frames": 5,
                "confidence_threshold": 0.6,
                "max_association_distance": 100,
                "max_angle_diff": 30
            }
        }
    
    def detect_lanes(self, frame):
        """
        检测图像中的车道线
        
        参数:
            frame: 输入图像
            
        返回:
            detected_lanes: 检测到的车道线列表
            annotated_frame: 标注了车道线的图像
        """
        if not self.detection_enabled:
            return [], frame
        
        try:
            # 创建图像副本
            annotated_frame = frame.copy()
            
            # 1. 预处理
            processed_frame, debug_info = self.preprocessor.preprocess(frame)
            
            # 2. 边缘检测与双向分离
            edge_result = self.edge_pipeline.detect(processed_frame, self.vanishing_point)
            
            # 3. 车道线拟合
            separation = edge_result['separation']
            fitted_lanes, vanishing_point = self.fitting_pipeline.fit_all_lanes(
                separation, frame.shape
            )
            
            # 更新消失点
            if vanishing_point is not None:
                self.vanishing_point = vanishing_point
            
            # 4. 识别应急车道
            emergency_lane = self.fitting_pipeline.identify_emergency_lane(
                fitted_lanes, frame.shape
            )
            
            # 5. 跟踪与平滑
            if fitted_lanes:
                tracked_lanes, self.vanishing_point = self.tracker.track(
                    fitted_lanes, frame, edge_result['combined_edges']
                )
            else:
                tracked_lanes = []
            
            # 6. 标注结果
            annotated_frame = self._draw_lanes(annotated_frame, tracked_lanes)
            
            # 调试显示
            if self.debug:
                self._show_debug_info(frame, processed_frame, edge_result, tracked_lanes)
            
            return tracked_lanes, annotated_frame
            
        except Exception as e:
            print(f"[{self.stream_id}] Error in lane detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], frame
    
    def _draw_lanes(self, frame, lanes):
        """
        在图像上绘制车道线
        
        参数:
            frame: 输入图像
            lanes: 车道线列表
            
        返回:
            frame: 标注后的图像
        """
        # 车道线颜色
        lane_colors = {
            self.LANE_TYPE_EMERGENCY: (0, 0, 255),    # 红色
            self.LANE_TYPE_LEFT: (0, 255, 0),         # 绿色
            self.LANE_TYPE_RIGHT: (255, 0, 0),        # 蓝色
            self.LANE_TYPE_MIDDLE: (0, 255, 255)      # 青色
        }
        
        for lane in lanes:
            lane_type = lane.get('type', 'unknown')
            color = lane_colors.get(lane_type, (255, 255, 255))
            points = lane.get('points', [])
            
            if len(points) >= 2:
                # 绘制车道线
                cv2.line(frame, points[0], points[1], color, 3)
                
                # 添加类型标签
                x, y = points[1]
                confidence = lane.get('confidence', 0.0)
                label = f"{lane_type} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 如果是预测值，用虚线表示
                if lane.get('predicted', False):
                    cv2.circle(frame, points[0], 5, (0, 0, 255), -1)
                    cv2.circle(frame, points[1], 5, (0, 0, 255), -1)
        
        # 绘制消失点
        if self.vanishing_point is not None:
            vp_x, vp_y = int(self.vanishing_point[0]), int(self.vanishing_point[1])
            cv2.circle(frame, (vp_x, vp_y), 8, (0, 165, 255), -1)
            cv2.putText(frame, "VP", (vp_x + 10, vp_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        return frame
    
    def _show_debug_info(self, original, processed, edge_result, lanes):
        """显示调试信息"""
        # 边缘图像
        cv2.imshow(f"{self.stream_id} - Edges", edge_result['edges'])
        cv2.imshow(f"{self.stream_id} - Combined Edges", edge_result['combined_edges'])
        
        # 颜色掩码
        color_masks = edge_result.get('color_masks', {})
        if 'white' in color_masks:
            cv2.imshow(f"{self.stream_id} - White Mask", color_masks['white'])
        if 'yellow' in color_masks:
            cv2.imshow(f"{self.stream_id} - Yellow Mask", color_masks['yellow'])
        
        # 打印检测信息
        print(f"[{self.stream_id}] Detected {len(lanes)} lanes")
        for lane in lanes:
            print(f"  - {lane.get('type', 'unknown')}: "
                  f"conf={lane.get('confidence', 0):.2f}, "
                  f"predicted={lane.get('predicted', False)}")
    
    def toggle_detection(self):
        """切换检测状态"""
        self.detection_enabled = not self.detection_enabled
        status = "ENABLED" if self.detection_enabled else "DISABLED"
        print(f"[{self.stream_id}] Lane detection {status}")
        return self.detection_enabled
    
    def set_parameters(self, **kwargs):
        """设置检测参数"""
        # 更新各模块参数
        if 'camera_offset' in kwargs or 'center_margin' in kwargs:
            self.edge_pipeline.set_parameters(**kwargs)
        
        if any(k in kwargs for k in ['clip_limit', 'grid_size', 'kernel_size', 'k']):
            self.preprocessor.set_parameters(**kwargs)
    
    def get_parameters(self):
        """获取当前参数"""
        return {
            'version': self.version,
            'detection_enabled': self.detection_enabled,
            'vanishing_point': self.vanishing_point,
            'config': self.config
        }
    
    def reset(self):
        """重置检测器状态"""
        self.tracker.reset()
        self.vanishing_point = None
        print(f"[{self.stream_id}] Lane detector reset")


# 保持向后兼容的别名
LaneDetector = LaneDetectorV2
