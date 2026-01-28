import cv2
import numpy as np
import time
import json
import os

class LaneDetector:
    def __init__(self, stream_id="default", debug=False, **kwargs):
        self.stream_id = stream_id
        self.debug = debug
        self.detection_enabled = True
        
        # 车道线类型定义
        self.LANE_TYPE_EMERGENCY = "emergency"
        self.LANE_TYPE_LEFT = "left"
        self.LANE_TYPE_RIGHT = "right"
        self.LANE_TYPE_MIDDLE = "middle"
        
        # 车道线历史记录，用于平滑
        self.lane_history = {}
        
        # 1. 设置代码默认值
        self._set_default_parameters()
        
        # 2. 从配置文件加载参数
        self._load_config_parameters()
        
        # 3. 应用启动参数（优先级最高）
        self.set_parameters(**kwargs)
        
        print(f"[{self.stream_id}] Lane detector initialized with parameters loaded")
    
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
            
            # 预处理图像
            preprocessed = self._preprocess_frame(frame)
            
            # 检测边缘
            edges = self._detect_edges(preprocessed)
            
            # 创建感兴趣区域掩码
            mask = self._create_roi_mask(frame.shape)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # 检测直线
            lines = self._detect_lines(masked_edges)
            
            # 处理车道线
            detected_lanes = self._process_lanes(lines, frame.shape)
            
            # 标注车道线
            annotated_frame = self._draw_lanes(annotated_frame, detected_lanes)
            
            if self.debug:
                # 显示中间结果
                cv2.imshow(f"{self.stream_id} - Edges", edges)
                cv2.imshow(f"{self.stream_id} - Masked Edges", masked_edges)
            
            return detected_lanes, annotated_frame
            
        except Exception as e:
            print(f"[{self.stream_id}] Error in lane detection: {str(e)}")
            return [], frame
    
    def _preprocess_frame(self, frame):
        """预处理图像"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def _detect_edges(self, frame):
        """检测边缘"""
        edges = cv2.Canny(frame, self.canny_threshold1, self.canny_threshold2)
        return edges
    
    def _create_roi_mask(self, frame_shape):
        """创建感兴趣区域掩码"""
        height, width = frame_shape[:2]
        
        # 定义梯形感兴趣区域
        roi_vertices = [
            (0, height),
            (width // 3, height // 2),
            (2 * width // 3, height // 2),
            (width, height)
        ]
        
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, np.array([roi_vertices], np.int32), 255)
        
        return mask
    
    def _detect_lines(self, edges):
        """检测直线"""
        lines = cv2.HoughLinesP(
            edges,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )
        
        return lines
    
    def _process_lanes(self, lines, frame_shape):
        """处理检测到的直线，识别车道线"""
        if lines is None or len(lines) == 0:
            return []
        
        height, width = frame_shape[:2]
        center_x = width // 2
        
        # 分类直线为左车道、右车道和中间线
        left_lines = []
        right_lines = []
        middle_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 计算斜率
            if x2 - x1 == 0:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            
            # 过滤垂直和水平线
            if abs(slope) < 0.1 or abs(slope) > 10:
                continue
            
            # 计算直线在底部的x坐标
            if y1 != y2:
                x_at_bottom = x1 + (height - y1) * (x2 - x1) / (y2 - y1)
            else:
                x_at_bottom = (x1 + x2) / 2
            
            # 分类车道线
            if slope < 0:  # 左车道线
                left_lines.append(line[0])
            elif slope > 0:  # 右车道线
                right_lines.append(line[0])
            
            # 检测中间线（接近图像中心）
            if abs(x_at_bottom - center_x) < width // 8:
                middle_lines.append(line[0])
        
        # 拟合车道线
        detected_lanes = []
        
        # 处理左车道线
        if left_lines:
            left_lane = self._fit_lane(left_lines, frame_shape)
            if left_lane:
                left_lane['type'] = self.LANE_TYPE_LEFT
                detected_lanes.append(left_lane)
        
        # 处理右车道线
        if right_lines:
            right_lane = self._fit_lane(right_lines, frame_shape)
            if right_lane:
                right_lane['type'] = self.LANE_TYPE_RIGHT
                detected_lanes.append(right_lane)
        
        # 处理中间线
        if middle_lines:
            middle_lane = self._fit_lane(middle_lines, frame_shape)
            if middle_lane:
                middle_lane['type'] = self.LANE_TYPE_MIDDLE
                detected_lanes.append(middle_lane)
        
        # 检测应急车道线（通常在最左侧或最右侧）
        emergency_lane = self._detect_emergency_lane(detected_lanes, frame_shape)
        if emergency_lane:
            detected_lanes.append(emergency_lane)
        
        # 平滑车道线检测结果
        detected_lanes = self._smooth_lanes(detected_lanes)
        
        return detected_lanes
    
    def _fit_lane(self, lines, frame_shape):
        """拟合车道线"""
        if not lines:
            return None
        
        height, width = frame_shape[:2]
        
        # 收集所有点
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        # 线性拟合
        if len(x_coords) > 1:
            coefficients = np.polyfit(y_coords, x_coords, 1)
            poly = np.poly1d(coefficients)
            
            # 计算车道线在图像底部和中部的点
            y1 = height
            y2 = height // 2
            x1 = int(poly(y1))
            x2 = int(poly(y2))
            
            return {
                'points': [(x1, y1), (x2, y2)],
                'coefficients': coefficients.tolist()
            }
        
        return None
    
    def _detect_emergency_lane(self, detected_lanes, frame_shape):
        """检测应急车道线"""
        height, width = frame_shape[:2]
        
        # 检查是否有车道线在图像边缘
        for lane in detected_lanes:
            x1, y1 = lane['points'][0]
            x2, y2 = lane['points'][1]
            
            # 检查是否靠近左侧边缘
            if x1 < width // 6 and x2 < width // 6:
                # 计算系数，与其他车道线保持一致
                if y1 != y2:
                    coefficients = np.polyfit([y1, y2], [x1, x2], 1).tolist()
                else:
                    coefficients = [0, (x1 + x2) / 2]
                
                return {
                    'points': [(x1, y1), (x2, y2)],
                    'type': self.LANE_TYPE_EMERGENCY,
                    'coefficients': coefficients
                }
            
            # 检查是否靠近右侧边缘
            if x1 > 5 * width // 6 and x2 > 5 * width // 6:
                # 计算系数，与其他车道线保持一致
                if y1 != y2:
                    coefficients = np.polyfit([y1, y2], [x1, x2], 1).tolist()
                else:
                    coefficients = [0, (x1 + x2) / 2]
                
                return {
                    'points': [(x1, y1), (x2, y2)],
                    'type': self.LANE_TYPE_EMERGENCY,
                    'coefficients': coefficients
                }
        
        return None
    
    def _smooth_lanes(self, detected_lanes):
        """平滑车道线检测结果"""
        smoothed_lanes = []
        
        for lane in detected_lanes:
            lane_type = lane['type']
            
            # 更新历史记录
            if lane_type not in self.lane_history:
                self.lane_history[lane_type] = []
            
            self.lane_history[lane_type].append(lane)
            
            # 保持历史记录长度
            if len(self.lane_history[lane_type]) > self.history_length:
                self.lane_history[lane_type].pop(0)
            
            # 平滑车道线
            if len(self.lane_history[lane_type]) > 1:
                smoothed_lane = self._average_lanes(self.lane_history[lane_type])
                smoothed_lanes.append(smoothed_lane)
            else:
                smoothed_lanes.append(lane)
        
        return smoothed_lanes
    
    def _average_lanes(self, lanes):
        """平均多条车道线"""
        if not lanes:
            return None
        
        # 平均点
        avg_x1 = int(np.mean([lane['points'][0][0] for lane in lanes]))
        avg_y1 = int(np.mean([lane['points'][0][1] for lane in lanes]))
        avg_x2 = int(np.mean([lane['points'][1][0] for lane in lanes]))
        avg_y2 = int(np.mean([lane['points'][1][1] for lane in lanes]))
        
        # 平均系数（如果存在）
        if 'coefficients' in lanes[0]:
            try:
                avg_coefficients = np.mean([lane['coefficients'] for lane in lanes], axis=0).tolist()
            except:
                avg_coefficients = []
        else:
            avg_coefficients = []
        
        return {
            'points': [(avg_x1, avg_y1), (avg_x2, avg_y2)],
            'type': lanes[0]['type'],
            'coefficients': avg_coefficients
        }
    
    def _draw_lanes(self, frame, lanes):
        """在图像上绘制车道线"""
        # 车道线颜色
        lane_colors = {
            self.LANE_TYPE_EMERGENCY: (0, 0, 255),  # 红色
            self.LANE_TYPE_LEFT: (0, 255, 0),       # 绿色
            self.LANE_TYPE_RIGHT: (255, 0, 0),      # 蓝色
            self.LANE_TYPE_MIDDLE: (0, 255, 255)    # 青色
        }
        
        for lane in lanes:
            color = lane_colors.get(lane['type'], (255, 255, 255))
            points = lane['points']
            
            # 绘制车道线
            cv2.line(frame, points[0], points[1], color, 3)
            
            # 添加车道线类型标签
            x, y = points[1]
            cv2.putText(frame, lane['type'], (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def toggle_detection(self):
        """切换检测状态"""
        self.detection_enabled = not self.detection_enabled
        status = "ENABLED" if self.detection_enabled else "DISABLED"
        print(f"[{self.stream_id}] Lane detection {status}")
        return self.detection_enabled
    
    def _set_default_parameters(self):
        """设置代码默认参数"""
        # 车道线检测参数
        self.canny_threshold1 = 40
        self.canny_threshold2 = 120
        self.hough_rho = 1
        self.hough_theta = np.pi / 180
        self.hough_threshold = 80
        self.hough_min_line_length = 80
        self.hough_max_line_gap = 50
        
        # 车道线颜色阈值
        self.white_lower = np.array([200, 200, 200])
        self.white_upper = np.array([255, 255, 255])
        self.yellow_lower = np.array([100, 100, 0])
        self.yellow_upper = np.array([200, 200, 100])
        
        # 车道线历史记录，用于平滑
        self.history_length = 5
    
    def _load_config_parameters(self):
        """从配置文件加载参数"""
        # 配置文件路径
        config_path = os.path.join(os.path.dirname(__file__), "..", "Config", "lane_detection_config.json")
        config_path = os.path.abspath(config_path)
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 加载参数
                if 'parameters' in config:
                    params = config['parameters']
                    # 跳过注释键（以#开头的键）
                    for key, value in params.items():
                        if not key.startswith('#') and hasattr(self, key):
                            # 处理颜色阈值的特殊情况（需要转换为numpy数组）
                            if key in ['white_lower', 'white_upper', 'yellow_lower', 'yellow_upper']:
                                if isinstance(value, list):
                                    setattr(self, key, np.array(value))
                            else:
                                setattr(self, key, value)
                    print(f"[{self.stream_id}] Lane detection parameters loaded from config file")
            except Exception as e:
                print(f"[{self.stream_id}] Error loading config file: {str(e)}")
        else:
            print(f"[{self.stream_id}] Config file not found at {config_path}, using default parameters")
    
    def set_parameters(self, **kwargs):
        """设置检测参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        print(f"[{self.stream_id}] Lane detection parameters updated")
    
    def get_parameters(self):
        """获取当前参数"""
        return {
            'canny_threshold1': self.canny_threshold1,
            'canny_threshold2': self.canny_threshold2,
            'hough_rho': self.hough_rho,
            'hough_theta': self.hough_theta,
            'hough_threshold': self.hough_threshold,
            'hough_min_line_length': self.hough_min_line_length,
            'hough_max_line_gap': self.hough_max_line_gap
        }
