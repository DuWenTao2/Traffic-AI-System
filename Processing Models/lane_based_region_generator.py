import cv2
import numpy as np
import time
from areas import AreaType

class LaneAnalyzer:
    """分析车道线几何关系的类"""
    def __init__(self):
        self.lane_lines = []  # 检测到的车道线
        self.vanishing_point = None  # 消失点
        self.lane_widths = []  # 车道宽度
        self.lane_directions = []  # 车道方向向量
        self.emergency_lane_index = -1  # 应急车道索引
        self.frame_shape = None  # 帧尺寸
        
        # 默认参数
        self.lane_extension_factor = 10.0  # 车道线延长系数
        self.default_lane_width = 100  # 默认车道宽度（像素）
        self.emergency_lane_width_threshold = 1.2  # 应急车道宽度阈值
    
    def set_parameters(self, **kwargs):
        """设置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def analyze_lanes(self, detected_lanes, frame_shape):
        """分析车道线几何关系"""
        # 1. 存储车道线
        self.lane_lines = detected_lanes
        self.frame_shape = frame_shape
        
        # 2. 计算消失点
        self._calculate_vanishing_point()
        
        # 3. 计算车道宽度和方向
        self._calculate_lane_parameters()
        
        # 4. 识别应急车道
        self._identify_emergency_lane()
    
    def _calculate_vanishing_point(self):
        """计算消失点"""
        if len(self.lane_lines) < 2 or not self.frame_shape:
            return
        
        # 收集所有车道线的延长线
        extended_lines = []
        for lane in self.lane_lines:
            points = lane['points']
            if len(points) == 2:
                # 延长车道线到图像顶部
                p1 = np.array(points[0])
                p2 = np.array(points[1])
                # 计算参数方程
                t = -p1[1] / (p2[1] - p1[1]) if (p2[1] - p1[1]) != 0 else self.lane_extension_factor
                extended_p = p1 + t * (p2 - p1)
                extended_lines.append((p1, extended_p))
        
        # 计算所有延长线的交点区域
        intersections = []
        for i in range(len(extended_lines)):
            for j in range(i+1, len(extended_lines)):
                p1, p2 = extended_lines[i]
                p3, p4 = extended_lines[j]
                # 计算两直线交点
                intersection = self._line_intersection(p1, p2, p3, p4)
                if intersection is not None:
                    intersections.append(intersection)
        
        # 计算交点的平均值作为消失点
        if intersections:
            self.vanishing_point = np.mean(intersections, axis=0)
    
    def _line_intersection(self, p1, p2, p3, p4):
        """计算两直线交点"""
        # 解线性方程组
        x_diff = np.array([p1[0] - p2[0], p3[0] - p4[0]])
        y_diff = np.array([p1[1] - p2[1], p3[1] - p4[1]])
        
        div = np.linalg.det(np.array([x_diff, y_diff]))
        if div == 0:
            return None  # 直线平行
        
        d = np.array([
            np.linalg.det(np.array([p1, p2])),
            np.linalg.det(np.array([p3, p4]))
        ])
        
        x = np.linalg.det(np.array([d, x_diff])) / div
        y = np.linalg.det(np.array([d, y_diff])) / div
        
        # 检查交点是否在图像范围内
        height, width = self.frame_shape[:2]
        if x >= 0 and x <= width and y >= 0 and y <= height:
            return np.array([x, y])
        return None
    
    def _calculate_lane_parameters(self):
        """计算车道宽度和方向"""
        self.lane_widths = []
        self.lane_directions = []
        
        if len(self.lane_lines) < 1:
            return
        
        # 计算每条车道的方向和相邻车道的宽度
        for i, lane in enumerate(self.lane_lines):
            points = lane['points']
            if len(points) == 2:
                # 计算车道方向向量
                p1 = np.array(points[0])
                p2 = np.array(points[1])
                direction = p2 - p1
                direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else np.array([0, 1])
                self.lane_directions.append(direction)
                
                # 计算与下一条车道的宽度
                if i < len(self.lane_lines) - 1:
                    next_lane = self.lane_lines[i + 1]
                    next_points = next_lane['points']
                    if len(next_points) == 2:
                        # 计算两条车道线在图像底部的距离
                        bottom_y = self.frame_shape[0]
                        lane_x = self._get_lane_x_at_y(lane, bottom_y)
                        next_lane_x = self._get_lane_x_at_y(next_lane, bottom_y)
                        if lane_x is not None and next_lane_x is not None:
                            width = abs(next_lane_x - lane_x)
                            self.lane_widths.append(width)
    
    def _get_lane_x_at_y(self, lane, y):
        """计算车道线在指定y坐标的x值"""
        points = lane['points']
        if len(points) != 2:
            return None
        
        p1 = np.array(points[0])
        p2 = np.array(points[1])
        
        # 确保y值在两点之间
        if (y < min(p1[1], p2[1])) or (y > max(p1[1], p2[1])):
            # 延长直线
            t = (y - p1[1]) / (p2[1] - p1[1]) if (p2[1] - p1[1]) != 0 else 0
            x = p1[0] + t * (p2[0] - p1[0])
            return x
        
        # 使用线性插值
        t = (y - p1[1]) / (p2[1] - p1[1]) if (p2[1] - p1[1]) != 0 else 0
        x = p1[0] + t * (p2[0] - p1[0])
        return x
    
    def _identify_emergency_lane(self):
        """识别应急车道"""
        if len(self.lane_lines) < 1:
            return
        
        # 按车道线在图像底部的x坐标排序
        lane_positions = []
        for i, lane in enumerate(self.lane_lines):
            bottom_y = self.frame_shape[0]
            x = self._get_lane_x_at_y(lane, bottom_y)
            if x is not None:
                lane_positions.append((x, i))
        
        if lane_positions:
            # 找到最左侧和最右侧的车道
            lane_positions.sort()
            leftmost_idx = lane_positions[0][1]
            rightmost_idx = lane_positions[-1][1]
            
            # 检查最左侧车道是否可能是应急车道
            # 应急车道通常比普通车道窄，或者位于道路边缘
            if len(self.lane_widths) > 0:
                avg_width = np.mean(self.lane_widths)
                # 检查最左侧车道与相邻车道的宽度
                if leftmost_idx < len(self.lane_widths):
                    if self.lane_widths[leftmost_idx] > avg_width * self.emergency_lane_width_threshold:
                        # 左侧有较宽的空间，可能是应急车道
                        self.emergency_lane_index = leftmost_idx
                        return
                
                # 检查最右侧车道
                if rightmost_idx > 0 and (rightmost_idx - 1) < len(self.lane_widths):
                    if self.lane_widths[rightmost_idx - 1] > avg_width * self.emergency_lane_width_threshold:
                        # 右侧有较宽的空间，可能是应急车道
                        self.emergency_lane_index = rightmost_idx
                        return
            
            # 默认识别最左侧车道为应急车道
            self.emergency_lane_index = leftmost_idx

class LaneBasedRegionGenerator:
    """基于车道线几何关系的区域生成器"""
    def __init__(self, stream_id="default"):
        self.stream_id = stream_id
        self.lane_analyzer = LaneAnalyzer()
        self.last_generation_time = 0
        
        # 1. 设置代码默认值
        self._set_default_parameters()
        
        # 2. 从配置文件加载参数
        self._load_config_parameters()
        
        print(f"[{self.stream_id}] Lane-based region generator initialized with parameters loaded")
    
    def _set_default_parameters(self):
        """设置代码默认参数"""
        # 基本参数
        self.generation_interval = 2.0  # 生成间隔，避免频繁计算
        self.region_update_interval = 5.0  # 区域更新间隔
        
        # 消失点计算参数
        self.lane_extension_factor = 10.0  # 车道线延长系数
        
        # 车道宽度计算参数
        self.default_lane_width = 100  # 默认车道宽度（像素）
        
        # 应急车道识别参数
        self.emergency_lane_width_threshold = 1.2  # 应急车道宽度阈值
        
        # 速度检测线生成参数
        self.speed_line_count = 5  # 速度检测线数量
        self.speed_line_spacing_factor = 6  # 速度检测线间距因子
        
        # 区域更新参数
        self.region_update_enabled = True  # 是否启用区域更新
    
    def _load_config_parameters(self):
        """从配置文件加载参数"""
        import os
        import json
        
        # 配置文件路径
        config_path = os.path.join(os.path.dirname(__file__), "..", "Models", "Config", "lane_based_region_generator_config.json")
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
                            setattr(self, key, value)
                    print(f"[{self.stream_id}] Lane-based region generator parameters loaded from config file")
            except Exception as e:
                print(f"[{self.stream_id}] Error loading config file: {str(e)}")
        else:
            print(f"[{self.stream_id}] Config file not found at {config_path}, using default parameters")
    
    def generate_regions(self, detected_lanes, frame_shape, area_manager):
        """基于检测到的车道线生成所有需要的区域"""
        # 1. 传递配置参数给 LaneAnalyzer
        self.lane_analyzer.set_parameters(
            lane_extension_factor=self.lane_extension_factor,
            default_lane_width=self.default_lane_width,
            emergency_lane_width_threshold=self.emergency_lane_width_threshold
        )
        
        # 2. 分析车道线几何关系
        self.lane_analyzer.analyze_lanes(detected_lanes, frame_shape)
        
        # 2. 生成速度检测线
        speed_lines = self._generate_speed_lines()
        
        # 3. 生成逆向检测线
        wrong_dir_lines = self._generate_wrong_direction_lines()
        
        # 4. 生成应急车道区域
        emergency_area = self._generate_emergency_lane_area()
        
        # 5. 更新区域管理器
        self._update_area_manager(area_manager, speed_lines, wrong_dir_lines, emergency_area)
        
        return True
    
    def _generate_speed_lines(self):
        """生成速度检测线"""
        speed_lines = []
        
        if not self.lane_analyzer.lane_lines or not self.lane_analyzer.frame_shape:
            return speed_lines
        
        # 计算平均车道宽度
        avg_lane_width = np.mean(self.lane_analyzer.lane_widths) if self.lane_analyzer.lane_widths else self.default_lane_width
        
        # 在图像垂直方向均匀分布速度线
        # 从图像底部向上生成，覆盖主要的车辆行驶区域
        height = self.lane_analyzer.frame_shape[0]
        width = self.lane_analyzer.frame_shape[1]
        
        # 生成指定数量的速度线
        for i in range(self.speed_line_count):
            # 计算速度线的y坐标
            y = height - int((i + 1) * height / self.speed_line_spacing_factor)
            
            # 找到最左侧和最右侧的车道线
            leftmost_x = 0
            rightmost_x = width
            
            for lane in self.lane_analyzer.lane_lines:
                points = lane['points']
                if len(points) == 2:
                    # 计算车道线在当前y坐标的x值
                    x = self.lane_analyzer._get_lane_x_at_y(lane, y)
                    if x is not None:
                        leftmost_x = max(leftmost_x, x - avg_lane_width)
                        rightmost_x = min(rightmost_x, x + avg_lane_width)
            
            # 创建速度线
            if leftmost_x < rightmost_x:
                speed_lines.append({
                    'points': [(int(leftmost_x), int(y)), (int(rightmost_x), int(y))],
                    'type': 'SPEED',
                    'enabled': True,
                    'properties': {}
                })
        
        return speed_lines
    
    def _generate_wrong_direction_lines(self):
        """生成逆向检测所需的车道线"""
        wrong_dir_lines = {
            'LEFT_LANE': [],
            'CENTER_LANE': [],
            'RIGHT_LANE': []
        }
        
        if not self.lane_analyzer.lane_lines:
            return wrong_dir_lines
        
        # 按照车道线的x坐标排序
        sorted_lanes = []
        for lane in self.lane_analyzer.lane_lines:
            bottom_y = self.lane_analyzer.frame_shape[0]
            x = self.lane_analyzer._get_lane_x_at_y(lane, bottom_y)
            if x is not None:
                sorted_lanes.append((x, lane))
        
        # 按x坐标排序
        sorted_lanes.sort(key=lambda x: x[0])
        sorted_lane_objects = [lane for _, lane in sorted_lanes]
        
        # 生成左、中、右三条车道线
        if len(sorted_lane_objects) >= 3:
            # 左车道线
            wrong_dir_lines['LEFT_LANE'].append({
                'points': sorted_lane_objects[0]['points'],
                'type': 'LEFT_LANE',
                'enabled': True,
                'properties': {}
            })
            
            # 中心车道线
            wrong_dir_lines['CENTER_LANE'].append({
                'points': sorted_lane_objects[len(sorted_lane_objects)//2]['points'],
                'type': 'CENTER_LANE',
                'enabled': True,
                'properties': {}
            })
            
            # 右车道线
            wrong_dir_lines['RIGHT_LANE'].append({
                'points': sorted_lane_objects[-1]['points'],
                'type': 'RIGHT_LANE',
                'enabled': True,
                'properties': {}
            })
        elif len(sorted_lane_objects) == 2:
            # 左车道线
            wrong_dir_lines['LEFT_LANE'].append({
                'points': sorted_lane_objects[0]['points'],
                'type': 'LEFT_LANE',
                'enabled': True,
                'properties': {}
            })
            
            # 右车道线
            wrong_dir_lines['RIGHT_LANE'].append({
                'points': sorted_lane_objects[1]['points'],
                'type': 'RIGHT_LANE',
                'enabled': True,
                'properties': {}
            })
        elif len(sorted_lane_objects) == 1:
            # 中心车道线
            wrong_dir_lines['CENTER_LANE'].append({
                'points': sorted_lane_objects[0]['points'],
                'type': 'CENTER_LANE',
                'enabled': True,
                'properties': {}
            })
        
        return wrong_dir_lines
    
    def _generate_emergency_lane_area(self):
        """生成应急车道区域"""
        emergency_area = []
        
        if self.lane_analyzer.emergency_lane_index == -1 or not self.lane_analyzer.frame_shape:
            return emergency_area
        
        # 获取应急车道
        emergency_lane = self.lane_analyzer.lane_lines[self.lane_analyzer.emergency_lane_index]
        
        # 获取应急车道的两个点
        points = emergency_lane['points']
        if len(points) != 2:
            return emergency_area
        
        # 计算应急车道的宽度
        avg_lane_width = np.mean(self.lane_analyzer.lane_widths) if self.lane_analyzer.lane_widths else 100
        
        # 确定应急车道的方向
        is_left_emergency = emergency_lane.get('type') == 'left' or \
                           (self.lane_analyzer.emergency_lane_index == 0 and len(self.lane_analyzer.lane_lines) > 1)
        
        # 生成应急车道区域多边形
        p1 = np.array(points[0])
        p2 = np.array(points[1])
        
        # 计算垂直于车道线的方向向量
        lane_dir = p2 - p1
        lane_dir = lane_dir / np.linalg.norm(lane_dir) if np.linalg.norm(lane_dir) > 0 else np.array([0, 1])
        
        # 垂直方向向量
        perpendicular_dir = np.array([-lane_dir[1], lane_dir[0]])
        
        # 根据应急车道位置调整方向
        if is_left_emergency:
            perpendicular_dir = -perpendicular_dir
        
        # 计算应急车道区域的四个点
        p3 = p1 + perpendicular_dir * avg_lane_width
        p4 = p2 + perpendicular_dir * avg_lane_width
        
        # 确保点按顺序排列
        emergency_area.append({
            'points': [tuple(p1.astype(int)), tuple(p2.astype(int)), tuple(p4.astype(int)), tuple(p3.astype(int)), tuple(p1.astype(int))],
            'type': 'EMERGENCY_LANE',
            'enabled': True,
            'properties': {}
        })
        
        return emergency_area
    
    def _update_area_manager(self, area_manager, speed_lines, wrong_dir_lines, emergency_area):
        """更新区域管理器"""
        # 1. 清空现有区域
        if AreaType.SPEED not in area_manager.areas:
            area_manager.areas[AreaType.SPEED] = []
        else:
            area_manager.areas[AreaType.SPEED] = []
        
        if AreaType.LEFT_LANE not in area_manager.areas:
            area_manager.areas[AreaType.LEFT_LANE] = []
        else:
            area_manager.areas[AreaType.LEFT_LANE] = []
        
        if AreaType.CENTER_LANE not in area_manager.areas:
            area_manager.areas[AreaType.CENTER_LANE] = []
        else:
            area_manager.areas[AreaType.CENTER_LANE] = []
        
        if AreaType.RIGHT_LANE not in area_manager.areas:
            area_manager.areas[AreaType.RIGHT_LANE] = []
        else:
            area_manager.areas[AreaType.RIGHT_LANE] = []
        
        if AreaType.EMERGENCY_LANE not in area_manager.areas:
            area_manager.areas[AreaType.EMERGENCY_LANE] = []
        else:
            area_manager.areas[AreaType.EMERGENCY_LANE] = []
        
        # 2. 添加速度检测线
        for line in speed_lines:
            area_manager.areas[AreaType.SPEED].append(line)
        
        # 3. 添加逆向检测线
        for line in wrong_dir_lines['LEFT_LANE']:
            area_manager.areas[AreaType.LEFT_LANE].append(line)
        for line in wrong_dir_lines['CENTER_LANE']:
            area_manager.areas[AreaType.CENTER_LANE].append(line)
        for line in wrong_dir_lines['RIGHT_LANE']:
            area_manager.areas[AreaType.RIGHT_LANE].append(line)
        
        # 4. 添加应急车道区域
        for area in emergency_area:
            area_manager.areas[AreaType.EMERGENCY_LANE].append(area)
        
        # 5. 保存区域配置
        area_manager.save_areas()
        print(f"[{self.stream_id}] Updated regions based on lane detection")
