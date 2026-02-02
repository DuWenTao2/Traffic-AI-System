import cv2
import numpy as np
import time
from areas import AreaType

class LaneRegionGenerator:
    """基于车道线检测结果的区域生成器
    
    优化版区域生成器，简化几何计算，提高执行效率
    保持与原 LaneBasedRegionGenerator 相同的接口
    """
    
    def __init__(self, stream_id="default"):
        """初始化区域生成器
        
        参数:
            stream_id: 流ID，用于日志和配置
        """
        self.stream_id = stream_id
        self.last_generation_time = 0
        
        # 设置默认参数
        self._set_default_parameters()
        
        # 从配置文件加载参数
        self._load_config_parameters()
        
        print(f"[{self.stream_id}] Lane region generator initialized with optimized parameters")
    
    def _set_default_parameters(self):
        """设置默认参数"""
        # 基本参数
        self.generation_interval = 2.0  # 生成间隔
        self.region_update_interval = 5.0  # 区域更新间隔
        
        # 速度检测线参数
        self.speed_line_count = 2  # 每条车道的速度检测线数量
        self.speed_line_positions = [0.4, 0.6]  # 速度检测线在图像中的相对位置
        self.speed_line_width_factor = 0.8  # 速度检测线长度与车道宽度的比例
        
        # 应急车道参数
        self.emergency_lane_width_factor = 0.8  # 应急车道宽度与普通车道的比例
        self.default_lane_width = 80  # 默认车道宽度（像素）
        
        # 区域更新参数
        self.region_update_enabled = True  # 是否启用区域更新
        
        # 双向道路参数
        self.enable_bidirectional_separation = True  # 启用双向分离
        self.center_margin_ratio = 0.1  # 中心区域边距比例
    
    def _load_config_parameters(self):
        """从配置文件加载参数"""
        import os
        import json
        
        # 配置文件路径
        config_path = os.path.join(os.path.dirname(__file__), "..", "Models", "Config", "lane_region_generator_config.json")
        config_path = os.path.abspath(config_path)
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 加载参数
                if 'parameters' in config:
                    params = config['parameters']
                    for key, value in params.items():
                        if not key.startswith('#') and hasattr(self, key):
                            setattr(self, key, value)
                    print(f"[{self.stream_id}] Lane region generator parameters loaded from config file")
            except Exception as e:
                print(f"[{self.stream_id}] Error loading config file: {str(e)}")
        else:
            print(f"[{self.stream_id}] Config file not found at {config_path}, using default parameters")
    
    def generate_regions(self, detected_lanes, frame_shape, area_manager):
        """基于检测到的车道线生成所有需要的区域
        
        参数:
            detected_lanes: 检测到的车道线列表
            frame_shape: 图像形状
            area_manager: 区域管理器
            
        返回:
            bool: 是否成功生成
        """
        if not detected_lanes or not frame_shape:
            return False
        
        try:
            # 1. 排序车道线（从左到右）
            sorted_lanes = self._sort_lanes_by_position(detected_lanes, frame_shape)
            
            if not sorted_lanes:
                return False
            
            # 2. 计算车道宽度
            lane_widths = self._calculate_lane_widths(sorted_lanes, frame_shape)
            avg_lane_width = np.mean(lane_widths) if lane_widths else self.default_lane_width
            
            # 3. 生成速度检测线
            speed_lines = self._generate_speed_lines(sorted_lanes, frame_shape, avg_lane_width)
            
            # 4. 生成逆向检测线
            wrong_dir_lines = self._generate_wrong_direction_lines(sorted_lanes)
            
            # 5. 生成应急车道区域
            emergency_area = self._generate_emergency_lane_area(sorted_lanes, frame_shape, avg_lane_width)
            
            # 6. 更新区域管理器
            self._update_area_manager(area_manager, speed_lines, wrong_dir_lines, emergency_area)
            
            return True
        except Exception as e:
            print(f"[{self.stream_id}] Error generating regions: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _sort_lanes_by_position(self, lanes, frame_shape):
        """按车道线在图像底部的位置排序（从左到右）
        
        参数:
            lanes: 车道线列表
            frame_shape: 图像形状
            
        返回:
            sorted_lanes: 排序后的车道线列表
        """
        if not lanes:
            return []
        
        height = frame_shape[0]
        
        # 计算每条车道线在图像底部的x坐标
        lane_positions = []
        for lane in lanes:
            points = lane.get('points', [])
            if len(points) == 2:
                x = self._get_lane_x_at_y(lane, height)
                if x is not None:
                    lane_positions.append((x, lane))
        
        # 按x坐标排序
        lane_positions.sort(key=lambda x: x[0])
        sorted_lanes = [lane for _, lane in lane_positions]
        
        return sorted_lanes
    
    def _get_lane_x_at_y(self, lane, y):
        """计算车道线在指定y坐标的x值
        
        参数:
            lane: 车道线字典
            y: y坐标
            
        返回:
            x: x坐标值
        """
        points = lane.get('points', [])
        if len(points) != 2:
            return None
        
        p1 = np.array(points[0])
        p2 = np.array(points[1])
        
        # 计算直线方程
        if p2[1] - p1[1] == 0:
            return (p1[0] + p2[0]) / 2
        
        # 计算x值
        t = (y - p1[1]) / (p2[1] - p1[1])
        x = p1[0] + t * (p2[0] - p1[0])
        
        return x
    
    def _calculate_lane_widths(self, sorted_lanes, frame_shape):
        """计算车道宽度
        
        参数:
            sorted_lanes: 排序后的车道线列表
            frame_shape: 图像形状
            
        返回:
            lane_widths: 车道宽度列表
        """
        if len(sorted_lanes) < 2:
            return []
        
        height = frame_shape[0]
        lane_widths = []
        
        # 计算相邻车道线之间的宽度
        for i in range(len(sorted_lanes) - 1):
            lane1 = sorted_lanes[i]
            lane2 = sorted_lanes[i + 1]
            
            x1 = self._get_lane_x_at_y(lane1, height)
            x2 = self._get_lane_x_at_y(lane2, height)
            
            if x1 is not None and x2 is not None:
                width = abs(x2 - x1)
                lane_widths.append(width)
        
        return lane_widths
    
    def _generate_speed_lines(self, sorted_lanes, frame_shape, avg_lane_width):
        """生成速度检测线 - 确保稳定显示4根红色速度检测线
        
        参数:
            sorted_lanes: 排序后的车道线列表
            frame_shape: 图像形状
            avg_lane_width: 平均车道宽度
            
        返回:
            speed_lines: 速度检测线列表
        """
        speed_lines = []
        
        # 强制使用 fallback 机制生成速度检测线，确保显示
        fallback_lines = self._generate_fallback_speed_lines(frame_shape)
        speed_lines.extend(fallback_lines)
        
        # 如果有车道线数据，再生成基于车道线的检测线
        if sorted_lanes and len(sorted_lanes) >= 2:
            generated_lines = self._generate_lines_for_lanes(
                sorted_lanes, frame_shape, avg_lane_width
            )
            # 优先使用基于车道线的检测线
            if generated_lines:
                speed_lines = generated_lines[:4]
                # 确保至少4根检测线
                if len(speed_lines) < 4:
                    speed_lines.extend(fallback_lines[:4 - len(speed_lines)])
        
        return speed_lines
    
    def _generate_lines_for_lanes(self, sorted_lanes, frame_shape, avg_lane_width):
        """为车道生成速度检测线
        
        参数:
            sorted_lanes: 排序后的车道线列表
            frame_shape: 图像形状
            avg_lane_width: 平均车道宽度
            
        返回:
            speed_lines: 速度检测线列表
        """
        speed_lines = []
        height, width = frame_shape[:2]
        
        # 为每条车道生成速度检测线
        for i in range(len(sorted_lanes) - 1):
            left_lane = sorted_lanes[i]
            right_lane = sorted_lanes[i + 1]
            
            # 计算车道中心线
            lane_center_x = lambda y: (self._get_lane_x_at_y(left_lane, y) + self._get_lane_x_at_y(right_lane, y)) / 2
            
            # 生成指定数量的速度检测线
            for pos in self.speed_line_positions:
                y = int(height * (1 - pos))
                
                # 计算车道中心线在当前y坐标的x值
                center_x = lane_center_x(y)
                if center_x is None:
                    continue
                
                # 计算速度检测线的左右端点
                line_length = avg_lane_width * self.speed_line_width_factor
                x1 = int(center_x - line_length / 2)
                x2 = int(center_x + line_length / 2)
                
                # 确保检测线在图像范围内
                x1 = max(0, x1)
                x2 = min(width - 1, x2)
                
                if x1 < x2:
                    speed_lines.append({
                        'points': [(x1, y), (x2, y)],
                        'type': 'SPEED',
                        'enabled': True,
                        'properties': {
                            'lane_id': i,
                            'position': pos,
                            'source': 'lane_based'
                        }
                    })
        
        return speed_lines
    
    def _generate_fallback_speed_lines(self, frame_shape):
        """生成 fallback 速度检测线 - 确保稳定显示4根红色速度检测线
        
        参数:
            frame_shape: 图像形状
            
        返回:
            speed_lines: 速度检测线列表
        """
        speed_lines = []
        height, width = frame_shape[:2]
        
        # 生成4根均匀分布的速度检测线，确保在图像中心
        positions = [0.3, 0.4, 0.6, 0.7]
        line_length = width * 0.3  # 增加线长，确保可见
        center_x = width // 2
        
        for i, pos in enumerate(positions):
            y = int(height * (1 - pos))
            x1 = int(center_x - line_length / 2)
            x2 = int(center_x + line_length / 2)
            
            # 确保检测线在图像范围内
            x1 = max(0, x1)
            x2 = min(width - 1, x2)
            
            if x1 < x2:
                speed_lines.append({
                    'points': [(x1, y), (x2, y)],
                    'type': 'SPEED',
                    'enabled': True,
                    'properties': {
                        'lane_id': i % 2,
                        'position': pos,
                        'source': 'fallback',
                        'color': (255, 0, 0)  # 明确设置为红色
                    }
                })
        
        return speed_lines
    
    def _generate_wrong_direction_lines(self, sorted_lanes):
        """生成逆向检测所需的车道线 - 确保生成左右两侧逆向检测线
        
        参数:
            sorted_lanes: 排序后的车道线列表
            
        返回:
            wrong_dir_lines: 逆向检测线字典
        """
        wrong_dir_lines = {
            'LEFT_LANE': [],
            'CENTER_LANE': [],
            'RIGHT_LANE': []
        }
        
        if not sorted_lanes:
            return wrong_dir_lines
        
        # 强制生成左侧逆向检测线
        if len(sorted_lanes) >= 1:
            left_lane = sorted_lanes[0]
            self._add_wrong_direction_line(wrong_dir_lines, 'LEFT_LANE', left_lane)
        
        # 生成中心逆向检测线
        if len(sorted_lanes) >= 2:
            center_index = len(sorted_lanes) // 2
            center_lane = sorted_lanes[center_index]
            self._add_wrong_direction_line(wrong_dir_lines, 'CENTER_LANE', center_lane)
        
        # 强制生成右侧逆向检测线
        if len(sorted_lanes) >= 1:
            right_lane = sorted_lanes[-1]
            self._add_wrong_direction_line(wrong_dir_lines, 'RIGHT_LANE', right_lane)
        
        return wrong_dir_lines
    
    def _add_wrong_direction_line(self, wrong_dir_lines, lane_type, lane):
        """添加逆向检测线并计算方向属性 - 确保生成逆向检测线
        
        参数:
            wrong_dir_lines: 逆向检测线字典
            lane_type: 车道类型
            lane: 车道线字典
        """
        points = lane.get('points', [])
        if not points:
            # 如果没有点数据，创建默认点
            points = [(0, 0), (100, 100)]
        
        # 计算车道线方向属性
        direction_properties = self._calculate_lane_direction(lane)
        
        wrong_dir_lines[lane_type].append({
            'points': points,
            'type': lane_type,
            'enabled': True,
            'properties': direction_properties
        })
    
    def _calculate_lane_direction(self, lane):
        """计算车道线方向属性
        
        参数:
            lane: 车道线字典
            
        返回:
            properties: 方向属性字典
        """
        points = lane.get('points', [])
        if len(points) < 2:
            return {'direction': 'unknown'}
        
        p1 = np.array(points[0])
        p2 = np.array(points[1])
        
        # 计算斜率
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # 计算方向角
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # 计算长度
        length = np.linalg.norm(p2 - p1)
        
        # 确定方向
        if dy < 0:  # 向上延伸
            direction = 'up'
        else:  # 向下延伸
            direction = 'down'
        
        return {
            'direction': direction,
            'angle': angle,
            'length': length,
            'slope': dx / dy if dy != 0 else float('inf'),
            'confidence': self._calculate_direction_confidence(lane)
        }
    
    def _calculate_direction_confidence(self, lane):
        """计算方向置信度
        
        参数:
            lane: 车道线字典
            
        返回:
            confidence: 置信度值 (0-1)
        """
        points = lane.get('points', [])
        if len(points) < 2:
            return 0.0
        
        p1 = np.array(points[0])
        p2 = np.array(points[1])
        
        # 基于长度和斜率计算置信度
        length = np.linalg.norm(p2 - p1)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # 长度置信度 (越长越可信)
        length_confidence = min(length / 500, 1.0)
        
        # 斜率置信度 (接近垂直车道线更可信)
        if dy != 0:
            slope_abs = abs(dx / dy)
            slope_confidence = max(0, 1 - slope_abs / 2)
        else:
            slope_confidence = 1.0
        
        # 综合置信度
        confidence = (length_confidence + slope_confidence) / 2
        
        return confidence
    
    def _generate_emergency_lane_area(self, sorted_lanes, frame_shape, avg_lane_width):
        """生成应急车道区域 - 支持双向道路，强制生成左右两侧
        
        参数:
            sorted_lanes: 排序后的车道线列表
            frame_shape: 图像形状
            avg_lane_width: 平均车道宽度
            
        返回:
            emergency_area: 应急车道区域列表
        """
        emergency_area = []
        
        if len(sorted_lanes) < 1:
            return emergency_area
        
        height, width = frame_shape[:2]
        
        # 强制生成左侧应急车道区域
        left_emergency = self._create_emergency_area(
            sorted_lanes[0], frame_shape, avg_lane_width, expand_left=True
        )
        if left_emergency:
            emergency_area.append(left_emergency)
        
        # 强制生成右侧应急车道区域
        right_emergency = self._create_emergency_area(
            sorted_lanes[-1], frame_shape, avg_lane_width, expand_left=False
        )
        if right_emergency:
            emergency_area.append(right_emergency)
        
        return emergency_area
    
    def _create_emergency_area(self, lane, frame_shape, avg_lane_width, expand_left=True):
        """创建单个应急车道区域 - 修正扩展方向
        
        参数:
            lane: 车道线字典
            frame_shape: 图像形状
            avg_lane_width: 平均车道宽度
            expand_left: 是否向左扩展
            
        返回:
            emergency_area: 应急车道区域字典
        """
        height, width = frame_shape[:2]
        
        # 获取车道线的两个点
        points = lane.get('points', [])
        if len(points) != 2:
            return None
        
        p1 = np.array(points[0])
        p2 = np.array(points[1])
        
        # 计算车道线方向向量
        lane_dir = p2 - p1
        lane_dir = lane_dir / np.linalg.norm(lane_dir) if np.linalg.norm(lane_dir) > 0 else np.array([0, 1])
        
        # 计算垂直方向向量（向右为正）
        perpendicular_dir = np.array([-lane_dir[1], lane_dir[0]])
        
        # 根据扩展方向调整（左侧应急车道向左扩展，右侧应急车道向右扩展）
        if expand_left:
            perpendicular_dir = -perpendicular_dir
        
        # 计算应急车道区域的四个点
        emergency_width = avg_lane_width * self.emergency_lane_width_factor
        
        p3 = p1 + perpendicular_dir * emergency_width
        p4 = p2 + perpendicular_dir * emergency_width
        
        # 确保点在图像范围内
        p3 = np.clip(p3, [0, 0], [width - 1, height - 1])
        p4 = np.clip(p4, [0, 0], [width - 1, height - 1])
        
        return {
            'points': [tuple(p1.astype(int)), tuple(p2.astype(int)), 
                       tuple(p4.astype(int)), tuple(p3.astype(int)), 
                       tuple(p1.astype(int))],
            'type': 'EMERGENCY_LANE',
            'enabled': True,
            'properties': {
                'side': 'left' if expand_left else 'right',
                'direction': 'outward'  # 明确标记向外扩展
            }
        }
    
    def _update_area_manager(self, area_manager, speed_lines, wrong_dir_lines, emergency_area):
        """更新区域管理器
        
        参数:
            area_manager: 区域管理器
            speed_lines: 速度检测线列表
            wrong_dir_lines: 逆向检测线字典
            emergency_area: 应急车道区域列表
        """
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
        print(f"[{self.stream_id}] Updated regions based on lane detection: {len(speed_lines)} speed lines, {len(wrong_dir_lines['LEFT_LANE']) + len(wrong_dir_lines['CENTER_LANE']) + len(wrong_dir_lines['RIGHT_LANE'])} direction lines, {len(emergency_area)} emergency areas")
