import cv2
import numpy as np


class AdaptiveEdgeDetector:
    """
    自适应边缘检测器
    根据图像亮度统计动态计算Canny阈值
    """
    
    def __init__(self, sigma_normal=0.33, sigma_low=0.50, sigma_high=0.20):
        """
        初始化自适应边缘检测器
        
        参数:
            sigma_normal: 正常光照条件下的sigma值
            sigma_low: 低光照条件下的sigma值（降低阈值提高敏感度）
            sigma_high: 高光照条件下的sigma值（提高阈值减少噪声）
        """
        self.sigma_normal = sigma_normal
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
    
    def estimate_lighting_condition(self, gray_image):
        """
        估计光照条件
        
        参数:
            gray_image: 灰度图像
            
        返回:
            condition: 'normal', 'low', 'high'
        """
        mean_brightness = np.mean(gray_image)
        
        if mean_brightness < 80:
            return 'low'
        elif mean_brightness > 200:
            return 'high'
        else:
            return 'normal'
    
    def compute_adaptive_thresholds(self, gray_image):
        """
        自适应计算Canny阈值
        
        基于图像中位数的统计方法
        
        参数:
            gray_image: 灰度图像
            
        返回:
            lower: 低阈值
            upper: 高阈值
            condition: 光照条件
        """
        # 计算中位数
        median = np.median(gray_image)
        
        # 根据光照条件选择sigma
        condition = self.estimate_lighting_condition(gray_image)
        sigma_map = {
            'normal': self.sigma_normal,
            'low': self.sigma_low,
            'high': self.sigma_high
        }
        sigma = sigma_map.get(condition, self.sigma_normal)
        
        # 计算高低阈值
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        
        return lower, upper, condition
    
    def detect(self, frame):
        """
        执行自适应边缘检测
        
        参数:
            frame: 输入BGR图像
            
        返回:
            edges: 边缘图像
            thresholds: (lower, upper, condition) 使用的阈值和光照条件
        """
        # 1. 灰度化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. 自适应阈值
        lower, upper, condition = self.compute_adaptive_thresholds(blurred)
        
        # 4. Canny边缘检测
        edges = cv2.Canny(blurred, lower, upper)
        
        return edges, (lower, upper, condition)


class ColorBasedDetector:
    """
    基于颜色的车道线检测器
    利用白色/黄色车道线的颜色特征进行分割
    """
    
    def __init__(self, white_lower=None, white_upper=None, 
                 yellow_lower=None, yellow_upper=None):
        """
        初始化颜色检测器
        
        参数:
            white_lower: 白色下限 (RGB)
            white_upper: 白色上限 (RGB)
            yellow_lower: 黄色下限 (HSV)
            yellow_upper: 黄色上限 (HSV)
        """
        self.white_lower = np.array(white_lower or [200, 200, 200])
        self.white_upper = np.array(white_upper or [255, 255, 255])
        self.yellow_lower = np.array(yellow_lower or [15, 70, 120])
        self.yellow_upper = np.array(yellow_upper or [35, 255, 255])
    
    def detect_white_lanes(self, frame):
        """
        检测白色车道线
        
        在RGB空间直接检测高亮度区域
        
        参数:
            frame: 输入BGR图像
            
        返回:
            white_mask: 白色车道线掩码
        """
        # 颜色掩码
        white_mask = cv2.inRange(frame, self.white_lower, self.white_upper)
        
        # 形态学操作连接断裂区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        return white_mask
    
    def detect_yellow_lanes(self, frame):
        """
        检测黄色车道线
        
        在HSV空间检测黄色区域（更稳定）
        
        参数:
            frame: 输入BGR图像
            
        返回:
            yellow_mask: 黄色车道线掩码
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        
        return yellow_mask
    
    def fuse_with_edges(self, edge_mask, color_mask):
        """
        融合边缘掩码和颜色掩码
        
        策略：边缘掩码与颜色掩码取交集
        
        参数:
            edge_mask: 边缘检测掩码
            color_mask: 颜色分割掩码
            
        返回:
            fused_mask: 融合后的掩码
        """
        fused_mask = cv2.bitwise_and(edge_mask, color_mask)
        return fused_mask
    
    def detect(self, frame, edge_mask=None):
        """
        执行颜色检测
        
        参数:
            frame: 输入BGR图像
            edge_mask: 可选的边缘掩码，用于融合
            
        返回:
            result: 检测结果掩码
            masks: 各颜色掩码字典
        """
        white_mask = self.detect_white_lanes(frame)
        yellow_mask = self.detect_yellow_lanes(frame)
        
        # 合并颜色掩码
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # 如果提供了边缘掩码，进行融合
        if edge_mask is not None:
            result = self.fuse_with_edges(edge_mask, color_mask)
        else:
            result = color_mask
        
        masks = {
            'white': white_mask,
            'yellow': yellow_mask,
            'combined': color_mask,
            'result': result
        }
        
        return result, masks


class HoughLineDetector:
    """
    Hough直线检测器
    封装HoughLinesP检测
    """
    
    def __init__(self, rho=1, theta=np.pi/180, threshold=80, 
                 min_line_length=80, max_line_gap=50):
        """
        初始化Hough检测器
        
        参数:
            rho: 距离分辨率
            theta: 角度分辨率
            threshold: 累加器阈值
            min_line_length: 最小线段长度
            max_line_gap: 最大线段间隙
        """
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
    
    def detect(self, edge_mask):
        """
        检测直线
        
        参数:
            edge_mask: 边缘掩码图像
            
        返回:
            lines: 检测到的线段列表，每项为 (x1, y1, x2, y2)
        """
        lines = cv2.HoughLinesP(
            edge_mask,
            self.rho,
            self.theta,
            self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return []
        
        # 转换为列表格式
        return [line[0] for line in lines]
    
    def set_parameters(self, **kwargs):
        """动态调整参数"""
        if 'rho' in kwargs:
            self.rho = kwargs['rho']
        if 'theta' in kwargs:
            self.theta = kwargs['theta']
        if 'threshold' in kwargs:
            self.threshold = kwargs['threshold']
        if 'min_line_length' in kwargs:
            self.min_line_length = kwargs['min_line_length']
        if 'max_line_gap' in kwargs:
            self.max_line_gap = kwargs['max_line_gap']


class BidirectionalSeparator:
    """
    双向车道分离器
    解决双向高速场景左右车道混淆问题
    """
    
    def __init__(self, camera_offset=0, center_margin=0.1):
        """
        初始化双向分离器
        
        参数:
            camera_offset: 相机偏移量（像素）
            center_margin: 中心线边距比例
        """
        self.camera_offset = camera_offset
        self.center_margin = center_margin
        self.vanishing_point = None
    
    def compute_dynamic_roi(self, frame_shape, vanishing_point=None):
        """
        计算动态ROI顶点
        
        基于消失点位置调整梯形区域
        
        参数:
            frame_shape: 图像形状 (height, width)
            vanishing_point: 消失点坐标 (x, y)，None则使用默认值
            
        返回:
            roi_vertices: ROI顶点列表
        """
        height, width = frame_shape[:2]
        
        if vanishing_point is None:
            # 默认消失点位置（图像中心偏上）
            vp_x, vp_y = width // 2, height // 2
        else:
            vp_x, vp_y = vanishing_point
        
        # 存储消失点供后续使用
        self.vanishing_point = (vp_x, vp_y)
        
        # 动态ROI顶点
        roi_vertices = [
            (0, height),                           # 左下角
            (int(vp_x - width * 0.15), int(vp_y)), # 左上（消失点左侧）
            (int(vp_x + width * 0.15), int(vp_y)), # 右上（消失点右侧）
            (width, height)                        # 右下角
        ]
        
        return roi_vertices
    
    def create_roi_mask(self, frame_shape, vanishing_point=None):
        """
        创建动态ROI掩码
        
        参数:
            frame_shape: 图像形状
            vanishing_point: 消失点坐标
            
        返回:
            mask: ROI掩码
        """
        height, width = frame_shape[:2]
        roi_vertices = self.compute_dynamic_roi(frame_shape, vanishing_point)
        
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, np.array([roi_vertices], np.int32), 255)
        
        return mask
    
    def separate_lanes(self, lines, frame_shape):
        """
        双向车道分离
        
        核心：结合斜率和位置信息进行分类
        
        参数:
            lines: 检测到的线段列表 [(x1, y1, x2, y2), ...]
            frame_shape: 图像形状 (height, width)
            
        返回:
            left_lanes: 左侧车道线列表
            right_lanes: 右侧车道线列表
            center_x: 中心线位置
        """
        height, width = frame_shape[:2]
        center_x = width // 2 + self.camera_offset
        margin = int(width * self.center_margin)
        
        left_lanes = []
        right_lanes = []
        center_lines = []  # 中间隔离带
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # 计算斜率
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            # 过滤水平线
            if abs(slope) < 0.2:
                continue
            
            # 过滤过于垂直的线（可能是非车道线）
            if abs(slope) > 10:
                continue
            
            # 计算底部x坐标
            if y2 != y1:
                x_at_bottom = x1 + (height - y1) * (x2 - x1) / (y2 - y1)
            else:
                x_at_bottom = (x1 + x2) / 2
            
            # 双向分离逻辑
            # 左侧车道线：底部在中心左侧，斜率为负（从左上到右下）
            if x_at_bottom < center_x - margin and slope < 0:
                left_lanes.append(line)
            # 右侧车道线：底部在中心右侧，斜率为正（从右上到左下）
            elif x_at_bottom > center_x + margin and slope > 0:
                right_lanes.append(line)
            # 中间区域：可能是隔离带或中间车道线
            elif abs(x_at_bottom - center_x) < margin:
                center_lines.append(line)
        
        return left_lanes, right_lanes, center_lines, center_x
    
    def sort_lanes(self, left_lanes, right_lanes, frame_shape):
        """
        对车道线进行排序
        
        按底部x坐标从左到右排序
        
        参数:
            left_lanes: 左侧车道线列表
            right_lanes: 右侧车道线列表
            frame_shape: 图像形状
            
        返回:
            left_sorted: 排序后的左侧车道线（从右到左，靠近中心的在前）
            right_sorted: 排序后的右侧车道线（从左到右，靠近中心的在前）
        """
        height = frame_shape[0]
        
        def get_bottom_x(line):
            x1, y1, x2, y2 = line
            if y2 != y1:
                return x1 + (height - y1) * (x2 - x1) / (y2 - y1)
            else:
                return (x1 + x2) / 2
        
        # 左侧车道线从右到左排序（靠近中心的在前）
        left_sorted = sorted(left_lanes, key=get_bottom_x, reverse=True)
        
        # 右侧车道线从左到右排序（靠近中心的在前）
        right_sorted = sorted(right_lanes, key=get_bottom_x)
        
        return left_sorted, right_sorted
    
    def process(self, lines, frame_shape):
        """
        完整的双向分离流程
        
        参数:
            lines: 检测到的线段列表
            frame_shape: 图像形状
            
        返回:
            result: 分离结果字典
        """
        # 1. 分离左右车道
        left_lanes, right_lanes, center_lines, center_x = self.separate_lanes(
            lines, frame_shape
        )
        
        # 2. 排序
        left_sorted, right_sorted = self.sort_lanes(left_lanes, right_lanes, frame_shape)
        
        return {
            'left_lanes': left_sorted,
            'right_lanes': right_sorted,
            'center_lines': center_lines,
            'center_x': center_x,
            'vanishing_point': self.vanishing_point
        }


class EdgeDetectionPipeline:
    """
    边缘检测流程整合
    整合自适应边缘检测、颜色检测、Hough变换、双向分离
    """
    
    def __init__(self, config=None):
        """
        初始化检测流程
        
        参数:
            config: 配置字典
        """
        config = config or {}
        
        self.edge_detector = AdaptiveEdgeDetector(
            sigma_normal=config.get('sigma_normal', 0.33),
            sigma_low=config.get('sigma_low', 0.50),
            sigma_high=config.get('sigma_high', 0.20)
        )
        
        self.color_detector = ColorBasedDetector(
            white_lower=config.get('white_lower'),
            white_upper=config.get('white_upper'),
            yellow_lower=config.get('yellow_lower'),
            yellow_upper=config.get('yellow_upper')
        )
        
        self.hough_detector = HoughLineDetector(
            rho=config.get('hough_rho', 1),
            theta=config.get('hough_theta', np.pi/180),
            threshold=config.get('hough_threshold', 80),
            min_line_length=config.get('hough_min_line_length', 80),
            max_line_gap=config.get('hough_max_line_gap', 50)
        )
        
        self.separator = BidirectionalSeparator(
            camera_offset=config.get('camera_offset', 0),
            center_margin=config.get('center_margin', 0.1)
        )
        
        # 功能开关
        self.enable_color_filter = config.get('enable_color_filter', True)
        self.enable_dynamic_roi = config.get('enable_dynamic_roi', True)
    
    def detect(self, frame, vanishing_point=None):
        """
        执行完整检测流程
        
        参数:
            frame: 输入图像
            vanishing_point: 消失点坐标（用于动态ROI）
            
        返回:
            result: 检测结果字典
        """
        # 1. 自适应边缘检测
        edges, (lower, upper, condition) = self.edge_detector.detect(frame)
        
        # 2. 颜色检测与融合
        if self.enable_color_filter:
            color_mask, color_masks = self.color_detector.detect(frame)
            combined_edges = cv2.bitwise_and(edges, color_mask)
        else:
            combined_edges = edges
            color_masks = {}
        
        # 3. 应用动态ROI
        if self.enable_dynamic_roi:
            roi_mask = self.separator.create_roi_mask(frame.shape, vanishing_point)
            combined_edges = cv2.bitwise_and(combined_edges, roi_mask)
        
        # 4. Hough直线检测
        lines = self.hough_detector.detect(combined_edges)
        
        # 5. 双向分离
        separation_result = self.separator.process(lines, frame.shape)
        
        return {
            'edges': edges,
            'combined_edges': combined_edges,
            'color_masks': color_masks,
            'lines': lines,
            'separation': separation_result,
            'thresholds': {
                'lower': lower,
                'upper': upper,
                'condition': condition
            }
        }
    
    def set_parameters(self, **kwargs):
        """动态调整参数"""
        self.hough_detector.set_parameters(**kwargs)
        
        if 'camera_offset' in kwargs:
            self.separator.camera_offset = kwargs['camera_offset']
        if 'center_margin' in kwargs:
            self.separator.center_margin = kwargs['center_margin']
