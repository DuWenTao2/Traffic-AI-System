import cv2
import numpy as np
import random
import warnings


class LineFitter:
    """
    直线拟合器
    使用最小二乘法进行线性拟合
    """
    
    def fit(self, points):
        """
        线性拟合
        使用y作为自变量避免垂直线问题
        
        参数:
            points: 点列表 [(x, y), ...]
            
        返回:
            coefficients: 拟合系数 [a, b]，表示 x = a*y + b
            residuals: 残差
        """
        if len(points) < 2:
            return None, float('inf')
        
        y_coords = [p[1] for p in points]
        x_coords = [p[0] for p in points]
        
        # 一次多项式拟合：x = a*y + b
        # Suppress RankWarning for poorly conditioned matrix

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            coeffs = np.polyfit(y_coords, x_coords, 1)
        
        # 计算残差
        predicted = np.polyval(coeffs, y_coords)
        residuals = np.mean((np.array(x_coords) - predicted) ** 2)
        
        return coeffs, residuals
    
    def get_line_points(self, coeffs, y_range):
        """
        根据拟合系数获取线段端点
        
        参数:
            coeffs: 拟合系数 [a, b]
            y_range: y坐标范围 (y_min, y_max)
            
        返回:
            points: 线段端点 [(x1, y1), (x2, y2)]
        """
        y_min, y_max = y_range
        x_min = int(np.polyval(coeffs, y_min))
        x_max = int(np.polyval(coeffs, y_max))
        
        return [(x_min, int(y_min)), (x_max, int(y_max))]


class CurveFitter:
    """
    曲线拟合器
    使用二次多项式拟合支持弯道检测
    """
    
    def fit(self, points):
        """
        二次曲线拟合
        模型：x = a*y^2 + b*y + c
        
        参数:
            points: 点列表 [(x, y), ...]
            
        返回:
            coefficients: 拟合系数 [a, b, c]
            residuals: 残差
        """
        if len(points) < 3:
            # 点数不足，退化为直线拟合
            line_fitter = LineFitter()
            return line_fitter.fit(points)
        
        y_coords = [p[1] for p in points]
        x_coords = [p[0] for p in points]
        
        # 二次多项式拟合
        # Suppress RankWarning for poorly conditioned matrix
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            coeffs = np.polyfit(y_coords, x_coords, 2)
        
        # 计算残差
        predicted = np.polyval(coeffs, y_coords)
        residuals = np.mean((np.array(x_coords) - predicted) ** 2)
        
        return coeffs, residuals
    
    def get_curve_points(self, coeffs, y_range, num_points=20):
        """
        根据拟合系数获取曲线点序列
        
        参数:
            coeffs: 拟合系数 [a, b, c]
            y_range: y坐标范围 (y_min, y_max)
            num_points: 采样点数
            
        返回:
            points: 曲线点列表 [(x, y), ...]
        """
        y_min, y_max = y_range
        y_values = np.linspace(y_min, y_max, num_points)
        x_values = np.polyval(coeffs, y_values)
        
        points = [(int(x), int(y)) for x, y in zip(x_values, y_values)]
        return points


class RANSACFitter:
    """
    RANSAC鲁棒拟合器
    去除离群点影响
    """
    
    def __init__(self, threshold=5.0, max_iterations=100, min_inliers_ratio=0.5):
        """
        初始化RANSAC拟合器
        
        参数:
            threshold: 内点阈值（像素）
            max_iterations: 最大迭代次数
            min_inliers_ratio: 最小内点比例
        """
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.min_inliers_ratio = min_inliers_ratio
    
    def fit(self, points, model_type='line'):
        """
        RANSAC鲁棒拟合
        
        参数:
            points: 点列表 [(x, y), ...]
            model_type: 'line' 或 'curve'
            
        返回:
            best_model: 最佳模型系数
            inliers: 内点列表
        """
        if len(points) < 3:
            fitter = LineFitter()
            coeffs, _ = fitter.fit(points)
            return coeffs, points
        
        best_inliers = []
        best_model = None
        
        for _ in range(self.max_iterations):
            # 随机采样
            sample_size = 2 if model_type == 'line' else 3
            if len(points) < sample_size:
                continue
            
            sample = random.sample(points, sample_size)
            
            # 拟合模型
            if model_type == 'curve':
                fitter = CurveFitter()
            else:
                fitter = LineFitter()
            
            model, _ = fitter.fit(sample)
            if model is None:
                continue
            
            # 计算内点
            inliers = []
            for p in points:
                y, x = p[1], p[0]
                predicted = np.polyval(model, y)
                
                if abs(x - predicted) < self.threshold:
                    inliers.append(p)
            
            # 更新最佳模型
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = model
        
        # 用所有内点重新拟合
        if len(best_inliers) >= 3 and model_type == 'curve':
            fitter = CurveFitter()
            best_model, _ = fitter.fit(best_inliers)
        elif len(best_inliers) >= 2:
            fitter = LineFitter()
            best_model, _ = fitter.fit(best_inliers)
        
        return best_model, best_inliers


class AdaptiveFitter:
    """
    自适应拟合器
    根据残差自动选择直线或曲线拟合
    """
    
    def __init__(self, curve_threshold=0.7):
        """
        初始化自适应拟合器
        
        参数:
            curve_threshold: 曲线拟合阈值（曲线残差/直线残差）
        """
        self.curve_threshold = curve_threshold
    
    def fit(self, points):
        """
        自适应拟合
        
        参数:
            points: 点列表 [(x, y), ...]
            
        返回:
            coeffs: 拟合系数
            model_type: 'line' 或 'curve'
            residuals: 残差
        """
        if len(points) < 5:
            # 点数不足，使用直线拟合
            fitter = LineFitter()
            coeffs, residuals = fitter.fit(points)
            return coeffs, 'line', residuals
        
        # 尝试直线拟合
        line_fitter = LineFitter()
        line_coeffs, line_residual = line_fitter.fit(points)
        
        # 尝试曲线拟合
        curve_fitter = CurveFitter()
        curve_coeffs, curve_residual = curve_fitter.fit(points)
        
        # 选择残差小的模型
        if curve_residual < line_residual * self.curve_threshold:
            return curve_coeffs, 'curve', curve_residual
        else:
            return line_coeffs, 'line', line_residual


class GeometryValidator:
    """
    几何约束验证器
    确保检测结果符合道路物理特性
    """
    
    def __init__(self, parallel_threshold=15, width_ratio_range=(0.7, 1.4), 
                 vanishing_tolerance=50):
        """
        初始化几何验证器
        
        参数:
            parallel_threshold: 平行性角度阈值（度）
            width_ratio_range: 车道宽度比例范围
            vanishing_tolerance: 消失点偏差容忍度（像素）
        """
        self.parallel_threshold = parallel_threshold
        self.width_ratio_range = width_ratio_range
        self.vanishing_tolerance = vanishing_tolerance
    
    def check_parallelism(self, lanes):
        """
        检查车道线平行性
        
        参数:
            lanes: 车道线列表，每项包含 'points' 键
            
        返回:
            is_valid: 是否通过验证
            score: 平行性得分（0-1）
        """
        if len(lanes) < 2:
            return True, 1.0
        
        angles = []
        for lane in lanes:
            p1, p2 = lane['points']
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi
            angles.append(angle)
        
        # 计算角度差异
        angle_diffs = []
        for i in range(len(angles)):
            for j in range(i+1, len(angles)):
                diff = abs(angles[i] - angles[j])
                # 归一化到0-180
                diff = min(diff, 360 - diff)
                angle_diffs.append(diff)
        
        max_diff = max(angle_diffs) if angle_diffs else 0
        score = max(0, 1 - max_diff / self.parallel_threshold)
        
        return max_diff < self.parallel_threshold, score
    
    def check_equal_width(self, lanes, frame_shape):
        """
        检查车道等宽性
        
        参数:
            lanes: 车道线列表
            frame_shape: 图像形状
            
        返回:
            is_valid: 是否通过验证
            score: 等宽性得分
        """
        if len(lanes) < 2:
            return True, 1.0
        
        height = frame_shape[0]
        widths = []
        
        # 计算相邻车道线间距（图像底部）
        for i in range(len(lanes) - 1):
            x1 = self._get_x_at_y(lanes[i], height)
            x2 = self._get_x_at_y(lanes[i+1], height)
            if x1 is not None and x2 is not None:
                widths.append(abs(x2 - x1))
        
        if len(widths) < 2:
            return True, 1.0
        
        # 检查宽度比例
        avg_width = np.mean(widths)
        ratios = [w / avg_width for w in widths]
        
        valid_ratios = [r for r in ratios 
                       if self.width_ratio_range[0] <= r <= self.width_ratio_range[1]]
        score = len(valid_ratios) / len(ratios)
        
        return score > 0.8, score
    
    def check_convergence(self, lanes, vanishing_point):
        """
        检查车道线汇聚性
        
        参数:
            lanes: 车道线列表
            vanishing_point: 消失点坐标 (x, y)
            
        返回:
            is_valid: 是否通过验证
            score: 汇聚性得分
        """
        if len(lanes) < 2 or vanishing_point is None:
            return True, 1.0
        
        intersections = []
        for i in range(len(lanes)):
            for j in range(i+1, len(lanes)):
                intersect = self._line_intersection(
                    lanes[i]['points'][0], lanes[i]['points'][1],
                    lanes[j]['points'][0], lanes[j]['points'][1]
                )
                if intersect is not None:
                    intersections.append(intersect)
        
        if not intersections:
            return False, 0.0
        
        # 计算与消失点的偏差
        vp_x, vp_y = vanishing_point
        deviations = [np.sqrt((ix-vp_x)**2 + (iy-vp_y)**2) 
                     for ix, iy in intersections]
        avg_deviation = np.mean(deviations)
        
        score = max(0, 1 - avg_deviation / self.vanishing_tolerance)
        
        return avg_deviation < self.vanishing_tolerance, score
    
    def validate(self, lanes, frame_shape, vanishing_point):
        """
        综合几何约束验证
        
        参数:
            lanes: 车道线列表
            frame_shape: 图像形状
            vanishing_point: 消失点坐标
            
        返回:
            is_valid: 是否通过验证
            total_score: 综合得分
            details: 各项得分详情
        """
        scores = {}
        
        # 平行性
        _, parallel_score = self.check_parallelism(lanes)
        scores['parallelism'] = parallel_score
        
        # 等宽性
        _, width_score = self.check_equal_width(lanes, frame_shape)
        scores['equal_width'] = width_score
        
        # 汇聚性
        _, converge_score = self.check_convergence(lanes, vanishing_point)
        scores['convergence'] = converge_score
        
        # 综合评分
        total_score = np.mean(list(scores.values()))
        
        return total_score >= 0.6, total_score, scores
    
    def _get_x_at_y(self, lane, y):
        """获取车道线在指定y坐标的x值"""
        points = lane.get('points', [])
        if len(points) != 2:
            return None
        
        p1, p2 = points
        x1, y1 = p1
        x2, y2 = p2
        
        if y2 == y1:
            return (x1 + x2) / 2
        
        t = (y - y1) / (y2 - y1)
        x = x1 + t * (x2 - x1)
        return x
    
    def _line_intersection(self, p1, p2, p3, p4):
        """计算两直线交点"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if denom == 0:
            return None
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        
        x = x1 + t*(x2-x1)
        y = y1 + t*(y2-y1)
        
        return (x, y)


class LaneFittingPipeline:
    """
    车道线拟合流程整合
    整合RANSAC拟合、自适应拟合选择、几何约束验证
    """
    
    def __init__(self, config=None):
        """
        初始化拟合流程
        
        参数:
            config: 配置字典
        """
        config = config or {}
        
        self.ransac_fitter = RANSACFitter(
            threshold=config.get('ransac_threshold', 5.0),
            max_iterations=config.get('ransac_max_iterations', 100)
        )
        
        self.adaptive_fitter = AdaptiveFitter(
            curve_threshold=config.get('curve_threshold', 0.7)
        )
        
        self.validator = GeometryValidator(
            parallel_threshold=config.get('parallel_threshold', 15),
            width_ratio_range=tuple(config.get('width_ratio_range', [0.7, 1.4])),
            vanishing_tolerance=config.get('vanishing_tolerance', 50)
        )
        
        # 车道线类型定义
        self.LANE_TYPE_LEFT = "left"
        self.LANE_TYPE_RIGHT = "right"
        self.LANE_TYPE_MIDDLE = "middle"
        self.LANE_TYPE_EMERGENCY = "emergency"
    
    def fit_lane_group(self, lines, frame_shape, lane_type):
        """
        拟合一组车道线
        
        参数:
            lines: 线段列表 [(x1, y1, x2, y2), ...]
            frame_shape: 图像形状
            lane_type: 车道线类型
            
        返回:
            lane: 拟合后的车道线字典
        """
        if not lines:
            return None
        
        height = frame_shape[0]
        
        # 收集所有点
        points = []
        for line in lines:
            x1, y1, x2, y2 = line
            points.append((x1, y1))
            points.append((x2, y2))
        
        # RANSAC鲁棒拟合
        coeffs, inliers = self.ransac_fitter.fit(points, model_type='line')
        
        if coeffs is None:
            return None
        
        # 使用自适应拟合检查是否需要曲线
        if len(inliers) >= 5:
            coeffs, model_type, _ = self.adaptive_fitter.fit(inliers)
        else:
            model_type = 'line'
        
        # 生成车道线点
        if model_type == 'curve' and len(coeffs) == 3:
            curve_fitter = CurveFitter()
            lane_points = curve_fitter.get_curve_points(coeffs, (height//2, height))
        else:
            line_fitter = LineFitter()
            lane_points = line_fitter.get_line_points(coeffs, (height//2, height))
        
        return {
            'points': [lane_points[0], lane_points[-1]],
            'coefficients': coeffs.tolist() if hasattr(coeffs, 'tolist') else coeffs,
            'model_type': model_type,
            'type': lane_type,
            'inliers': len(inliers)
        }
    
    def fit_all_lanes(self, separation_result, frame_shape):
        """
        拟合所有车道线
        
        参数:
            separation_result: 双向分离结果
            frame_shape: 图像形状
            
        返回:
            lanes: 拟合后的车道线列表
            vanishing_point: 计算的消失点
        """
        lanes = []
        
        # 拟合左侧车道线
        left_lanes = separation_result.get('left_lanes', [])
        if left_lanes:
            left_lane = self.fit_lane_group(left_lanes, frame_shape, self.LANE_TYPE_LEFT)
            if left_lane:
                lanes.append(left_lane)
        
        # 拟合右侧车道线
        right_lanes = separation_result.get('right_lanes', [])
        if right_lanes:
            right_lane = self.fit_lane_group(right_lanes, frame_shape, self.LANE_TYPE_RIGHT)
            if right_lane:
                lanes.append(right_lane)
        
        # 拟合中间车道线
        center_lines = separation_result.get('center_lines', [])
        if center_lines:
            center_lane = self.fit_lane_group(center_lines, frame_shape, self.LANE_TYPE_MIDDLE)
            if center_lane:
                lanes.append(center_lane)
        
        # 计算消失点
        vanishing_point = self._calculate_vanishing_point(lanes)
        
        # 几何约束验证
        if lanes and vanishing_point:
            is_valid, score, details = self.validator.validate(
                lanes, frame_shape, vanishing_point
            )
            
            # 为每条车道线添加置信度
            for lane in lanes:
                lane['confidence'] = score
                lane['geometry_score'] = details
        
        return lanes, vanishing_point
    
    def _calculate_vanishing_point(self, lanes):
        """计算消失点"""
        if len(lanes) < 2:
            return None
        
        intersections = []
        for i in range(len(lanes)):
            for j in range(i+1, len(lanes)):
                p1 = lanes[i]['points'][0]
                p2 = lanes[i]['points'][1]
                p3 = lanes[j]['points'][0]
                p4 = lanes[j]['points'][1]
                
                intersect = self.validator._line_intersection(p1, p2, p3, p4)
                if intersect is not None:
                    intersections.append(intersect)
        
        if intersections:
            avg_x = np.mean([p[0] for p in intersections])
            avg_y = np.mean([p[1] for p in intersections])
            return (avg_x, avg_y)
        
        return None
    
    def identify_emergency_lane(self, lanes, frame_shape):
        """
        识别应急车道
        
        策略：最左侧或最右侧的车道线
        """
        if not lanes:
            return None
        
        height = frame_shape[0]
        width = frame_shape[1]
        
        # 按底部x坐标排序
        def get_bottom_x(lane):
            return lane['points'][0][0]
        
        sorted_lanes = sorted(lanes, key=get_bottom_x)
        
        # 最左侧车道线
        leftmost = sorted_lanes[0]
        leftmost_x = get_bottom_x(leftmost)
        
        # 如果靠近左边缘，标记为应急车道
        if leftmost_x < width * 0.15:
            leftmost['type'] = self.LANE_TYPE_EMERGENCY
            return leftmost
        
        # 最右侧车道线
        rightmost = sorted_lanes[-1]
        rightmost_x = get_bottom_x(rightmost)
        
        # 如果靠近右边缘，标记为应急车道
        if rightmost_x > width * 0.85:
            rightmost['type'] = self.LANE_TYPE_EMERGENCY
            return rightmost
        
        return None
