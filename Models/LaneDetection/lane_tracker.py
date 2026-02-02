import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanLaneTracker:
    """
    基于卡尔曼滤波的车道线跟踪器
    为每条车道线维护一个卡尔曼滤波器
    """
    
    def __init__(self, lane_id, initial_state, process_noise=0.01, measurement_noise=0.1):
        """
        初始化卡尔曼跟踪器
        
        参数:
            lane_id: 车道线ID
            initial_state: 初始状态 [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
            process_noise: 过程噪声协方差
            measurement_noise: 观测噪声协方差
        """
        self.lane_id = lane_id
        self.kf = cv2.KalmanFilter(8, 4)
        
        # 状态转移矩阵（匀速模型）
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 观测矩阵
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # 过程噪声协方差
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * process_noise
        
        # 观测噪声协方差
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * measurement_noise
        
        # 初始状态
        self.kf.statePost = np.array(initial_state, dtype=np.float32).reshape(8, 1)
        
        # 跟踪状态
        self.missed_frames = 0
        self.confidence = 1.0
        self.history = []
    
    def predict(self):
        """
        预测下一状态
        
        返回:
            prediction: 预测状态 [x1, y1, x2, y2]
        """
        prediction = self.kf.predict()
        return prediction.flatten()[:4]
    
    def update(self, measurement):
        """
        更新状态
        
        参数:
            measurement: 观测值 [x1, y1, x2, y2]
            
        返回:
            updated: 更新后的状态 [x1, y1, x2, y2]
        """
        measurement = np.array(measurement, dtype=np.float32).reshape(4, 1)
        self.kf.correct(measurement)
        return self.kf.statePost.flatten()[:4]
    
    def get_state(self):
        """获取当前状态"""
        return self.kf.statePost.flatten()[:4]
    
    def get_lane_points(self):
        """获取车道线端点"""
        state = self.get_state()
        return [(int(state[0]), int(state[1])), (int(state[2]), int(state[3]))]


class LaneAssociator:
    """
    车道线关联器
    使用匈牙利算法关联预测和检测结果
    """
    
    def __init__(self, max_distance=100, max_angle_diff=30):
        """
        初始化关联器
        
        参数:
            max_distance: 最大关联距离（像素）
            max_angle_diff: 最大角度差（度）
        """
        self.max_distance = max_distance
        self.max_angle_diff = max_angle_diff
    
    def compute_cost(self, prediction, detection):
        """
        计算关联代价
        
        参数:
            prediction: 预测状态 [x1, y1, x2, y2]
            detection: 检测结果字典
            
        返回:
            cost: 关联代价
        """
        det_points = detection.get('points', [])
        if len(det_points) != 2:
            return float('inf')
        
        # 位置代价（端点距离的平均）
        pred_points = [(prediction[0], prediction[1]), (prediction[2], prediction[3])]
        pos_cost = 0
        for pp, dp in zip(pred_points, det_points):
            pos_cost += np.sqrt((pp[0]-dp[0])**2 + (pp[1]-dp[1])**2)
        pos_cost /= 2
        
        # 角度代价
        pred_angle = np.arctan2(prediction[3]-prediction[1], prediction[2]-prediction[0]) * 180 / np.pi
        det_angle = np.arctan2(det_points[1][1]-det_points[0][1], 
                               det_points[1][0]-det_points[0][0]) * 180 / np.pi
        angle_diff = abs(pred_angle - det_angle)
        angle_diff = min(angle_diff, 360 - angle_diff)
        angle_cost = angle_diff * 5  # 角度权重
        
        # 总代价
        total_cost = pos_cost + angle_cost
        
        # 如果超过最大距离，返回无穷大
        if pos_cost > self.max_distance or angle_diff > self.max_angle_diff:
            return float('inf')
        
        return total_cost
    
    def associate(self, predictions, detections):
        """
        关联预测和检测（分层匹配策略）
        
        参数:
            predictions: 预测列表 [(tracker_id, prediction), ...]
            detections: 检测结果列表
            
        返回:
            matches: 匹配列表 [(tracker_id, detection_index), ...]
            unmatched_trackers: 未匹配的跟踪器ID列表
            unmatched_detections: 未匹配的检测索引列表
        """
        if not predictions or not detections:
            return [], [p[0] for p in predictions], list(range(len(detections)))
        
        # 构建代价矩阵
        cost_matrix = np.zeros((len(predictions), len(detections)))
        
        for i, (tracker_id, prediction) in enumerate(predictions):
            for j, detection in enumerate(detections):
                cost_matrix[i, j] = self.compute_cost(prediction, detection)
        
        # 尝试使用匈牙利算法
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 过滤高代价匹配
            matches = []
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < float('inf'):
                    matches.append((predictions[r][0], c))
                    
            # 精细化验证匹配结果
            refined_matches = self.refine_matches(matches, predictions, detections)
                    
            # 找出未匹配的
            matched_trackers = [m[0] for m in refined_matches]
            matched_detections = [m[1] for m in refined_matches]
            
            unmatched_trackers = [p[0] for p in predictions if p[0] not in matched_trackers]
            unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
            
            return refined_matches, unmatched_trackers, unmatched_detections
        except ValueError as e:
            if "infeasible" in str(e):
                # 如果匈牙利算法失败，使用贪心算法作为备选
                print(f"Hungarian algorithm failed, using greedy assignment")
                greedy_matches, unmatched_trackers, unmatched_detections = self.greedy_assignment(predictions, detections)
                
                # 对贪心算法的结果也进行精细化验证
                refined_matches = self.refine_matches(greedy_matches, predictions, detections)
                
                # 更新未匹配的项目
                matched_trackers = [m[0] for m in refined_matches]
                matched_detections = [m[1] for m in refined_matches]
                
                unmatched_trackers = [p[0] for p in predictions if p[0] not in matched_trackers]
                unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
                
                return refined_matches, unmatched_trackers, unmatched_detections
            else:
                raise
    
    def greedy_assignment(self, predictions, detections):
        """
        使用贪心算法进行匹配
        
        参数:
            predictions: 预测列表 [(tracker_id, prediction), ...]
            detections: 检测结果列表
            
        返回:
            matches: 匹配列表 [(tracker_id, detection_index), ...]
            unmatched_trackers: 未匹配的跟踪器ID列表
            unmatched_detections: 未匹配的检测索引列表
        """
        matches = []
        used_detections = set()
        
        # 按照预测的置信度排序（如果有的话），或者按顺序处理
        for i, (tracker_id, prediction) in enumerate(predictions):
            min_cost = float('inf')
            best_match = -1
            
            for j, detection in enumerate(detections):
                if j in used_detections:
                    continue
                    
                cost = self.compute_cost(prediction, detection)
                if cost < min_cost and cost < self.max_distance * 2:  # 放宽阈值
                    min_cost = cost
                    best_match = j
                    
            if best_match != -1:
                matches.append((tracker_id, best_match))
                used_detections.add(best_match)
        
        # 找出未匹配的
        matched_trackers = [m[0] for m in matches]
        matched_detections = [m[1] for m in matches]
        
        unmatched_trackers = [p[0] for p in predictions if p[0] not in matched_trackers]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
        
        return matches, unmatched_trackers, unmatched_detections

    def refine_matches(self, matches, predictions, detections):
        """
        对初步匹配结果进行精细化验证和优化
        
        参数:
            matches: 初步匹配结果 [(tracker_id, detection_index), ...]
            predictions: 预测列表
            detections: 检测结果列表
            
        返回:
            refined_matches: 优化后的匹配结果
        """
        refined_matches = []
        
        for tracker_id, detection_idx in matches:
            # 获取对应的预测和检测
            prediction = next((pred for tid, pred in predictions if tid == tracker_id), None)
            detection = detections[detection_idx] if 0 <= detection_idx < len(detections) else None
            
            if prediction is None or detection is None:
                continue
            
            # 重新计算匹配质量
            cost = self.compute_cost(prediction, detection)
            
            # 根据成本判断匹配质量，如果质量足够好则保留
            if cost < self.max_distance * 1.5:  # 略高于粗匹配的阈值
                refined_matches.append((tracker_id, detection_idx))
        
        return refined_matches


class ConfidenceEvaluator:
    """
    置信度评估器
    多维度评估检测可靠性
    """
    
    def __init__(self, weights=None):
        """
        初始化评估器
        
        参数:
            weights: 各维度权重字典
        """
        self.weights = weights or {
            'edge': 0.3,
            'color': 0.2,
            'geometry': 0.3,
            'temporal': 0.2
        }
    
    def evaluate_edge_quality(self, lane, edge_image):
        """
        评估边缘质量
        
        参数:
            lane: 车道线字典
            edge_image: 边缘图像
            
        返回:
            score: 边缘质量得分（0-1）
        """
        points = lane.get('points', [])
        if len(points) != 2:
            return 0.0
        
        # 在车道线附近采样点
        p1, p2 = points
        num_samples = 20
        scores = []
        
        for i in range(num_samples):
            t = i / (num_samples - 1)
            x = int(p1[0] + t * (p2[0] - p1[0]))
            y = int(p1[1] + t * (p2[1] - p1[1]))
            
            # 检查边缘强度
            if 0 <= x < edge_image.shape[1] and 0 <= y < edge_image.shape[0]:
                # 在点附近采样小区域
                roi = edge_image[max(0, y-2):min(edge_image.shape[0], y+3),
                                max(0, x-2):min(edge_image.shape[1], x+3)]
                scores.append(np.mean(roi) / 255.0)
        
        return np.mean(scores) if scores else 0.0
    
    def evaluate_color_consistency(self, lane, frame):
        """
        评估颜色一致性
        
        参数:
            lane: 车道线字典
            frame: 原始图像
            
        返回:
            score: 颜色一致性得分（0-1）
        """
        # 简化的颜色评估：检查车道线区域亮度
        points = lane.get('points', [])
        if len(points) != 2:
            return 0.5
        
        # 转换到灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 在车道线附近采样亮度
        p1, p2 = points
        brightness_values = []
        
        for i in range(10):
            t = i / 9
            x = int(p1[0] + t * (p2[0] - p1[0]))
            y = int(p1[1] + t * (p2[1] - p1[1]))
            
            if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                brightness_values.append(gray[y, x])
        
        if not brightness_values:
            return 0.5
        
        # 车道线应该较亮（白色/黄色）
        avg_brightness = np.mean(brightness_values)
        score = min(avg_brightness / 200.0, 1.0)
        
        return score
    
    def evaluate_geometry_consistency(self, lane, all_lanes):
        """
        评估几何一致性
        
        参数:
            lane: 当前车道线
            all_lanes: 所有车道线列表
            
        返回:
            score: 几何一致性得分（0-1）
        """
        # 使用已有的几何得分
        geometry_score = lane.get('geometry_score', {})
        if geometry_score:
            return np.mean(list(geometry_score.values()))
        return 0.5
    
    def evaluate_temporal_consistency(self, lane, history):
        """
        评估时序一致性
        
        参数:
            lane: 当前车道线
            history: 历史状态列表
            
        返回:
            score: 时序一致性得分（0-1）
        """
        if not history or len(history) < 2:
            return 1.0
        
        # 计算与历史位置的偏差
        current_points = lane.get('points', [])
        if len(current_points) != 2:
            return 0.5
        
        # 与最近历史比较
        last_state = history[-1]
        deviations = []
        
        for cp, lp in zip(current_points, last_state):
            dev = np.sqrt((cp[0]-lp[0])**2 + (cp[1]-lp[1])**2)
            deviations.append(dev)
        
        avg_deviation = np.mean(deviations)
        
        # 偏差越小得分越高
        score = max(0, 1 - avg_deviation / 50.0)
        
        return score
    
    def compute_confidence(self, lane, frame, edge_image, all_lanes, history):
        """
        计算综合置信度
        
        参数:
            lane: 车道线字典
            frame: 原始图像
            edge_image: 边缘图像
            all_lanes: 所有车道线列表
            history: 历史状态列表
            
        返回:
            total_score: 综合置信度
            details: 各维度得分详情
        """
        scores = {}
        
        # 边缘质量
        scores['edge'] = self.evaluate_edge_quality(lane, edge_image)
        
        # 颜色一致性
        scores['color'] = self.evaluate_color_consistency(lane, frame)
        
        # 几何一致性
        scores['geometry'] = self.evaluate_geometry_consistency(lane, all_lanes)
        
        # 时序一致性
        scores['temporal'] = self.evaluate_temporal_consistency(lane, history)
        
        # 加权综合
        total_score = sum(scores[k] * self.weights[k] for k in scores)
        
        return total_score, scores


class LaneTracker:
    """
    车道线跟踪器主类
    整合卡尔曼滤波、匈牙利关联、置信度评估
    """
    
    def __init__(self, config=None):
        """
        初始化跟踪器
        
        参数:
            config: 配置字典
        """
        config = config or {}
        
        self.trackers = {}  # lane_id -> KalmanLaneTracker
        self.next_id = 0
        self.max_missed = config.get('max_missed_frames', 5)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        
        self.associator = LaneAssociator(
            max_distance=config.get('max_association_distance', 100),
            max_angle_diff=config.get('max_angle_diff', 30)
        )
        
        self.confidence_evaluator = ConfidenceEvaluator(
            weights=config.get('confidence_weights')
        )
        
        # 历史记录
        self.lane_history = {}
        self.vanishing_point = None
    
    def _lane_to_state(self, lane):
        """将车道线转换为状态向量"""
        points = lane.get('points', [])
        if len(points) != 2:
            return None
        
        p1, p2 = points
        # 状态: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        return [p1[0], p1[1], p2[0], p2[1], 0, 0, 0, 0]
    
    def _state_to_lane(self, state, lane_type, lane_id):
        """将状态向量转换为车道线"""
        return {
            'points': [(int(state[0]), int(state[1])), (int(state[2]), int(state[3]))],
            'type': lane_type,
            'id': lane_id,
            'tracked': True
        }
    
    def track(self, detected_lanes, frame, edge_image):
        """
        跟踪车道线
        
        参数:
            detected_lanes: 检测到的车道线列表
            frame: 原始图像
            edge_image: 边缘图像
            
        返回:
            tracked_lanes: 跟踪后的车道线列表
            vanishing_point: 消失点
        """
        # 1. 获取所有预测
        predictions = []
        for lane_id, tracker in self.trackers.items():
            prediction = tracker.predict()
            predictions.append((lane_id, prediction))
        
        # 2. 匈牙利关联
        matches, unmatched_trackers, unmatched_detections = self.associator.associate(
            predictions, detected_lanes
        )
        
        # 3. 更新匹配的跟踪器
        tracked_lanes = []
        
        for lane_id, det_idx in matches:
            tracker = self.trackers[lane_id]
            detection = detected_lanes[det_idx]
            
            # 更新卡尔曼滤波器
            measurement = self._lane_to_state(detection)
            if measurement:
                tracker.update(measurement[:4])
                tracker.missed_frames = 0
                
                # 更新历史
                if lane_id not in self.lane_history:
                    self.lane_history[lane_id] = []
                self.lane_history[lane_id].append(tracker.get_lane_points())
                if len(self.lane_history[lane_id]) > 10:
                    self.lane_history[lane_id].pop(0)
                
                # 创建跟踪结果
                lane = self._state_to_lane(
                    tracker.get_state(),
                    detection.get('type', 'unknown'),
                    lane_id
                )
                
                # 计算置信度
                confidence, conf_details = self.confidence_evaluator.compute_confidence(
                    lane, frame, edge_image, detected_lanes, self.lane_history[lane_id]
                )
                lane['confidence'] = confidence
                lane['confidence_details'] = conf_details
                
                tracked_lanes.append(lane)
        
        # 4. 处理未匹配的跟踪器（使用预测值）
        for lane_id in unmatched_trackers:
            tracker = self.trackers[lane_id]
            tracker.missed_frames += 1
            
            if tracker.missed_frames <= self.max_missed:
                # 使用预测值
                lane = self._state_to_lane(
                    tracker.get_state(),
                    'unknown',
                    lane_id
                )
                lane['predicted'] = True
                lane['confidence'] = max(0.3, 1.0 - tracker.missed_frames * 0.2)
                tracked_lanes.append(lane)
            else:
                # 删除丢失的跟踪器
                del self.trackers[lane_id]
                if lane_id in self.lane_history:
                    del self.lane_history[lane_id]
        
        # 5. 处理未匹配的检测（创建新跟踪器）
        for det_idx in unmatched_detections:
            detection = detected_lanes[det_idx]
            state = self._lane_to_state(detection)
            
            if state:
                # 创建新跟踪器
                tracker = KalmanLaneTracker(
                    self.next_id,
                    state,
                    process_noise=0.01,
                    measurement_noise=0.1
                )
                self.trackers[self.next_id] = tracker
                
                lane = self._state_to_lane(
                    tracker.get_state(),
                    detection.get('type', 'unknown'),
                    self.next_id
                )
                lane['confidence'] = detection.get('confidence', 0.5)
                tracked_lanes.append(lane)
                
                self.next_id += 1
        
        # 6. 计算消失点
        self.vanishing_point = self._calculate_vanishing_point(tracked_lanes)
        
        return tracked_lanes, self.vanishing_point
    
    def _calculate_vanishing_point(self, lanes):
        """计算消失点"""
        if len(lanes) < 2:
            return self.vanishing_point
        
        intersections = []
        for i in range(len(lanes)):
            for j in range(i+1, len(lanes)):
                p1 = lanes[i]['points'][0]
                p2 = lanes[i]['points'][1]
                p3 = lanes[j]['points'][0]
                p4 = lanes[j]['points'][1]
                
                # 计算交点
                denom = (p1[0]-p2[0])*(p3[1]-p4[1]) - (p1[1]-p2[1])*(p3[0]-p4[0])
                if denom == 0:
                    continue
                
                t = ((p1[0]-p3[0])*(p3[1]-p4[1]) - (p1[1]-p3[1])*(p3[0]-p4[0])) / denom
                x = p1[0] + t*(p2[0]-p1[0])
                y = p1[1] + t*(p2[1]-p1[1])
                
                intersections.append((x, y))
        
        if intersections:
            avg_x = np.mean([p[0] for p in intersections])
            avg_y = np.mean([p[1] for p in intersections])
            return (avg_x, avg_y)
        
        return self.vanishing_point
    
    def get_tracked_lanes(self, min_confidence=None):
        """
        获取跟踪的车道线
        
        参数:
            min_confidence: 最小置信度阈值
            
        返回:
            lanes: 车道线列表
        """
        lanes = []
        for lane_id, tracker in self.trackers.items():
            lane = self._state_to_lane(
                tracker.get_state(),
                'unknown',
                lane_id
            )
            lane['confidence'] = max(0.3, 1.0 - tracker.missed_frames * 0.2)
            
            if min_confidence is None or lane['confidence'] >= min_confidence:
                lanes.append(lane)
        
        return lanes
    
    def reset(self):
        """重置跟踪器"""
        self.trackers = {}
        self.next_id = 0
        self.lane_history = {}
        self.vanishing_point = None
