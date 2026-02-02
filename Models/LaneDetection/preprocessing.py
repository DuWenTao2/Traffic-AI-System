import cv2
import numpy as np


class ColorEnhancer:
    """
    颜色空间增强器
    使用Lab颜色空间和CLAHE自适应对比度增强
    """
    
    def __init__(self, clip_limit=2.0, grid_size=(8, 8)):
        """
        初始化颜色增强器
        
        参数:
            clip_limit: CLAHE对比度限制阈值，防止噪声放大
            grid_size: 局部处理块大小
        """
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    
    def enhance(self, frame):
        """
        对输入图像进行颜色增强
        
        参数:
            frame: 输入BGR图像
            
        返回:
            frame_enhanced: 增强后的BGR图像
            l_enhanced: 增强后的L通道（用于后续处理）
        """
        # 1. RGB转Lab颜色空间
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        
        # 2. 对L通道应用CLAHE增强
        l_enhanced = self.clahe.apply(l)
        
        # 3. 重建Lab图像
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        
        # 4. 转回BGR空间
        frame_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_Lab2BGR)
        
        return frame_enhanced, l_enhanced
    
    def set_parameters(self, clip_limit=None, grid_size=None):
        """动态调整CLAHE参数"""
        if clip_limit is not None:
            self.clip_limit = clip_limit
        if grid_size is not None:
            self.grid_size = grid_size
        
        # 重新创建CLAHE对象
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, 
            tileGridSize=self.grid_size
        )


class ShadowRemover:
    """
    阴影检测与去除器
    基于HSV颜色空间的V通道进行自适应阴影检测和补偿
    """
    
    def __init__(self, kernel_size=15, k=0.5):
        """
        初始化阴影去除器
        
        参数:
            kernel_size: 局部统计计算的核大小
            k: 阈值调节系数，越大检测到的阴影越少
        """
        self.kernel_size = kernel_size
        self.k = k
    
    def detect_shadows(self, frame):
        """
        检测图像中的阴影区域
        
        原理：阴影区域亮度低但饱和度变化不大
        
        参数:
            frame: 输入BGR图像
            
        返回:
            shadow_mask: 阴影掩码（255表示阴影区域）
        """
        # 转换到HSV空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2].astype(np.float32)
        
        # 计算局部均值
        local_mean = cv2.blur(v_channel, (self.kernel_size, self.kernel_size))
        
        # 计算局部标准差
        mean_sq = cv2.blur(v_channel ** 2, (self.kernel_size, self.kernel_size))
        variance = mean_sq - local_mean ** 2
        local_std = np.sqrt(np.maximum(variance, 0))
        
        # 自适应阈值：均值 - k * 标准差
        shadow_threshold = local_mean - self.k * local_std
        
        # 生成阴影掩码
        shadow_mask = (v_channel < shadow_threshold).astype(np.uint8) * 255
        
        # 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        return shadow_mask
    
    def compensate_shadows(self, frame, shadow_mask):
        """
        对阴影区域进行亮度补偿
        
        策略：提升阴影区域亮度至非阴影区域平均水平
        
        参数:
            frame: 输入BGR图像
            shadow_mask: 阴影掩码
            
        返回:
            frame_compensated: 补偿后的图像
        """
        # 转换到Lab空间处理亮度
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # 计算阴影和非阴影区域的平均亮度
        shadow_pixels = l_channel[shadow_mask > 0]
        non_shadow_pixels = l_channel[shadow_mask == 0]
        
        shadow_mean = np.mean(shadow_pixels) if len(shadow_pixels) > 0 else 128
        non_shadow_mean = np.mean(non_shadow_pixels) if len(non_shadow_pixels) > 0 else 128
        
        # 计算补偿系数
        if shadow_mean > 0 and non_shadow_mean > shadow_mean:
            compensation_ratio = min(non_shadow_mean / shadow_mean, 2.0)
        else:
            compensation_ratio = 1.5
        
        # 应用补偿（仅对阴影区域）
        l_compensated = l_channel.copy()
        l_compensated[shadow_mask > 0] *= compensation_ratio
        l_compensated = np.clip(l_compensated, 0, 255).astype(np.uint8)
        
        # 重建图像
        lab[:, :, 0] = l_compensated
        frame_compensated = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        
        return frame_compensated
    
    def remove_shadows(self, frame):
        """
        完整的阴影去除流程
        
        参数:
            frame: 输入BGR图像
            
        返回:
            frame_compensated: 阴影补偿后的图像
            shadow_mask: 阴影掩码（用于调试）
        """
        shadow_mask = self.detect_shadows(frame)
        frame_compensated = self.compensate_shadows(frame, shadow_mask)
        return frame_compensated, shadow_mask


class MultiScaleProcessor:
    """
    多尺度处理框架
    构建图像金字塔，在不同尺度下检测特征
    """
    
    def __init__(self, levels=3):
        """
        初始化多尺度处理器
        
        参数:
            levels: 金字塔层数（默认3层：1x, 0.5x, 0.25x）
        """
        self.levels = levels
    
    def build_pyramid(self, frame):
        """
        构建高斯金字塔
        
        参数:
            frame: 输入图像
            
        返回:
            pyramid: 图像金字塔列表
        """
        pyramid = [frame]
        current = frame.copy()
        
        for i in range(self.levels - 1):
            # 高斯模糊后下采样
            current = cv2.pyrDown(current)
            pyramid.append(current)
        
        return pyramid
    
    def scale_detection(self, detection, scale_factor):
        """
        将检测结果从当前尺度映射回原图坐标
        
        参数:
            detection: 检测结果（点坐标列表或线段）
            scale_factor: 缩放因子
            
        返回:
            scaled_detection: 映射后的检测结果
        """
        if isinstance(detection, dict) and 'points' in detection:
            # 车道线格式
            scaled_points = []
            for x, y in detection['points']:
                scaled_points.append((int(x * scale_factor), int(y * scale_factor)))
            
            scaled_detection = detection.copy()
            scaled_detection['points'] = scaled_points
            return scaled_detection
        elif isinstance(detection, (list, tuple)) and len(detection) == 4:
            # 线段格式 (x1, y1, x2, y2)
            x1, y1, x2, y2 = detection
            return (
                int(x1 * scale_factor),
                int(y1 * scale_factor),
                int(x2 * scale_factor),
                int(y2 * scale_factor)
            )
        else:
            return detection
    
    def compute_iou(self, det1, det2):
        """
        计算两个检测结果的IoU（用于NMS）
        
        参数:
            det1, det2: 检测结果
            
        返回:
            iou: IoU值
        """
        # 简化的线段IoU计算
        def get_line_bbox(det):
            if isinstance(det, dict) and 'points' in det:
                points = det['points']
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            return (0, 0, 0, 0)
        
        box1 = get_line_bbox(det1)
        box2 = get_line_bbox(det2)
        
        # 计算BBox IoU
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def fuse_detections(self, detections, iou_threshold=0.5):
        """
        多尺度检测结果融合（NMS去重）
        
        参数:
            detections: 检测结果列表，每项包含'detection'和'scale'
            iou_threshold: IoU阈值，超过则认为是重复检测
            
        返回:
            fused_detections: 融合后的检测结果
        """
        if not detections:
            return []
        
        # 按置信度排序（如果有）或按尺度（优先保留大尺度）
        detections_sorted = sorted(
            detections,
            key=lambda x: (x.get('confidence', 0), -x.get('scale', 1)),
            reverse=True
        )
        
        keep = []
        while detections_sorted:
            current = detections_sorted.pop(0)
            keep.append(current)
            
            # 移除与当前检测IoU过高的其他检测
            detections_sorted = [
                det for det in detections_sorted
                if self.compute_iou(current['detection'], det['detection']) < iou_threshold
            ]
        
        return [k['detection'] for k in keep]
    
    def process(self, frame, detector_func):
        """
        多尺度处理流程
        
        参数:
            frame: 输入图像
            detector_func: 单尺度检测函数，接收图像返回检测结果
            
        返回:
            fused_detections: 融合后的检测结果
        """
        # 1. 构建金字塔
        pyramid = self.build_pyramid(frame)
        
        # 2. 多尺度检测
        all_detections = []
        for level, image in enumerate(pyramid):
            detections = detector_func(image)
            scale_factor = 2 ** level
            
            for det in detections:
                scaled_det = self.scale_detection(det, scale_factor)
                all_detections.append({
                    'detection': scaled_det,
                    'level': level,
                    'scale': scale_factor
                })
        
        # 3. NMS融合
        fused = self.fuse_detections(all_detections)
        
        return fused


class Preprocessor:
    """
    预处理器主类
    整合颜色增强、阴影去除、多尺度处理
    """
    
    def __init__(self, config=None):
        """
        初始化预处理器
        
        参数:
            config: 配置字典，包含各模块参数
        """
        config = config or {}
        
        # 初始化各模块
        self.color_enhancer = ColorEnhancer(
            clip_limit=config.get('clahe_clip_limit', 2.0),
            grid_size=tuple(config.get('clahe_grid_size', (8, 8)))
        )
        
        self.shadow_remover = ShadowRemover(
            kernel_size=config.get('shadow_kernel_size', 15),
            k=config.get('shadow_k', 0.5)
        )
        
        self.multi_scale = MultiScaleProcessor(
            levels=config.get('multiscale_levels', 3)
        )
        
        # 功能开关
        self.enable_color_enhance = config.get('enable_color_enhance', True)
        self.enable_shadow_removal = config.get('enable_shadow_removal', True)
    
    def preprocess(self, frame):
        """
        完整的预处理流程
        
        参数:
            frame: 输入BGR图像
            
        返回:
            processed: 预处理后的图像
            debug_info: 调试信息字典
        """
        debug_info = {}
        processed = frame.copy()
        
        # 1. 颜色增强
        if self.enable_color_enhance:
            processed, l_channel = self.color_enhancer.enhance(processed)
            debug_info['l_channel'] = l_channel
        
        # 2. 阴影去除
        if self.enable_shadow_removal:
            processed, shadow_mask = self.shadow_remover.remove_shadows(processed)
            debug_info['shadow_mask'] = shadow_mask
        
        debug_info['processed'] = processed
        return processed, debug_info
    
    def set_parameters(self, **kwargs):
        """动态调整参数"""
        if 'clip_limit' in kwargs or 'grid_size' in kwargs:
            self.color_enhancer.set_parameters(
                clip_limit=kwargs.get('clip_limit'),
                grid_size=kwargs.get('grid_size')
            )
        
        if 'kernel_size' in kwargs:
            self.shadow_remover.kernel_size = kwargs['kernel_size']
        
        if 'k' in kwargs:
            self.shadow_remover.k = kwargs['k']
