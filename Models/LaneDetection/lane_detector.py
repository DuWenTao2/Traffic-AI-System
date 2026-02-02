"""
车道线检测器 - 优化版本 V2.0

本文件为重构后的车道线检测器入口，整合以下优化模块：
- preprocessing.py: 预处理模块（颜色增强、阴影去除、多尺度）
- edge_detection.py: 边缘检测模块（自适应Canny、颜色分割、双向分离）
- lane_fitting.py: 车道线拟合模块（RANSAC、曲线拟合、几何约束）
- lane_tracker.py: 跟踪模块（卡尔曼滤波、匈牙利关联、置信度评估）

旧版本代码已备份至: lane_detector_v1_backup.py
"""

import os
import sys

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加当前目录到路径（确保可以导入同级模块）
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 从优化版本导入
from lane_detector_v2 import LaneDetectorV2, LaneDetector

# 保持向后兼容
__all__ = ['LaneDetector', 'LaneDetectorV2']
