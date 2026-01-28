#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试车道线检测配置系统
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), "Models", "LaneDetection"))

from lane_detector import LaneDetector

def test_config_system():
    """测试配置系统"""
    print("=== 测试车道线检测配置系统 ===")
    
    # 测试1: 默认初始化（应该加载配置文件中的参数）
    print("\n1. 测试默认初始化（加载配置文件参数）:")
    detector1 = LaneDetector(stream_id="test1")
    params1 = detector1.get_parameters()
    print("  当前参数:", params1)
    
    # 测试2: 传入启动参数（应该覆盖配置文件参数）
    print("\n2. 测试传入启动参数（覆盖配置文件参数）:")
    detector2 = LaneDetector(
        stream_id="test2",
        canny_threshold1=100,
        hough_threshold=150,
        history_length=10
    )
    params2 = detector2.get_parameters()
    print("  当前参数:", params2)
    print("  验证是否覆盖成功:")
    print(f"    canny_threshold1 = {params2['canny_threshold1']} (预期: 100)")
    print(f"    hough_threshold = {params2['hough_threshold']} (预期: 150)")
    
    # 测试3: 测试set_parameters方法
    print("\n3. 测试set_parameters方法:")
    detector3 = LaneDetector(stream_id="test3")
    detector3.set_parameters(
        canny_threshold2=200,
        hough_min_line_length=120
    )
    params3 = detector3.get_parameters()
    print("  当前参数:", params3)
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_config_system()
