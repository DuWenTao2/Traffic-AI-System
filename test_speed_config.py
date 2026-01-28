#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试速度检测配置系统
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), "Models", "Speed Model"))

from speed_detector import SpeedDetector

def test_config_system():
    """测试配置系统"""
    print("=== 测试速度检测配置系统 ===")
    
    # 测试1: 默认初始化（应该加载配置文件中的参数）
    print("\n1. 测试默认初始化（加载配置文件参数）:")
    detector1 = SpeedDetector(stream_id="test1")
    params1 = detector1.get_parameters()
    print("  当前参数:", params1)
    print("  验证是否从配置文件加载:")
    print(f"    pixels_per_meter = {params1['pixels_per_meter']} (预期: 30)")
    print(f"    speed_limit = {params1['speed_limit']} (预期: 10.0)")
    print(f"    process_noise = {params1['process_noise']} (预期: 0.1)")
    
    # 测试2: 传入自定义参数（应该覆盖配置文件中的参数）
    print("\n2. 测试传入自定义参数（覆盖配置文件）:")
    detector2 = SpeedDetector(
        stream_id="test2",
        pixels_per_meter=50,
        speed_limit=20.0,
        smoothing_window=15
    )
    params2 = detector2.get_parameters()
    print("  当前参数:", params2)
    print("  验证是否覆盖成功:")
    print(f"    pixels_per_meter = {params2['pixels_per_meter']} (预期: 50)")
    print(f"    speed_limit = {params2['speed_limit']} (预期: 20.0)")
    print(f"    smoothing_window = {params2['smoothing_window']} (预期: 15)")
    
    # 测试3: 测试set_parameters方法
    print("\n3. 测试set_parameters方法:")
    detector3 = SpeedDetector(stream_id="test3")
    detector3.set_parameters(
        min_speed=10.0,
        max_speed_change=10.0,
        process_noise=0.2,
        measurement_noise=2.0
    )
    params3 = detector3.get_parameters()
    print("  当前参数:", params3)
    print("  验证是否修改成功:")
    print(f"    min_speed = {params3['min_speed']} (预期: 10.0)")
    print(f"    max_speed_change = {params3['max_speed_change']} (预期: 10.0)")
    print(f"    process_noise = {params3['process_noise']} (预期: 0.2)")
    print(f"    measurement_noise = {params3['measurement_noise']} (预期: 2.0)")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_config_system()
