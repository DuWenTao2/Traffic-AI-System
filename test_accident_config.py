#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试事故检测配置系统
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), "Models", "Accident_det"))

from Accident_Detector import AccidentDetector

def test_config_system():
    """测试配置系统"""
    print("=== 测试事故检测配置系统 ===")
    
    # 测试1: 默认初始化（应该加载配置文件中的参数）
    print("\n1. 测试默认初始化（加载配置文件参数）:")
    detector1 = AccidentDetector(stream_id="test1")
    params1 = detector1.get_parameters()
    print("  当前参数:", params1)
    
    # 测试2: 传入启动参数（应该覆盖配置文件参数）
    print("\n2. 测试传入启动参数（覆盖配置文件参数）:")
    detector2 = AccidentDetector(
        stream_id="test2",
        conf_threshold=0.5,
        cooldown=60,
        frame_skip=5
    )
    params2 = detector2.get_parameters()
    print("  当前参数:", params2)
    print("  验证是否覆盖成功:")
    print(f"    conf_threshold = {params2['conf_threshold']} (预期: 0.5)")
    print(f"    cooldown = {params2['cooldown']} (预期: 60)")
    print(f"    frame_skip = {params2['frame_skip']} (预期: 5)")
    
    # 测试3: 测试set_parameters方法
    print("\n3. 测试set_parameters方法:")
    detector3 = AccidentDetector(stream_id="test3")
    detector3.set_parameters(
        auto_disable_count=5,
        auto_disable_window=120,
        accident_area_radius=200
    )
    params3 = detector3.get_parameters()
    print("  当前参数:", params3)
    print("  验证是否修改成功:")
    print(f"    auto_disable_count = {params3['auto_disable_count']} (预期: 5)")
    print(f"    auto_disable_window = {params3['auto_disable_window']} (预期: 120)")
    print(f"    accident_area_radius = {params3['accident_area_radius']} (预期: 200)")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_config_system()
