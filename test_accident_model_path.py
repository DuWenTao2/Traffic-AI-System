#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试事故检测配置系统中的model_path参数
"""

import sys
import os
import unittest
from unittest.mock import patch

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), "Models", "Accident_det"))

class TestAccidentConfig(unittest.TestCase):
    """测试事故检测配置系统"""
    
    @patch('Models.Accident_det.Accident_Detector.YOLO')
    def test_default_initialization(self, mock_yolo):
        """测试默认初始化（加载配置文件中的model_path）"""
        from Accident_Detector import AccidentDetector
        
        print("1. 测试默认初始化（加载配置文件中的model_path）:")
        
        # 模拟YOLO类，避免实际加载模型
        mock_instance = mock_yolo.return_value
        mock_instance.names = {0: 'Accident'}
        
        detector = AccidentDetector(stream_id="test1")
        params = detector.get_parameters()
        
        print("  当前model_path:", params['model_path'])
        print("  预期: weights/Accident_Detection/accident_multi.pt")
        
        # 验证model_path是否正确加载
        self.assertEqual(params['model_path'], "weights/Accident_Detection/accident_multi.pt")
    
    @patch('Models.Accident_det.Accident_Detector.YOLO')
    def test_custom_model_path(self, mock_yolo):
        """测试传入自定义model_path（覆盖配置文件）"""
        from Accident_Detector import AccidentDetector
        
        print("\n2. 测试传入自定义model_path（覆盖配置文件）:")
        
        # 模拟YOLO类，避免实际加载模型
        mock_instance = mock_yolo.return_value
        mock_instance.names = {0: 'Accident'}
        
        custom_model_path = "weights/Accident_Detection/custom_model.pt"
        detector = AccidentDetector(
            stream_id="test2",
            model_path=custom_model_path
        )
        params = detector.get_parameters()
        
        print("  当前model_path:", params['model_path'])
        print("  预期:", custom_model_path)
        
        # 验证model_path是否正确覆盖
        self.assertEqual(params['model_path'], custom_model_path)
    
    @patch('Models.Accident_det.Accident_Detector.YOLO')
    def test_other_parameters(self, mock_yolo):
        """测试其他参数是否正常工作"""
        from Accident_Detector import AccidentDetector
        
        print("\n3. 测试其他参数是否正常工作:")
        
        # 模拟YOLO类，避免实际加载模型
        mock_instance = mock_yolo.return_value
        mock_instance.names = {0: 'Accident'}
        
        detector = AccidentDetector(
            stream_id="test3",
            conf_threshold=0.5,
            cooldown=60
        )
        params = detector.get_parameters()
        
        print("  当前model_path:", params['model_path'])
        print("  当前conf_threshold:", params['conf_threshold'])
        print("  当前cooldown:", params['cooldown'])
        
        # 验证其他参数是否正确
        self.assertEqual(params['conf_threshold'], 0.5)
        self.assertEqual(params['cooldown'], 60)
        # 验证model_path是否仍从配置文件加载
        self.assertEqual(params['model_path'], "weights/Accident_Detection/accident_multi.pt")

def test_model_path_config():
    """运行测试"""
    print("=== 测试事故检测配置系统中的model_path参数 ===")
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAccidentConfig)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n=== 测试完成 ===")
    return result.wasSuccessful()

if __name__ == "__main__":
    success = test_model_path_config()
    sys.exit(0 if success else 1)
