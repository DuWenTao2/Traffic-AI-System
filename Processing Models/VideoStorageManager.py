import cv2
import os
import time
import threading
import json
from datetime import datetime
from collections import deque

class VideoStorageManager:
    def __init__(self, stream_id, camera_location="Unknown", config=None):
        """
        初始化视频存储管理器
        
        Args:
            stream_id (str): 视频流ID
            camera_location (str): 摄像头位置
            config (dict): 视频存储配置
        """
        self.stream_id = stream_id
        self.camera_location = camera_location
        
        # 默认配置
        default_config = {
            "storage_enabled": True,
            "storage_path": "videos",
            "video_format": "mp4",
            "codec": "mp4v",
            "fps": 30.0,
            "video_duration": 60,  # 单个视频文件最大时长（秒）
            "pre_detection_buffer": 5,  # 检测前缓冲时间（秒）
            "post_detection_buffer": 10,  # 检测后缓冲时间（秒）
            "max_storage_size": 1024,  # 最大存储大小（MB）
            "cleanup_enabled": True  # 启用自动清理
        }
        
        # 合并配置
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        # 确保存储目录存在
        self.storage_path = os.path.join(os.getcwd(), self.config["storage_path"])
        os.makedirs(self.storage_path, exist_ok=True)
        
        # 初始化变量
        self.is_recording = False
        self.out = None
        self.current_video_path = None
        self.start_time = None
        self.detection_start_time = None
        self.last_detection_time = None
        
        # 帧缓冲队列 (使用deque提高性能)
        self.frame_buffer = deque(maxlen=int(self.config["pre_detection_buffer"] * self.config["fps"]))
        self.buffer_size = int(self.config["pre_detection_buffer"] * self.config["fps"])
        
        # 锁用于线程安全
        self.lock = threading.Lock()
        
        # Thread for periodically checking storage size
        self.cleanup_thread = threading.Thread(target=self._check_storage_periodically, daemon=True)
        self.cleanup_thread.start()
    
    def _generate_video_filename(self):
        """
        Generate video filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.stream_id}_{timestamp}.{self.config['video_format']}"
        return os.path.join(self.storage_path, filename)
    
    def _start_recording(self, frame):
        """
        Start recording video
        """
        with self.lock:
            if not self.is_recording:
                try:
                    # Generate filename
                    self.current_video_path = self._generate_video_filename()
                    
                    # Get frame size
                    height, width = frame.shape[:2]
                    
                    # Create VideoWriter
                    fourcc = cv2.VideoWriter_fourcc(*self.config["codec"])
                    self.out = cv2.VideoWriter(
                        self.current_video_path,
                        fourcc,
                        self.config["fps"],
                        (width, height)
                    )
                    
                    if not self.out.isOpened():
                        raise Exception("Failed to create video writer")
                    
                    # Write buffered frames
                    for buffered_frame in self.frame_buffer:
                        self.out.write(buffered_frame)
                    
                    # Clear buffer
                    self.frame_buffer.clear()
                    
                    self.is_recording = True
                    self.start_time = time.time()
                    self.detection_start_time = time.time()
                    self.last_detection_time = time.time()
                    
                    print(f"[{self.stream_id}] Started recording: {self.current_video_path}")
                    
                except Exception as e:
                    print(f"[{self.stream_id}] Failed to start recording: {str(e)}")
                    self._stop_recording()
    
    def _stop_recording(self):
        """
        Stop recording video
        """
        with self.lock:
            if self.is_recording:
                try:
                    if self.out:
                        self.out.release()
                        self.out = None
                    
                    duration = time.time() - self.start_time
                    print(f"[{self.stream_id}] Stopped recording, duration: {duration:.2f}s")
                    
                except Exception as e:
                    print(f"[{self.stream_id}] Failed to stop recording: {str(e)}")
                finally:
                    self.is_recording = False
                    self.current_video_path = None
                    self.start_time = None
                    self.detection_start_time = None
                    self.last_detection_time = None
    
    def _check_storage_periodically(self):
        """
        Check storage size periodically and clean up old files
        """
        while True:
            time.sleep(3600)  # Check every hour
            
            if self.config["cleanup_enabled"]:
                self._cleanup_old_files()
    
    def _cleanup_old_files(self):
        """
        Clean up old files to free up storage space
        """
        try:
            # Get all video files
            video_files = []
            for file in os.listdir(self.storage_path):
                if file.endswith(f".{self.config['video_format']}"):
                    file_path = os.path.join(self.storage_path, file)
                    if os.path.isfile(file_path):
                        video_files.append((file_path, os.path.getmtime(file_path)))
            
            # Sort by modification time (oldest first)
            video_files.sort(key=lambda x: x[1])
            
            # Calculate current storage usage
            current_size = sum(os.path.getsize(file[0]) for file in video_files) / (1024 * 1024)  # MB
            
            # Clean up old files until below max storage limit
            while current_size > self.config["max_storage_size"] and video_files:
                oldest_file = video_files.pop(0)
                file_path = oldest_file[0]
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                
                try:
                    os.remove(file_path)
                    current_size -= file_size
                    print(f"[{self.stream_id}] Cleaned up old video file: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"[{self.stream_id}] Failed to clean up file: {str(e)}")
                    
        except Exception as e:
            print(f"[{self.stream_id}] Failed to clean up storage: {str(e)}")
    
    def process_frame(self, frame, has_detection=False):
        """
        Process video frame, decide whether to record based on detection status
        
        Args:
            frame (numpy.ndarray): Video frame
            has_detection (bool): Whether violation is detected
        """
        if not self.config["storage_enabled"]:
            return
        
        # Update last detection time
        if has_detection:
            self.last_detection_time = time.time()
        
        # Not recording: maintain buffer for pre-recording functionality
        if not self.is_recording:
            # Add current frame to buffer for potential pre-recording
            # Using deque with maxlen automatically manages buffer size
            self.frame_buffer.append(frame.copy())
            
            # If there's detection, start recording with buffered frames
            if has_detection:
                self._start_recording(frame)
        else:
            # Recording: write frame
            with self.lock:
                if self.out and self.is_recording:
                    self.out.write(frame)
                    
                    # Check if need to stop recording
                    current_time = time.time()
                    if self.last_detection_time:
                        # Post-detection buffer time has passed
                        if current_time - self.last_detection_time > self.config["post_detection_buffer"]:
                            self._stop_recording()
                    
                    # Check single video file duration
                    if current_time - self.start_time > self.config["video_duration"]:
                        # Stop current recording and start new one
                        self._stop_recording()
                        if has_detection:
                            self._start_recording(frame)
    
    def manual_start_recording(self, frame):
        """
        Manually start recording
        """
        self._start_recording(frame)
    
    def manual_stop_recording(self):
        """
        Manually stop recording
        """
        self._stop_recording()
    
    def get_recording_status(self):
        """
        Get recording status
        """
        return self.is_recording
    
    def get_current_video_path(self):
        """
        Get current recording video path
        """
        return self.current_video_path
    
    def cleanup(self):
        """
        Clean up resources
        """
        self._stop_recording()
        
        # Clear frame buffer
        self.frame_buffer.clear()
        
        print(f"[{self.stream_id}] Video storage manager cleaned up")
