import cv2
import numpy as np
import time
import os
import json
from datetime import datetime

class KalmanFilter:
    """
    Kalman Filter implementation for vehicle speed estimation
    Used to reduce measurement noise and improve speed accuracy
    """
    def __init__(self, initial_speed=0.0, process_noise=0.1, measurement_noise=1.0, error_covariance=1.0):
        # State vector [speed]
        self.x = np.array([[initial_speed]])
        
        # State transition matrix
        self.F = np.array([[1.0]])
        
        # Measurement matrix
        self.H = np.array([[1.0]])
        
        # Process noise covariance
        self.Q = np.array([[process_noise]])
        
        # Measurement noise covariance
        self.R = np.array([[measurement_noise]])
        
        # Error covariance matrix
        self.P = np.array([[error_covariance]])
    
    def predict(self):
        # Predict next state
        self.x = np.dot(self.F, self.x)
        
        # Update error covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return self.x[0, 0]
    
    def update(self, measurement):
        # Calculate Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update state estimate
        measurement = np.array([[measurement]])
        self.x = self.x + np.dot(K, (measurement - np.dot(self.H, self.x)))
        
        # Update error covariance
        I = np.eye(self.F.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        
        return self.x[0, 0]
    
    def get_current_state(self):
        return self.x[0, 0]

class SpeedDetector:
    def __init__(self, stream_id="default", pixels_per_meter=30, smoothing_window=10, font_style="plain", violation_manager=None, **kwargs):
       
        self.stream_id = stream_id
        self.violation_manager = violation_manager
        self.enabled = False  # Only enable after speed lines are defined
        
        # Vehicle tracking data
        self.vehicles = {}  # Store vehicle position history
        self.speeds = {}    # Store calculated speeds
        self.recorded_speeds = {}  # Store speeds recorded at the measurement line
        self.speed_filters = {}  # Store Kalman filters for each vehicle
        
        # Keep track of violations to avoid duplicates
        self.speed_violations = set()  # Set of vehicle IDs that have already been recorded for violation
        
        # Modified output directory handling for snapshots when needed
        self.output_dir = None
        
        # 1. 设置代码默认参数
        self._set_default_parameters()
        
        # 2. 从配置文件加载参数
        self._load_config_parameters()
        
        # 3. 应用启动参数（优先级最高）
        self.pixels_per_meter = pixels_per_meter
        self.smoothing_window = smoothing_window
        self.font_style = font_style
        
        # 应用额外的启动参数
        self.set_parameters(**kwargs)
        
        # Set font style based on parameter
        if self.font_style.lower() == "simplex":
            self.font = cv2.FONT_HERSHEY_SIMPLEX
            self.font_scale = 0.6
            self.font_thickness = 2
        else:  # Default to plain
            self.font = cv2.FONT_HERSHEY_PLAIN
            self.font_scale = 1.2
            self.font_thickness = 1
        
        # 初始化最后一帧时间
        self.last_frame_time = time.time()
        
        print(f"[{self.stream_id}] Speed detector initialized with calibration: {self.pixels_per_meter} pixels/meter")
        print(f"[{self.stream_id}] Speed limit set to: {self.speed_limit} km/h")
        
    def calculate_speed(self, frame, tracked_objects, fps=30):
        """Calculate speed for all tracked vehicles and identify violations"""
        if not self.enabled or not tracked_objects:
            return frame
        
        # Store current frame for violation snapshots
        self.current_frame = frame.copy()  # Make a clean copy without any annotations
        self.tracked_objects = tracked_objects  # Store reference to all tracked objects
            
        # Calculate time difference using fixed step for stable FPS
        current_time = time.time()
        time_diff = self.fixed_time_step  # Always use 1/30 seconds between frames
        self.last_frame_time = current_time
        
        # Track IDs we've seen in this frame
        current_ids = set()
        
        for obj_id, track_data in tracked_objects.items():
            current_ids.add(obj_id)
            
            if 'center_points' not in track_data or len(track_data['center_points']) < 2:
                continue
                
            # Get current position (latest center point)
            current_pos = track_data['center_points'][-1]
            
            # Initialize tracking for new vehicles
            if obj_id not in self.vehicles:
                # Create Kalman filter for this vehicle
                if obj_id not in self.speed_filters:
                    self.speed_filters[obj_id] = KalmanFilter(**self.kalman_config)
                
                self.vehicles[obj_id] = {
                    'positions': track_data['center_points'].copy(),  # Copy existing points
                    'timestamps': [current_time - self.fixed_time_step * i for i in range(len(track_data['center_points'])-1, -1, -1)],
                    'speeds': [],     # List of calculated speeds
                    'last_speed': 0,  # Last stable speed
                    'crossed_lines': set(),  # Lines already crossed (copy from tracked_objects)
                    'line_speeds': {},  # Store speeds from each crossed line
                }
                if 'crossed_lines' in track_data:
                    self.vehicles[obj_id]['crossed_lines'] = track_data['crossed_lines'].copy()
            else:
                # Update positions and timestamps
                self.vehicles[obj_id]['positions'].append(current_pos)
                self.vehicles[obj_id]['timestamps'].append(current_time)
                
                # Limit history size
                max_history = self.smoothing_window * 2
                if len(self.vehicles[obj_id]['positions']) > max_history:
                    self.vehicles[obj_id]['positions'] = self.vehicles[obj_id]['positions'][-max_history:]
                    self.vehicles[obj_id]['timestamps'] = self.vehicles[obj_id]['timestamps'][-max_history:]
                    self.vehicles[obj_id]['speeds'] = self.vehicles[obj_id]['speeds'][-max_history:]
                
                # Check if vehicle crossed speed lines recently and calculate speed
                speed_line_crossings = [(line_id, 1) for line_id, line_type in track_data.get('crossed_lines', set()) 
                                     if line_type == 'speed' and (line_id, 'speed') not in self.vehicles[obj_id].get('processed_lines', set())]
                
                if speed_line_crossings:
                    for speed_line_id, direction in speed_line_crossings:
                        # Calculate speed at this crossing line
                        self._calculate_vehicle_speed(obj_id, speed_line_id)
                        
                        # Mark line as processed
                        if 'processed_lines' not in self.vehicles[obj_id]:
                            self.vehicles[obj_id]['processed_lines'] = set()
                        self.vehicles[obj_id]['processed_lines'].add((speed_line_id, 'speed'))
            
            # Get box coordinates for drawing
            if 'box' in track_data:
                x1, y1, x2, y2 = track_data['box']
            else:
                # Estimate box from center points if not available
                center_x, center_y = current_pos
                box_size = 50  # Default box size
                x1, y1 = center_x - box_size//2, center_y - box_size//2
                x2, y2 = center_x + box_size//2, center_y + box_size//2
            
            # Draw speed if available
            if obj_id in self.speeds:
                # Only display speed for active vehicles or for a limited time (1 second only)
                is_visible = obj_id in current_ids
                was_recently_visible = False
                
                # Check if vehicle was recently visible (within last 1 second)
                if 'last_seen_time' in track_data:
                    was_recently_visible = (current_time - track_data['last_seen_time'] < 1.0)  # Changed from 3.0 to 1.0
                
                if is_visible or was_recently_visible:
                    # Use different display for vehicles that have recorded speeds
                    if obj_id in self.recorded_speeds:
                        speed_text = f"{self.recorded_speeds[obj_id]:.1f} km/h"
                        text_color = (0, 0, 255)  # Red for recorded speeds
                    else:
                        speed_text = f"{self.speeds[obj_id]:.1f} km/h"
                        text_color = (0, 255, 0)  # Green for real-time speeds
                    
                    # Use selected font style
                    cv2.putText(
                        frame,
                        speed_text,
                        (x1, y1 - 10),
                        self.font,
                        self.font_scale,
                        text_color,
                        self.font_thickness,
                        cv2.LINE_AA  # Anti-aliased line for smoother appearance
                    )
        
        # Remove vehicles that are no longer in the frame for a while
        ids_to_remove = []
        for vehicle_id in self.vehicles:
            if vehicle_id not in current_ids:
                # If vehicle hasn't been seen for at least 30 frames, remove it
                if len(self.vehicles[vehicle_id]['timestamps']) > 0:
                    if current_time - self.vehicles[vehicle_id]['timestamps'][-1] > 1.0:  # 1 second timeout
                        ids_to_remove.append(vehicle_id)
        
        # Remove stale vehicle records
        for vehicle_id in ids_to_remove:
            if vehicle_id in self.vehicles:
                del self.vehicles[vehicle_id]
            if vehicle_id in self.speeds:
                del self.speeds[vehicle_id]
            if vehicle_id in self.speed_filters:
                del self.speed_filters[vehicle_id]
            if vehicle_id in self.recorded_speeds:
                del self.recorded_speeds[vehicle_id]
                
        return frame
    
    def _calculate_vehicle_speed(self, vehicle_id, speed_line_id=None):
        """Calculate speed for a specific vehicle"""
        if vehicle_id not in self.vehicles or len(self.vehicles[vehicle_id]['positions']) < 5:
            return
            
        positions = self.vehicles[vehicle_id]['positions']
        timestamps = self.vehicles[vehicle_id]['timestamps']
        
        # Use at least 5 positions for more stable measurements
        history_index = min(5, len(positions) - 1)
        
        # Calculate pixel distance between first and last position in history
        first_pos = positions[-history_index-1]
        last_pos = positions[-1]
        
        pixel_distance = np.sqrt(
            (last_pos[0] - first_pos[0]) ** 2 + 
            (last_pos[1] - first_pos[1]) ** 2
        )
        
        # Only calculate speed if movement is significant (reduces noise)
        if pixel_distance > self.min_distance_for_calc:
            # Convert to real-world distance (meters)
            distance = pixel_distance / self.pixels_per_meter
            
            # Speed = distance / time (m/s)
            time_elapsed = timestamps[-1] - timestamps[-history_index-1]
            if time_elapsed > 0:
                speed_ms = distance / time_elapsed
                
                # Convert to km/h
                speed_kmh = speed_ms * 3.6
                
                # Limit maximum speed for realism
                speed_kmh = min(speed_kmh, 200.0)
                
                # Apply stability constraint (limit rate of change)
                last_speed = self.vehicles[vehicle_id]['last_speed']
                if last_speed > 0:
                    # Limit speed change to prevent jumps
                    if abs(speed_kmh - last_speed) > self.max_speed_change:
                        if speed_kmh > last_speed:
                            speed_kmh = last_speed + self.max_speed_change
                        else:
                            speed_kmh = last_speed - self.max_speed_change
                
                # Store the calculated speed
                self.vehicles[vehicle_id]['speeds'].append(speed_kmh)
                self.vehicles[vehicle_id]['last_speed'] = speed_kmh
                
                # Apply Kalman Filter to improve speed accuracy
                if vehicle_id in self.speed_filters:
                    # Predict next state
                    self.speed_filters[vehicle_id].predict()
                    # Update with measurement
                    filtered_speed = self.speed_filters[vehicle_id].update(speed_kmh)
                else:
                    # Fallback if Kalman filter not available
                    filtered_speed = speed_kmh
                
                # Apply smoothing - calculate weighted average of recent speed measurements
                # Use exponential weighting to favor more recent measurements
                window_size = min(self.smoothing_window, len(self.vehicles[vehicle_id]['speeds']))
                if window_size > 0:
                    weights = np.exp(np.linspace(0, 1, window_size))
                    weighted_speeds = np.array(self.vehicles[vehicle_id]['speeds'][-window_size:]) * weights
                    smoothed_speed = weighted_speeds.sum() / weights.sum()
                    
                    # Combine smoothed speed with Kalman filtered speed for best results
                    final_speed = (smoothed_speed + filtered_speed) / 2
                    
                    # If this was a speed line crossing, record the speed for this line
                    if speed_line_id is not None:
                        # Store speed for this specific line
                        self.vehicles[vehicle_id]['line_speeds'][speed_line_id] = final_speed
                        
                        # Calculate multi-line speed using all crossed lines
                        all_line_speeds = list(self.vehicles[vehicle_id]['line_speeds'].values())
                        if len(all_line_speeds) > 0:
                            # Use median speed for better robustness against outliers
                            multi_line_speed = np.median(all_line_speeds)
                        else:
                            multi_line_speed = final_speed
                        
                        # Update speed for display and recording
                        self.speeds[vehicle_id] = multi_line_speed
                        self.recorded_speeds[vehicle_id] = multi_line_speed
                        
                        print(f"[{self.stream_id}] Vehicle {vehicle_id} speed at line {speed_line_id}: {final_speed:.1f} km/h (multi-line: {multi_line_speed:.1f} km/h)")
                        
                        # Create a unique identifier for this vehicle and speed line crossing
                        violation_key = f"{vehicle_id}_{speed_line_id}"
                        
                        # Check for speed violations (over or under speed limit) using multi-line speed
                        violation_type = None
                        if multi_line_speed > self.speed_limit:
                            violation_type = "over_speed"
                        elif multi_line_speed > 0 and multi_line_speed < self.min_speed:
                            violation_type = "under_speed"
                        
                        # Record speed violation if any type and not already recorded
                        if violation_type and violation_key not in self.speed_violations:
                            try:
                                # Add to violations set to prevent duplicate recordings
                                self.speed_violations.add(violation_key)
                                
                                # Get the vehicle bbox from tracked objects
                                bbox = self._get_vehicle_bbox(vehicle_id)
                                
                                if bbox is not None and hasattr(self, 'current_frame'):
                                    # Record violation through unified manager if available
                                    if self.violation_manager:
                                        violation_id, snapshot_paths = self.violation_manager.record_speed_violation(
                                            self.current_frame, 
                                            vehicle_id, 
                                            bbox,
                                            speed=multi_line_speed,
                                            violation_type=violation_type
                                        )
                                        if violation_type == "over_speed":
                                            print(f"[{self.stream_id}] Recorded SPEED VIOLATION for vehicle {vehicle_id}: " +
                                                 f"{multi_line_speed:.1f} km/h (above limit: {self.speed_limit} km/h) at line {speed_line_id}")
                                        else:
                                            print(f"[{self.stream_id}] Recorded SPEED VIOLATION for vehicle {vehicle_id}: " +
                                                 f"{multi_line_speed:.1f} km/h (below limit: {self.min_speed} km/h) at line {speed_line_id}")
                                    else:
                                        # Standalone approach
                                        self._save_violation_snapshot(self.current_frame, bbox, vehicle_id, multi_line_speed)
                            except Exception as e:
                                print(f"[{self.stream_id}] Error recording speed violation: {str(e)}")
                    else:
                        # Update speed for display using Kalman filtered speed
                        self.speeds[vehicle_id] = final_speed

    def _get_vehicle_bbox(self, vehicle_id):
        """Get the vehicle's bounding box from multiple possible sources"""
        # First try to get from tracked_objects (most reliable and up-to-date)
        if hasattr(self, 'tracked_objects') and vehicle_id in self.tracked_objects:
            if 'box' in self.tracked_objects[vehicle_id]:
                return self.tracked_objects[vehicle_id]['box']
            elif 'bbox' in self.tracked_objects[vehicle_id]:
                return self.tracked_objects[vehicle_id]['bbox']
        
        # Then try from our internal vehicle data
        if vehicle_id in self.vehicles:
            if 'box' in self.vehicles[vehicle_id]:
                return self.vehicles[vehicle_id]['box']
            elif 'bbox' in self.vehicles[vehicle_id]:
                return self.vehicles[vehicle_id]['bbox']
        
        # If we get here, we couldn't find a valid bbox
        return None

    def _save_violation_snapshot(self, frame, bbox, vehicle_id, speed):
        """Save snapshots of speeding vehicle (closeup and context)"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Make a clean copy of the frame
            clean_frame = frame.copy()
            
            # Extract vehicle bbox coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Calculate expanded box dimensions for more context
            width, height = x2 - x1, y2 - y1
            padding_x, padding_y = int(width * 0.6), int(height * 0.6)
            
            # Ensure expanded box stays within frame bounds
            h, w = clean_frame.shape[:2]
            ex1 = max(0, x1 - padding_x)
            ey1 = max(0, y1 - padding_y)
            ex2 = min(w-1, x2 + padding_x)
            ey2 = min(h-1, y2 + padding_y)
            
            if ex2 > ex1 and ey2 > ey1:
                # Crop vehicle image with wider context
                vehicle_img = clean_frame[ey1:ey2, ex1:ex2]
                
                # Add violation text to the snapshot with font thickness 1
                cv2.putText(vehicle_img, f"SPEED: {speed:.1f} km/h", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)  # Changed from 2 to 1
                cv2.putText(vehicle_img, f"LIMIT: {self.speed_limit} km/h", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)  # Changed from 2 to 1
                cv2.putText(vehicle_img, f"Vehicle ID: {vehicle_id}", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                # Create directories if they don't exist
                snapshots_dir = os.path.join(self.output_dir, "snapshots")
                os.makedirs(snapshots_dir, exist_ok=True)
                
                # Continue with the existing snapshot saving logic
                # Save closeup snapshot
                closeup_path = os.path.join(snapshots_dir, f"speed_violation_closeup_{self.stream_id}_veh{vehicle_id}_{timestamp}.jpg")
                cv2.imwrite(closeup_path, vehicle_img)
                
                # Mark the violation on the full context image
                cv2.rectangle(clean_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Changed from 2 to 1
                cv2.putText(clean_frame, f"SPEED: {speed:.1f} km/h", 
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Save full frame context
                full_path = os.path.join(snapshots_dir, f"speed_violation_full_{self.stream_id}_veh{vehicle_id}_{timestamp}.jpg")
                cv2.imwrite(full_path, clean_frame)
                
                print(f"[{self.stream_id}] Saved speed violation snapshots for vehicle {vehicle_id} at {speed:.1f} km/h")
                return True
            else:
                print(f"[{self.stream_id}] Invalid crop dimensions for vehicle {vehicle_id}")
                return False
                
        except Exception as e:
            print(f"[{self.stream_id}] Error saving speed violation snapshot: {str(e)}")
            return False
    
    def enable_csv_logging(self, enable=True):
        """Enable or disable CSV logging (disabled by default)"""
        self.disable_csv_logging = not enable
        status = "enabled" if enable else "disabled"
        print(f"[{self.stream_id}] CSV logging {status}")
        return not self.disable_csv_logging
    
    def set_calibration(self, pixels_per_meter):
        """Set calibration parameter for speed calculation"""
        self.pixels_per_meter = pixels_per_meter
        print(f"[{self.stream_id}] Speed calibration set to {pixels_per_meter} pixels/meter")
    
    def set_speed_limit(self, limit_kmh):
        """Set the speed limit in km/h"""
        self.speed_limit = float(limit_kmh)
        print(f"[{self.stream_id}] Speed limit set to {self.speed_limit} km/h")
    
    def set_speed_limits(self, max_speed, min_speed=None):
        """Set both maximum and minimum speed limits in km/h"""
        self.speed_limit = float(max_speed)
        if min_speed is not None:
            self.min_speed = float(min_speed)
        print(f"[{self.stream_id}] Speed limits updated: max={self.speed_limit} km/h, min={self.min_speed} km/h")
    
    def reset_violations(self):
        """Reset the list of recorded speed violations"""
        self.speed_violations.clear()
        print(f"[{self.stream_id}] Speed violation records cleared")
        
    def set_font_style(self, font_style):
       
        if font_style.lower() == "simplex":
            self.font = cv2.FONT_HERSHEY_SIMPLEX
            self.font_scale = 0.4
            self.font_thickness = 1
        else:  # Default to plain
            self.font = cv2.FONT_HERSHEY_PLAIN
            self.font_scale = 0.4
            self.font_thickness = 1
    
    def enable(self):
        """Enable speed detection after speed lines are defined"""
        self.enabled = True
        print(f"[{self.stream_id}] Speed detection enabled")
        
    def disable(self):
        """Disable speed detection"""
        self.enabled = False
        print(f"[{self.stream_id}] Speed detection disabled")
        
    def is_enabled(self):
        """Check if speed detection is enabled"""
        return self.enabled
    
    def _set_default_parameters(self):
        """设置代码默认参数"""
        # 基本参数
        self.pixels_per_meter = 30  # Calibration parameter
        self.smoothing_window = 10
        self.font_style = "plain"
        self.speed_limit = 10.0  # Maximum speed limit
        self.min_speed = 5.0    # Minimum speed limit
        
        # 卡尔曼滤波器参数
        self.kalman_config = {
            'process_noise': 0.1,
            'measurement_noise': 1.0,
            'error_covariance': 1.0
        }
        
        # 速度计算参数
        self.fixed_time_step = 1.0 / 30.0  # Fixed time step in seconds
        self.max_speed_change = 5.0  # Maximum allowed change in speed between calculations (km/h)
        self.min_distance_for_calc = 3.0  # Minimum pixel distance to calculate speed (reduces noise)
        
        # 其他参数
        self.disable_csv_logging = True
    
    def _load_config_parameters(self):
        """从配置文件加载参数"""
        # 配置文件路径
        config_path = os.path.join(os.path.dirname(__file__), "..", "Config", "speed_detection_config.json")
        config_path = os.path.abspath(config_path)
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 加载参数
                if 'parameters' in config:
                    params = config['parameters']
                    # 跳过注释键（以#开头的键）
                    for key, value in params.items():
                        if not key.startswith('#'):
                            # 处理卡尔曼滤波器参数的特殊情况
                            if key in ['process_noise', 'measurement_noise', 'error_covariance']:
                                if 'kalman_config' not in self.__dict__:
                                    self.kalman_config = {}
                                self.kalman_config[key] = value
                            else:
                                if hasattr(self, key):
                                    setattr(self, key, value)
                    print(f"[{self.stream_id}] Speed detection parameters loaded from config file")
            except Exception as e:
                print(f"[{self.stream_id}] Error loading config file: {str(e)}")
        else:
            print(f"[{self.stream_id}] Config file not found at {config_path}, using default parameters")
    
    def set_parameters(self, **kwargs):
        """设置检测参数"""
        for key, value in kwargs.items():
            # 处理卡尔曼滤波器参数的特殊情况
            if key in ['process_noise', 'measurement_noise', 'error_covariance']:
                if 'kalman_config' not in self.__dict__:
                    self.kalman_config = {}
                self.kalman_config[key] = value
            else:
                if hasattr(self, key):
                    setattr(self, key, value)
        print(f"[{self.stream_id}] Speed detection parameters updated")
    
    def get_parameters(self):
        """获取当前参数"""
        params = {
            'pixels_per_meter': self.pixels_per_meter,
            'smoothing_window': self.smoothing_window,
            'font_style': self.font_style,
            'speed_limit': self.speed_limit,
            'min_speed': self.min_speed,
            'fixed_time_step': self.fixed_time_step,
            'max_speed_change': self.max_speed_change,
            'min_distance_for_calc': self.min_distance_for_calc,
            'disable_csv_logging': self.disable_csv_logging
        }
        # 添加卡尔曼滤波器参数
        params.update(self.kalman_config)
        return params