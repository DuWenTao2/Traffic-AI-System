# Vision Patrol - Driver Violation Detection Module
import cv2
import numpy as np
import os
from datetime import datetime
import csv
from ultralytics import YOLO
import cvzone
import traceback

class DriverViolationDetector:
    def __init__(self, stream_id="default", violation_manager=None):
        self.stream_id = stream_id
        self.violated_ids = []
        self.violation_timeout = 100  # Frames before allowing new violation for same ID
        self.violation_counters = {}  # Track violation counter per ID
        self.tracked_vehicles = {}  # Track vehicles in detection area
        
        # Violation tracking to prevent duplicates
        self.processed_violations = set()  # IDs already logged
        self.permanent_violations = set()  # IDs permanently flagged
        self.violation_timestamps = {}     # Violation occurrence timestamps
        self.last_inference_time = {}      # Last inference time per vehicle
        self.min_inference_interval = 1.0  # Minimum seconds between inferences
        
        # Unified violation manager integration
        self.violation_manager = violation_manager
        
        # Load configuration to set default enabled state
        self.detection_enabled = self._load_configured_state()
        
        # Load driver violation detection model
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "weights", "Driver_Violation_Detection", "best.pt")
        # Make sure model defaults to None if loading fails
        try:
            self.model = YOLO(self.model_path)
            print(f"[{self.stream_id}] Driver violation detector model loaded from {self.model_path}")
            status = "ENABLED" if self.detection_enabled else "DISABLED"
            print(f"[{self.stream_id}] Driver violation detection is {status} by configuration")
        except Exception as e:
            print(f"[{self.stream_id}] Error loading driver violation detection model: {str(e)}")
            self.model = None
            self.detection_enabled = False
            
        # Class names from model
        self.driver_class_names = {
            0: 'mobile',
            1: 'person',
            2: 'seatbelt'
        }
        
        # Don't set up local directories
        self.output_dir = None
        self.log_file = None
        
        # Detection status flag
        self.has_detection = False
    
    def _load_configured_state(self):
        """Load configured enabled state from unified config file"""
        import json
        config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "所有检测功能开关控制配置文件.json")
        
        # Default to False if config file not found or parsing fails
        default_enabled = False
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if "detection_settings" in config:
                        settings = config["detection_settings"]
                        if "driver_violation_detection" in settings:
                            return settings["driver_violation_detection"]
            print(f"[{self.stream_id}] Config file not found or driver_violation_detection setting not present, using default: {default_enabled}")
        except Exception as e:
            print(f"[{self.stream_id}] Error loading driver violation detection configuration: {str(e)}")
        
        return default_enabled
    
    def setup_output_directory(self, base_dir=None):
        """Set directory reference but don't create it"""
        # Only create directory if using violation_manager
        if self.violation_manager:
            if base_dir is None:
                # Get the directory where this script is located
                current_dir = os.path.dirname(os.path.abspath(__file__))
                base_dir = os.path.join(current_dir, 'violations')
            
            today_date = datetime.now().strftime('%Y-%m-%d')
            self.output_dir = os.path.join(base_dir, today_date)
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"[{self.stream_id}] Driver violation images will be saved to: {self.output_dir}")
        else:
            self.output_dir = None
            print(f"[{self.stream_id}] Local storage of driver violations is disabled")
        
        return self.output_dir
    
    def setup_log_file(self):
        """Set log file reference but don't create the file"""
        # Only set up log file if using violation_manager
        if self.violation_manager:
            # Get the directory where this script is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(current_dir, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            # Create a log file name with date
            today_date = datetime.now().strftime('%Y-%m-%d')
            self.log_file = os.path.join(logs_dir, f'driver_violations_{today_date}.csv')
            
            # Create log file with headers if it doesn't exist
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'Date', 
                        'Time', 
                        'Violation Type', 
                        'Camera ID', 
                        'Vehicle ID', 
                        'Vehicle Type',
                        'Image Path'
                    ])
            
            print(f"[{self.stream_id}] Driver violations will be logged to: {self.log_file}")
        else:
            self.log_file = None
            print(f"[{self.stream_id}] Local logging of driver violations is disabled")
            
        return self.log_file
    
    def log_violation(self, violation_type, vehicle_id, image_path, vehicle_type="car"):
        """Log a violation - use unified manager if available"""
        # Skip if already permanently processed or in current session
        if vehicle_id in self.permanent_violations or vehicle_id in self.processed_violations:
            print(f"[{self.stream_id}] Skipping duplicate driver violation for vehicle {vehicle_id} (already processed)")
            return False
            
        # Record current violation time
        current_time = datetime.now().timestamp()
        self.violation_timestamps[vehicle_id] = current_time
            
        # Skip local logging if using unified manager
        if self.violation_manager:
            # Mark as processed and permanent
            self.processed_violations.add(vehicle_id)
            self.permanent_violations.add(vehicle_id)  # Add to permanent violations
            print(f"[{self.stream_id}] Vehicle {vehicle_id} added to permanent violations list")
            return True
            
        # Otherwise, use local file for snapshots only - the CSV part is removed
        # Mark as processed and permanent
        self.processed_violations.add(vehicle_id)
        self.permanent_violations.add(vehicle_id)
        print(f"[{self.stream_id}] Vehicle {vehicle_id} added to permanent violations list")
        return True
    
    def process_frame(self, frame, tracked_objects, area_manager):
        """Process the current frame for driver violations"""
        # Early return if conditions aren't met
        if not self.detection_enabled or frame is None or self.model is None:
            # Reset detection status
            self.has_detection = False
            return frame
        
        # Reset detection status for each frame
        self.has_detection = False
        
        try:
            # Create a defensive copy of frame to prevent issues
            result_frame = frame.copy() if frame is not None else None
            if result_frame is None:
                return frame
                
            # Directly run driver violation detection on the entire frame
            driver_results = self.model(result_frame, verbose=False)
            
            # Check for driver violation detection results
            if len(driver_results) > 0 and hasattr(driver_results[0], 'boxes') and driver_results[0].boxes is not None:
                if len(driver_results[0].boxes) > 0:
                    boxes = driver_results[0].boxes.xyxy.cpu().numpy()
                    cls = driver_results[0].boxes.cls.cpu().numpy()
                    confs = driver_results[0].boxes.conf.cpu().numpy()
                    
                    # Process all driver violation detections
                    violations = {}
                    violation_boxes = {}
                    for i, box in enumerate(boxes):
                        d_x1, d_y1, d_x2, d_y2 = box.astype(int)
                        d_cls = int(cls[i])
                        d_conf = float(confs[i])
                        
                        # Get class name and color
                        d_label = self.driver_class_names.get(d_cls, f"Class {d_cls}")
                        
                        # Set color based on violation type
                        if d_cls == 0:  # mobile - using phone (violation)
                            d_color = (0, 0, 255)  # Red
                            violations['mobile'] = d_conf
                            violation_boxes['mobile'] = (d_x1, d_y1, d_x2, d_y2)
                        elif d_cls == 1:  # person - driver present (normal)
                            d_color = (0, 255, 0)  # Green
                            violations['person'] = d_conf
                        elif d_cls == 2:  # seatbelt - wearing seatbelt (normal)
                            d_color = (0, 255, 0)  # Green
                            violations['seatbelt'] = d_conf
                            violation_boxes['seatbelt'] = (d_x1, d_y1, d_x2, d_y2)
                        else:
                            d_color = (255, 255, 0)  # Yellow
                        
                        # Draw detection box and label
                        cv2.rectangle(result_frame, (d_x1, d_y1), (d_x2, d_y2), d_color, 2)
                        cvzone.putTextRect(result_frame, f"{d_label}: {d_conf:.2f}", (d_x1, d_y1 - 10), 
                                         scale=0.8, thickness=1, colorR=d_color)
                    
                    # Check if both mobile and seatbelt violations are detected
                    # Note: seatbelt not detected means no seatbelt is being worn
                    if 'mobile' in violations and 'seatbelt' not in violations:
                        # Create a unique vehicle ID for this violation
                        vehicle_id = f"driver_violation_{datetime.now().timestamp()}"
                        
                        # Skip if already processed
                        if vehicle_id in self.processed_violations or vehicle_id in self.permanent_violations:
                            print(f"[{self.stream_id}] Skipping duplicate driver violation")
                            return result_frame
                        
                        # Save a snapshot of the violation
                        image_path = self.save_violation_evidence(result_frame, vehicle_id, tracked_objects)
                        
                        # Log the violation - internally handles marking as processed
                        violation_type = "DRIVER_VIOLATION"
                        if self.log_violation(violation_type, vehicle_id, image_path, "car"):
                            # Set detection status to True
                            self.has_detection = True
                            
                            # Add to violated IDs if not already there
                            if vehicle_id not in self.violated_ids:
                                self.violated_ids.append(vehicle_id)
                            
                            # Draw violation box on frame
                            if 'mobile' in violation_boxes:
                                x1, y1, x2, y2 = violation_boxes['mobile']
                                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cvzone.putTextRect(result_frame, f"DRIVER VIOLATION", (x1, y1 - 10), 
                                                 scale=1, thickness=2, colorR=(0, 0, 255))
                            
                            print(f"[{self.stream_id}] Driver violation detected: mobile usage without seatbelt")
            
            # Display violation stats on frame
            cv2.putText(result_frame, f"Driver Violations: {len(self.permanent_violations)}", 
                      (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return result_frame
            
        except Exception as e:
            print(f"[{self.stream_id}] Error in driver violation detection process_frame: {str(e)}")
            traceback.print_exc()
            return frame
    
    def save_violation_evidence(self, frame, vehicle_id, tracked_objects):
        """Process violation but save only when using violation_manager"""
        try:
            # Use unified violation manager if available
            if self.violation_manager:
                # Determine vehicle box and type
                vehicle_box = None
                vehicle_type = "car"
                
                # Try to get vehicle box from tracked_objects if available
                if isinstance(tracked_objects, dict) and vehicle_id in tracked_objects and 'box' in tracked_objects[vehicle_id]:
                    vehicle_box = tracked_objects[vehicle_id]['box']
                    if 'class_id' in tracked_objects[vehicle_id]:
                        class_id = tracked_objects[vehicle_id]['class_id']
                        if class_id == 3:
                            vehicle_type = "truck"
                        elif class_id == 5:
                            vehicle_type = "bus"
                        elif class_id == 7:
                            vehicle_type = "motorcycle"
                        else:
                            vehicle_type = "car"
                else:
                    # If no vehicle box available, use the entire frame
                    vehicle_box = (0, 0, frame.shape[1], frame.shape[0])
                
                # Expand the box to capture the entire vehicle or frame
                x1, y1, x2, y2 = vehicle_box
                width, height = x2 - x1, y2 - y1
                
                # Add padding to ensure the entire vehicle is captured
                expanded_box = (
                    max(0, x1 - int(width * 0.1)),  # left
                    max(0, y1 - int(height * 0.1)),  # top
                    min(frame.shape[1], x2 + int(width * 0.1)),  # right
                    min(frame.shape[0], y2 + int(height * 0.1))  # bottom
                )
                
                try:
                    # Use the same method signature as helmet detection for compatibility
                    violation_id, snapshot_paths = self.violation_manager.record_driver_violation(
                        frame.copy(),  # Use a copy to prevent modifications
                        vehicle_id, 
                        expanded_box, 
                        vehicle_type=vehicle_type
                    )
                    
                    # Improved handling of different return types
                    if snapshot_paths is None:
                        return f"driver_{self.stream_id}_{violation_id}_none"
                    elif isinstance(snapshot_paths, dict) and 'full' in snapshot_paths:
                        return snapshot_paths['full']
                    elif isinstance(snapshot_paths, str):
                        return snapshot_paths
                    else:
                        return f"driver_{self.stream_id}_{violation_id}"
                    
                except Exception as e:
                    print(f"[{self.stream_id}] Error in violation manager: {str(e)}")
                    traceback.print_exc()
                    # Return a placeholder path
                    return f"error_driver_{vehicle_id}_{datetime.now().strftime('%H%M%S')}"
            
            # Just return a placeholder path if not using violation_manager
            return f"disabled_local_storage_{vehicle_id}_{datetime.now().strftime('%H%M%S')}"
            
        except Exception as e:
            print(f"[{self.stream_id}] Error processing driver violation evidence: {str(e)}")
            traceback.print_exc()
            return f"error_saving_violation_{vehicle_id}_{datetime.now().strftime('%H%M%S')}"
    
    def toggle_detection(self):
        """Toggle driver violation detection on/off"""
        self.detection_enabled = not self.detection_enabled
        status = "ENABLED" if self.detection_enabled else "DISABLED"
        print(f"[{self.stream_id}] Driver violation detection {status}")
        
        # If detection was just disabled, clear all tracking data
        if not self.detection_enabled:
            self.tracked_vehicles = {}
            self.last_inference_time = {}  # Clear inference times
            # Don't clear permanent violations when disabling - 
            # they should persist until manually reset
            print(f"[{self.stream_id}] Cleared vehicle tracking data")
            
        return self.detection_enabled
    
    def reset_violations(self):
        """Reset all violation counters and tracking data"""
        self.violated_ids = []
        self.tracked_vehicles = {}
        self.processed_violations = set()  # Clear processed violations
        self.permanent_violations = set()  # Clear permanent violations
        self.violation_timestamps = {}     # Clear violation timestamps
        self.last_inference_time = {}      # Clear inference times
        print(f"[{self.stream_id}] Reset all driver violation records and tracking data")
