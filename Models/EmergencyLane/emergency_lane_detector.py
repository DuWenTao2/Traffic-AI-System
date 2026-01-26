import cv2
import time
import os
import datetime
from areas import AreaType
from pathlib import Path

class EmergencyLaneDetector:
    """
    Class to detect vehicles violating emergency lane
    """
    
    def __init__(self, stream_id="default", violation_time_limit=2, violation_manager=None):
        """
        Initialize the emergency lane detector
        
        Args:
            stream_id (str): Identifier for the video stream
            violation_time_limit (int): Time limit in seconds for emergency lane violation
            violation_manager: Reference to the violation manager for logging
        """
        self.stream_id = stream_id
        self.violation_time_limit = violation_time_limit
        # Track emergency lane vehicles: {track_id: {'start_time': timestamp, 'violation': bool, 'location': (x,y)}}
        self.emergency_lane_vehicles = {}
        self.violation_manager = violation_manager
        
        # Don't create log file or output directory directly
        self.log_file = None
        self.output_dir = None
        
        print(f"[{self.stream_id}] Emergency lane detector initialized with {violation_time_limit} second limit")
    
    def _setup_output_directory(self):
        """Set output directory reference but don't create it"""
        # Only create directory if using violation_manager
        if self.violation_manager:
            self.output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "emergency_lane_violations"
            self.output_dir.mkdir(exist_ok=True)
        else:
            self.output_dir = None
    
    def _initialize_log_file(self):
        """Don't create log file, just return reference"""
        # No file creation, just return a path reference
        log_file_path = f"EmergencyLane/emergency_lane_violations_{self.stream_id}.txt"
        return log_file_path
    
    def process_objects(self, frame, tracked_objects, area_manager):
        """
        Process tracked objects and detect emergency lane violations
        
        Args:
            frame: Current video frame
            tracked_objects: Dictionary of tracked objects
            area_manager: Area manager for checking if objects are in emergency lane zones
            
        Returns:
            frame: Frame with emergency lane violation annotations
        """
        current_time = time.time()
        
        # Track vehicles currently in emergency lane areas
        current_emergency_vehicles = set()
        
        # First check all tracked objects
        for track_id, track_info in tracked_objects.items():
            if len(track_info['center_points']) > 0:
                # Get the latest center point
                center_x, center_y = track_info['center_points'][-1]
                
                # Check if the object is a vehicle (class_id 2 or 3)
                if 'class_id' in track_info:
                    class_id = track_info['class_id']
                    # Consider cars, buses, trucks as vehicles
                    if class_id in [2, 3, 5, 7]:  # Car, Bus, Truck, Motorcycle
                        # Check if the object is in an emergency lane area
                        if area_manager.is_in_area(center_x, center_y, AreaType.EMERGENCY_LANE):
                            # Add to the current emergency lane vehicles
                            current_emergency_vehicles.add(track_id)
                            
                            # If this is a new vehicle in the emergency lane, add it to our tracking dict
                            if track_id not in self.emergency_lane_vehicles:
                                self.emergency_lane_vehicles[track_id] = {
                                    'start_time': current_time,
                                    'violation': False,
                                    'location': (center_x, center_y),
                                    'box': track_info['box'],
                                    'snapshot_taken': False,  # Flag to track if a snapshot was taken
                                    'class_id': class_id
                                }
                            else:
                                # Update location if it's an existing vehicle
                                self.emergency_lane_vehicles[track_id]['location'] = (center_x, center_y)
                                self.emergency_lane_vehicles[track_id]['box'] = track_info['box']
                                self.emergency_lane_vehicles[track_id]['class_id'] = class_id
        
        # Now process all vehicles we're tracking for emergency lane violations
        vehicles_to_remove = []
        
        for track_id, vehicle_data in self.emergency_lane_vehicles.items():
            # If vehicle is no longer in an emergency lane area or no longer tracked, mark for removal
            if track_id not in current_emergency_vehicles:
                vehicles_to_remove.append(track_id)
                continue
            
            # Calculate how long the vehicle has been in the emergency lane
            violation_duration = current_time - vehicle_data['start_time']
            
            # Check for violation
            if violation_duration >= self.violation_time_limit and not vehicle_data['violation']:
                # Mark as violation and log it
                vehicle_data['violation'] = True
                self._log_violation(track_id, vehicle_data['location'], vehicle_data['class_id'])
                
                # Take a snapshot of the violation if it hasn't been taken yet
                if not vehicle_data.get('snapshot_taken', False) and 'box' in vehicle_data:
                    self._save_violation_snapshot(frame, vehicle_data['box'], track_id, vehicle_data['class_id'])
                    vehicle_data['snapshot_taken'] = True
            
            # Draw information on the frame
            self._draw_violation_info(frame, track_id, vehicle_data, violation_duration)
        
        # Remove vehicles that are no longer in emergency lane areas
        for track_id in vehicles_to_remove:
            del self.emergency_lane_vehicles[track_id]
        
        return frame
    
    def _draw_violation_info(self, frame, track_id, vehicle_data, violation_duration):
        """
        Draw emergency lane violation information on the frame
        
        Args:
            frame: Current video frame
            track_id: Tracked vehicle ID
            vehicle_data: Vehicle data dictionary
            violation_duration: Duration of violation in seconds
            
        Returns:
            frame: Frame with annotations
        """
        x1, y1, x2, y2 = vehicle_data['box']
        center_x, center_y = vehicle_data['location']
        
        # Format duration as seconds
        duration_text = f"{int(violation_duration)}s"
        
        # Choose color based on violation status
        if vehicle_data['violation']:
            color = (0, 0, 255)  # Red for violation
            text = f"EMERGENCY LANE VIOLATION {duration_text}"
        else:
            # Gradient from green to yellow to red as time increases
            ratio = min(1.0, violation_duration / self.violation_time_limit)
            b = 0
            g = int(255 * (1 - ratio))
            r = int(255 * ratio)
            color = (b, g, r)
            text = f"EMERGENCY LANE {duration_text}"
        
        # Draw box around vehicle with appropriate color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add time text above the bounding box
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # If violation, show a violation indicator
        if vehicle_data['violation']:
            cv2.circle(frame, (x2-10, y1+10), 5, (0, 0, 255), -1)  # Red dot indicator
        
        return frame
    
    def _save_violation_snapshot(self, frame, bbox, vehicle_id, class_id):
        """
        Process violation but save only when using violation_manager
        
        Args:
            frame: Current video frame
            bbox: Bounding box of the violating vehicle
            vehicle_id: Tracked vehicle ID
            class_id: Class ID of the violating vehicle
            
        Returns:
            bool: Success status
        """
        try:
            # Use unified violation manager if available
            if self.violation_manager:
                # Determine vehicle type
                vehicle_type = "car" if class_id == 2 else "bus" if class_id == 5 else "truck" if class_id == 7 else "motorcycle"
                
                violation_id, snapshot_path = self.violation_manager.record_emergency_lane_violation(
                    frame, vehicle_id, bbox, vehicle_type
                )
                return True
                
            # Otherwise, just log the event without saving files
            print(f"[{self.stream_id}] Emergency lane violation detected for {vehicle_type} {vehicle_id} (local saving disabled)")
            return True
            
        except Exception as e:
            print(f"[{self.stream_id}] Error processing emergency lane violation: {str(e)}")
            return False
    
    def _log_violation(self, track_id, location, class_id):
        """
        Log an emergency lane violation but don't write to file directly
        
        Args:
            track_id: Tracked vehicle ID
            location: Location of the violation
            class_id: Class ID of the violating vehicle
            
        Returns:
            bool: Success status
        """
        try:
            # Use unified violation manager if available
            if self.violation_manager:
                # Violation will be logged centrally through the violation manager
                return True
            
            # Just print to console without file logging
            vehicle_type = "car" if class_id == 2 else "bus" if class_id == 5 else "truck" if class_id == 7 else "motorcycle"
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{self.stream_id}] Emergency lane violation detected for {vehicle_type} {track_id} at {timestamp}")
            return True
        except Exception as e:
            print(f"[{self.stream_id}] Error logging emergency lane violation: {str(e)}")
            return False
    
    def set_violation_time_limit(self, seconds):
        """
        Update the time limit for emergency lane violations
        
        Args:
            seconds: New time limit in seconds
            
        Returns:
            bool: Success status
        """
        self.violation_time_limit = seconds
        print(f"[{self.stream_id}] Emergency lane violation time limit updated to {seconds} seconds")
        return True
    
    def toggle_detection(self):
        """
        Toggle emergency lane detection on/off
        
        Returns:
            bool: New detection status
        """
        # This method is kept for consistency with other detectors
        # Detection is always on when this method is called
        return True