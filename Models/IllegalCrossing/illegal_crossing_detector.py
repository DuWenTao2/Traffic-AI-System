import cv2
import time
import os
import datetime
from areas import AreaType
from pathlib import Path

class IllegalCrossingDetector:
    """
    Class to detect illegal crossings by pedestrians and non-motor vehicles
    """
    
    def __init__(self, stream_id="default", violation_time_limit=5, violation_manager=None):
        """
        Initialize the illegal crossing detector
        
        Args:
            stream_id (str): Identifier for the video stream
            violation_time_limit (int): Time limit in seconds for illegal crossing
            violation_manager: Reference to the violation manager for logging
        """
        self.stream_id = stream_id
        self.violation_time_limit = violation_time_limit
        # Track illegal crossing objects: {track_id: {'start_time': timestamp, 'violation': bool, 'location': (x,y)}}
        self.crossing_objects = {}
        self.violation_manager = violation_manager
        
        # Don't create log file or output directory directly
        self.log_file = None
        self.output_dir = None
        
        print(f"[{self.stream_id}] Illegal crossing detector initialized with {violation_time_limit} second limit")
    
    def _setup_output_directory(self):
        """Set output directory reference but don't create it"""
        # Only create directory if using violation_manager
        if self.violation_manager:
            self.output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "illegal_crossing_violations"
            self.output_dir.mkdir(exist_ok=True)
        else:
            self.output_dir = None
    
    def _initialize_log_file(self):
        """Don't create log file, just return reference"""
        # No file creation, just return a path reference
        log_file_path = f"IllegalCrossing/illegal_crossing_violations_{self.stream_id}.txt"
        return log_file_path
    
    def process_objects(self, frame, tracked_objects, area_manager):
        """
        Process tracked objects and detect illegal crossings
        
        Args:
            frame: Current video frame
            tracked_objects: Dictionary of tracked objects
            area_manager: Area manager for checking if objects are in illegal crossing zones
            
        Returns:
            frame: Frame with illegal crossing annotations
        """
        current_time = time.time()
        
        # Track objects currently in illegal crossing areas
        current_crossing_ids = set()
        
        # First check all tracked objects
        for track_id, track_info in tracked_objects.items():
            if len(track_info['center_points']) > 0:
                # Get the latest center point
                center_x, center_y = track_info['center_points'][-1]
                
                # Check if the object is a person or non-motor vehicle (class_id 0 or 4)
                if 'class_id' in track_info:
                    class_id = track_info['class_id']
                    if class_id in [0, 4]:  # person or bicycle
                        # Check if the object is in an illegal crossing area
                        if area_manager.is_in_area(center_x, center_y, AreaType.ILLEGAL_CROSSING):
                            # Add to the current crossing objects
                            current_crossing_ids.add(track_id)
                            
                            # If this is a new object in the crossing area, add it to our tracking dict
                            if track_id not in self.crossing_objects:
                                self.crossing_objects[track_id] = {
                                    'start_time': current_time,
                                    'violation': False,
                                    'location': (center_x, center_y),
                                    'box': track_info['box'],
                                    'snapshot_taken': False,  # Flag to track if a snapshot was taken
                                    'class_id': class_id
                                }
                            else:
                                # Update location if it's an existing object
                                self.crossing_objects[track_id]['location'] = (center_x, center_y)
                                self.crossing_objects[track_id]['box'] = track_info['box']
                                self.crossing_objects[track_id]['class_id'] = class_id
        
        # Now process all objects we're tracking for illegal crossing violations
        objects_to_remove = []
        
        for track_id, object_data in self.crossing_objects.items():
            # If object is no longer in an illegal crossing area or no longer tracked, mark for removal
            if track_id not in current_crossing_ids:
                objects_to_remove.append(track_id)
                continue
            
            # Calculate how long the object has been in the crossing area
            crossing_duration = current_time - object_data['start_time']
            
            # Check for violation
            if crossing_duration >= self.violation_time_limit and not object_data['violation']:
                # Mark as violation and log it
                object_data['violation'] = True
                self._log_violation(track_id, object_data['location'], object_data['class_id'])
                
                # Take a snapshot of the violation if it hasn't been taken yet
                if not object_data.get('snapshot_taken', False) and 'box' in object_data:
                    self._save_violation_snapshot(frame, object_data['box'], track_id, object_data['class_id'])
                    object_data['snapshot_taken'] = True
            
            # Draw information on the frame
            self._draw_crossing_info(frame, track_id, object_data, crossing_duration)
        
        # Remove objects that are no longer in illegal crossing areas
        for track_id in objects_to_remove:
            del self.crossing_objects[track_id]
        
        return frame
    
    def _draw_crossing_info(self, frame, track_id, object_data, crossing_duration):
        """
        Draw minimal illegal crossing information on the frame
        
        Args:
            frame: Current video frame
            track_id: Tracked object ID
            object_data: Object data dictionary
            crossing_duration: Duration of crossing in seconds
            
        Returns:
            frame: Frame with annotations
        """
        x1, y1, x2, y2 = object_data['box']
        center_x, center_y = object_data['location']
        
        # Format duration as seconds
        duration_text = f"{int(crossing_duration)}s"
        
        # Choose color based on violation status
        if object_data['violation']:
            color = (0, 0, 255)  # Red for violation
            text = f"ILLEGAL CROSSING {duration_text}"
        else:
            # Gradient from green to yellow to red as time increases
            ratio = min(1.0, crossing_duration / self.violation_time_limit)
            b = 0
            g = int(255 * (1 - ratio))
            r = int(255 * ratio)
            color = (b, g, r)
            text = f"CROSSING {duration_text}"
        
        # Draw box around object with appropriate color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add time text above the bounding box
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # If violation, show a violation indicator
        if object_data['violation']:
            cv2.circle(frame, (x2-10, y1+10), 5, (0, 0, 255), -1)  # Red dot indicator
        
        return frame
    
    def _save_violation_snapshot(self, frame, bbox, object_id, class_id):
        """
        Process violation but save only when using violation_manager
        
        Args:
            frame: Current video frame
            bbox: Bounding box of the violating object
            object_id: Tracked object ID
            class_id: Class ID of the violating object
            
        Returns:
            bool: Success status
        """
        try:
            # Use unified violation manager if available
            if self.violation_manager:
                # Determine object type
                object_type = "person" if class_id == 0 else "non-motor vehicle"
                
                violation_id, snapshot_path = self.violation_manager.record_illegal_crossing_violation(
                    frame, object_id, bbox, object_type
                )
                return True
                
            # Otherwise, just log the event without saving files
            print(f"[{self.stream_id}] Illegal crossing violation detected for {object_type} {object_id} (local saving disabled)")
            return True
            
        except Exception as e:
            print(f"[{self.stream_id}] Error processing illegal crossing violation: {str(e)}")
            return False
    
    def _log_violation(self, track_id, location, class_id):
        """
        Log an illegal crossing violation but don't write to file directly
        
        Args:
            track_id: Tracked object ID
            location: Location of the violation
            class_id: Class ID of the violating object
            
        Returns:
            bool: Success status
        """
        try:
            # Use unified violation manager if available
            if self.violation_manager:
                # Violation will be logged centrally through the violation manager
                return True
            
            # Just print to console without file logging
            object_type = "person" if class_id == 0 else "non-motor vehicle"
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{self.stream_id}] Illegal crossing violation detected for {object_type} {track_id} at {timestamp}")
            return True
        except Exception as e:
            print(f"[{self.stream_id}] Error logging illegal crossing violation: {str(e)}")
            return False
    
    def set_violation_time_limit(self, seconds):
        """
        Update the time limit for illegal crossing violations
        
        Args:
            seconds: New time limit in seconds
            
        Returns:
            bool: Success status
        """
        self.violation_time_limit = seconds
        print(f"[{self.stream_id}] Illegal crossing violation time limit updated to {seconds} seconds")
        return True
    
    def toggle_detection(self):
        """
        Toggle illegal crossing detection on/off
        
        Returns:
            bool: New detection status
        """
        # This method is kept for consistency with other detectors
        # Detection is always on when this method is called
        return True