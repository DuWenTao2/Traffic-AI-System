import cv2
import numpy as np
import os
import logging
import time
from datetime import datetime
from enum import Enum
import sys
from pathlib import Path

# Import the AreaType with proper error handling
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
# Add Processing Models directory to path
processing_models_dir = os.path.join(parent_dir, "Processing Models")
sys.path.append(processing_models_dir)

try:
    from areas import AreaType
except ImportError:
    try:
        # Try importing from Processing Models directory
        from areas import AreaType
    except ImportError:
        # Define a backup AreaType if import fails
        class AreaType(Enum):
            DETECTION = 1
            SPEED = 2
            WRONG_DIRECTION = 3
            PARKING = 4
            TRAFFIC_LINE = 5
            TRAFFIC_SIGN = 6
            ILLEGAL_CROSSING = 7
            EMERGENCY_LANE = 8
            LEFT_LANE = 9
            CENTER_LANE = 10
            RIGHT_LANE = 11
            CUSTOM = 12

class Direction(Enum):
    """Enum for vehicle direction relative to a line pair"""
    FORWARD = 1    # Vehicle moving in allowed direction
    BACKWARD = -1  # Vehicle moving in wrong direction
    UNKNOWN = 0    # Direction not yet determined

class WrongDirectionDetector:
    """
    Detector for vehicles moving in the wrong direction through defined line pairs.
    Uses AreaManager's line definitions to create virtual gates that detect wrong-way traffic.
    """
    def __init__(self, stream_id="default", conf_threshold=0.0, violation_manager=None):
        # Basic configuration
        self.stream_id = stream_id
        self.conf_threshold = conf_threshold
        self.detection_enabled = False
        self.violation_manager = violation_manager
        
        # Disable logging setup
        self.logger = self._setup_dummy_logger()
        
        # State tracking for multi-lane detection
        self.lane_config = {}               # {lane_id: {left_line, center_line, right_line, direction}}
        self.vehicle_lane_assignment = {}   # {vehicle_id: lane_id}
        self.vehicle_direction_vector = {}   # {vehicle_id: (dx, dy)}
        self.lane_directions = {}            # {lane_id: expected_direction_vector}
        self.wrong_way_vehicles = set()     # Set of vehicle IDs going wrong way
        self.snapshots_counter = 0
        
        self.logger.info(f"Wrong direction detector initialized")
        self.logger.info(f"Detection will auto-enable when lane lines are configured")
    
    def _setup_dummy_logger(self):
        """Set up a dummy logger that doesn't write to file"""
        logger = logging.getLogger(f"wrong_direction_{self.stream_id}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Add a NullHandler which does nothing
            logger.addHandler(logging.NullHandler())
        
        return logger
    

    
    def configure_lane_lines(self, area_manager):
        """Configure lane lines based on lines defined in the AreaManager"""
        try:
            # Skip reconfiguration if lane lines haven't changed
            if hasattr(self, '_last_area_config_hash'):
                current_hash = hash(str(area_manager.areas.get(AreaType.LEFT_LANE, [])) + 
                                   str(area_manager.areas.get(AreaType.CENTER_LANE, [])) + 
                                   str(area_manager.areas.get(AreaType.RIGHT_LANE, [])))
                if current_hash == self._last_area_config_hash:
                    return True
            
            self.lane_config = {}
            self.lane_directions = {}
            
            if not self._validate_area_manager(area_manager):
                return False
            
            # Get lane lines
            left_lanes = self._get_lane_lines(area_manager, AreaType.LEFT_LANE)
            center_lanes = self._get_lane_lines(area_manager, AreaType.CENTER_LANE)
            right_lanes = self._get_lane_lines(area_manager, AreaType.RIGHT_LANE)
            
            # Validate lane lines
            if not self._validate_lane_lines(left_lanes, center_lanes, right_lanes):
                return False
            
            # Create lane configurations
            success = self._create_lane_configurations(left_lanes, center_lanes, right_lanes)
            
            # Store hash of current configuration
            if success:
                self._last_area_config_hash = hash(str(left_lanes) + str(center_lanes) + str(right_lanes))
                self.detection_enabled = True
                self.logger.info(f"Wrong direction detection automatically ENABLED with {len(self.lane_config)} lanes")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error configuring lane lines: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _validate_area_manager(self, area_manager):
        """Validate the area manager is properly initialized"""
        if area_manager is None or not hasattr(area_manager, 'areas'):
            self.logger.error("Invalid area manager")
            return False
        return True
    

    
    def _get_lane_lines(self, area_manager, area_type):
        """Get lane lines from area manager"""
        # Try to get area by name if enum comparison fails
        area_found = False
        area_name = area_type.name if hasattr(area_type, 'name') else str(area_type)
        
        # Check if area exists by direct comparison
        if area_type in area_manager.areas:
            area_found = True
            lane_lines = area_manager.areas[area_type]
        else:
            # Try to find area by name
            for key in area_manager.areas:
                key_name = key.name if hasattr(key, 'name') else str(key)
                if key_name == area_name:
                    area_found = True
                    lane_lines = area_manager.areas[key]
                    break
            else:
                # Area not found
                area_manager.areas[area_type] = []
                return []
        
        if not isinstance(lane_lines, list):
            return []
        
        return lane_lines
    
    def _validate_lane_lines(self, left_lanes, center_lanes, right_lanes):
        """Validate lane lines configuration"""
        # Check if we have at least one lane configuration
        has_left = len(left_lanes) > 0
        has_center = len(center_lanes) > 0
        has_right = len(right_lanes) > 0
        
        return has_left or has_center or has_right
    
    def _create_lane_configurations(self, left_lanes, center_lanes, right_lanes):
        """Create lane configurations based on lane lines"""
        # Create a single lane configuration that considers all three lane types together
        lane_id = 0
        
        # We'll create one combined lane system considering all three types
        if left_lanes or center_lanes or right_lanes:
            # Use the first available lane line to determine the general direction
            main_direction = None
            
            # Prioritize center lane, then left, then right for determining direction
            if center_lanes:
                points = center_lanes[0].get('points', [])
                if len(points) >= 2:
                    main_direction = self._calculate_lane_direction(points)
            elif left_lanes:
                points = left_lanes[0].get('points', [])
                if len(points) >= 2:
                    main_direction = self._calculate_lane_direction(points)
            elif right_lanes:
                points = right_lanes[0].get('points', [])
                if len(points) >= 2:
                    main_direction = self._calculate_lane_direction(points)
            
            if main_direction:
                self.lane_config[lane_id] = {
                    'left_line': left_lanes[0].get('points', []) if left_lanes else [],
                    'center_line': center_lanes[0].get('points', []) if center_lanes else [],
                    'right_line': right_lanes[0].get('points', []) if right_lanes else [],
                    'type': 'combined_lane'
                }
                self.lane_directions[lane_id] = main_direction
                lane_id += 1
        
        return len(self.lane_config) > 0
    
    def _calculate_lane_direction(self, lane_points):
        """Calculate the expected direction vector for a lane"""
        if len(lane_points) < 2:
            return (1, 0)  # Default direction: right
        
        # Calculate direction vector from lane line
        p1, p2 = lane_points[0], lane_points[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Normalize vector
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            return (dx/length, dy/length)
        
        return (1, 0)
    

    
    def process_frame(self, frame, tracked_objects, area_manager):
        """Process a frame to detect vehicles moving in wrong direction"""
        if frame is None:
            return None
        
        # Show detection status
        processed_frame = self._display_status(frame)
        
        if not self.detection_enabled:
            return processed_frame
        
        # Configure lane lines if needed
        if not self.lane_config:
            # Configure lane lines
            lane_success = self.configure_lane_lines(area_manager)
            if not lane_success:
                return processed_frame
        
        # Draw lane lines
        if self.lane_config:
            self._draw_lane_lines(processed_frame, area_manager)
        
        # Validate tracked_objects
        if not self._validate_tracked_objects(tracked_objects):
            return processed_frame
        
        # Process each vehicle for wrong direction
        if self.lane_config:
            self._process_vehicles_multi_lane(processed_frame, tracked_objects, area_manager)
        
        # Display count of wrong way vehicles
        self._display_wrong_way_count(processed_frame)
        
        return processed_frame
    
    def _display_status(self, frame):
        """Display detection status on frame"""
        processed_frame = frame.copy()
        status_text = "WRONG DIRECTION: " + ("ENABLED" if self.detection_enabled else "DISABLED")
        color = (0, 255, 0) if self.detection_enabled else (0, 0, 255)
        cv2.putText(processed_frame, status_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return processed_frame
    
    def _draw_lane_lines(self, frame, area_manager):
        """Draw all lane lines on the frame"""
        # Draw left lane lines
        if AreaType.LEFT_LANE in area_manager.areas:
            for lane in area_manager.areas[AreaType.LEFT_LANE]:
                points = lane.get('points', [])
                if len(points) == 2:
                    cv2.line(frame, points[0], points[1], (0, 128, 255), 1)
                    for point in points:
                        cv2.circle(frame, point, 2, (255, 255, 255), -1)
        
        # Draw center lane lines
        if AreaType.CENTER_LANE in area_manager.areas:
            for lane in area_manager.areas[AreaType.CENTER_LANE]:
                points = lane.get('points', [])
                if len(points) == 2:
                    cv2.line(frame, points[0], points[1], (255, 128, 0), 1)
                    for point in points:
                        cv2.circle(frame, point, 2, (255, 255, 255), -1)
        
        # Draw right lane lines
        if AreaType.RIGHT_LANE in area_manager.areas:
            for lane in area_manager.areas[AreaType.RIGHT_LANE]:
                points = lane.get('points', [])
                if len(points) == 2:
                    cv2.line(frame, points[0], points[1], (0, 255, 128), 1)
                    for point in points:
                        cv2.circle(frame, point, 2, (255, 255, 255), -1)
    
    def _validate_tracked_objects(self, tracked_objects):
        """Validate that tracked_objects is a valid dictionary"""
        return tracked_objects and isinstance(tracked_objects, dict)
    
    def _process_vehicles_multi_lane(self, frame, tracked_objects, area_manager):
        """Process each vehicle to detect wrong direction movement in multi-lane scenario"""
        for vehicle_id, vehicle_info in tracked_objects.items():
            if not self._has_sufficient_tracking_points(vehicle_info):
                continue
            
            # Get current and previous position
            current_pos = vehicle_info['center_points'][-1]
            prev_pos = vehicle_info['center_points'][-2]
            
            # Assign vehicle to lane
            lane_id = self._assign_vehicle_to_lane(vehicle_id, current_pos, area_manager)
            if lane_id is None:
                continue
            
            # Calculate vehicle direction vector
            direction_vector = self._calculate_vehicle_direction(prev_pos, current_pos)
            self.vehicle_direction_vector[vehicle_id] = direction_vector
            
            # Check if vehicle is moving in wrong direction
            if self._is_wrong_direction(vehicle_id, lane_id, direction_vector):
                # Handle wrong direction violation
                if vehicle_id not in self.wrong_way_vehicles:
                    self._handle_new_violation(frame, vehicle_id, lane_id, vehicle_info)
            
            # Highlight vehicle if it's going wrong way
            if vehicle_id in self.wrong_way_vehicles and 'bbox' in vehicle_info:
                self._highlight_wrong_way_vehicle(frame, vehicle_info['bbox'])
        
        # Clean up old lane assignments to save memory
        current_time = time.time()
        vehicles_to_remove = []
        for vehicle_id, assignment in self.vehicle_lane_assignment.items():
            if isinstance(assignment, dict) and 'last_assigned' in assignment:
                if current_time - assignment['last_assigned'] > 60:  # Remove assignments older than 60 seconds
                    vehicles_to_remove.append(vehicle_id)
        
        for vehicle_id in vehicles_to_remove:
            if vehicle_id in self.vehicle_lane_assignment:
                del self.vehicle_lane_assignment[vehicle_id]
    
    def _has_sufficient_tracking_points(self, vehicle_info):
        """Check if vehicle has enough tracking points for analysis"""
        return 'center_points' in vehicle_info and len(vehicle_info['center_points']) >= 2
    

    
    def _assign_vehicle_to_lane(self, vehicle_id, position, area_manager):
        """Assign vehicle to a specific lane based on its position"""
        # Check if vehicle was recently assigned to a lane and position hasn't changed much
        if vehicle_id in self.vehicle_lane_assignment:
            # Simple distance check - if vehicle hasn't moved far, keep the same lane assignment
            if 'last_position' in self.vehicle_lane_assignment.get(vehicle_id, {}):
                last_pos = self.vehicle_lane_assignment[vehicle_id].get('last_position', position)
                distance_moved = ((position[0] - last_pos[0]) ** 2 + (position[1] - last_pos[1]) ** 2) ** 0.5
                if distance_moved < 10:  # If vehicle moved less than 10 pixels
                    return self.vehicle_lane_assignment[vehicle_id]['lane_id']
        
        # Since we now have a single combined lane configuration, return the first (and only) lane if it exists
        if self.lane_config:
            lane_id = next(iter(self.lane_config.keys()))  # Get the first lane ID
            # Store lane assignment with timestamp and position
            self.vehicle_lane_assignment[vehicle_id] = {
                'lane_id': lane_id,
                'last_position': position,
                'last_assigned': time.time()
            }
            return lane_id
        
        return None
    
    def _distance_from_line(self, point, line_start, line_end):
        """Calculate distance from a point to a line"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        
        if denominator == 0:
            return float('inf')
        
        return numerator / denominator
    
    def _calculate_vehicle_direction(self, prev_pos, current_pos):
        """Calculate vehicle direction vector"""
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        
        # Normalize vector
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length > 0:
            return (dx / length, dy / length)
        
        return (0, 0)
    
    def _is_wrong_direction(self, vehicle_id, lane_id, direction_vector):
        """Check if vehicle is moving in wrong direction"""
        if lane_id not in self.lane_directions:
            return False
        
        expected_direction = self.lane_directions[lane_id]
        
        # Calculate dot product to determine direction similarity
        dot_product = direction_vector[0] * expected_direction[0] + direction_vector[1] * expected_direction[1]
        
        # Use a more conservative threshold to reduce false positives
        # A dot product close to -1 means completely opposite directions
        # Using -0.7 instead of -0.5 to reduce sensitivity
        return dot_product < -0.75  # More conservative threshold for wrong direction
    
    def _highlight_wrong_way_vehicle(self, frame, bbox):
        """Highlight a vehicle going the wrong way with thinner box"""
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)  # Reduced thickness to 1
        cv2.putText(frame, "WRONG WAY", 
                   (int(bbox[0]), int(bbox[1]) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)  # Smaller font
    

    
    def _handle_new_violation(self, frame, vehicle_id, pair_id, vehicle_info):
        """Handle a newly detected wrong way violation"""
        self.wrong_way_vehicles.add(vehicle_id)
        
        # Add detection timestamp to the vehicle info
        if 'detection_time' not in vehicle_info:
            vehicle_info['detection_time'] = time.time()
        
        # Log the violation
        self._log_wrong_direction(vehicle_id, pair_id)
        
        # Take snapshot if bbox available
        if 'bbox' in vehicle_info:
            self._save_wrong_way_vehicle(frame, vehicle_info['bbox'], vehicle_id, pair_id)
    
    def _display_wrong_way_count(self, frame):
        """Display count of wrong way vehicles on the frame"""
        cv2.putText(frame, f"Wrong Way Vehicles: {len(self.wrong_way_vehicles)}", 
                  (frame.shape[1] - 200, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    def _line_segments_intersect(self, p1, p2, p3, p4):
        """Check if line segments (p1,p2) and (p3,p4) intersect"""
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0: return 0  # Collinear
            return 1 if val > 0 else 2  # Clockwise or Counterclockwise
        
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                   q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)
        
        # General case
        if o1 != o2 and o3 != o4:
            return True
        
        # Special cases
        if o1 == 0 and on_segment(p1, p3, p2): return True
        if o2 == 0 and on_segment(p1, p4, p2): return True
        if o3 == 0 and on_segment(p3, p1, p4): return True
        if o4 == 0 and on_segment(p3, p2, p4): return True
        
        return False
    
    def _log_wrong_direction(self, vehicle_id, pair_id):
        """Log a wrong direction violation to the log file"""
        log_message = f"Wrong direction violation - Vehicle ID: {vehicle_id}, Line pair: {pair_id}"
        self.logger.info(log_message)
    
    def _save_wrong_way_vehicle(self, frame, bbox, vehicle_id, pair_id=None):
        """Save a snapshot of a vehicle going the wrong way with expanded context"""
        try:
            # Use unified violation manager if available
            if self.violation_manager:
                violation_id, snapshot_path = self.violation_manager.record_wrong_direction(
                    frame, vehicle_id, bbox, line_pair=pair_id
                )
                self.snapshots_counter += 1
                self.logger.info(f"Saved wrong-way violation through manager: {violation_id}")
                return True
                
            # Otherwise, just log that snapshot would have been saved
            self.snapshots_counter += 1
            self.logger.info(f"Would save wrong-way vehicle snapshot (local saving disabled)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing wrong-way vehicle: {str(e)}")
            return False
    
    def clear_tracking_data(self):
        """Clear all tracking data (useful when changing scenes)"""
        self.wrong_way_vehicles = set()
        # Clear multi-lane tracking data
        self.vehicle_lane_assignment = {}
        self.vehicle_direction_vector = {}
        # Clear configuration hash to force reconfiguration
        if hasattr(self, '_last_area_config_hash'):
            delattr(self, '_last_area_config_hash')
        self.logger.info("Wrong direction tracking data cleared")
    
    def debug_area_manager(self, area_manager):
        """Print diagnostic information about the area manager configuration"""
        if not hasattr(area_manager, 'areas'):
            self.logger.error("Area manager has no 'areas' attribute")
            return
            
        # Log area counts
        area_type_to_use = AreaType
        if hasattr(area_manager, 'AreaType'):
            area_type_to_use = area_manager.AreaType
            
        for area_type in area_type_to_use:
            if area_type in area_manager.areas:
                count = len(area_manager.areas[area_type])
                self.logger.info(f"{area_type.name}: {count} areas defined")
