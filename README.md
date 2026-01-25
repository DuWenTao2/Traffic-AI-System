# Vision Patrol - AI Traffic Monitoring System

An advanced AI-powered traffic monitoring system that detects various traffic violations and incidents using computer vision and deep learning technologies.

## Features

- **Traffic Violation Detection**: Detects traffic light violations, wrong-way driving, illegal parking, and other violations
- **Accident Detection**: Real-time detection of accidents and emergency situations
- **Helmet Detection**: Identifies motorcyclists and cyclists without helmets
- **Speed Detection**: Monitors vehicle speeds using calibrated distance/time calculations
- **Wrong Direction Detection**: Identifies vehicles traveling in restricted directions
- **License Plate Recognition**: Automatic license plate detection and recognition
- **Violation Logging**: Comprehensive logging system with evidence capture

## Architecture

The system uses a modular architecture with:
- YOLOv8 for object detection and tracking
- Custom models for specific violation types
- Area-based configuration for detection zones
- Real-time video processing with multi-threading
- Web-based interface for configuration and monitoring

## Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the required model files (yolov8s.pt will be downloaded automatically)
4. Configure the system using `config.conf`

## Usage

Run the main application:
```
python Main_mp.py
```

## Configuration

- Video sources: Configure in `config.conf`
- Detection areas: Define using the GUI by pressing 'c' key
- Model settings: Adjust in the configuration files

## Model Files

The system requires several model files for different detection tasks:
- `yolov8s.pt` - Base YOLOv8 model for object detection
- Additional specialized models for specific tasks

## Important Notes

- Large model files (`.pt`) are excluded from the Git repository due to size limitations
- These will be downloaded automatically when running the system for the first time
- For development purposes, ensure you have sufficient disk space and a compatible GPU

## Contributing

Please read the contributing guidelines before submitting pull requests.

## License

See the LICENSE file for licensing information.