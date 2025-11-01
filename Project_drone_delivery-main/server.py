# server.py
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import threading
import time
from drone_control import get_controller, get_lastest_frame
import json
from planner import run_planner  # Import planner
from dronekit import VehicleMode, LocationGlobalRelative
import math
import cv2
import numpy as np
import signal
import sys
import atexit
import socket
import os

app = Flask(__name__)
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   logger=False, 
                   engineio_logger=False,
                   async_mode='threading')

# Global variables
controller = None
prev_loc = None
distance_traveled = 0.0
aruco_markers = {}
zone_center_offset = [0, 0]
zone_width = 800
zone_height = 600

def initialize_controller():
    """Initialize drone controller with error handling"""
    global controller
    try:
        controller = get_controller(connection_str='/dev/ttyS0', takeoff_height=3)  # Changed to ttyS0
        print("✅ Drone controller initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize drone controller: {e}")
        controller = None
        return False

def safe_start_camera():
    """Safely start camera with error handling"""
    try:
        if controller:
            controller.start_image_stream()
            print("✅ Camera started successfully")
            return True
    except Exception as e:
        print(f"⚠️ Camera start warning: {e}")
        return False

def safe_start_aruco():
    """Safely start ArUco processing with error handling"""
    try:
        if controller:
            controller.start_aruco_processing()
            print("✅ ArUco processing started successfully")
            return True
    except Exception as e:
        print(f"⚠️ ArUco start warning: {e}")
        return False

def get_network_info():
    """Get network information"""
    try:
        hostname = socket.gethostname()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            main_ip = s.getsockname()[0]
            s.close()
        except:
            main_ip = "127.0.0.1"
        return hostname, main_ip
    except Exception as e:
        return "unknown", "127.0.0.1"

def cleanup():
    """Cleanup function to be called on exit"""
    print("🧹 Cleaning up resources...")
    try:
        if controller:
            controller.stop_aruco_processing()
            controller.stop_image_stream()
    except Exception as e:
        print("❌ Error during cleanup:", e)

# Register cleanup function
atexit.register(cleanup)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n🛑 Shutting down server...')
    cleanup()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def mjpeg_generator():
    """Generator that serves MJPEG frames"""
    while True:
        try:
            frame_jpeg = get_lastest_frame()
            if frame_jpeg is None:
                # Create a simple placeholder frame
                placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No Camera", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', placeholder)
                if ret:
                    frame_jpeg = jpeg.tobytes()
                else:
                    time.sleep(0.1)
                    continue
                    
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n')
            time.sleep(0.033)  # ~30fps
            
        except Exception as e:
            print(f"❌ Error in MJPEG generator: {e}")
            time.sleep(0.1)

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'drone_connected': controller is not None and controller.is_connected(),
        'camera_running': get_lastest_frame() is not None
    })

@app.route('/arm', methods=['POST'])
def arm_drone():
    try:
        if controller:
            success = controller.arm_drone()
            if success:
                return jsonify({'status': 'armed', 'message': 'Drone armed successfully'})
            else:
                return jsonify({'error': 'Failed to arm drone'}), 500
        else:
            return jsonify({'error': 'Controller not available'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/start_aruco_detection', methods=['POST'])
def start_aruco_detection():
    try:
        if safe_start_aruco():
            return jsonify({'status': 'success', 'message': 'ArUco detection started'})
        else:
            return jsonify({'error': 'Failed to start ArUco detection'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_aruco_detection', methods=['POST'])
def stop_aruco_detection():
    try:
        if controller:
            controller.stop_aruco_processing()
            return jsonify({'status': 'success', 'message': 'ArUco detection stopped'})
        else:
            return jsonify({'error': 'Controller not available'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/set_aruco_ids', methods=['POST'])
def set_aruco_ids():
    try:
        payload = request.get_json(silent=True) or {}
        ids = payload.get('ids', [])
        if not isinstance(ids, list) or not all(isinstance(id, int) for id in ids):
            return jsonify({'error': 'ArUco IDs must be a list of integers'}), 400
        
        if controller:
            controller.set_find_aruco(ids)
            return jsonify({'status': 'success', 'ids': ids})
        else:
            return jsonify({'error': 'Controller not available'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_gps_stations', methods=['GET'])
def get_gps_stations():
    try:
        with open('file_gps_station.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('change_mode')
def handle_change_mode(data):
    mode = data.get('mode')
    if mode in ['LAND', 'GUIDED']:
        try:
            if controller and controller.vehicle:
                controller.vehicle.mode = VehicleMode(mode)
                emit('mission_status', {'status': f'mode_changed_to_{mode}'})
            else:
                emit('mission_status', {'status': 'error', 'error': 'Controller not available'})
        except Exception as e:
            emit('mission_status', {'status': 'error', 'error': str(e)})
    else:
        emit('mission_status', {'status': 'invalid_mode'})

def telemetry_loop():
    """Send telemetry data to clients periodically"""
    while True:
        try:
            if controller and controller.vehicle and controller.is_connected():
                v = controller.vehicle
                data = {
                    'lat': v.location.global_frame.lat if v.location.global_frame else 0,
                    'lon': v.location.global_frame.lon if v.location.global_frame else 0,
                    'alt': v.location.global_relative_frame.alt if v.location.global_relative_frame else 0,
                    'mode': v.mode.name if v.mode else 'UNKNOWN',
                    'velocity': math.sqrt(v.velocity[0]**2 + v.velocity[1]**2 + v.velocity[2]**2) if v.velocity else 0,
                    'distance_traveled': distance_traveled,
                    'armed': v.armed
                }
                socketio.emit('telemetry', data)
            time.sleep(1)  # Reduced frequency to 1Hz
        except Exception as e:
            print(f"❌ Telemetry error: {e}")
            time.sleep(1)

if __name__ == '__main__':
    print("🚀 Starting Drone Control Server...")
    
    # Initialize controller
    if not initialize_controller():
        print("⚠️ Continuing without drone connection...")
    
    # Start camera (non-critical)
    if not safe_start_camera():
        print("⚠️ Continuing without camera...")
    
    hostname, main_ip = get_network_info()
    
    print("🔍 Network Information:")
    print(f"   Hostname: {hostname}")
    print(f"   IP: {main_ip}")
    
    print("📊 Starting telemetry thread...")
    telemetry_thread = threading.Thread(target=telemetry_loop, daemon=True)
    telemetry_thread.start()
    
    print("=" * 50)
    print("🎯 DRONE CONTROL SERVER READY")
    print("=" * 50)
    print(f"📍 Local: http://localhost:5000")
    print(f"🌐 Network: http://{main_ip}:5000")
    print(f"📹 Video: http://{main_ip}:5000/video_feed")
    print(f"❤️ Health: http://{main_ip}:5000/health")
    print("=" * 50)
    
    try:
        socketio.run(
            app, 
            host='0.0.0.0', 
            port=5000, 
            debug=False,
            allow_unsafe_werkzeug=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        cleanup()