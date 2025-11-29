
import cv2
import mediapipe as mp
import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Lip landmark indices (MediaPipe Face Mesh)
LIP_LANDMARKS = {
    'outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375],
    'inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
}

# Upload folder configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_lip_landmarks(landmarks, image_shape):
    """Extract lip landmarks from MediaPipe results"""
    h, w = image_shape[:2]
    lip_data = {}
    
    # Extract outer lip points
    outer_lips = []
    for idx in LIP_LANDMARKS['outer']:
        landmark = landmarks.landmark[idx]
        outer_lips.append({
            'index': idx,
            'x': round(landmark.x, 6),
            'y': round(landmark.y, 6),
            'z': round(landmark.z, 6),
            'pixel_x': int(landmark.x * w),
            'pixel_y': int(landmark.y * h)
        })
    
    # Extract inner lip points
    inner_lips = []
    for idx in LIP_LANDMARKS['inner']:
        landmark = landmarks.landmark[idx]
        inner_lips.append({
            'index': idx,
            'x': round(landmark.x, 6),
            'y': round(landmark.y, 6),
            'z': round(landmark.z, 6),
            'pixel_x': int(landmark.x * w),
            'pixel_y': int(landmark.y * h)
        })
    
    lip_data['outer_lip'] = outer_lips
    lip_data['inner_lip'] = inner_lips
    lip_data['total_landmarks'] = len(outer_lips) + len(inner_lips)
    
    return lip_data


@app.route('/')
def home():
    """Home route with API documentation"""
    return jsonify({
        'status': 'API Active',
        'version': '1.0',
        'endpoints': {
            'POST /process_image': 'Send image (base64 or file) for lip detection',
            'POST /process_video_frame': 'Send video frame for real-time processing',
            'GET /health': 'Check API health',
            'POST /batch_process': 'Process multiple images'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'AI Lip Reading Backend',
        'mediapipe': 'active'
    })


@app.route('/process_image', methods=['POST'])
def process_image():
    """
    Process image for lip detection
    Accepts: multipart/form-data with 'file' or 'image' (base64)
    Returns: JSON with lip landmarks
    """
    try:
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename and allowed_file(file.filename):
                # Read image from uploaded file
                image_data = Image.open(file.stream)
                image_np = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
            else:
                return jsonify({'error': 'Invalid file format. Allowed: JPG, PNG, GIF, BMP'}), 400
        
        # Handle base64 image
        elif 'image' in request.json:
            base64_str = request.json['image']
            # Remove data URL prefix if present
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            
            image_data = base64.b64decode(base64_str)
            image_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            return jsonify({'error': 'No image provided. Send as file or base64'}), 400

        if image_np is None:
            return jsonify({'error': 'Could not decode image'}), 400

        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Run face mesh detection
        results = face_mesh.process(image_rgb)
        
        response_data = {
            'success': False,
            'face_detected': False,
            'image_shape': image_np.shape,
            'landmarks': None
        }
        
        # Extract lip landmarks if face detected
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            lip_data = extract_lip_landmarks(face_landmarks, image_np.shape)
            
            response_data['success'] = True
            response_data['face_detected'] = True
            response_data['landmarks'] = lip_data
            response_data['message'] = 'Lip landmarks detected successfully'
        else:
            response_data['message'] = 'No face detected in image'
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/process_video_frame', methods=['POST'])
def process_video_frame():
    """
    Process single video frame for real-time lip detection
    Accepts: base64 encoded frame
    Returns: Lip landmarks with performance metrics
    """
    try:
        data = request.json
        
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame provided'}), 400
        
        # Decode base64 frame
        frame_data = base64.b64decode(data['frame'].split(',')[1] if ',' in data['frame'] else data['frame'])
        frame_np = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        
        if frame_np is None:
            return jsonify({'error': 'Could not decode frame'}), 400
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        
        # Detect face landmarks
        results = face_mesh.process(frame_rgb)
        
        response_data = {
            'success': False,
            'face_detected': False,
            'frame_shape': frame_np.shape,
            'landmarks': None
        }
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            lip_data = extract_lip_landmarks(face_landmarks, frame_np.shape)
            
            response_data['success'] = True
            response_data['face_detected'] = True
            response_data['landmarks'] = lip_data
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/batch_process', methods=['POST'])
def batch_process():
    """
    Process multiple images in batch
    Accepts: array of base64 images
    Returns: Array of results
    """
    try:
        data = request.json
        
        if not data or 'images' not in data or not isinstance(data['images'], list):
            return jsonify({'error': 'Invalid request. Send array of base64 images'}), 400
        
        results = []
        
        for idx, img_base64 in enumerate(data['images']):
            try:
                # Decode image
                img_data = base64.b64decode(img_base64.split(',')[1] if ',' in img_base64 else img_base64)
                img_np = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                
                if img_np is None:
                    results.append({
                        'index': idx,
                        'success': False,
                        'error': 'Could not decode image'
                    })
                    continue
                
                # Process image
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                face_results = face_mesh.process(img_rgb)
                
                if face_results.multi_face_landmarks:
                    lip_data = extract_lip_landmarks(face_results.multi_face_landmarks[0], img_np.shape)
                    results.append({
                        'index': idx,
                        'success': True,
                        'face_detected': True,
                        'landmarks': lip_data
                    })
                else:
                    results.append({
                        'index': idx,
                        'success': False,
                        'face_detected': False,
                        'error': 'No face detected'
                    })
            
            except Exception as e:
                results.append({
                    'index': idx,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'total_processed': len(results),
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/get_lip_indices', methods=['GET'])
def get_lip_indices():
    """Get MediaPipe lip landmark indices mapping"""
    return jsonify({
        'lip_landmarks': LIP_LANDMARKS,
        'description': 'MediaPipe Face Mesh lip indices',
        'outer_count': len(LIP_LANDMARKS['outer']),
        'inner_count': len(LIP_LANDMARKS['inner']),
        'total': len(LIP_LANDMARKS['outer']) + len(LIP_LANDMARKS['inner'])
    })


if __name__ == '__main__':
    print("🚀 AI Lip Reading Backend Starting...")
    print("📍 Server running on: http://localhost:5000")
    print("📚 API Documentation: http://localhost:5000/")
    print("\n✅ MediaPipe Face Mesh loaded successfully")
    print(f"👄 Lip Landmarks: Outer={len(LIP_LANDMARKS['outer'])}, Inner={len(LIP_LANDMARKS['inner'])}")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)