from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Initialize MediaPipe face mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def get_face_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        landmarks = np.array([
            (int(lm.x * image.shape[1]), int(lm.y * image.shape[0]))
            for lm in results.multi_face_landmarks[0].landmark
        ])
        return landmarks
    return None

def get_eye_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark

    # Indices of landmarks around the eyes from MediaPipe
    eye_indices = list(range(33, 133)) + list(range(362, 462))
    eye_points = [(int(landmarks[i].x * image.shape[1]), int(landmarks[i].y * image.shape[0])) for i in eye_indices]
    return np.array(eye_points, dtype=np.float32)

def align_images(source_image, target_image):
    source_points = get_face_landmarks(source_image)
    target_points = get_face_landmarks(target_image)

    if source_points is None or target_points is None:
        return None  # No eyes detected in one of the images

    # Compute the transformation matrix
    mat, _ = cv2.estimateAffinePartial2D(target_points,source_points)
    transformed_image = cv2.warpAffine(target_image, mat, (target_image.shape[1], target_image.shape[0]))
    return transformed_image

def normalize_face(image):
    # Process image through MediaPipe
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None  # No face detected
    landmarks = results.multi_face_landmarks[0].landmark
    # Get landmarks for eyes for rotation and scaling
    eye_left = np.array([landmarks[33].x * image.shape[1], landmarks[33].y * image.shape[0]])
    eye_right = np.array([landmarks[133].x * image.shape[1], landmarks[133].y * image.shape[0]])
    eye_center = (eye_left + eye_right) / 2
    # eye_left_pts = landmarks[36:42]  # Increase points for more robustness
    # eye_right_pts = landmarks[42:48]
    # eye_center = (np.mean(eye_left_pts, axis=0) + np.mean(eye_right_pts, axis=0)) / 2
    angle = np.degrees(np.arctan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0]))
    # Calculate scale based on eye distance
    eye_distance = np.linalg.norm(eye_right - eye_left)
    desired_eye_distance = image.shape[1] * 0.05  # Desired distance in pixels
    scale = desired_eye_distance / eye_distance
    # Calculate the original center of the image
    original_center = (image.shape[1] / 2, image.shape[0] / 2)
    # Get rotation matrix for alignment
    M = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale)
    # Update translation components of the matrix to re-center the image
    # Compute where the center of the original image should map to after transformation
    transformed_center = (M[0, 0] * original_center[0] + M[0, 1] * original_center[1] + M[0, 2],
                          M[1, 0] * original_center[0] + M[1, 1] * original_center[1] + M[1, 2])
    # Calculate the translation needed to bring the image center back to the original center
    translation_x = original_center[0] - transformed_center[0]
    translation_y = original_center[1] - transformed_center[1]
    M[0, 2] += translation_x
    M[1, 2] += translation_y
    # Apply the transformation
    transformed = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return transformed

    # eye_left_pts = landmarks[36:42]  # Increase points for more robustness
    # eye_right_pts = landmarks[42:48]
    # eye_center = (np.mean(eye_left_pts, axis=0) + np.mean(eye_right_pts, axis=0)) / 2


def calculate_ebh(landmarks):
    # Calculate vertical distances between eyelid and nearest eyebrow
    left_ebh = abs(landmarks[159][1] - landmarks[53][1])  # vertical distance for left eye
    right_ebh = abs(landmarks[386][1] - landmarks[283][1])  # vertical distance for right eye
    return left_ebh, right_ebh

def calculate_ocular_area(landmarks, indices):
    # Gather the landmarks based on provided indices
    contour = landmarks[indices]
    # Calculate the convex hull of the contour to ensure all points are included properly
    hull = cv2.convexHull(contour.astype(np.int32))
    # Return the area of the convex hull
    return cv2.contourArea(hull)

def visualize_measurements(image, landmarks):
    # Draw lines for EBH
    cv2.line(image, (landmarks[159][0], landmarks[159][1]), (landmarks[159][0], landmarks[53][1]), (0, 0, 255), 1)  # Red line left
    cv2.line(image, (landmarks[386][0], landmarks[386][1]), (landmarks[386][0], landmarks[283][1]), (0, 0, 255), 1)  # Red line right

    # Draw polygon for ocular surface area
    left_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

    cv2.polylines(image, [landmarks[left_eye_indices].astype(np.int32)], True, (0, 255, 0), 1)  # Blue polyline left eye
    cv2.polylines(image, [landmarks[right_eye_indices].astype(np.int32)], True, (0, 255, 0), 1)  # Blue polyline right eye

    return image

def crop_to_eyes(image, landmarks):
    margin_x = int(0.05 * image.shape[1])  # 4% of image width
    margin_y = int(0.05 * image.shape[0])  # 4% of image height
    x_coordinates = landmarks[[33, 133, 362, 263, 70, 105, 334, 300], 0]
    y_coordinates = landmarks[[33, 133, 362, 263, 70, 105, 334, 300], 1]
    x_min, x_max = np.min(x_coordinates), np.max(x_coordinates)
    y_min, y_max = np.min(y_coordinates), np.max(y_coordinates)
    x_min = max(x_min - margin_x, 0)
    y_min = max(y_min - margin_y, 0)
    x_max = min(x_max + margin_x, image.shape[1])
    y_max = min(y_max + margin_y, image.shape[0])
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image

def equalize_dimensions(cropped1, cropped2):
    max_height = max(cropped1.shape[0], cropped2.shape[0])
    max_width = max(cropped1.shape[1], cropped2.shape[1])

    padded1 = cv2.copyMakeBorder(cropped1,
                                 top=0,
                                 bottom=max_height - cropped1.shape[0],
                                 left=0,
                                 right=max_width - cropped1.shape[1],
                                 borderType=cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])

    padded2 = cv2.copyMakeBorder(cropped2,
                                 top=0,
                                 bottom=max_height - cropped2.shape[0],
                                 left=0,
                                 right=max_width - cropped2.shape[1],
                                 borderType=cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
    return padded1, padded2

def analyze_image(image1, image2):
    landmarks1 = get_face_landmarks(image1)
    landmarks2 = get_face_landmarks(image2)

    left_ebh1, right_ebh1 = calculate_ebh(landmarks1)
    left_ebh2, right_ebh2 = calculate_ebh(landmarks2)

    left_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 183, 155, 154, 153, 145, 144, 163, 7]
    right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]

    ocular_area1_left = calculate_ocular_area(landmarks1, left_eye_indices)
    ocular_area1_right = calculate_ocular_area(landmarks1, right_eye_indices)
    ocular_area2_left = calculate_ocular_area(landmarks2, left_eye_indices)
    ocular_area2_right = calculate_ocular_area(landmarks2, right_eye_indices)

    def safe_item(value):
        if isinstance(value, np.generic):
            return value.item()
        return value

    return {
        'left_ebh1': safe_item(left_ebh1), 'left_ebh2': safe_item(left_ebh2), 'left_ebh_change': safe_item((left_ebh2 - left_ebh1) / left_ebh1 * 100),
        'right_ebh1': safe_item(right_ebh1), 'right_ebh2': safe_item(right_ebh2), 'right_ebh_change': safe_item((right_ebh2 - right_ebh1) / right_ebh1 * 100),
        'ocular_area1_left': safe_item(ocular_area1_left), 'ocular_area2_left': safe_item(ocular_area2_left),
        'ocular_left_change': safe_item((ocular_area2_left - ocular_area1_left) / ocular_area1_left * 100),
        'ocular_area1_right': safe_item(ocular_area1_right), 'ocular_area2_right': safe_item(ocular_area2_right),
        'ocular_right_change': safe_item((ocular_area2_right - ocular_area1_right) / ocular_area1_right * 100)
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file1 = request.files['image1']
        file2 = request.files['image2']

        image1 = Image.open(file1.stream)
        image2 = Image.open(file2.stream)

        # Convert PIL images to OpenCV format
        image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)

        image1 = normalize_face(image1)
        aligned_image = normalize_face(image2)

        # aligned_image = align_images(image1, image2)

        # Get landmarks and process images
        landmarks1 = get_face_landmarks(image1)
        if landmarks1 is None:
            raise ValueError("Could not find landmarks in Pre-operative Image.")

        landmarks2 = get_face_landmarks(aligned_image)
        if landmarks2 is None:
            raise ValueError("Could not find landmarks in Post-operative Image:")

        image1_visualize = visualize_measurements(image1, landmarks1)
        image2_visualize = visualize_measurements(aligned_image, landmarks2)
        cropped1 = crop_to_eyes(image1_visualize, landmarks1)
        cropped2 = crop_to_eyes(image2_visualize, landmarks2)
        padded1, padded2 = equalize_dimensions(cropped1, cropped2)

        # analyze images
        analysis_results = analyze_image(image1, aligned_image)

        # Convert images to data URL to send back to browser
        _, buffer1 = cv2.imencode('.jpg', cropped1)
        _, buffer2 = cv2.imencode('.jpg', cropped2)
        image1_data = base64.b64encode(buffer1).decode('utf-8')
        image2_data = base64.b64encode(buffer2).decode('utf-8')

        return jsonify({'Before': f'data:image/jpeg;base64,{image1_data}', 'After': f'data:image/jpeg;base64,{image2_data}', 'results': analysis_results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
