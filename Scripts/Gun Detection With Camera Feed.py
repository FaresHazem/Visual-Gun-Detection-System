# --- Real-Time Gun Detection using Video Feed (Webcam Simulation) ---
import cv2
import numpy as np
import os

# Path to video file (simulate webcam)
video_path = 'Video.mp4'  # Change to your video file
ref_desc_dir = 'Processing/GunFeatures'
resize_dim = (400, 300)
num_clusters = 3
ratio_thresh = 0.7

def preprocess_image(image):
    image = cv2.resize(image, resize_dim)
    image = cv2.medianBlur(image, 5)
    return image

def color_segmentation(image, k=3):
    img_reshaped = image.reshape((-1, 3))
    img_reshaped = np.float32(img_reshaped)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, labels, centers = cv2.kmeans(img_reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented = centers[labels.flatten()]
    segmented_image = segmented.reshape(image.shape).astype(np.uint8)
    return segmented_image

def morphological_process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    result = cv2.bitwise_and(image, image, mask=closed)
    return result

def extract_freak_descriptors(image, mask=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use mask to focus on likely gun region
    if mask is not None:
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(gray, mask)
    else:
        harris_corners = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
        harris_corners = cv2.dilate(harris_corners, None)
        keypoints = [cv2.KeyPoint(float(pt[1]), float(pt[0]), 1) for pt in np.argwhere(harris_corners > 0.01 * harris_corners.max())]
    if len(keypoints) == 0:
        return None, None
    try:
        freak = cv2.xfeatures2d.FREAK_create()
    except AttributeError:
        print('FREAK not available in your OpenCV build.')
        return None, None
    keypoints, descriptors = freak.compute(gray, keypoints)
    return keypoints, descriptors

def match_descriptors(desc1, desc2, ratio_thresh=0.7):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    return good

# Load all reference descriptors
print('Loading reference descriptors from', ref_desc_dir)
ref_descs = []
for ref_file in os.listdir(ref_desc_dir):
    if ref_file.endswith('_desc.npy'):
        desc = np.load(os.path.join(ref_desc_dir, ref_file))
        if desc is not None and len(desc) > 0:
            ref_descs.append(desc)
print(f'Loaded {len(ref_descs)} reference descriptors.')

print('Opening video:', video_path)
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print('Error: Could not open video.')
else:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print('End of video or cannot read frame.')
            break
        frame_count += 1
        # Only process every 10th frame
        if frame_count % 10 != 0:
            continue
        print(f'Processing frame {frame_count}...')
        # --- Pipeline ---
        preprocessed = preprocess_image(frame)
        segmented = color_segmentation(preprocessed, k=num_clusters)
        morphed = morphological_process(segmented)

        # Create a mask from morphed image (nonzero pixels)
        mask = cv2.cvtColor(morphed, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Extract keypoints and descriptors only in the mask region
        kps, desc = extract_freak_descriptors(morphed, mask=mask)

        # Draw keypoints directly on the preprocessed frame (no yellow overlay)
        if kps is not None and len(kps) > 0:
            frame_with_kp = cv2.drawKeypoints(preprocessed, kps, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            frame_with_kp = preprocessed.copy()

        # Print descriptor extraction info
        if desc is not None:
            print(f'Frame {frame_count}: Extracted {len(desc)} descriptors')
        else:
            print(f'Frame {frame_count}: No descriptors extracted')

        # --- Gun Detection Matching ---
        is_gun = False
        if desc is not None and len(desc) > 0 and len(ref_descs) > 0:
            total_matches = 0
            for idx, ref_desc in enumerate(ref_descs):
                if ref_desc is not None and len(ref_desc) > 0 and desc.dtype == ref_desc.dtype and desc.shape[1] == ref_desc.shape[1]:
                    matches = match_descriptors(desc, ref_desc, ratio_thresh)
                    print(f'Frame {frame_count}: Matches with reference {idx}: {len(matches)}')
                    total_matches += len(matches)
                else:
                    print('Warning: Descriptor shape/type mismatch, skipping this reference descriptor.')
            print(f'Frame {frame_count}: Total matches = {total_matches}, Absolute threshold = 20')
            if total_matches >= 20:
                is_gun = True
        print(f'Frame {frame_count}: {"Gun Detected" if is_gun else "No Gun"}')

        # --- Display Results ---
        label = 'Gun Detected' if is_gun else 'No Gun'
        color = (0, 0, 255) if is_gun else (0, 255, 0)
        cv2.putText(frame_with_kp, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.imshow('Real-Time Gun Detection', frame_with_kp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Quitting on user request.')
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Video processing finished.')