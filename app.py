import streamlit as st
from ultralytics import YOLO


@st.cache_resource
def load_model():
    """
    Initialize and return YOLOv8 nano model.
    Uses Streamlit's cache_resource decorator to cache the model across reruns.
    
    Returns:
        YOLO: YOLOv8 model object
        
    Raises:
        Exception: If model loading fails
    """
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Failed to load YOLOv8 model: {e}")
        raise

def process_frame(frame, model):
    """
    Process a frame to detect people and draw bounding boxes.
    
    Args:
        frame: NumPy array (BGR image from OpenCV)
        model: YOLOv8 model object
        
    Returns:
        tuple: (processed_frame, people_count)
            - processed_frame: Frame with bounding boxes drawn
            - people_count: Number of people detected in the frame
    """
    # Run YOLOv8 inference on the frame
    results = model(frame, verbose=False)
    
    # Get the first result (single frame inference)
    result = results[0]
    
    # Initialize people count
    people_count = 0
    
    # Filter detections for person class (class_id == 0 in COCO dataset)
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            
            # Only process person detections (class_id == 0)
            if class_id == 0:
                people_count += 1
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get confidence score
                confidence = float(box.conf[0])
                
                # Draw bounding box with visible color (green) and 2+ pixel thickness
                import cv2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with confidence score
                label = f"Person {confidence:.2f}"
                
                # Draw label background for better visibility
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame, people_count

def main():
    """
    Main Streamlit application entry point.
    Creates the UI layout and handles user interactions.
    """
    # Create centered title
    st.markdown("<h1 style='text-align: center;'>Real-Time People Detection System</h1>", unsafe_allow_html=True)
    
    # Add centered description text below title
    st.markdown("<p style='text-align: center;'>Live detection of people in a video stream using OpenCV, with bounding boxes and real-time count.<br>- Yug Agarwal (2023CSAI258)</p>", unsafe_allow_html=True)

    
    # Add spacing
    st.markdown("---")
    
    # Initialize model using load_model()
    model = load_model()
    
    # Implement radio button for source selection (Camera/Video File)
    source_option = st.radio(
        "Select Video Source:",
        ("Camera", "Video File"),
        horizontal=True
    )
    
    # Add conditional file uploader for video file option
    uploaded_file = None
    if source_option == "Video File":
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a video file to perform people detection"
        )
    
    # Add spacing
    st.markdown("---")
    
    # Create st.empty() placeholder for video frame display
    frame_placeholder = st.empty()
    
    # Add metric display for people count with "People Detected: X" label
    count_placeholder = st.empty()
    count_placeholder.metric(label="People Detected", value=0)
    
    # Add stop button for detection control
    stop_button_placeholder = st.empty()
    stop_button = stop_button_placeholder.button("Stop Detection", type="primary")
    
    # Initialize video capture based on selected source
    cap = None
    temp_file_path = None
    
    try:
        if source_option == "Camera":
            # For camera source, use device index 0
            import cv2
            cap = cv2.VideoCapture(0)
            
            # Check if camera is available
            if not cap.isOpened():
                st.error("Unable to access camera. Please check that your camera is connected and not being used by another application.")
                return
            
            st.success("Camera initialized successfully!")
            
        elif source_option == "Video File":
            # Check if file is uploaded
            if uploaded_file is None:
                st.warning("Please upload a video file to begin detection.")
                return
            
            # Save uploaded file temporarily
            import tempfile
            import os
            import cv2
            
            # Create a temporary file with the same extension as uploaded file
            file_extension = os.path.splitext(uploaded_file.name)[1]
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            temp_file_path = temp_file.name
            
            # Write uploaded file content to temporary file
            temp_file.write(uploaded_file.read())
            temp_file.close()
            
            # Initialize VideoCapture with temporary file path
            cap = cv2.VideoCapture(temp_file_path)
            
            # Check if video file is valid
            if not cap.isOpened():
                st.error("Unable to open video file. Please ensure the file is a valid video format (mp4, avi, mov, mkv).")
                # Clean up temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                return
            
            st.success(f"Video file '{uploaded_file.name}' loaded successfully!")
        
        # Implement frame processing loop that reads frames continuously
        import cv2
        import time
        
        # Track previous people count to only update when it changes
        previous_count = 0
        
        # Track frame timing to maintain at least 15 FPS
        frame_time_target = 1.0 / 15.0  # Target: 15 FPS minimum
        
        # Processing loop
        while cap.isOpened():
            start_time = time.time()
            
            # Check if stop button was clicked
            if stop_button:
                st.info("Detection stopped by user.")
                break
            
            # Read frame from video source
            ret, frame = cap.read()
            
            # Check if frame was read successfully
            if not ret:
                st.warning("End of video or unable to read frame.")
                break
            
            # Call process_frame() for each frame
            processed_frame, people_count = process_frame(frame, model)
            
            # Convert BGR frames to RGB for Streamlit display
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Update video placeholder with processed frames
            frame_placeholder.image(rgb_frame, channels="RGB", width='stretch')
            
            # Update people count display when count changes
            if people_count != previous_count:
                count_placeholder.metric(label="People Detected", value=people_count)
                previous_count = people_count
            
            # Maintain at least 15 FPS frame rate
            elapsed_time = time.time() - start_time
            if elapsed_time < frame_time_target:
                time.sleep(frame_time_target - elapsed_time)
        
    except Exception as e:
        st.error(f"An error occurred during video processing: {e}")
        
    finally:
        # Implement proper resource cleanup (release video capture)
        if cap is not None:
            cap.release()
        
        # Remove temporary file if it was created
        if temp_file_path:
            import os
            if os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass  # Ignore cleanup errors


if __name__ == "__main__":
    main()
