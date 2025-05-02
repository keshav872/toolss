import cv2
import dlib
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk
import os
import face_recognition

class FaceSwapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Face Swapper")
        self.root.geometry("1000x700")
        
        # Variables
        self.source_image = None
        self.target_video_path = None
        self.output_video_path = "output.mp4"
        self.processing = False
        self.preview_mode = False
        
        # UI Setup
        self.setup_ui()
        
        # Face detection models
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
    def setup_ui(self):
        # Source Image Section
        source_frame = tk.LabelFrame(self.root, text="Source Face", padx=10, pady=10)
        source_frame.pack(fill="x", padx=10, pady=5)
        
        self.source_label = tk.Label(source_frame, text="No image selected", width=40, height=10, bg="white")
        self.source_label.pack(side="left", padx=5)
        
        source_btn_frame = tk.Frame(source_frame)
        source_btn_frame.pack(side="left", padx=10)
        
        tk.Button(source_btn_frame, text="Load Image", command=self.load_source_image).pack(fill="x", pady=5)
        tk.Button(source_btn_frame, text="Clear", command=self.clear_source_image).pack(fill="x", pady=5)
        
        # Target Video Section
        target_frame = tk.LabelFrame(self.root, text="Target Video", padx=10, pady=10)
        target_frame.pack(fill="x", padx=10, pady=5)
        
        self.target_label = tk.Label(target_frame, text="No video selected", width=40, height=10, bg="white")
        self.target_label.pack(side="left", padx=5)
        
        target_btn_frame = tk.Frame(target_frame)
        target_btn_frame.pack(side="left", padx=10)
        
        tk.Button(target_btn_frame, text="Load Video", command=self.load_target_video).pack(fill="x", pady=5)
        tk.Button(target_btn_frame, text="Preview", command=self.toggle_preview).pack(fill="x", pady=5)
        
        # Output Section
        output_frame = tk.LabelFrame(self.root, text="Output", padx=10, pady=10)
        output_frame.pack(fill="x", padx=10, pady=5)
        
        self.output_label = tk.Label(output_frame, text="Output will appear here", width=40, height=10, bg="white")
        self.output_label.pack(side="left", padx=5)
        
        output_btn_frame = tk.Frame(output_frame)
        output_btn_frame.pack(side="left", padx=10)
        
        tk.Button(output_btn_frame, text="Process", command=self.process_video).pack(fill="x", pady=5)
        tk.Button(output_btn_frame, text="Save", command=self.save_output).pack(fill="x", pady=5)
        
        # Progress Bar
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)
        
        # Status Bar
        self.status = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(fill="x", pady=5)
        
    def load_source_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.source_image = cv2.imread(file_path)
            self.source_image = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2RGB)
            
            # Detect face in source image
            face_locations = face_recognition.face_locations(self.source_image)
            if len(face_locations) == 0:
                self.status.config(text="No face detected in source image!")
                return
            
            # Display the image
            img = Image.fromarray(self.source_image)
            img = img.resize((300, 300), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)
            self.source_label.config(image=img)
            self.source_label.image = img
            self.source_label.config(text="")
            self.status.config(text="Source image loaded successfully")
    
    def clear_source_image(self):
        self.source_image = None
        self.source_label.config(image="", text="No image selected")
        self.status.config(text="Source image cleared")
    
    def load_target_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.target_video_path = file_path
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((300, 300), Image.LANCZOS)
                img = ImageTk.PhotoImage(img)
                self.target_label.config(image=img)
                self.target_label.image = img
                self.target_label.config(text="")
                self.status.config(text="Target video loaded successfully")
            cap.release()
    
    def toggle_preview(self):
        if not self.source_image or not self.target_video_path:
            self.status.config(text="Please load both source image and target video first!")
            return
            
        self.preview_mode = not self.preview_mode
        if self.preview_mode:
            self.status.config(text="Preview mode activated - press 'q' to stop")
            self.preview_face_swap()
        else:
            self.status.config(text="Preview mode deactivated")
    
    def process_video(self):
        if not self.source_image or not self.target_video_path:
            self.status.config(text="Please load both source image and target video first!")
            return
            
        self.processing = True
        self.status.config(text="Processing video...")
        
        # Get output path
        output_path = filedialog.asksaveasfilename(defaultextension=".mp4", 
                                                  filetypes=[("MP4 files", "*.mp4")])
        if not output_path:
            self.status.config(text="Processing cancelled")
            self.processing = False
            return
            
        self.output_video_path = output_path
        
        # Process video in a separate thread (simplified here)
        self.process_face_swap_video()
    
    def save_output(self):
        if not os.path.exists(self.output_video_path):
            self.status.config(text="No output video to save!")
            return
            
        save_path = filedialog.asksaveasfilename(defaultextension=".mp4", 
                                              filetypes=[("MP4 files", "*.mp4")])
        if save_path:
            os.replace(self.output_video_path, save_path)
            self.status.config(text=f"Video saved to {save_path}")
    
    def preview_face_swap(self):
        cap = cv2.VideoCapture(self.target_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps)
        
        while self.preview_mode:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform face swap
            swapped_frame = self.swap_faces(self.source_image, frame_rgb)
            
            # Display in UI
            img = Image.fromarray(swapped_frame)
            img = img.resize((300, 300), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)
            self.output_label.config(image=img)
            self.output_label.image = img
            self.output_label.config(text="")
            
            # Check for quit command
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
                
            self.root.update()
            
        cap.release()
        cv2.destroyAllWindows()
        self.preview_mode = False
        self.status.config(text="Preview stopped")
    
    def process_face_swap_video(self):
        cap = cv2.VideoCapture(self.target_video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
        
        current_frame = 0
        while self.processing and current_frame < frame_count:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform face swap
            swapped_frame = self.swap_faces(self.source_image, frame_rgb)
            
            # Convert back to BGR for writing
            swapped_frame_bgr = cv2.cvtColor(swapped_frame, cv2.COLOR_RGB2BGR)
            out.write(swapped_frame_bgr)
            
            # Update progress
            current_frame += 1
            progress = (current_frame / frame_count) * 100
            self.progress["value"] = progress
            self.status.config(text=f"Processing... {current_frame}/{frame_count} frames")
            self.root.update()
            
            # Display preview
            if current_frame % 10 == 0:  # Update preview every 10 frames
                img = Image.fromarray(swapped_frame)
                img = img.resize((300, 300), Image.LANCZOS)
                img = ImageTk.PhotoImage(img)
                self.output_label.config(image=img)
                self.output_label.image = img
                self.output_label.config(text="")
            
        cap.release()
        out.release()
        self.processing = False
        self.progress["value"] = 0
        self.status.config(text=f"Processing complete! Video saved to {self.output_video_path}")
    
    def swap_faces(self, source_img, target_img):
        # Get face encodings
        source_face_encoding = face_recognition.face_encodings(source_img)[0]
        target_face_locations = face_recognition.face_locations(target_img)
        
        if len(target_face_locations) == 0:
            return target_img  # No faces found in target
            
        # Find all face encodings in target image
        target_face_encodings = face_recognition.face_encodings(target_img, target_face_locations)
        
        # Find best match
        matches = face_recognition.compare_faces(target_face_encodings, source_face_encoding)
        best_match_index = matches.index(True) if True in matches else 0
        
        # Get landmarks for both faces
        source_landmarks = self.get_landmarks(source_img)
        target_landmarks = self.get_landmarks(target_img, target_face_locations[best_match_index])
        
        if not source_landmarks or not target_landmarks:
            return target_img
            
        # Perform face swap
        swapped_image = self.apply_face_swap(source_img, target_img, source_landmarks, target_landmarks)
        
        return swapped_image
    
    def get_landmarks(self, image, face_location=None):
        if face_location is None:
            # For source image
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) == 0:
                return None
            face_location = face_locations[0]
        
        # Convert face_recognition location to dlib rectangle
        top, right, bottom, left = face_location
        rect = dlib.rectangle(left, top, right, bottom)
        
        # Convert image to grayscale for landmark detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Get landmarks
        landmarks = self.predictor(gray, rect)
        
        # Convert to list of (x,y) tuples
        landmarks_points = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
            
        return landmarks_points
    
    def apply_face_swap(self, source_img, target_img, source_landmarks, target_landmarks):
        # Convert images to numpy arrays
        source_np = np.array(source_img, dtype=np.uint8)
        target_np = np.array(target_img, dtype=np.uint8)
        
        # Create masks
        hull_source = cv2.convexHull(np.array(source_landmarks))
        hull_target = cv2.convexHull(np.array(target_landmarks))
        
        # Calculate Delaunay triangles
        rect = cv2.boundingRect(hull_target)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(target_landmarks)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)
        
        # Get triangle indices
        indices = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            
            index_pt1 = np.where((np.array(target_landmarks) == pt1).all(axis=1))[0][0]
            index_pt2 = np.where((np.array(target_landmarks) == pt2).all(axis=1))[0][0]
            index_pt3 = np.where((np.array(target_landmarks) == pt3).all(axis=1))[0][0]
            
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                indices.append((index_pt1, index_pt2, index_pt3))
        
        # Warp and blend
        warped_img = np.copy(target_np)
        
        for i, (tr1, tr2, tr3) in enumerate(indices):
            # Triangles from source and target images
            src_tri = np.float32([source_landmarks[tr1], source_landmarks[tr2], source_landmarks[tr3]])
            tgt_tri = np.float32([target_landmarks[tr1], target_landmarks[tr2], target_landmarks[tr3]])
            
            # Calculate affine transform
            transform = cv2.getAffineTransform(src_tri, tgt_tri)
            
            # Apply transform to source image
            warped_tri = cv2.warpAffine(source_np, transform, (target_np.shape[1], target_np.shape[0]), 
                                       None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            
            # Create mask for the triangle
            mask = np.zeros(target_np.shape[:2], dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(tgt_tri), (1.0, 1.0, 1.0), 16, 0)
            
            # Blend the triangle
            warped_img = warped_img * (1 - mask[:, :, np.newaxis]) + warped_tri * mask[:, :, np.newaxis]
        
        # Create mask for seamless cloning
        mask = np.zeros(target_np.shape, dtype=target_np.dtype)
        cv2.fillConvexPoly(mask, np.int32(hull_target), (255, 255, 255))
        
        # Find center point
        center = np.mean(hull_target, axis=0, dtype=np.int32).ravel()
        
        # Seamless clone
        output = cv2.seamlessClone(np.uint8(warped_img), target_np, mask, tuple(center), cv2.NORMAL_CLONE)
        
        return output

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceSwapApp(root)
    root.mainloop()
