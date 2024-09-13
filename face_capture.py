import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
import detector  # Importing your detector functions
import face_recognition
import pickle
from collections import Counter
class FaceCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Capture App")

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        # Buttons
        self.capture_button = tk.Button(root, text="Capture Photo", command=self.capture_photo)
        self.capture_button.pack(side=tk.LEFT, padx=10)
        
        self.encode_button = tk.Button(root, text="Encode Faces", command=self.encode_faces)
        self.encode_button.pack(side=tk.LEFT, padx=10)

        self.validate_button = tk.Button(root, text="Validate Person", command=self.validate_person)
        self.validate_button.pack(side=tk.LEFT, padx=10)

        # Start video feed
        self.update_video_feed()

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert color from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            img = Image.fromarray(frame)
            # Convert to ImageTk format
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.root.after(10, self.update_video_feed)  # Update every 10 ms
            self.imgtk = imgtk  # Keep reference to avoid garbage collection

    def capture_photo(self):
        folder_name = simpledialog.askstring("Input", "Enter the folder name for saving photo:")
        if folder_name:
            Path(f"training/{folder_name}").mkdir(parents=True, exist_ok=True)
            ret, frame = self.cap.read()
            if ret:
                cv2.imwrite(f"training/{folder_name}/photo.jpg", frame)
                messagebox.showinfo("Success", "Photo captured and saved!")
        else:
            messagebox.showwarning("Input Error", "Folder name cannot be empty!")

    def encode_faces(self):
        detector.encode_known_faces()  # Call the function to encode faces
        messagebox.showinfo("Success", "Faces encoded successfully!")

    def validate_person(self):
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture frame for validation.")
            return

        # Convert color from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Save the frame as a temporary image file for validation
        temp_path = Path("temp_frame.jpg")
        cv2.imwrite(str(temp_path), frame)

        # Perform the recognition and validation
        result = self.perform_validation(str(temp_path))

        # Delete the temporary image file
        temp_path.unlink()

        if result:
            messagebox.showinfo("Validation Result", f"Validated person: {result}")
        else:
            messagebox.showinfo("Validation Result", "No recognized person found.")

    def perform_validation(self, image_path: str):
        """ Perform validation on the given image path """
        encodings_location = Path("output/encodings.pkl")
        # Use the recognize_faces function to validate the captured photo
        with encodings_location.open(mode="rb") as f:
            loaded_encodings = pickle.load(f)

        input_image = face_recognition.load_image_file(image_path)
        input_face_locations = face_recognition.face_locations(input_image, model="hog")
        input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

        if not input_face_encodings:
            return None  # No faces found

        # Check each face in the image
        for unknown_encoding in input_face_encodings:
            name = self._recognize_face(unknown_encoding, loaded_encodings)
            if name:
                return name
        
        return None

    def _recognize_face(self, unknown_encoding, loaded_encodings):
        boolean_matches = face_recognition.compare_faces(
            loaded_encodings["encodings"], unknown_encoding
        )
        votes = Counter(
            name
            for match, name in zip(boolean_matches, loaded_encodings["names"])
            if match
        )
        if votes:
            return votes.most_common(1)[0][0]
        return None

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceCaptureApp(root)
    root.mainloop()
