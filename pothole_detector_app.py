
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os

class PotholeDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Road Surface Anomaly Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2c3e50")

        # Load the trained model
        self.model = YOLO(r"/content/drive/My Drive/Projects/Road_Surface_Anomaly_Detection_using_AI-Driven_CPS/training_results/pothole_detection_v12/weights/best.pt")
        self.current_image = None
        self.current_image_path = None

        self.setup_ui()

    def setup_ui(self):
        # [Previous UI code remains the same]
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        title_label = ttk.Label(main_frame,
                               text=" Road Surface Anomaly Detection System",
                               font=("Arial", 16, "bold"),
                               foreground="#3498db")
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        upload_btn = ttk.Button(main_frame,
                               text=" Upload Road Image",
                               command=self.upload_image)
        upload_btn.grid(row=1, column=0, padx=10, pady=10)

        detect_btn = ttk.Button(main_frame,
                               text=" Detect Potholes",
                               command=self.detect_potholes,
                               state="disabled")
        detect_btn.grid(row=1, column=1, padx=10, pady=10)
        self.detect_btn = detect_btn

        clear_btn = ttk.Button(main_frame,
                              text=" Clear",
                              command=self.clear_all)
        clear_btn.grid(row=1, column=2, padx=10, pady=10)

        image_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        image_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.original_label = ttk.Label(image_frame, text="Original Image")
        self.original_label.grid(row=0, column=0, padx=10, pady=5)

        self.original_img_label = ttk.Label(image_frame)
        self.original_img_label.grid(row=1, column=0, padx=10, pady=10)

        self.detected_label = ttk.Label(image_frame, text="Detection Result")
        self.detected_label.grid(row=0, column=1, padx=10, pady=5)

        self.detected_img_label = ttk.Label(image_frame)
        self.detected_img_label.grid(row=1, column=1, padx=10, pady=10)

        results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky=(tk.W, tk.E))

        self.results_text = tk.Text(results_frame, height=8, width=80, font=("Arial", 10))
        self.results_text.grid(row=0, column=0, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Road Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            self.current_image_path = file_path
            self.display_original_image(file_path)
            self.detect_btn.config(state="normal")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Image loaded successfully! Click 'Detect Potholes' to analyze.\n")

    def display_original_image(self, image_path):
        image = Image.open(image_path)
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)

        self.original_img_label.configure(image=photo)
        self.original_img_label.image = photo
        self.current_image = image

    def detect_potholes(self):
        if not self.current_image_path:
            messagebox.showerror("Error", "Please upload an image first!")
            return

        try:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, " Analyzing image for potholes...\n")
            self.root.update()

            results = self.model(self.current_image_path)

            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    detections.append({
                        "confidence": float(conf),
                        "class": "pothole"
                    })

            self.results_text.insert(tk.END, f"\n Detection Results:\n")
            self.results_text.insert(tk.END, f"Potholes detected: {len(detections)}\n\n")

            for i, det in enumerate(detections):
                self.results_text.insert(tk.END,
                    f"Pothole {i+1}: Confidence {det['confidence']:.2%}\n")

            for r in results:
                im_array = r.plot()
                im_array_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
                detected_image = Image.fromarray(im_array_rgb)
                detected_image.thumbnail((400, 400))
                detected_photo = ImageTk.PhotoImage(detected_image)

                self.detected_img_label.configure(image=detected_photo)
                self.detected_img_label.image = detected_photo

            self.results_text.insert(tk.END, f"\n Analysis complete!\n")

        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")

    def clear_all(self):
        self.current_image_path = None
        self.current_image = None
        self.original_img_label.configure(image="")
        self.detected_img_label.configure(image="")
        self.results_text.delete(1.0, tk.END)
        self.detect_btn.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = PotholeDetectorApp(root)
    root.mainloop()
