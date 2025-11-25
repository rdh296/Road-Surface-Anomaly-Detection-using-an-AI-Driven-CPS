Road Surface Anomaly Detection using an AI-Driven CPS

A low-cost, camera-based system to detect potholes, cracks, and uneven road surfaces.

Why This Project?

Road inspections take a lot of time, money, and human effort. Most cities still rely on manual surveys or public complaints.
We wanted to create something simple, practical, and affordable — a system that can look at road images and automatically detect potholes, cracks, and other damage.
All you need is a camera + GPS + AI model.
This makes it perfect for a Cyber-Physical System (CPS) setup.
Key Features
•	Detects potholes, cracks, uneven surfaces, and more
•	Works on normal phone/camera/dashcam images
•	Supports GPS extraction (from image EXIF)
•	Generates annotated images
•	Creates an interactive map of road defects
•	Produces a simple dashboard inside the notebook
•	No external server, no ngrok — everything runs locally
•	Completely low-cost and easy to deploy

How the System Works (Simple Overview)

Camera → Capture Road Images → YOLOv8 Model → Detect Damage
         ↓                                ↓
       GPS (optional)               Annotated Image
         ↓                                ↓
        Map Plotting ← Dashboard ← Summary CSV
The notebook handles:
•	dataset prep
•	annotation conversion
•	model training
•	evaluation
•	dashboard generation
Everything is inside one runnable .ipynb.

Project Structure (Recommended)
/
├── README.md
├── notebook.ipynb   ← Your uploaded notebook
├── technical_summary.pdf
├── models/
│   └── best.pt      ← Your trained model
├── project_outputs/
│   ├── annotated/   ← Bounding box images
│   ├── detections_summary.csv
│   ├── map.html
│   └── final_dashboard.html
├── input_images/
├── requirements.txt
└── src/
    ├── inference.py
    ├── api_main.py
    └── streamlit_app.py
Model Training
We used YOLOv8 because it’s fast, accurate, and simple to train.
Training is done inside the notebook using:
•	img size: 640×640
•	epochs: 50–100
•	optimizer: Adam / SGD
•	metrics: mAP50, mAP50-95, Precision, Recall
The notebook walks through the entire process.

Dashboard & Output Files

The notebook automatically creates:
✔ Annotated images
✔ A detections CSV
✔ An interactive Folium map (map.html)
✔ A simple dashboard (final_dashboard.html)
These can be directly shown in the demo or uploaded to the repository.

Dataset
You can use:
•	India Road Damage Dataset (IRDD)
•	Japan Road Damage Dataset
•	Pothole-600
•	Self-collected images
Annotations were created using LabelImg / Roboflow and converted to YOLO format.

If your dataset is too large, include:
•	a small sample
•	OR a link in the README

How to Run
1. Upload the Notebook to Google Colab
Open notebook.ipynb in Colab and run all cells.
2. Upload the Model Weight
Upload best.pt into the Colab environment.
3. Add Test Images
Put images in the input_images/ folder (or upload in Colab).
4. Run the Inference/Visualization Section
This will generate:
•	annotated images
•	CSV
•	map
•	dashboard html
All files saved into project_outputs/.
Future Extensions
•	Turn into a mobile app for traffic police / citizens
•	Install on municipal vehicles for automatic monitoring
•	Live dashboard with backend API
•	Drone imagery support
•	Severity analysis of potholes
•	Predictive road maintenance

Technical Summary
A 4-page humanized technical summary is included in the repo as:
technical_summary.pdf
This covers:
•	motivation
•	dataset
•	preprocessing
•	model
•	dashboard
•	CPS integration

The dataset we used:
Fernández, A .(2022).Pothole Detection Dataset. Kaggle. 
https://www.kaggle.com/datasets/andrewmvd/pothole-detection

Our complete Project can be viewed here: 
https://drive.google.com/drive/folders/1yacWcX4WAx8_-HLBSeA6oCFF-VC2-Rzc?usp=sharing

Team & Declaration
We confirm that the work submitted (code, model, documentation, summary) is original and belongs to our team.

Submission Notes
This repository is intended for the Code Submission Deadline:
25 November 2025 (11:59 PM IST)
and corresponds to the Google Form submission:
https://forms.gle/a79kYUkJMt5FApex7

