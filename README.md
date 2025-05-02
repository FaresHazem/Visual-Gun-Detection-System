# Visual Gun Detection System

## 🛠️ Overview
The Visual Gun Detection System is a project designed to detect guns in images using advanced image processing and machine learning techniques. The system preprocesses images, extracts features, and applies clustering algorithms to identify and classify gun-related objects.

## 🎯 Project Objective
The goal of this project is to implement a simplified visual gun detection system using the following techniques:
- Color-based segmentation (using K-means clustering)
- Harris Corner Detection for interest point detection
- FREAK Descriptors for feature extraction
- Feature matching using SSD (Sum of Squared Differences) and ratio testing

This system is designed to detect guns in static images, making it applicable to surveillance systems, security cameras, and baggage scanning.

## 📂 Project Structure
- **Data/**: Contains raw image data and annotations.
- **Notebook/**: Includes Jupyter Notebooks for experimentation and development.
- **Processing/**: Stores processed data, features, and results.
  - **Features/**: Contains extracted features and keypoints for images.
  - **GunFeatures/**: Stores descriptors for gun images.
  - **MorphProcessed/**: Contains morphologically processed images.
  - **Preprocessed/**: Stores preprocessed images.
  - **Reference/**: Contains reference gun images.
  - **Segmented/**: Stores segmented images.

## ✨ Key Features
- Image preprocessing with resizing and blurring.
- Feature extraction using keypoints and descriptors.
- Clustering using KMeans for color analysis.
- Morphological operations to close gaps in segmented blobs.
- Feature matching using SSD and ratio testing.
- Performance evaluation focusing on true positive and false positive rates.

## ⚙️ Configuration
The system allows configuration of:
- Input directories for raw and reference images.
- Output directories for processed data and features.
- Image resizing dimensions.
- Number of clusters for KMeans.

## 👥 Team Members
- Fares Hazem (ID: 20221443356)
- Ali Ashraf (ID: 2103106)
- Ahmed Dawood (ID: 20221454408)
- Ahmed Yousri (ID: 2103108)
- Ahmed Ashraf (ID: 2103134)

## ▶️ How to Run
1. Ensure all dependencies are installed (e.g., OpenCV, NumPy).
2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the paths and parameters in the notebook or script.
4. Run the Jupyter Notebook `Visual_Gun_Detection_System.ipynb` for processing and analysis.

## 📦 Dependencies
- Python 3.x
- OpenCV
- NumPy
- Matplotlib