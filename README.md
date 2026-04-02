**Brain Tumor Classification using Deep Learning**
This project is a Computer Vision-based Brain Tumor Classification System developed using Deep Learning techniques. It classifies brain MRI images into multiple tumor categories using a trained Convolutional Neural Network (CNN).

**The model can identify:**
-Pituitary Tumor
 -Meningioma
 -Glioma
 -No Tumor
 
 **Features**
 -Image-based tumor classification
 -Deep Learning model (CNN)
 -Real-time prediction (Webcam support)
 -Web interface for easy interaction
 -Confidence score display
 -Supports MRI scan inputs
 
** Tech Stack**
 -Programming & Libraries
 -Python
-TensorFlow
-Keras
-OpenCV
-NumPy

** Web Framework**
-Flask (for deployment & UI)

**Model**
-Convolutional Neural Network (CNN)
-Trained using image dataset (Teachable Machine / custom dataset)

 **Model Architecture**
-Input Layer: 224 × 224 × 3 images
-Convolutional Layers (feature extraction)
-Pooling Layers
-Fully Connected Layers
-Output Layer (4 classes)

**How It Works**
-Input image (MRI scan or webcam frame)
- Image preprocessing (resize + normalization)
-CNN model analyzes image
 -Outputs class probabilities
- Displays predicted tumor type + confidence
- 
  **Conclusion**
  This project shows how Deep Learning and Computer Vision can help in detecting brain tumors from MRI images. Using a CNN built with Keras and TensorFlow, the model can classify different tumor types effectively. With OpenCV, it also supports real-time prediction.
