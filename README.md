# Face Mask Detection, Classification, and Segmentation

## Author: K. S. Tarun , Amruth Gadepalli, Kadimetla Adarsha Reddy 
**Student ID:** IMT2022034, iMT2022065, IMT2022069   

## i. Introduction
This project aims to develop a computer vision solution for classifying and segmenting face masks in images. It involves both traditional machine learning methods and deep learning techniques to analyze the presence of face masks and segment mask regions. The tasks include feature extraction, classification using machine learning models, training a CNN for classification, and segmentation using both traditional and deep learning-based methods.

---

## ii. Dataset

The dataset consists of labeled images of individuals with and without face masks. It is sourced from:  

1. [Face Mask Detection Dataset](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset) for classification tasks.  
2. [Masked Face Segmentation Dataset](https://github.com/sadjadrz/MFSD) for segmentation tasks.  

We use the first dataset for the questions (a) and (b), and the second dataset for the questions (c) and (d).  

The first dataset contains two categories:  
1. *With Mask:* Images of individuals wearing face masks.  
2. *Without Mask:* Images of individuals without face masks.  
---
## iii.*Part (a)*
### Binary Classification Using Handcrafted Features and ML Classifiers (4 Marks)  
&nbsp;&nbsp;&nbsp;&nbsp; i. Extract handcrafted features from the dataset.  
&nbsp;&nbsp;&nbsp;&nbsp; ii. Train and evaluate at least two machine learning classifiers (e.g., SVM, Neural network) to classify faces as *"with mask"* or *"without mask."*  
&nbsp;&nbsp;&nbsp;&nbsp; iii. Report and compare the accuracy of the classifiers.  


## Methodology


- Describe the feature extraction process.  
- Mention the machine learning classifiers used (e.g., SVM, Neural Network).  
- Steps for training and evaluating the classifiers.

### 1. *Feature Extraction:*  
&nbsp;&nbsp;&nbsp;&nbsp; a. *Support Vector Machine* (SVM) with a linear kernel.
&nbsp;&nbsp;&nbsp;&nbsp; b. *Multilayer Perceptron*(MLP) with two hidden layers (128 and 64 neurons) and a maximum of 500 iterations.
python


![image](https://github.com/user-attachments/assets/1a06ee5f-9f98-4fe6-a758-1379d970cdbd)



### 2. *Model Training:*  

&nbsp;&nbsp;&nbsp;&nbsp; a. Images were converted to grayscale and resized to *64x64* pixels.  
&nbsp;&nbsp;&nbsp;&nbsp; b. Histogram of Oriented Gradients (HOG) was used to extract texture-based features.  

![image](https://github.com/user-attachments/assets/242fbed2-2183-4ce2-92a8-88f2e2687961)


The dataset was split into training (80%) and testing (20%) sets.

### 3. *Evaluation:*  

Accuracy scores were used to compare the performance of both classifiers.

### 4. *Results* (Model Accuracy):  
*SVM:*	0.88  
*MLP:*	0.92  
The results indicate that MLP performed better in classifying face masks.




---
## iv. *Part (b)*

### Binary Classification Using CNN (3 Marks) 
&nbsp;&nbsp;i.	Design and train a Convolutional Neural Network (CNN) to perform binary classification on the same dataset. <br>
&nbsp;&nbsp;ii. Try a few hyper-parameter variations (e.g., learning rate, batch size, optimizer, activation function in the classification layer) and report the results. <br>
&nbsp;&nbsp;iii. Compare the CNN's performance with the ML classifiers.<br>

## *Methodology*

### *1.	Data Preprocessing:*<br>
&nbsp;&nbsp;&nbsp;&nbsp;Images were resized to 64x64 and normalized.<br>
### *2.	Model Design:*<br>
&nbsp;&nbsp;&nbsp;&nbsp;A CNN architecture was implemented with three convolutional layers, max-pooling layers, and fully connected layers.<br>


![image](https://github.com/user-attachments/assets/3403b636-80e7-4bd1-9b7f-300f4f876f9d)


### *3.	Hyperparameter Experiments:*<br>
&nbsp;&nbsp;&nbsp;&nbsp;We experimented with different hyperparameters to optimize the CNN model. Different values of batch size, learning rate, activation function, and optimizers were tested. The variations included:<br>

![image](https://github.com/user-attachments/assets/1433d153-b372-44d9-a182-15a4ff606112)


&nbsp;&nbsp;&nbsp;&nbsp;Since we tested all combinations of these hyperparameters, the total number of experiments conducted was 36 different runs.<br>

### *4.	Training and Evaluation:*<br>
&nbsp;&nbsp;&nbsp;&nbsp;The CNN model was trained for 10 epochs on the dataset, with validation accuracy recorded for different hyperparameter settings<br>

![image](https://github.com/user-attachments/assets/d910aaf4-6e22-40dc-86c5-3a2ede0f08f1)



&nbsp;&nbsp;&nbsp;&nbsp;Here’s a summary of the top results from your CNN hyperparameter experiments:<br>

### *5. Results:*<br>
All the 36 results are there in the python notebook submitted.<br>
### *Best Performing Model:*<br>
&nbsp;&nbsp;Batch Size: 32, Activation: ReLU, Learning Rate: 0.001, Optimizer: Adam<br>
&nbsp;&nbsp;Final Training Accuracy: 97.89%<br>
&nbsp;&nbsp;Final Validation Accuracy: 95.85%<br>
&nbsp;&nbsp;Final Validation Loss: 0.1243**<br>

![image](https://github.com/user-attachments/assets/4d656cd0-6183-498a-8f1e-cd1778dac06b)


### *Other Notable Results:*<br>
&nbsp;&nbsp;Batch Size: 128, Activation: ReLU, Learning Rate: 0.001, Optimizer: Adam<br>
&nbsp;&nbsp;&nbsp;&nbsp;Final Validation Accuracy: 95.85%<br>
&nbsp;&nbsp;&nbsp;&nbsp;Final Validation Loss: 0.1440<br>
&nbsp;&nbsp;Batch Size: 64, Activation: Leaky ReLU, Learning Rate: 0.0001, Optimizer: Adam<br>
&nbsp;&nbsp;&nbsp;&nbsp;Final Validation Accuracy: 92.64%<br>
&nbsp;&nbsp;&nbsp;&nbsp;Final Validation Loss: 0.2106<br>
### *Worst Performing Model:*<br>
&nbsp;&nbsp;Batch Size: 32, Activation: ReLU, Learning Rate: 0.0001, Optimizer: SGD<br>
&nbsp;&nbsp;&nbsp;&nbsp;Final Validation Accuracy: 54.00%<br>
&nbsp;&nbsp;&nbsp;&nbsp;Final Validation Loss: 0.6924<br>

### *6. Observation and Analysis*

The Adam optimizer with a learning rate of 0.001 consistently yielded the best results. ReLU and Leaky ReLU both performed well, but ReLU with Adam at 0.001 LR showed the highest accuracy. On the other hand, models trained with SGD and low learning rates performed significantly worse.
Adam outperformed SGD because it adapts the learning rate for each parameter using moment estimates, enabling faster convergence and better handling of complex loss surfaces. It efficiently adjusts updates based on past gradients, making it more suitable for deep networks. In contrast, SGD with a fixed learning rate struggles with slow convergence and can get stuck in local minima, leading to suboptimal performance, especially in non-convex problems like CNN training. Even the best performance of SGD stopped at 0.83, highlighting its limitations in reaching higher accuracy compared to Adam.


### Comparison of CNN Performance with ML Classifiers


The CNN-based model slightly outperformed traditional ML classifiers in face mask classification. The key comparisons are:
#### 1.	Feature Extraction:
ML classifiers relied on handcrafted features which may not fully capture complex facial mask patterns.
CNN learned hierarchical features automatically, making it more robust to variations in mask shape, lighting, and occlusions.

#### 2.	Classification Accuracy:
&nbsp;&nbsp;The best ML classifier achieved an accuracy of 92% (mention exact value).
&nbsp;&nbsp;The CNN model achieved 96%, demonstrating a substantial improvement due to its ability to learn rich feature representations.
#### 3.	Computational Complexity:
&nbsp;&nbsp;ML classifiers were lightweight and required minimal computational resources.
&nbsp;&nbsp;CNN, while computationally expensive, leveraged GPUs for efficient training and inference.
Overall, CNN demonstrated superior accuracy and generalization, making it a more effective approach for face mask classification compared to traditional ML classifiers

---
## v. *Part (c)*
### Region Segmentation Using Traditional Techniques (3 Marks)  <br>
&nbsp;&nbsp;i. Implement a region-based segmentation method (e.g., thresholding, edge detection) to segment the mask regions for faces identified as "with mask."  <br>
&nbsp;&nbsp;ii. Visualize and evaluate the segmentation results. <br>


##  Methodology <br>

### *1. Segmentation Process* <br>
The segmentation is performed using the segment_mask function, which applies the following steps <br>
&nbsp;&nbsp;&nbsp;&nbsp;**Grayscale Conversion:** The input face image is converted to grayscale. <br>
&nbsp;&nbsp;&nbsp;&nbsp;**Intensity Analysis:** The mean pixel intensity determines the thresholding approach (binary or inverse binary). <br>
&nbsp;&nbsp;&nbsp;&nbsp;**Gaussian Blurring:** A 3x3 Gaussian blur is applied to reduce noise.<br>
&nbsp;&nbsp;&nbsp;&nbsp;**Otsu’s Thresholding:** Adaptive thresholding is applied to generate a binary mask.<br>
&nbsp;&nbsp;&nbsp;&nbsp;**Edge Detection:** Canny edge detection is used to highlight boundaries.<br>
&nbsp;&nbsp;&nbsp;&nbsp;*8Mask Refinement:** Morphological closing is performed to refine the segmented mask.<br>
&nbsp;&nbsp;&nbsp;&nbsp;**Saving Results:** The generated masks are saved in the designated output directory.<br>

![image](https://github.com/user-attachments/assets/fce93a05-d6e9-4f5d-9993-eb76e0673d66)


### *2. Dice Coefficient Calculation*<br>

To evaluate segmentation accuracy, the Dice coefficient is computed, measuring the overlap between the segmented output and the ground truth mask.<br>
&nbsp;&nbsp;&nbsp;&nbsp;Resizing Masks: The predicted mask is resized to match the ground truth dimensions.<br>
&nbsp;&nbsp;&nbsp;&nbsp;Binarization: Thresholding ensures binary mask representation.<br>
&nbsp;&nbsp;&nbsp;&nbsp;Intersection Calculation: The logical AND operation finds overlapping pixels.<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dice Score Computation: The Dice coefficient formula is applied.<br>

![image](https://github.com/user-attachments/assets/500e38f6-746d-40a6-8176-7fbc5b0005ee)


### *3. Implementation Details*<br>
•	Input Data: The images are loaded from input_dir.<br>
•	Processing Pipeline: The segmentation function is applied to all images in the directory.<br>
•	Output Storage: The processed masks are saved in output_dir.<br>
•	Evaluation: The segmented results are compared against ground_truth_dir.<br>
•	Error Handling: If an image fails to load, it is skipped with a warning message.<br>

### *4. Obseravtion and Analysis*<br>

During the segmentation process, we encountered a challenge where the background of the face crop images varied in intensity—sometimes appearing darker than the face mask and other times lighter. The appropriate thresholding method depends on this variation: cv2.THRESH_BINARY is suitable when the background is lighter, while cv2.THRESH_BINARY_INV is required when the background is darker.<br>

Light Background Image<br>
![image](https://github.com/user-attachments/assets/3bf1590a-1b2a-4620-b3c8-483645bd8754)


Segmented Image<br>
![image](https://github.com/user-attachments/assets/45ef9b98-7e41-485d-997d-f4acb633111b)


Dark Background Image<br>
![image](https://github.com/user-attachments/assets/03767cf9-4eec-4471-a900-091f197faa26)


Segmented Image<br>
![image](https://github.com/user-attachments/assets/67e805a6-36e3-456c-b629-f321dd2d0645)


 To address this, we calculate the average intensity of the image. If the average intensity exceeds 127, the background is classified as light; otherwise, it is classified as dark. Based on this classification, the appropriate thresholding technique is selected to ensure accurate segmentation.

 ### *5. Results*<br>
Using this segmentation process the Dice score we achieved is 46.08%, so to improve this accuracy we use U-Net. The accuracy we achieved using U-Net is and the process is explained below
<br>

---
## *Part (d):* <br>
## Mask Segmentation Using U-Net (5 Marks)<br>
&nbsp;&nbsp;i. Train a U-Net model for precise segmentation of mask regions in the images.<br>
&nbsp;&nbsp;ii. Compare the performance of U-Net with the traditional segmentation method using metrics like IoU or Dice score.<br>


##  Methodology <br>

### *1.Preprocessing*<br>
&nbsp;&nbsp;&nbsp;&nbsp;a.	Images are read in RGB format and normalized to the range [0,1].<br>
&nbsp;&nbsp;&nbsp;&nbsp;b.	Masks are converted to grayscale, resized with nearest-neighbor interpolation, and binarized.<br>
&nbsp;&nbsp;&nbsp;&nbsp;c.	Data is split into training and testing sets using an 80-20 split.<br>

### *2.Model Architecture*<br>
&nbsp;&nbsp;&nbsp;&nbsp;a.	Backbone: ResNet-50 (pretrained on ImageNet)<br>
&nbsp;&nbsp;&nbsp;&nbsp;b.	Input shape: (128,128,3)<br>
&nbsp;&nbsp;&nbsp;&nbsp;c.	Activation function: Sigmoid (for binary segmentation)<br>
&nbsp;&nbsp;&nbsp;&nbsp;d.	Loss function: Binary cross-entropy<br>
&nbsp;&nbsp;&nbsp;&nbsp;e.	Optimizer: Adam<br>
&nbsp;&nbsp;&nbsp;&nbsp;f.	Metrics: Accuracy<br>
<br>
### *3.Training Setup*<br>
&nbsp;&nbsp;&nbsp;&nbsp;a.	Epochs: 10<br>
&nbsp;&nbsp;&nbsp;&nbsp;b.	Batch size: 8<br>
&nbsp;&nbsp;&nbsp;&nbsp;c.	Validation split: 20% of training data<br>
<br>
### *4. Results and Evaluation*<br>
&nbsp;&nbsp;&nbsp;&nbsp;Quantitative Evaluation<br>
&nbsp;&nbsp;&nbsp;&nbsp;The U-Net segmentation performance is evaluated using IoU (Intersection over Union) and Dice Coefficient:<br>
&nbsp;&nbsp;&nbsp;&nbsp;IoU Score: 0.8862<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dice Score: 0.9329<br>


### *5. Observation and analysis*<br>

### Comparison of U-Net and Traditional Segmentation<br>
The segmentation performance was evaluated using the Dice coefficient, which measures the similarity between the predicted and ground truth masks. A higher Dice score indicates better segmentation accuracy.<br>
#### 1. Performance Difference<br>
&nbsp;&nbsp;•	U-Net achieved a Dice coefficient of 94%, indicating highly accurate segmentation with minimal deviation from ground truth masks.<br>
&nbsp;&nbsp;•	Traditional segmentation methods scored 46%, showing significantly lower accuracy due to limitations in handling variations in lighting, textures, and complex facial structures.<br>
#### 2. Strengths of U-Net<br>
&nbsp;&nbsp;•	Deep learning-based U-Net effectively captures intricate mask boundaries by leveraging learned hierarchical features.<br>
&nbsp;&nbsp;•	It generalizes well across diverse images and adapts to variations in illumination and facial features.<br>
&nbsp;&nbsp;•	U-Net incorporates skip connections, which help retain fine-grained spatial details, leading to precise mask delineation.<br>
#### 3. Limitations of Traditional Segmentation<br>
&nbsp;&nbsp;•	Otsu thresholding and edge detection struggle with noisy backgrounds, shadows, and varying face tones.<br>
&nbsp;&nbsp;•	Handcrafted filters lack adaptability, resulting in over-segmentation or under-segmentation in challenging cases.<br>
&nbsp;&nbsp;•	Morphological operations improve mask refinement but cannot match the flexibility of a data-driven deep learning approach.
#### 4. Practical Implications<br>
&nbsp;&nbsp;•	U-Net is better suited for real-world applications where accuracy is critical, such as medical imaging or face mask detection.<br>
&nbsp;&nbsp;•	Traditional methods, while computationally inexpensive, are not robust enough for complex segmentation tasks.<br>
#### 5. Conclusion<br>

### The results demonstrate that deep learning-based segmentation significantly outperforms traditional methods. While handcrafted features can provide a quick, interpretable approach, they lack the adaptability and precision offered by neural networks like U-Net.

<br>
---

## vi. *How to run the files?*<br>
## To successfully run this project, ensure you have the following libraries installed in your Python environment.<br>
&nbsp;&nbsp;&nbsp;&nbsp;### 1.	numpy <br>
&nbsp;&nbsp;&nbsp;&nbsp;### 2.	pandas <br>
&nbsp;&nbsp;&nbsp;&nbsp;### 3.	pandas <br>
&nbsp;&nbsp;&nbsp;&nbsp;### 4.	os<br>
&nbsp;&nbsp;&nbsp;&nbsp;### 5.	opencv-python (cv2) <br>
&nbsp;&nbsp;&nbsp;&nbsp;### 6.	scikit-image (skimage.feature, skimage.metrics)<br>
&nbsp;&nbsp;&nbsp;&nbsp;### 7.	scikit-learn (sklearn)<br>
&nbsp;&nbsp;&nbsp;&nbsp;### 8.	tensorflow & keras<br>
&nbsp;&nbsp;&nbsp;&nbsp;### 9.	 segmentation_models – Pretrained segmentation models (Requires installation)<br>
&nbsp;&nbsp;&nbsp;&nbsp;### 10.	matplotlib – For plotting results<br>

## 1. *Running questionab.ipynb*<br>
### Before running the notebook, make sure the dataset is stored in the following directory structure:<br>
&nbsp;Modify the dataset paths in the notebook to match your local directory:<br>
&nbsp;&nbsp;&nbsp;**mask_path = "/path/to/dataset/with_mask"**<br>
&nbsp;&nbsp;&nbsp;**no_mask_path = "/path/to/dataset/without_mask"**<br>
&nbsp;Replace **/path/to/dataset/** with the actual location of your dataset on your computer.<br>
### If you want to run the notebook directly as a Python script without opening Jupyter Notebook, Convert the Jupyter Notebook into a Python script and run it:<br>
&nbsp;&nbsp;&nbsp;jupyter nbconvert --to script questionab.ipynb<br>
&nbsp;&nbsp;&nbsp;python questionab.py<br>
<br>
## 2. *Running cdquestion.ipynb*<br>
### Update the paths inside the notebook to reflect your local directory structure:<br>
&nbsp;&nbsp;**input_dir = "/path/to/dataset/MSFD/1/face_crop"**<br>
&nbsp;&nbsp;**output_dir = "/path/to/output/segmented_mask"**<br>
&nbsp;&nbsp;**ground_truth_dir = "/path/to/dataset/MSFD/1/face_crop_segmentation"**<br>
&nbsp;Replace **/path/to/dataset/** with the actual location of your dataset.<br>
### If you want to run the notebook directly as a Python script without opening Jupyter Notebook, Convert the Jupyter Notebook into a Python script and run it:<br>
&nbsp;&nbsp;&nbsp;jupyter nbconvert --to script cdquestion.ipynb<br>
&nbsp;&nbsp;&nbsp;python cdquestion.py<br>
## This will process the images and save the output in the specified directory.<br>
---
