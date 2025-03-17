OBJECTIVE:

Developed a robust machine learning system to automate handwritten digit recognition (0–9) using the MNIST dataset, addressing real-world applications like check processing and postal automation. Achieved 97.05% accuracy in digit classification.

KEY CONTRIBUTIONS:

Data Pipeline: Engineered end-to-end data preprocessing:
Extracted pixel data from MNIST files (IDX3/IDX1 formats) using C++.
Normalized pixel values to [0, 1] and structured data into OpenCV matrices for model compatibility.
Model Implementation:
Random Forest: Trained with OpenCV’s RTrees, optimizing parameters (max depth, termination criteria).
K-Nearest Neighbors (KNN): Leveraged OpenCV’s KNearest for similarity-based classification.
Evaluation: Tested models on 10,000 unseen images, comparing predictions against ground-truth labels to measure accuracy.
Technical Stack

LANGUAGES: C++
Tools: OpenCV (data handling, model training/prediction)
Dataset: MNIST (60k training + 10k test images)

IMPACT & INSIGHTS

Demonstrated the effectiveness of ensemble methods (Random Forest) and instance-based learning (KNN) for image classification.
Highlighted preprocessing’s critical role in ML pipelines, including byte decoding and normalization.
Reproducible design: Clear steps for dataset setup, path configuration, and model training.
