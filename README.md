
# Chronic Kidney Disease Prediction Model

## Overview
This repository contains code for a machine learning model that predicts the likelihood of chronic kidney disease (CKD) based on various health parameters. The model utilizes an ensemble approach, combining three different algorithms: Random Forest, Gradient Boosting, and Logistic Regression. The input data is provided in CSV format.

## Dataset
The dataset used for training and testing the model is not provided in this repository due to privacy concerns. However, you can use your own dataset containing relevant health parameters such as age, blood pressure, serum creatinine levels, etc. Make sure the dataset is in CSV format and follows the same structure as described in the Data Preparation section below.

## Installation
To run the code in this repository, you'll need Python 3.x along with the following libraries:
- NumPy
- pandas
- scikit-learn

You can install these dependencies using pip:
```
pip install numpy pandas scikit-learn
```

## Usage
1. Clone this repository to your local machine:
```
git clone https://github.com/your-username/chronic-kidney-disease-prediction.git
```

2. Navigate to the repository directory:
```
cd chronic-kidney-disease-prediction
```

3. Place your dataset file (in CSV format) inside the `data` directory.

4. Run the `main.py` script to train the model and make predictions:
```
python main.py
```

5. Follow the on-screen instructions to provide the necessary input for prediction.

## Data Preparation
Ensure that your dataset follows the following structure:
- The first row contains column headers.
- Each subsequent row represents a sample.
- The last column contains the target variable (1 for positive cases of CKD, 0 for negative cases).

Example dataset structure:
```
age,gender,blood_pressure,serum_creatinine,albumin,diabetes,CKD
65,M,140/90,1.9,4.0,yes,1
45,F,130/80,1.1,0.7,no,0
...
```

## Model Evaluation
The model's performance can be evaluated using various metrics such as accuracy, precision, recall, and F1-score. These metrics will be displayed during the model training process and can also be calculated separately using the `evaluate_model()` function in `evaluation.py`.

## Contributing
Contributions to this project are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License
This project is not licensed. Feel free to copy.
