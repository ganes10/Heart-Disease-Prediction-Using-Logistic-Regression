# Heart Disease Prediction using Logistic Regression

## Overview
This project focuses on predicting the likelihood of heart disease in individuals using logistic regression. The model is trained on a dataset containing various health metrics and aims to provide accurate predictions to assist in early diagnosis and prevention.

## Features
- Data preprocessing and cleaning
- Exploratory data analysis (EDA)
- Logistic regression model implementation
- Model evaluation using metrics like accuracy, precision, recall, and F1-score
- Visualization of results

## Requirements
- Python 3.12.4
- Libraries:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn

## Installation
1. Clone the repository:
     ```bash
     git clone https://github.com/ganes10/Heart-Disease-Prediction-Using-Logistic-Regression.git
     ```
2. Navigate to the project directory:
     ```bash
     cd heart-disease-prediction
     ```
3. Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

## Usage
1. Prepare the dataset and place it in the `data/` directory.
2. Run the preprocessing script:
     ```bash
     python preprocess.py
     ```
3. Train the model:
     ```bash
     python train.py
     ```
4. Evaluate the model:
     ```bash
     python evaluate.py
     ```

## Dataset
The dataset used for this project should include features such as age, cholesterol levels, blood pressure, and other relevant health metrics. Ensure the dataset is properly formatted before use.

## Results
The logistic regression model achieves high accuracy in predicting heart disease. Detailed evaluation metrics and visualizations are provided in the `results/` directory.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [scikit-learn documentation](https://scikit-learn.org/)
- Publicly available heart disease datasets
- Inspiration from healthcare analytics projects
