### Healthcare Stroke Data Analysis and Model Evaluation

## Project Overview

This project involves the analysis and predictive modeling of a healthcare dataset focused on stroke data. The dataset is preprocessed, explored, and visualized to derive insights, followed by the application of logistic regression and support vector machine (SVM) models to predict the occurrence of strokes.

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Exploration and Preprocessing](#exploration-and-preprocessing)
4. [Data Visualization](#data-visualization)
5. [Modeling and Evaluation](#modeling-and-evaluation)
6. [Conclusion](#conclusion)

## Installation

To run this project, you need the following libraries:
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these using pip:
```sh
pip install pandas matplotlib seaborn scikit-learn
```

## Dataset

The dataset used in this project is `archhealthcare-dataset-stroke-data.csv`. Make sure to place this file in the same directory as the script.

## Exploration and Preprocessing

### Data Exploration
The data exploration phase includes:
- Displaying the dataset
- Checking the size and columns of the dataset
- Viewing the first 5 rows
- Inspecting data types and identifying missing values
- Descriptive statistics

### Data Preprocessing
The preprocessing steps include:
- Filling missing values in the 'bmi' column with the mean
- Replacing 'Other' in the gender column with 'Male'
- Encoding categorical variables
- Dropping unnecessary columns (`work_type` and `id`)

## Data Visualization

The following visualizations are created to understand the data better:
- Age distribution histogram
- Gender count plot
- BMI distribution by smoking status
- Stroke count plot
- Smoking status distribution pie chart
- Age vs. average glucose level scatter plot
- Correlation heatmap
- Age distribution by marital status box plot

## Modeling and Evaluation

Two machine learning models are used in this project:
1. **Logistic Regression**
2. **Support Vector Machine (SVM)**

### Logistic Regression Evaluation
- Confusion matrix
- Accuracy score
- F1 score
- Recall score

### SVM Evaluation
- Confusion matrix
- Accuracy score
- F1 score
- Recall score

## Conclusion

This project demonstrates the complete workflow of data exploration, preprocessing, visualization, and modeling using logistic regression and SVM for stroke prediction. The insights derived and the performance metrics of the models can be used to further improve predictive accuracy and make data-driven decisions in healthcare.

---

Feel free to modify any part of this README as per your specific project details!
