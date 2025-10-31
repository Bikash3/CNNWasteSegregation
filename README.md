# Waste Segregation
> This project aims to detect fraudulent insurance claims using machine learning models, helping insurers reduce financial losses and improve operational efficiency. The process began with thorough data preprocessing and feature engineering to clean the dataset, handle missing values, and create meaningful features that capture fraud patterns. Two models—Logistic Regression and Random Forest—were developed to classify claims as either fraudulent or legitimate. To enhance model performance, hyperparameter tuning and threshold optimization were performed using cross-validation and grid search techniques. The final models were evaluated using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC to ensure robustness and reliability. This end-to-end approach demonstrates how machine learning can be effectively leveraged for fraud detection in the insurance sector.

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

## General Information
> We at UPGRAD doing this Waste Segregation Case Study as a part of Model Selection.

> Global Insure, a leading insurance company, processes thousands of claims annually. However, a significant percentage of these claims turn out to be fraudulent, resulting in considerable financial losses. The company’s current process for identifying fraudulent claims involves manual inspections, which is time-consuming and inefficient. Fraudulent claims are often detected too late in the process, after the company has already paid out significant amounts. Global Insure wants to improve its fraud detection process using data-driven insights to classify claims as fraudulent or legitimate early in the approval process. This would minimise financial losses and optimise the overall claims handling process.

## Conclusions
> Random forest model is effective model for fraud detection. Careful preprocessing and feature selection improved model performance. This solution enables early detection of fraud, reducing financial loss and improving operational efficiency.

## Key Insights
* Fraudulent claims often exhibit distinct patterns in terms of claim timing, incident severity, and policy details.
* Behavioural features (e.g., insured relationship, hobbies, claim amount) are used to identify suspicious activity.
* The most predictive features are incident_severity, insured_relationship, property_damage, incident_city and auto_make
* Robust against overfitting and capable of handling both numerical and categorical data.
* Feature importance from the model provided interpretable insights into fraud indicators.
* Handling missing values, encoding categorical data, and transforming timestamps significantly boosted model performance.
* Multicollinearity checks and RFECV helped refine the model to include only the most informative features.

## Technologies Used
  Python Coding and Visualizations


## Acknowledgements
Give credit here.
- This project was inspired by Upgrad 
- References if any...
    https://matplotlib.org/stable/api/index.html
    https://seaborn.pydata.org/examples/index.html



## Contact
### Created by 
  * Jyothsna Devi Duggasani
  * Bikas Sarkar 
      


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
