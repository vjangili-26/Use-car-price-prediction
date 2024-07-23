# USED CAR PRICE PREDICTION

## Introduction
The project's goal is to create a reliable and accurate predictive model that uses a variety of features and historical data to estimate used car prices. It is now essential for both buyers and sellers to anticipate prices accurately due to the rising demand for used cars. To create a trustworthy model that can accurately estimate used car prices, this study makes use of machine learning techniques implemented in R Studio.

To train and assess the predictive models, the project will make use of a variety of linear continuous models and non-linear continuous models. To improve the models' performance and capacity for generalization, cross-validation techniques will be employed to refine and optimize them.

To measure the precision and dependability of the predictions, the models will be evaluated using metrics such as Root Mean Squared Error (RMSE) and R-squared. To add to the interpretability, the most successful model will be chosen and its insights into the significance of the features will be examined.

The dataset contains 8128 records of cars with 13 attributes, including the target variable describing the selling price of a particular car. The dataset is an open-source dataset found on Kaggle.

## Data Description

| Features       | Data        | Description                                   |
|----------------|-------------|-----------------------------------------------|
| Name           | Categorical | Name of the car                               |
| Year           | Continuous  | Year of manufacturing                         |
| Km_driven      | Continuous  | Total Kms the car was driven                  |
| fuel_type      | Categorical | Petrol, Diesel, LPG, CNG                      |
| seller_type    | Categorical | Type of seller, e.g., owner/dealer            |
| transmission   | Categorical | Automatic/Manual                              |
| owner          | Categorical | First or second owner                         |
| mileage        | Continuous  | Mileage offered by car                        |
| engine_cc      | Continuous  | The displacement volume of the engine in CC   |
| max_power      | Continuous  | Maximum power output                          |
| seats          | Continuous  | Number of seats in car                        |
| torque         | Continuous  | Rotational force engine can apply             |
| selling_price  | Continuous  | Price of the car                              |

## Pre-Processing of Data
Data preprocessing is an essential step to ensure the quality and dependability of the ensuing analyses or model predictions. It involves several methods and procedures to convert unprocessed data into a standardized, neat format.

### A. Handling Missing Data
- Missing values are identified and handled using the `na.omit()` function to remove rows with missing values.

### B. Transformation
- Continuous variables' distributions are analyzed using box plots and histograms.
- Centering and scaling are applied to account for skewness, outliers, and varying scales.
- Box-Cox transformation and spatial sign transformation are used to normalize the data.

### C. Dummy Variables
- Categorical data are numerically represented using dummy variables to incorporate information about categorical variables into models that need numerical input.

### D. Near Zero Variables
- Variables with near-zero variance, such as `fuel_typeCNG`, `fuel_typeLPG`, `seller_typeTrustmarkdealer`, `ownerFour/above`, and `ownerTest`, are removed from the dataset to reduce noise and improve model performance.

### E. Highly Correlated Variables
- Highly correlated variables are identified and removed to address multicollinearity, improving the stability and interpretability of the model.

After preprocessing, the dataset has 7906 observations and 12 predictors.

## Data Splitting
- The variable "selling price" is predicted using stratified random sampling, with 80% of each outcome used in the training set and 20% for testing.
- K-cross-validation is used to set up the training control parameter with 3 folds.

## Model Fitting
The data was used to train both linear and non-linear continuous models, and for any model with tuning parameters, additional figures are provided in the appendices. The outcomes of these models' predictions on the training set are shown below. Cross-validation produced the R-squared values for the most conservative results.

### Linear Continuous Models

| Model           | Best Tuning Parameter | Training R² |
|-----------------|-----------------------|-------------|
| Linear Model    | Intercept = True      | 0.668314    |
| Ridge           | Lambda = 0.0071       | 0.67237     |
| LASSO           | alpha=1, lambda=0.1   | 0.6712203   |
| E NET           | fraction = 0.95, lambda = 0.01 | 0.672509  |

### Non-Linear Continuous Models

| Model           | Best Tuning Parameter | Training R² |
|-----------------|-----------------------|-------------|
| Neural Network  | Size=1, Decay=0.1     | 0.69361     |
| SVM             | Sigma = 0.1, C=10     | 0.81239     |
| KNN             | k=4                   | 0.73249     |
| MARS            | Nprune = 21, degree=2 | 0.789187    |

The two top models are the Elastic Net model and the Support Vector Machine based on the R-squared values after predicting the test data set.

### Model Performance

| Model  | R-Squared | RMSE     |
|--------|-----------|----------|
| E Net  | 0.67710   | 45915.11 |
| SVM    | 0.81678   | 1.64328  |

Variable importance in LOESS is assessed using R-squared values, where higher values indicate a variable's influence on the response variable.

### Top 5 Important Variables of SVM Model

![image](https://github.com/user-attachments/assets/44b6112e-78be-4520-a013-6529420d791b)


## Conclusion
Our used car price prediction project employs advanced machine learning techniques in R Studio to deliver accurate and reliable price estimates. The user-friendly application enhances transparency in the used car market, aiding buyers and sellers with informed decision-making. The best model found during this analysis was the Support Vector Machine with an R-squared value of 0.81678 after predicting the test data set.

## References
- [Used Cars Price Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction)

## Appendices

### Appendix 1: Linear Continuous Models
#### A. Linear Regression
- Tuning parameter: Intercept = True
- 6327 samples, 12 predictors
- Resampling: Cross-Validated (3 fold)
- RMSE: 469908.5, R²: 0.668314, MAE: 278638.7

#### B. Ridge Regression
- Pre-processing: centered (12), scaled (12)
- Resampling: Cross-Validated (3 fold)
- Lambda: 0.007142857
- RMSE: 467700.0, R²: 0.6723782, MAE: 277875.8

#### C. LASSO
- Pre-processing: centered (12), scaled (12)
- Resampling: Cross-Validated (3 fold)
- Alpha: 1, Lambda: 0.1
- RMSE: 467514.8, R²: 0.6712203, MAE: 276882.5

#### D. Elastic Net
- Pre-processing: centered (12), scaled (12)
- Resampling: Cross-Validated (3 fold)
- Fraction: 0.95, Lambda: 0.01
- RMSE: 466493.7, R²: 0.6723535, MAE: 274709.9

### Appendix 2: Non-Linear Continuous Models
#### A. Neural Network
- Pre-processing: centered (12), scaled (12)
- Resampling: Cross-Validated (3 fold)
- Decay: 0.1, Size: 1
- RMSE: 2.196803, R²: 0.69361172, MAE: 1.697985

#### B. Support Vector Machine with Radial Basis Function Kernel
- Pre-processing: centered (12), scaled (12)
- Resampling: Cross-Validated (3 fold)
- Sigma: 0.1, C: 10
- RMSE: 1.676611, R²: 0.8123967, MAE: 1.170976

#### C. K-Nearest Neighbors
- Pre-processing: centered (12), scaled (12)
- Resampling: Cross-Validated (3 fold)
- K: 4
- RMSE: 2.003764, R²: 0.7324919, MAE: 1.369141

#### D. Multivariate Adaptive Regression Spline
- No pre-processing
- Resampling: Cross-Validated (3 fold)
- Degree: 2, Nprune: 21
- RMSE: 1.774042, R²: 0.7890059, MAE: 1.313411
