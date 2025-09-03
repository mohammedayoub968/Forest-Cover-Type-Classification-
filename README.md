# 🌲 Forest Cover Type Classification

This project predicts the **forest cover type** from cartographic variables using different **tree-based machine learning models**.  
The dataset is based on the **Covertype Dataset**, which includes various environmental and geographical features.  
The goal is to compare traditional decision trees with advanced ensemble methods and optimize them with hyperparameter tuning.

---

## 📂 Dataset
- **Source**: [Forest Cover Type Dataset on Kaggle](https://www.kaggle.com/datasets/zsinghrahulk/covertype-forest-cover-types)  
- **Target Variable**: `Cover_Type` (7 categories of forest cover types)  
- **Features**:  
  - Elevation, Aspect, Slope  
  - Hillshade measures (at 9am, Noon, 3pm)  
  - Distances to hydrology, roadways, and fire points  
  - Wilderness Area (one-hot encoded)  
  - Soil Type (40 categories, one-hot encoded)  

Data preprocessing steps:
- Dropped unnecessary columns (`Unnamed: 0, 1, 2`)  
- Checked and removed duplicates  
- No missing values were found  
- Train/test split (80/20) with **stratified sampling**  

---

## ⚙️ Project Features
- Load dataset directly with **kagglehub**
- Exploratory Data Analysis (EDA): distributions, correlations
- Train/test split with stratification
- Train and evaluate multiple ML models:
  - ✅ Decision Tree
  - ✅ Random Forest
  - ✅ Gradient Boosting (GBM)
  - ✅ CatBoost
- Feature importance ranking & visualization
- Experiment with **Top 15 features only**
- Hyperparameter tuning with:
  - `GridSearchCV` (Decision Tree)
  - `RandomizedSearchCV` (Random Forest)
- Model evaluation using:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)

---

## 📊 Model Performance Report

### 🔹 Decision Tree (All Features)
- **Accuracy**: `91.57%`
- **Classification Report**: Shows good precision and recall, but slightly lower for certain cover types.
- **Confusion Matrix**: Misclassifications mostly occur between similar vegetation types.
- **Feature Importance**: Top predictors include *Elevation*, *Hillshade_9am*, and *Slope*.

---

### 🔹 Decision Tree (Top 15 Features)
- **Accuracy**: `87.8%`
- **Notes**: Using fewer features reduces model complexity and training time but lowers accuracy.

---

### 🔹 Random Forest
- **Accuracy**: `93.6%` ✅ *(Best Model)*
- **Classification Report**: Balanced precision and recall across all classes.
- **Insights**: Robust against overfitting, performs consistently better than a single Decision Tree.
- **Top Features**: *Elevation*, *Slope*, and several *Soil_Type* variables.

---

### 🔹 Gradient Boosting (GBM)
- **Accuracy**: `70%`
- **Notes**: Underperformed compared to Random Forest. Likely requires deeper hyperparameter tuning and better handling of class imbalance.

---

### 🔹 CatBoost
- **Accuracy**: `70.4%`
- **Notes**: Similar to GBM, did not outperform Random Forest. Could benefit from tuning learning rate and depth.

---

### 🔹 Hyperparameter Tuning
- **Decision Tree (GridSearchCV)**
  - Best Params: `{max_depth=None, min_samples_leaf=1, min_samples_split=2}`
  - Accuracy: `91.58%`
- **Random Forest (RandomizedSearchCV)**
  - Best Params: `{n_estimators=300, min_samples_split=10, min_samples_leaf=2, max_depth=None}`
  - Accuracy: `92.35%`

---

## 📈 Summary Report

| Model                               | Accuracy   | Key Insights |
|-------------------------------------|------------|--------------|
| Decision Tree (All Features)        | **0.9157** | Strong baseline but slightly overfits |
| Decision Tree (Top 15 Features)     | 0.8780     | Faster, simpler, but less accurate |
| Random Forest                       | **0.9360** | ✅ Best performing model, robust and accurate |
| Gradient Boosting (GBM)             | 0.7000     | Weak performance, needs tuning |
| CatBoost                            | 0.7040     | Comparable to GBM, underperforms |
| Decision Tree (GridSearchCV tuned)  | 0.9158     | Same as baseline Decision Tree |
| Random Forest (RandomizedSearchCV)  | 0.9235     | Tuned version, slightly worse than default |

📌 **Conclusion**:  
- The **Random Forest Classifier** achieved the **highest accuracy (93.6%)** and is the most reliable model for this dataset.  
- Simpler models (Decision Trees) perform reasonably well but lack robustness.  
- Boosting methods (GBM, CatBoost) underperformed in this setup, but could improve with advanced tuning.  



```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost kagglehub
