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

