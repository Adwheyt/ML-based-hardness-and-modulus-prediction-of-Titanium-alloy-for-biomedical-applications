# Machine Learning-Based Prediction of Elastic Modulus in Titanium Alloys

This project applies **machine learning regression models** to predict the **Elastic Modulus (EM, in GPa)** of titanium alloys based on material properties. The goal is to achieve a predictive model with an **R² score above 60%**, improving upon baseline models like Random Forest.

---

## Dataset
The dataset used is `revised_titanium_alloy_data.csv`, which contains material properties such as:
- Tensile strength
- Yield strength
- Percentage elongation
- Density
- Carbon content

The target variable is:
- **Elastic Modulus (EM_GPa)**

---

## Models Explored
We evaluated several models from **scikit-learn**:
- Random Forest Regressor
- Gradient Boosting Regressor (GBR)
- Extra Trees Regressor
- HistGradientBoosting Regressor

Among these, **Gradient Boosting Regressor (GBR)** with tuned hyperparameters performed the best, achieving close to **R² ≈ 0.59**.

---

## Current Best Model (GBR)
The tuned **Gradient Boosting Regressor** uses the following parameters:
```python
GradientBoostingRegressor(
    subsample=0.9,
    n_estimators=1000,
    max_features=None,
    max_depth=6,
    learning_rate=0.005,
    random_state=42
)
```

Performance on test set:
- **R² Score**: ~0.59
- **RMSE**: ~11.3 GPa

---

## Installation & Usage
1. Clone the repository:
```bash
git clone https://github.com/your-username/titanium-alloy-ml.git
cd titanium-alloy-ml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training script:
```bash
python3 svg2.py
```

---

## Requirements
- Python 3.8+
- pandas
- scikit-learn
- numpy

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## Project Structure
```
├── revised_titanium_alloy_data.csv   # Dataset
├── svg2.py                           # Training & evaluation script
├── README.md                         # Project documentation
└── requirements.txt                  # Dependencies
```

---

## Future Work
- Explore **XGBoost/LightGBM** for further performance improvements.
- Try **feature engineering** (ratios, transformations) to boost predictive power.
- Implement **cross-validation + early stopping** for more stable results.

---
