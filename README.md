# Heart Disease Prediction 🫀 

## About the Project  
This project aims to predict the likelihood of heart disease in patients using supervised machine learning algorithms.  
The objective is to preprocess patient data, train multiple models, evaluate their performance, and identify the best model for accurate prediction.  

---

## Objectives  
- Load and preprocess the heart disease dataset  
- Perform exploratory data analysis (EDA)  
- Train multiple ML models (Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes)  
- Evaluate models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC  
- Compare models and highlight the best performing algorithm  

---

## Dataset  
- **Source:** UCI Repository / Kaggle  
- **File Name:** heart.csv  
- **Key Features:** Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Induced Angina, Oldpeak, Slope, Thal, Target (0 = No Disease, 1 = Disease)  

---

## Tools and Technologies  
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Models:** Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes  

---

## Recommendations  
- **Random Forest** is the most reliable for predictive accuracy  
- **Logistic Regression** is preferred for interpretability in medical decision-making  
- Focus on features like Chest Pain Type, Age, Cholesterol, and Max Heart Rate which strongly influence predictions  

---

## Challenges Faced  
- Dataset imbalance required careful evaluation using ROC-AUC  
- Handling categorical variables needed proper encoding  
- Avoiding overfitting in Decision Trees  

---

## Future Scope  
- Deploy the best model as a web application using Flask or Django  
- Experiment with Deep Learning (ANNs) for improved accuracy  
- Collect larger datasets for better generalization  
- Integrate with healthcare dashboards for real-time prediction  

---

> * This project strengthened my skills in supervised machine learning, model evaluation, and health data analytics to generate actionable medical insights. * 
