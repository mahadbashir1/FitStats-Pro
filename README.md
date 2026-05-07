# FitStats Pro: Ghost Busters

**FitStats Pro** is a comprehensive web-based interactive data analysis application designed for the Probability & Statistics course project. It analyzes a BodyFat Extended dataset, providing users with rich statistical insights, predictive modeling, and graphical representations natively in the browser.

## 🚀 Features

### 1. Data Overview
*   **Variable Dictionary:** Detailed context and domain interpretation for every dataset variable.
*   **Frequency Tables:** Generate relative, cumulative, and percentage frequency distributions for both numerical and categorical variables.
*   **Live Preview:** Instantly inspect the dataset's rows and shapes.

### 2. Visualizations
*   **Histograms & Distributions:** Visualize numeric columns and get auto-generated interpretations of skewness and shape.
*   **Scatter Plots:** Explore pairwise relationships between variables, easily grouped by a selected categorical variable.
*   **Bar Charts:** Compare categorical data accurately.
*   **Box Plots:** Automatically detect and flag physiological or data-entry outliers using the 1.5×IQR rule.

### 3. Descriptive Statistics
*   Calculates essential measures of central tendency (Mean, Median, Mode).
*   Calculates dispersion (Variance, Standard Deviation, Range, IQR).
*   Provides robust interpretive context for data shape including the coefficient of variation (CV) and skewness.

### 4. Probability Distributions
*   **Uniform & Normal Fitting:** Generates random theoretical samples and analyzes how well the target metrics conform to uniform or normal distributions using measures such as the Shapiro-Wilk test natively in Python.

### 5. Correlation & Regression Analysis
*   Generate full **Correlation Matrices** measuring predictors for body fat percentage.
*   **Pairwise Correlation testing:** Performs significance checks (p-values) for Pearson corellations.
*   **Multivariate OLS Regression:** Model variables predicting body fat. Formulates Ordinary Least Squares estimates utilizing constants and dummy categorical variables. Displays $R^2$, F-statistics, intervals, and categorical prediction tools.

### 6. Inferential Statistics
*   **Hypothesis Testing:** Run interactive single-sample t-tests (One-Sided/Two-Sided) against specified null hypothesis parameters directly in the user interface.
*   **Confidence Intervals:** Construct statistical confidence intervals for the population mean of physical metrics. 

## 🛠️ Setup Instructions

### Prerequisites
Make sure you have Python 3.8+ installed on your system.

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd "Prob Project (Python)"
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**
   Install the required libraries listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   Launch the Flask development server:
   ```bash
   python app.py
   # Or using Flask directly:
   # flask run
   ```

5. **Access the Web UI**
   Open your browser and navigate to: `http://127.0.0.1:5000/`

## 📦 Tech Stack
*   **Backend Framework:** Flask
*   **Data Processing:** pandas, numpy
*   **Statistical Analysis:** scipy, statsmodels
*   **Frontend UI:** HTML5, CSS3, Vanilla JS
*   **Charting Library:** Plotly.js
