# Credit Risk Modelling

A credit risk classification project focused on predicting whether a loan applicant is likely to be a good or bad borrower based on customer profile and credit attributes.

## Overview

This project uses a German credit dataset and explores the relationship between applicant characteristics, financial indicators, and risk outcome. The workflow includes:

- exploratory data analysis (EDA)
- feature engineering
- categorical encoding
- model training and comparison
- model selection with XGBoost
- a Streamlit web app for live prediction

## Why this project matters

Credit risk modelling is a classic supervised learning use case in financial analytics. Accurate prediction supports:

- better lending decisions
- reduced default risk
- more consistent underwriting processes
- improved business understanding of borrower behavior

## Demo

The GIFs below are kept in the local `gifs/` folder for demonstration purposes and are intentionally ignored by Git so the repository stays clean and lightweight.

> If your GIF files are stored locally in `gifs/`, they will render in the README from your workspace. They will not be included in Git pushes because of the ignore rule.

![App walkthrough](gifs/streamlit_demo.gif)
![Prediction flow](gifs/credit_risk_prediction.gif)

## Project structure

```text
credit_risk_modelling/
├── .gitignore
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── german_credit_data.csv
├── gifs/
│   └── .gitkeep
├── Notebooks/
│   └── EDA_model.ipynb
└── .venv/
```

## Tech stack

- Python 3.11
- pandas
- NumPy
- scikit-learn
- XGBoost
- seaborn
- matplotlib
- Streamlit

## Setup

1. Clone the repository.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Data

The project uses the German credit dataset stored in `data/german_credit_data.csv`.

## Modelling approach

The workflow in the notebook includes:

- data cleaning and missing value checks
- univariate and bivariate analysis
- categorical encoding with `LabelEncoder`
- train/test split
- comparison of models such as:
  - Decision Tree
  - Random Forest
  - Extra Trees
  - XGBoost
- final model selection based on validation performance

## Model output

The selected model is saved locally as a serialized artifact for use in inference and app deployment.

## Notes

- The GIF assets are intentionally ignored by Git to keep the repository clean.
- To see the demos in your local README preview, make sure the GIF files are present in the `gifs/` directory.
- If you want the GIFs uploaded to GitHub, remove the ignore rule for `gifs/*.gif`.

## License

This project is intended for educational and portfolio use.
