# FastAPI Machine Learning API

This project is a FastAPI-based application that allows users to upload a dataset, train a Logistic Regression model, and make predictions about downtime.

## Features

- Upload a CSV dataset.
- Train a Logistic Regression model.
- Predict downtime based on input features.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```
2. Set up a virtual environment and install dependencies:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   uvicorn main:app --reload
   ```

## Endpoints

- **`/upload`**** (POST)**: Upload a CSV file containing `Temperature`, `Run_Time`, and `Downtime_Flag`.
- **`/train`**** (POST)**: Train the model and get accuracy and F1 score.
- **`/predict`**** (POST)**: Predict downtime (`Yes`/`No`) with confidence score.

## Usage

1. Start the app:
   ```bash
   uvicorn main:app --reload
   ```
2. Access API docs:
   - Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)



