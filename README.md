# Insurance Price Prediction

This project is focused on predicting insurance premium prices based on various health and demographic features. The goal is to build a model that can predict the insurance price (premium) for individuals based on their medical history, age, height, weight, and other related features.

## Dataset

The dataset used for this project consists of 986 entries and 10 features, with the target variable being the insurance premium price. Below are the details of each column in the dataset:

| Column                     | Description                                              |
|----------------------------|----------------------------------------------------------|
| **Age**                     | The age of the individual (int)                          |
| **Diabetes**                | Indicates if the individual has diabetes (1 = Yes, 0 = No) |
| **BloodPressureProblems**   | Indicates if the individual has blood pressure problems (1 = Yes, 0 = No) |
| **AnyTransplants**          | Indicates if the individual has had any organ transplants (1 = Yes, 0 = No) |
| **AnyChronicDiseases**      | Indicates if the individual has any chronic diseases (1 = Yes, 0 = No) |
| **Height**                  | Height of the individual (in centimeters)                |
| **Weight**                  | Weight of the individual (in kilograms)                  |
| **KnownAllergies**          | Indicates if the individual has any known allergies (1 = Yes, 0 = No) |
| **HistoryOfCancerInFamily** | Indicates if there is a history of cancer in the individual’s family (1 = Yes, 0 = No) |
| **NumberOfMajorSurgeries**  | The number of major surgeries the individual has undergone (0, 1, 2, 3) |
| **PremiumPrice**            | The target variable: the insurance premium price (int)   |

## Project Overview

### Objectives
The goal of this project is to predict the **PremiumPrice** (insurance premium) based on the provided features. The model will be trained on the dataset and can be used to predict the premium price for new individuals based on their health and demographic characteristics.

### Features
The features include both medical and demographic information, such as:

- Age
- Presence of chronic conditions (e.g., diabetes, blood pressure problems)
- Medical history (e.g., transplants, surgeries)
- Physical attributes (height and weight)
- Family medical history (e.g., cancer)

### Target
The target variable is the **PremiumPrice**, which represents the amount of money an individual needs to pay for their insurance.

## Getting Started

### Prerequisites
To set up this project and install all the required dependencies, use the following steps.

1. **Clone the repository**:

   ```bash
   git clone https://github.com/wmujawar/Insurance-Price-Predictor.git
   cd Insurance-Price-Predictor
   ```

2. **Create a virtual environment (optional but recommended):**

   If you are using Python 3, create a virtual environment to avoid conflicts with other packages on your system:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`

   ```

3. **Install the dependencies:**

   Install the required dependencies listed in the requirements.txt file:

   ```bash
   pip install -r requirements.txt
   ```

   This will install all necessary packages, including:

    - numpy==1.26.4
    - pandas==2.2.2
    - scikit-learn==1.5.1
    - scipy==1.13.1
    - xgboost==2.1.1
    - flask==3.0.3
    - pytest==8.3.4
    
4. **Testing:**
   To run the tests, in terminal, execute
   
   ```bash
   pytest
   ```

5. **Flask API:**
   To start flask api, run

   ```bash
   python src/app.py
   ```

## View models and their performance using MLFlow

### 1. Install MLFlow
Install MLFlow

```bash
pip install mlflow
```

### 2. View Results
Run `notebooks/06_visualize_mlflow_matrix.ipynb` to visualize model performance

[Visualize Models performance](notebooks/06_visualize_mlflow_matrix.ipynb)


## Docker Setup: Build and Run a Container
To containerize the application and ensure it runs in any environment, you can use Docker. Below are the steps to build a Docker image and run a container.

### 1. **Build the Docker Image**

In the terminal, navigate to the directory containing the `Dockerfile` and run the following command to build the Docker image:

```bash
docker build -t insurance-price-prediction .
```

This command tells Docker to build an image with the tag `insurance-price-prediction` using the current directory (denoted by `.`) as the context.

### 2. **Run the Docker Container**

After the image is built, run the container:

```bash
docker run -d -p 5000:5000 insurance-price-prediction
```

This command starts a Docker container based on the `insurance-price-prediction` image and maps port 5000 of the container to port 5000 on your local machine. The Flask API will be accessible at `http://localhost:5000`.

## Running the Streamlit App

To run the Streamlit app, follow these steps:

### 1. **Ensure that you have Streamlit installed. If not, you can install it via pip:**

   ```bash
   pip install streamlit
   ```

### 2. **Navigate to the project directory where the `src/streamlit_app.py` file is located.**

### 3. **Run the Streamlit app using the following command:**

   ```bash
   streamlit run src/streamlit_app.py
   ```

### 4. **This will launch the app in your default web browser, where you can interact with it.**
