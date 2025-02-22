# Polynomial Curve Fitting with Ridge Regression

This project implements a polynomial curve fitting algorithm enhanced with ridge regression. It is designed for data analysis and prediction, leveraging cross-validation to select the best polynomial degree and ridge regularization parameter (lambda).

## Getting Started

### Clone the Repository

To get started, clone the repository using Git:

```bash
git clone https://github.com/smebellis/cis581_polynomial_curve.git
```

Then, navigate into the project directory:

```bash
cd cis581_polynomial_curve
```

### Prerequisites

Ensure you have the following Python packages installed:
- numpy
- pandas
- matplotlib
- scikit-learn

You can install these by running:

```bash
pip install -r requirements.txt
```

*Tip: It is recommended to use a virtual environment (e.g., using `venv` or `conda`) for dependency isolation.*

## Project Overview

The code in this repository performs the following tasks:

- **Data Loading:** Reads training and test datasets from specified files.
- **Polynomial Feature Generation:** Converts input features into polynomial features up to a specified degree.
- **Feature Standardization:** Standardizes features (excluding the bias term) for numerical stability.
- **Model Training:** 
  - Uses standard linear regression to train a polynomial model.
  - Applies ridge regression (with lambda regularization) to improve generalization.
- **Cross-Validation:** Employs k-fold cross-validation (with 12 folds) to:
  - Evaluate different polynomial degrees.
  - Find the optimal lambda value that minimizes the root-mean-square error (RMSE).
- **Plotting:** Optionally generates plots of the fitted curves for both training and test data.
- **Final Evaluation:** Trains a final model on the full training set and computes the test RMSE.

## How to Run the Code

There are three ways to execute this project:

### 1. Run as a Python Command Line Application

1. **Install Requirements:**  
   Ensure you have installed all the necessary packages (see [Prerequisites](#prerequisites)).
2. **Execute the Script:**  
   Run the Python script from the command line while providing the required arguments. For example:
   ```bash
   python ML_learning_algo.py --train path/to/train.dat --test path/to/test.dat --plot
   ```
   - `--train`: Path to the training dataset file.
   - `--test`: Path to the test dataset file.
   - `--plot`: Optional flag; if provided, the script generates plots of the fitted curves.

### 2. Run Using the Provided Bash Script

The project includes a bash script (e.g., `run.sh`) to streamline execution.

1. **Make the Script Executable (on Unix-like systems):**  
   On a new computer, you may need to change the script’s permissions:
   ```bash
   chmod +x run.sh
   ```
2. **Execute the Script:**  
   Run the script from your terminal:
   ```bash
   ./run.sh
   ```
3. **Windows Considerations:**  
   - On Windows, you cannot run bash scripts natively.
   - Install **Git Bash** or use **Windows Subsystem for Linux (WSL)** to run the script.

### 3. Run the PyInstaller Executable

An executable version of the application has been created using PyInstaller. Note the following:

- **Running the Executable:**  
  Simply run the executable file (e.g., `ML_learning_algo`) from your command line.
  
- **Windows Compatibility:**  
  Since the executable was compiled using WSL2, it may not run directly on native Windows. In that case, use **WSL2** or a **Linux environment** to execute it.

## Example Execution

#### Using Python Directly:
```bash
python ML_learning_algo.py --train /train.dat --test /test.dat --plot
```

#### Using the Bash Script:
1. On Linux/WSL/Git Bash:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

#### Using the PyInstaller Executable:
```bash
./ML_learning_algo
```

*(On Windows, run the executable from a Linux-like environment such as WSL2.)*

## Additional Details

- **Cross-Validation:**  
  The script performs 12-fold cross-validation to avoid overfitting and to select the best polynomial degree and lambda value.
  
- **Plotting:**  
  When the `--plot` flag is enabled, the program saves plots (e.g., `Train_plot.png` and `Test_plot.png`) that visually compare the data points with the regression curves.
  
- **Standardization:**  
  Features (except the bias term) are standardized using the training set’s mean and standard deviation, ensuring consistency between training and testing.
