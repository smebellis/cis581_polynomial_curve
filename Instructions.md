# Task

Overall we are trying to apply polynomial curve-fitting for regression learning using a dataset of U.S. Covid-19 cases.  We will be using 12 fold cross-validation for model selection to determine the optimal degree for low-degree polynomial and evaluate the Root Mean Square Error (RMSE) of the selected polynomial models. 

- Data Preparation
    -  Data is split into training and test sets.
    - Normalize the input and output data using standard scaling
    - Normalize the transformed polynomial features
- Model Selection
    - Implement k-fold cross-validation (specifically, 12-fold CV) on the training data
    - Condisder polynomial degrees from 0 to 28
    - For each degree, compute and record the RMSE for both the CV training and test sets on each fold
    - Calculate the average RMSE across all folds for each polynomial degree
    - Select the polynomial degree (d*) that achieves the minimum average RMSE on the CV tests sets
- Regularization (for degree 28): Additionally, for a polynomial of degree 28, explore different regularization parameters
    - Use the set of $\lambda$ values: {0, $e^{-30}$, $e^{-28}$, $e^{-26}$, ..., $e^{10}$}
    - Perform 12-fold CV to select the optimal $\lambda$ ($\lambda$*)
- Final Model Training and Evaluation: Train the final models using all the training data
    - Train a polynomial with optimal degree d*
    - Train a regularized 28-degree polynomial model using the optimal regularization parameter $\lambda$*
    - Report the coefficient-weights for d* and $\lambda$*
    - Calculate and report the training and test RMSE for the final learned polynomials
- Visualization: Plot the training data along with the resulting polynomial curves for d* and $\lambda$* over the range of weeks 1-91
- Reporting: Prepare a written report including the average CV RMSE values, optimal d* and $\lambda$*, coefficient-weights, training and test RMSE, plots, and a discussion of findings.
- Code Submission: Submit all code with instructions on how to run the program.  The code should be executable as a standalone program from the command line, using standard tools/compilers


