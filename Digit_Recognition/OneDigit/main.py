from tqdm.auto import tqdm
from kernel import *
from features import *
from softmax import *
from svm import *
from linear_regression import *
from utils import *
import sys
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
sys.path.append("..")

###############################################################################
# 1. Load MNIST data:
###############################################################################
train_x, train_y, test_x, test_y = get_MNIST_data()
# Plot the first 20 images of the training set.
plot_images(train_x[0:20, :])


###############################################################################
# 2. Linear Regression with Closed Form Solution
###############################################################################
def run_linear_regression_on_MNIST(lambda_factor=1):
    """
    Trains linear regression, classifies test data, computes test error on test set

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x_bias, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x_bias, test_y, theta)
    return test_error


# Print test error using linear_regression: Closed Form Solution
L = np.around(np.linspace(1e-2, 1, 20), decimals=2)
loop = tqdm(enumerate(L), total=len(L), leave=False)
print('Linear Regression test errors for different lambdas:')
errors = []
for index, l in loop:
    loop.set_description(f"Lambda{index + 1}={l}")
    e = run_linear_regression_on_MNIST(l)
    errors.append(e)
    loop.set_postfix({'error': e})
print(tabulate(np.array([L[:10], errors[:10], L[10:], errors[10:]]).T, headers=(
    'lambda', 'error', 'lambda', 'error')))


###############################################################################
# 3. Support Vector Machine (One vs. Rest and Multiclass)
###############################################################################
def run_svm_one_vs_rest_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    # Image class = 0 or 1 (if 1-9)
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return np.around(test_error, decimals=3)


print('\n\nSVM one vs. rest test error:', run_svm_one_vs_rest_on_MNIST())


def run_multiclass_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return np.around(test_error, decimals=3)


print('\n\nMulticlass SVM test error:', run_multiclass_svm_on_MNIST())


###############################################################################
# 4. Multinomial (Softmax) Regression and Gradient Descent
###############################################################################
def run_softmax_on_MNIST(temp_parameter=1):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(
        train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression.
    write_pickle_data(theta, "./theta.pkl.gz")
    return test_error


print('\n\nSoftmax test error=', run_softmax_on_MNIST())
T = [.5, 1.0, 2.0]
temp_parameter = tqdm(T, leave=False, total=3, desc="Temp Parameter loop")
errors = [run_softmax_on_MNIST(t) for t in temp_parameter]
print('\nSoftmax test errors for different temperature parameters:')
print(tabulate(np.array([T, errors]).T, headers=(
    'temp', 'error')))


###############################################################################
# 5. Changing Labels
###############################################################################
def run_softmax_on_MNIST_mod3(temp_parameter=1):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y, test_y = update_y(train_y, test_y)
    theta, cost_function_history = softmax_regression(
        train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error_mod3(test_x, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta_mod3.pkl.gz")
    return test_error


print('\n\nSoftMax % 3 test error=', run_softmax_on_MNIST())


###############################################################################
# 6. Classification Using Manually Crafted Non-Linear Features
###############################################################################

##############################################
# 6-A. Dimensionality reduction via PCA
##############################################

n_components = 18
pcs = principal_components(train_x)
train_pca = project_onto_PC(train_x, pcs, n_components)
test_pca = project_onto_PC(test_x, pcs, n_components)
# train_pca (and test_pca) is a representation of our training (and test) data
# after projecting each example onto the first 18 principal components.

theta, cost_function_history = softmax_regression(
    train_pca, train_y, temp_parameter=0.5, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
plot_cost_function_over_time(cost_function_history)
test_error = compute_test_error(test_pca, test_y, theta, temp_parameter=1)
# Save the model parameters theta obtained from calling softmax_regression to disk.
write_pickle_data(theta, "./theta_pca.pkl.gz")
print(f"\n\nSoftmax & PCA test error: {test_error}")

# Plot first 100 images of the dataset as projected onto PC1 and PC2
plot_PC(train_x[range(100), ], pcs, train_y[range(100)])

# Use the reconstruct_PC function in features.py to show
# the first and second MNIST images as reconstructed solely from
# their 18-dimensional principal component representation.
# Compare the reconstructed images with the originals.
firstimage_reconstructed = reconstruct_PC(
    train_pca[0, ], pcs, n_components, train_x)
plot_images(firstimage_reconstructed)
plot_images(train_x[0, ])

secondimage_reconstructed = reconstruct_PC(
    train_pca[1, ], pcs, n_components, train_x)
plot_images(secondimage_reconstructed)
plot_images(train_x[1, ])


##############################################
# 6-B. Cubic Kernel
##############################################

# 10-dimensional PCA representation of the training and test set
n_components = 10
pcs = principal_components(train_x)
train_pca10 = project_onto_PC(train_x, pcs, n_components)
test_pca10 = project_onto_PC(test_x, pcs, n_components)

# Apply cubic feature transformation on the 10-D PCA train and test sets
# train_cube (and test_cube) is a representation of our training (and test) data
# after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.

train_cube = cubic_features(train_pca10)
test_cube = cubic_features(test_pca10)

# Train the softmax regression model using (train_cube, train_y)
# and evaluate its accuracy on (test_cube, test_y).

theta, cost_function_history = softmax_regression(
    train_pca, train_y, temp_parameter=0.5, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
plot_cost_function_over_time(cost_function_history)
test_error = compute_test_error(test_pca, test_y, theta, temp_parameter=1)
# Save the model parameters theta obtained from calling softmax_regression to disk.
write_pickle_data(theta, "./theta_cubic.pkl.gz")
print(f"\n\nSoftmax on Cubic features test error: {test_error}")
