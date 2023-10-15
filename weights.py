"""
https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/#:~:text=Weight%20initialization%20is%20a%20procedure,of%20the%20neural%20network%20model.
https://www.deeplearning.ai/ai-notes/initialization/index.html

"""
import numpy as np

def he_normal(rows, cols):
    std_dev = np.sqrt(2.0 / rows)
    return std_dev * np.random.randn(rows, cols)

def he_uniform(rows, cols):
    limit = np.sqrt(6.0 / rows)
    return 2 * limit * np.random.rand(rows, cols) - limit

def xavier_normal(rows, cols):
    std_dev = np.sqrt(2.0 / (rows + cols))
    return std_dev * np.random.randn(rows, cols)

def xavier_uniform(rows, cols):
    limit = np.sqrt(6.0 / (rows + cols))
    return 2 * limit * np.random.rand(rows, cols) - limit

def normal(rows, cols):
    return np.random.randn(rows, cols)

def uniform(rows, cols):
    return np.random.rand(rows, cols)
