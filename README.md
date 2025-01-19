# Optimized CNN Model Project

This project implements a Convolutional Neural Network (CNN) for multi-class classification tasks. The project includes an optimized CNN model defined in Python and a Jupyter Notebook for experimentation, analysis, and visualization.

## Files

- **`HW1_319044434_314779166.ipynb`**: A Jupyter Notebook containing the following sections:
  1. **Train a Fully Connected Network on MNIST**: Demonstrates training a simple fully connected neural network on the MNIST dataset.
  2. **Implement and Train a CNN**: Implements and trains a convolutional neural network for multi-class classification.
  3. **Analyzing a Pre-trained CNN**: Explores and analyzes the performance of a pre-trained CNN model.
  - Includes numerous visualizations, such as accuracy and loss curves, confusion matrices, and feature maps.
- **`HW1_319044434_314779166.py`**: A Python script defining the CNN architecture as a PyTorch class.

## CNN Architecture

The CNN model includes the following features:

1. **Layer Composition**:
   - 5 convolutional layers with reduced filters to minimize computational cost.
   - Batch normalization to stabilize training.
   - ReLU activation for non-linearity.
   - Max-pooling for spatial down-sampling.
   - Global average pooling in the final convolutional layer.
2. **Fully Connected Layer**:
   - A dense layer with 64 units, followed by a dropout layer to reduce overfitting.
   - Final output layer for classification into 4 classes.

## How to Use

### Prerequisites

- Python 3.8 or above
- PyTorch 1.12 or above
- Jupyter Notebook

### Installation

1. Clone the repository or download the files.
2. Install the required dependencies:
   ```bash
   pip install torch torchvision notebook
   ```

### Running the Code

1. **Run the Jupyter Notebook**:
   Open `HW1_319044434_314779166.ipynb` to:
   - Train and evaluate models across different architectures and datasets.
   - Visualize metrics, predictions, and feature maps.

2. **Use the Python Script**:
   Import the `OptimizedCNN` class from `HW1_319044434_314779166.py` in your project:
   ```python
   from HW1_319044434_314779166 import OptimizedCNN

   model = OptimizedCNN(num_classes=4)
   print(model)
   ```

### Model Input

- **Input Shape**: The model expects input tensors of shape `(batch_size, 3, 128, 128)`.
  - 3 channels for RGB images.
  - Resized to 128x128 pixels.

### Training Details

- **Loss Function**: Cross-entropy loss for multi-class classification.
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Epochs**: Configurable in the notebook.

## Results

- Accuracy and loss metrics are logged during training and plotted for analysis.
- Visualizations include confusion matrices and feature maps from different layers.
