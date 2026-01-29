# Introduction to Deep Learning & Neural Networks with Keras

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-D00000.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📖 Course Overview

This course introduces you to deep learning and neural networks using Keras. You'll learn how neural networks work, how to train them effectively, and how to apply them to real-world problems in computer vision and beyond. The course covers everything from basic artificial neural networks to advanced architectures like CNNs and Transformers.

---

## 📚 Table of Contents

- [Module 1: Neural Network Fundamentals](#module-1-neural-network-fundamentals)
- [Module 2: Training Neural Networks](#module-2-training-neural-networks)
- [Module 3: Building Models with Keras](#module-3-building-models-with-keras)
- [Module 4: Advanced Architectures](#module-4-advanced-architectures)
- [Module 5: Final Project](#module-5-final-project)
- [Key Takeaways](#key-takeaways)

---

## Module 1: Neural Network Fundamentals

### 📊 Labs

1. **Artificial Neural Networks**  
   📂 [Artificial Neural Networks.ipynb](module%201/Artificial%20Neural%20Networks.ipynb)

### 🎯 Key Concepts

- **Deep Learning Applications**: Color restoration, speech synthesis, handwriting generation
- **Biological Inspiration**: Neural networks inspired by brain function

### 📝 Technical Highlights

**Neuron Anatomy:**
- **Soma**: Main body of the neuron
- **Dendrites**: Receive electrical impulses from other neurons
- **Axon**: Carries processed information to synapses
- **Synapses**: Connect to other neurons (output connections)

**Artificial Neural Networks:**
- **Input Layer**: Feeds data into the network
- **Hidden Layers**: Process information between input and output
- **Output Layer**: Provides network predictions
- **Forward Propagation**: Data flows from input to output through layers
- **Learning Process**: Repeatedly activating and reinforcing neural connections

---

## Module 2: Training Neural Networks

### 📊 Labs

1. **Activation Functions and Vanishing Gradients**  
   📂 [DL0101EN-2-1-Activation_functions_and_Vanishing-py-v1 0__1__.ipynb](module%202/DL0101EN-2-1-Activation_functions_and_Vanishing-py-v1%200__1__.ipynb)

2. **Backpropagation**  
   📂 [DL0101EN-2-1-Backpropagation-py-v1 0.ipynb](module%202/DL0101EN-2-1-Backpropagation-py-v1%200.ipynb)

### 🎯 Key Concepts

- **Gradient Descent**: Iterative optimization algorithm for finding function minima
- **Learning Rate**: Critical hyperparameter balancing convergence speed and stability

### 📝 Technical Highlights

**Training Process (Iterative Loop):**
1. Calculate network output using forward propagation
2. Calculate error between ground truth and predictions
3. Update weights and biases through backpropagation
4. Repeat until convergence or max epochs reached

**Common Challenges:**
- **Large Learning Rate**: May miss minimum with overshooting steps
- **Small Learning Rate**: Slow convergence, requires many iterations
- **Vanishing Gradient Problem**: Gradients become too small in earlier layers
  - Caused by sigmoid activation function issues
  - Earlier layers learn much slower than later layers
  - Compromises prediction accuracy and increases training time

**Activation Functions:**
- **Sigmoid**: Traditional but prone to vanishing gradients
- **Hyperbolic Tangent (tanh)**: Scaled sigmoid, symmetric over origin
- **ReLU**: Most widely used today, doesn't activate all neurons simultaneously
- **Softmax**: Ideal for output layer in classification tasks (provides probabilities)

---

## Module 3: Building Models with Keras

### 📊 Labs

1. **Regression with Keras**  
   📂 [DL0101EN-3-1-Regression-with-Keras-py-v1 0__2_.ipynb](module%203/DL0101EN-3-1-Regression-with-Keras-py-v1%200__2_.ipynb)

2. **Classification with Keras**  
   📂 [DL0101EN-3-2-Classification-with-Keras-py-v1 0__1.ipynb](module%203/DL0101EN-3-2-Classification-with-Keras-py-v1%200__1.ipynb)

### 🎯 Key Concepts

- **Popular DL Libraries**: TensorFlow, PyTorch, and Keras
- **Keras Advantages**: Ease of use, syntactic simplicity, fast development

### 📝 Technical Highlights

**Library Comparison:**
- **TensorFlow**: Production-focused, large community, steeper learning curve
- **PyTorch**: Academic research preference, GPU support, Torch framework based
- **Keras**: High-level API, runs on top of TensorFlow, easy to learn

**Keras Workflow:**
1. Prepare and format data properly
2. Transform target column (use `to_categorical` for classification)
3. Build neural network with few lines of code
4. Train and evaluate model
5. Make predictions

**Key Features:**
- Build complex networks with minimal code
- Clear, intuitive API design
- Seamless integration with TensorFlow backend
- Excellent for rapid prototyping

---

## Module 4: Advanced Architectures

### 📊 Labs

1. **Convolutional Neural Networks with Keras**  
   📂 [DL0101EN_4_1_Convolutional_Neural_Networks_with_Keras_py_v1.ipynb](module%204/DL0101EN_4_1_Convolutional_Neural_Networks_with_Keras_py_v1.ipynb)

2. **Transformers with Keras**  
   📂 [DL0101EN-4-1-Transformers-with-Keras-py-v1.ipynb](module%204/DL0101EN-4-1-Transformers-with-Keras-py-v1.ipynb)

### 🎯 Key Concepts

- **Shallow vs. Deep Networks**: One hidden layer vs. many layers
- **Deep Learning Boom**: Driven by algorithmic advances, data availability, and computational power

### 📝 Technical Highlights

**Convolutional Neural Networks (CNNs):**
- **Input Format**: 
  - Grayscale: (n × m × 1)
  - Color: (n × m × 3)
- **Architecture Components**:
  - **Convolutional Layer**: Applies filters to detect features
  - **ReLU**: Filters output, keeps only positive values
  - **Pooling Layer**: Reduces spatial dimensions (Max pooling, Average pooling)
  - **Fully Connected Layer**: Flattens and connects all nodes
- **Applications**: Image recognition, object detection, computer vision

**Recurrent Neural Networks (RNNs):**
- Take previous output as input (sequence modeling)
- **Applications**: Text, genomes, handwriting, stock markets
- **LSTM**: Popular RNN variant for long-term dependencies
- **Use Cases**: Image generation, handwriting generation, image captioning, video descriptions

**Autoencoders:**
- Data compression with learned encoding/decoding
- Data-specific compression
- **Applications**: Data denoising, dimensionality reduction, visualization
- **Restricted Boltzmann Machines**: Popular autoencoder type
- **Use Cases**: Fixing imbalanced datasets, estimating missing values, automatic feature extraction

---

## Module 5: Final Project

### 📊 Project

**Final Project: Classification and Captioning**  
📂 [Final_Project_Classification_and_Captioning_v1.ipynb](module%205/Final_Project_Classification_and_Captioning_v1.ipynb)

### 🎯 Project Objectives

Combine computer vision with natural language processing for comprehensive AI system:
- Aircraft damage assessment and classification
- Image captioning and summarization
- Multi-modal AI integration

### 📝 Technical Components

- **Feature Extraction**: Pre-trained VGG16 for transfer learning
- **Binary Classification**: Damage detection (dent vs. crack)
- **BLIP Model**: Bootstrapping Language-Image Pretraining for captioning
- **Custom Keras Layers**: Integrate external pre-trained models
- **Data Preprocessing**: Image preprocessing and augmentation techniques
- **Model Evaluation**: Metrics and visualization for performance assessment

---

## 🎓 Key Takeaways

### Neural Network Fundamentals
- Understand biological inspiration behind artificial neural networks
- Master forward propagation concepts
- Learn how neural networks process and learn from data

### Training Techniques
- Implement gradient descent and backpropagation
- Choose appropriate activation functions for different layers
- Overcome vanishing gradient problems
- Optimize learning rates for efficient training

### Keras Mastery
- Build neural networks quickly with high-level API
- Implement regression and classification models
- Prepare and transform data for deep learning
- Leverage pre-built layers and optimizers

### Advanced Architectures
- Apply CNNs to image recognition tasks
- Understand when to use RNNs for sequential data
- Implement autoencoders for dimensionality reduction
- Integrate transfer learning with pre-trained models
- Build multi-modal AI systems combining vision and language

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib tensorflow keras opencv-python pillow
```

### Running the Labs
```bash
cd "2 - Introduction to Deep Learning & Neural Networks with Keras"
jupyter notebook
```

---

## 📈 Next Steps

After completing this course, proceed to:
- **[Deep Learning with Keras and TensorFlow](../3%20-%20Deep%20Learning%20with%20Keras%20and%20Tensorflow/README.md)**

---

**Happy Learning!** 🎓✨
