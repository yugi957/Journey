# cuDNN clone by Atul Krishnadas (ATUL-NN)

ATUL-NN is a high-performance neural network library developed in C/C++/CUDA. I built this library to further enhance my understanding of neural network algorithms, and how we uilize GPUs to make deep learning applicable in the real world. ATUL-NN enables users to instantiate fully connected neural networks and train them using abstracted classes. The project includes parallelized forward and backward propagation, convolution operations, and mini-batch training over the GPU from scratch.

## Features

- **High-Performance Neural Networks**: Leverage CUDA to accelerate neural network training and inference.
- **User-Friendly Interface**: Interface similar to PyTorch/TensorFlow for ease of use.
- **Fully Connected Layers**: Easily instantiate and train fully connected neural networks.
- **Parallelized Operations**: Forward and backward propagation, convolution, and mini-batch training are fully parallelized over the GPU.
- **No External Libraries**: Implemented from scratch without relying on external deep learning or GPU libraries (cuBLAS,etc).


## Getting Started
I'm still working to clean up code and flesh out the library, and later on I'd like to provide more documentation on proper usage. But for now, I'd recommend taking a look at the codebase, and I can provide a live demo if you contact me via the email associated with my github!

### Prerequisites