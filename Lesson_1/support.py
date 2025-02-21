import os
import cv2
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

def download_mnist(save_dir='images/mnist'):
    os.makedirs(save_dir, exist_ok=True)
    # Download MNIST dataset (train split)
    transform = transforms.ToTensor()
    mnist_dataset = torchvision.datasets.MNIST(
        root='datasets/MNIST', train=True, download=True, transform=transform)
    
    # Save the first 10 images with their labels in the filename
    for i in range(10):
        img, label = mnist_dataset[i]
        # Convert tensor to a numpy array and scale to [0,255]
        img_np = (img.numpy().squeeze() * 255).astype(np.uint8)
        filename = os.path.join(save_dir, f'mnist_{i}_label_{label}.png')
        cv2.imwrite(filename, img_np)
        print(f"Saved {filename}")

def download_cifar10(save_dir='images/cifar10'):
    os.makedirs(save_dir, exist_ok=True)
    # Download CIFAR10 dataset (train split)
    transform = transforms.ToTensor()
    cifar_dataset = torchvision.datasets.CIFAR10(
        root='datasets/CIFAR10', train=True, download=True, transform=transform)
    
    # Save the first 10 images with their labels in the filename
    for i in range(10):
        img, label = cifar_dataset[i]
        # Convert tensor to numpy array and change channel order from CxHxW to HxWxC
        img_np = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        # OpenCV expects BGR order for color images
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        filename = os.path.join(save_dir, f'cifar10_{i}_label_{label}.png')
        cv2.imwrite(filename, img_bgr)
        print(f"Saved {filename}")

def illustrate_convolution(image_path, output_path):
    """
    Load an image, apply a Sobel filter (x-direction) to illustrate convolution,
    and save a side-by-side comparison of the original and filtered image.
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image not found:", image_path)
        return
    
    # Apply Sobel filter (detects horizontal edges)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    # Convert to absolute values and then to uint8
    sobelx = np.absolute(sobelx)
    sobelx = np.uint8(sobelx)
    
    # Plot the original and filtered images side-by-side using matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    axs[1].imshow(sobelx, cmap='gray')
    axs[1].set_title("Sobel Filter (x-direction)")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved convolution illustration to {output_path}")

def create_kernel_visualization(output_path):
    """
    Create and save an illustration of a 3x3 Sobel kernel (x-direction).
    """
    # Define a sample 3x3 Sobel kernel for x-direction edge detection
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    fig, ax = plt.subplots(figsize=(3, 3))
    cax = ax.matshow(kernel, cmap='coolwarm')
    for (i, j), val in np.ndenumerate(kernel):
        ax.text(j, i, f'{val}', ha='center', va='center', color='black')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(cax)
    plt.title("Sobel Kernel (X)")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved kernel visualization to {output_path}")

def main():
    # Create directories for output images
    os.makedirs('images', exist_ok=True)
    
    # Download datasets and save sample images
    download_mnist()
    download_cifar10()
    
    # Create a convolution illustration using one of the MNIST images
    mnist_example = 'images/mnist/mnist_0_label_5.png'
    convolution_output = 'images/convolution_illustration.png'
    illustrate_convolution(mnist_example, convolution_output)
    
    # Create a kernel visualization illustration
    kernel_output = 'images/kernel_visualization.png'
    create_kernel_visualization(kernel_output)

if __name__ == '__main__':
    main()