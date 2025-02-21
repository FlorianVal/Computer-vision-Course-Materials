import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2

# Create a directory for images if it doesn't exist
import os
os.makedirs('Lesson_1/images', exist_ok=True)

def plot_image_representation():
    # Create a simple 8x8 grayscale image
    img_gray = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])

    # Create RGB version
    img_rgb = np.zeros((8, 8, 3))
    img_rgb[:, :, 0] = img_gray  # Red channel
    img_rgb[:, :, 1] = 0        # Green channel
    img_rgb[:, :, 2] = 0        # Blue channel

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot grayscale image with values
    ax1.imshow(img_gray, cmap='gray')
    for i in range(8):
        for j in range(8):
            ax1.text(j, i, f'{img_gray[i,j]:.0f}', ha='center', va='center', color='red')
    ax1.set_title('Grayscale Image\nPixel Values')
    
    # Plot RGB channels separately
    ax2.imshow(img_rgb)
    ax2.set_title('RGB Image\n(Red Channel Only)')
    
    # Plot pixel grid for a small section
    #pixel_zoom = img_gray[2:5, 2:5]
    #ax3.imshow(pixel_zoom, cmap='gray', interpolation='nearest')
    #ax3.grid(True, which='major', color='white', linewidth=2)
    #ax3.set_title('Zoomed Pixel Grid\n3x3 Section')
    
    plt.tight_layout()
    plt.savefig('Lesson_1/images/image_representation.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_convolution():
    # Create a simple 6x6 image
    image = np.zeros((6, 6))
    image[2:4, 2:4] = 1  # White square in the middle

    # Define a 3x3 kernel
    kernel = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]]) / 9.0  # Mean filter

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    for i in range(6):
        for j in range(6):
            axes[0].text(j, i, f'{image[i,j]:.1f}', ha='center', va='center', color='red')

    # Kernel
    axes[1].imshow(kernel, cmap='gray')
    axes[1].set_title('3x3 Mean Filter Kernel')
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f'{kernel[i,j]:.2f}', ha='center', va='center', color='red')

    # Convolution result
    result = cv2.filter2D(image, -1, kernel)
    axes[2].imshow(result, cmap='gray')
    axes[2].set_title('Convolution Result')
    for i in range(6):
        for j in range(6):
            axes[2].text(j, i, f'{result[i,j]:.2f}', ha='center', va='center', color='red')

    plt.tight_layout()
    plt.savefig('Lesson_1/images/convolution_basic.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_different_kernels():
    # Create a sample image with some edges
    image = np.zeros((8, 8))
    image[2:6, 2:6] = 1

    # Define different kernels
    kernels = {
        'Mean Filter': np.ones((3, 3)) / 9.0,
        'Gaussian Filter': np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]]) / 16.0,
        'Horizontal Edge': np.array([[-1, -1, -1],
                                   [0, 0, 0],
                                   [1, 1, 1]]),
        'Vertical Edge': np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]])
    }

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')

    # Apply each kernel
    for idx, (name, kernel) in enumerate(kernels.items(), 1):
        result = cv2.filter2D(image, -1, kernel)
        axes[idx].imshow(result, cmap='gray')
        axes[idx].set_title(f'{name}\nResult')

    plt.tight_layout()
    plt.savefig('Lesson_1/images/different_kernels.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_convolution_animation():
    # Create a simple image and kernel
    image = np.zeros((6, 6))
    image[2:4, 2:4] = 1
    kernel = np.ones((3, 3)) / 9.0

    # Create visualization of kernel sliding
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    
    # Draw kernel position
    rect = plt.Rectangle((1, 1), 3, 3, fill=False, color='red', linewidth=2)
    ax.add_patch(rect)
    
    # Add arrows showing sliding direction
    ax.arrow(4, 1, 0.5, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
    ax.arrow(1, 4, 0, 0.5, head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    ax.set_title('Convolution: Kernel Sliding\nOver the Image')
    plt.savefig('Lesson_1/images/convolution_sliding.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all visualizations
plot_image_representation()
visualize_convolution()
visualize_different_kernels()
create_convolution_animation() 