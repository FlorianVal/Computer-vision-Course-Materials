import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
from torchvision import transforms
from skimage import feature
import cv2
import imageio
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for better compatibility
from tqdm import tqdm

class VisionIllustrator:
    def __init__(self, save_path="Lesson_1/images/generated", base_path="Lesson_1/images"):
        self.save_path = save_path
        self.base_path = base_path
        self._setup()
    
    def _setup(self):
        """Set up the illustration environment (e.g., creating necessary directories)."""
        import os
        os.makedirs(self.save_path, exist_ok=True)
    
    def generate_all(self):
        """Call all defined illustration functions."""
        illustration_methods = [method for method in dir(self) if method.startswith("illustrate_")]
        for method in tqdm(illustration_methods, desc="Generating illustrations", 
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"):
            print(f"Generating: {method}")
            getattr(self, method)()

    def illustrate_data_collection(self):
        """Illustrate data collection sources."""
        fig, ax = plt.subplots(figsize=(8, 6))
        sources = ['Images', 'Videos', 'Sensors', 'Databases']
        x = [0.2, 0.8, 0.2, 0.8]  # Moved boxes further out horizontally
        y = [0.8, 0.8, 0.2, 0.2]  # Moved boxes further out vertically
        
        central_x, central_y = 0.5, 0.5
        
        # Draw central node
        ax.add_patch(plt.Circle((central_x, central_y), 0.15, fc='lightgreen', ec='black'))
        ax.text(central_x, central_y, 'Data\nCollection', ha='center', va='center')
        
        # Draw source nodes and connections
        for i, (source, xi, yi) in enumerate(zip(sources, x, y)):
            ax.add_patch(plt.Rectangle((xi-0.1, yi-0.1), 0.2, 0.2, fc='lightblue', ec='black'))
            ax.text(xi, yi, source, ha='center', va='center')
            
            # Calculate arrow start point at edge of rectangle
            dx = central_x - xi
            dy = central_y - yi
            angle = np.arctan2(dy, dx)
            start_x = xi + 0.1 * np.cos(angle)  # 0.1 is half the rectangle width
            start_y = yi + 0.1 * np.sin(angle)
            
            # Calculate arrow end point before the circle
            end_x = central_x - 0.15 * np.cos(angle)  # 0.15 is circle radius
            end_y = central_y - 0.15 * np.sin(angle)
            
            ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                    head_width=0.03, head_length=0.03, fc='black', ec='black')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(f"{self.save_path}/data_collection.png", bbox_inches='tight', dpi=300)
        plt.close()

    def illustrate_data_processing(self):
        """Illustrate data processing steps."""
        fig, ax = plt.subplots(figsize=(10, 2))
        
        steps = ['Raw Data', 'Cleaning', 'Augmentation', 'Feature\nExtraction', 'Processed\nData']
        x = np.linspace(0.1, 0.9, len(steps))
        y = [0.5] * len(steps)
        
        # Draw process flow
        for i in range(len(steps)-1):
            ax.arrow(x[i]+0.08, y[i], 0.09, 0, head_width=0.05, head_length=0.02, fc='black', ec='black')
        
        colors = ['lightgray', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink']
        for i, (step, xi, color) in enumerate(zip(steps, x, colors)):
            ax.add_patch(plt.Rectangle((xi-0.08, y[i]-0.15), 0.16, 0.3, fc=color, ec='black'))
            ax.text(xi, y[i], step, ha='center', va='center', fontsize=10)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0.2, 0.8)
        ax.axis('off')
        plt.savefig(f"{self.save_path}/data_processing.png", bbox_inches='tight', dpi=300)
        plt.close()

    def illustrate_data_processing_mnist(self):
        """Illustrate data processing steps with MNIST data."""
        # Load MNIST data
        mnist_data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        
        # Get a sample image and convert to proper format
        img = mnist_data.data[0].numpy()
        
        # Create augmentations
        # For rotation, we need to handle the image properly
        img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
        rotated = transforms.functional.rotate(img_tensor, 15).squeeze().numpy()
        
        # Add noise
        noisy = img + np.random.normal(0, 25, img.shape)
        noisy = np.clip(noisy, 0, 255)
        
        # Feature extraction (Canny edge detection)
        edges = feature.canny(img.astype(np.float32) / 255.0, sigma=2)
        
        # Create the figure
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        
        # Original image
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Rotated image
        axes[1].imshow(rotated, cmap='gray')
        axes[1].set_title('Rotation (15°)')
        axes[1].axis('off')
        
        # Noisy image
        axes[2].imshow(noisy, cmap='gray')
        axes[2].set_title('Added Noise')
        axes[2].axis('off')
        
        # Edge detection
        axes[3].imshow(edges, cmap='gray')
        axes[3].set_title('Edge Detection')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/data_processing_mnist.png", bbox_inches='tight', dpi=300)
        plt.close()

    def illustrate_deployment(self):
        """Illustrate model deployment process."""
        fig, ax = plt.subplots(figsize=(10, 3))
        
        components = ['Model\nValidation', 'Model\nPackaging', 'Deployment', 'Monitoring']
        x = np.linspace(0.2, 0.8, len(components))
        y = [0.5] * len(components)
        
        # Draw components and connections
        for i in range(len(components)-1):
            ax.arrow(x[i]+0.08, y[i], 0.09, 0, head_width=0.05, head_length=0.02, fc='black', ec='black')
        
        colors = ['lightgreen', 'lightblue', 'lightyellow', 'lightpink']
        for i, (component, xi, color) in enumerate(zip(components, x, colors)):
            ax.add_patch(plt.Rectangle((xi-0.08, y[i]-0.15), 0.16, 0.3, fc=color, ec='black'))
            ax.text(xi, y[i], component, ha='center', va='center', fontsize=10)
        
        # Add feedback loop
        ax.arrow(x[-1], y[-1]-0.2, -0.6, 0, head_width=0.05, head_length=0.02, 
                fc='red', ec='red', linestyle='--')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0.2, 0.8)
        ax.axis('off')
        plt.savefig(f"{self.save_path}/deployment.png", bbox_inches='tight', dpi=300)
        plt.close()

    def illustrate_rgb_pixel(self):
        """Illustrate a single RGB pixel with its values."""
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Example RGB values
        rgb = [120, 50, 240]  # Purple-ish color
        
        # Left subplot: RGB values as text
        ax1.text(0.5, 0.6, f"RGB = [{rgb[0]}, {rgb[1]}, {rgb[2]}]", 
                ha='center', va='center', fontsize=14)
        ax1.text(0.5, 0.4, f"R = {rgb[0]}\nG = {rgb[1]}\nB = {rgb[2]}", 
                ha='center', va='center', fontsize=12)
        ax1.axis('off')
        
        # Right subplot: Colored square
        color_array = np.array(rgb).reshape(1, 1, 3)
        ax2.imshow(np.tile(color_array, (50, 50, 1)))
        ax2.axis('off')
        
        plt.savefig(f"{self.save_path}/rgb_pixel.png", bbox_inches='tight', dpi=300)
        plt.close()

    def illustrate_two_pixels(self):
        """Illustrate a 1x2 image with two different colored pixels."""
        # Create two different colors
        pixel1 = [255, 0, 0]  # Red
        pixel2 = [0, 255, 0]  # Green
        
        # Create the image array
        img = np.array([pixel1, pixel2]).reshape(1, 2, 3)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Left: Matrix representation
        ax1.text(0.5, 0.5, 
                f"Image (1×2×3) =\n\n"
                f"[[[{pixel1[0]}, {pixel1[1]}, {pixel1[2]}],\n"
                f"  [{pixel2[0]}, {pixel2[1]}, {pixel2[2]}]]]",
                ha='center', va='center', fontsize=12, family='monospace')
        ax1.axis('off')
        
        # Right: Visual representation
        ax2.imshow(img)
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0])
        ax2.grid(True)
        ax2.set_title("Visual Representation (enlarged)", pad=10)
        
        plt.savefig(f"{self.save_path}/two_pixels.png", bbox_inches='tight', dpi=300)
        plt.close()

    def illustrate_mnist_matrix(self):
        """Illustrate a MNIST digit with its matrix representation."""
        # Load MNIST data
        mnist_data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        
        # Get a simple digit (e.g., a 5)
        digit = mnist_data.data[2].numpy()  # Choose an example that looks like a 5
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Left: Show the digit
        ax1.imshow(digit, cmap='gray')
        ax1.set_title(f"MNIST Digit\nShape: {digit.shape}", pad=10)
        ax1.axis('off')
        
        # Right: Show part of the matrix
        # Select a 5x5 region from the middle of the digit
        center_y, center_x = digit.shape[0]//2, digit.shape[1]//2
        subset = digit[center_y-2:center_y+3, center_x-2:center_x+3]
        matrix_text = "Full matrix shape: 28×28\n\nCenter 5×5 subset:\n\n"
        matrix_text += str(subset)
        
        ax2.text(0.5, 0.5, matrix_text, 
                ha='center', va='center', fontsize=10, family='monospace')
        ax2.axis('off')
        
        plt.savefig(f"{self.save_path}/mnist_matrix.png", bbox_inches='tight', dpi=300)
        plt.close()

    def illustrate_natural_image(self):
        """Illustrate a natural image with a zoomed portion and its matrix."""
        # Create a simple synthetic image (since we don't have access to real photos)
        img = np.zeros((100, 100, 3))
        for i in range(100):
            for j in range(100):
                img[i,j] = [(i+j)/200*255, i/100*255, j/100*255]
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # Left: Full image
        ax1.imshow(img.astype(np.uint8))
        ax1.set_title(f"Full Image\nShape: {img.shape}", pad=10)
        ax1.axis('off')
        
        # Middle: Zoomed portion
        zoom_size = 10
        start_y, start_x = 45, 45
        zoomed = img[start_y:start_y+zoom_size, start_x:start_x+zoom_size]
        ax2.imshow(zoomed.astype(np.uint8))
        ax2.set_title(f"Zoomed Region\nShape: {zoomed.shape}", pad=10)
        ax2.axis('off')
        
        # Right: Matrix values for a 3x3 portion
        tiny = zoomed[:3,:3]
        matrix_text = "RGB values for 3×3 region:\n\n"
        matrix_text += str(tiny.astype(np.uint8))
          
        ax3.text(0.5, 0.5, matrix_text,
                ha='center', va='center', fontsize=8, family='monospace')
        ax3.axis('off')
        
        plt.savefig(f"{self.save_path}/natural_image.png", bbox_inches='tight', dpi=300)
        plt.close()

    def illustrate_video_representation(self):
        """Illustrate video as a 4D tensor with time dimension."""
        # Create a simple animation-like sequence (3 frames)
        frames = []
        size = 50
        for t in range(3):
            frame = np.zeros((size, size, 3))
            # Create a moving gradient
            for i in range(size):
                for j in range(size):
                    frame[i,j] = [(i+j+t*10)/100*255, i/size*255, j/size*255]
            frames.append(frame)
        
        # Create figure with four subplots
        fig = plt.figure(figsize=(15, 4))
        gs = plt.GridSpec(1, 4, width_ratios=[1, 1, 1, 1.5])
        
        # Show three consecutive frames
        for t in range(3):
            ax = fig.add_subplot(gs[t])
            ax.imshow(frames[t].astype(np.uint8))
            ax.set_title(f"Frame {t+1}", pad=10)
            ax.axis('off')
        
        # Add text explanation
        ax = fig.add_subplot(gs[3])
        text = "Video = 4D Tensor\n\n"
        text += "Dimensions:\n"
        text += "1. Temps (nombre de frames)\n"
        text += "2. Hauteur\n"
        text += "3. Largeur\n"
        text += "4. Canaux (RGB)\n\n"
        text += "Exemple HD 30fps:\n"
        text += "→ 30 frames/sec\n"
        text += "→ 1920×1080 pixels\n"
        text += "→ 3 canaux\n"
        text += "= 186M valeurs/sec !"
        
        ax.text(0.5, 0.5, text,
                ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/video_representation.png", bbox_inches='tight', dpi=300)
        plt.close()

    def illustrate_color_thresholding(self):
        """Illustrate color thresholding for classification."""
        # Download a sample image from ImageNet (a red apple)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # Create a sample image with a red object
        img = np.zeros((224, 224, 3))
        # Add a red circular object
        center = (112, 112)
        for i in range(224):
            for j in range(224):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2)
                if dist < 80:
                    img[i,j] = [0.8, 0.2, 0.2]  # Red object
                else:
                    img[i,j] = [0.7, 0.7, 0.7]  # Gray background
        
        # Convert to tensor format
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        # Apply red color thresholding
        red_mask = (img_tensor[0] > 0.6) & (img_tensor[1] < 0.3) & (img_tensor[2] < 0.3)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Original image
        ax1.imshow(img)
        ax1.set_title("Image originale", pad=10)
        ax1.axis('off')
        
        # Thresholded image
        ax2.imshow(red_mask.numpy(), cmap='gray')
        ax2.set_title("Masque après seuillage\n(détection du rouge)", pad=10)
        ax2.axis('off')
        
        plt.savefig(f"{self.save_path}/color_thresholding.png", bbox_inches='tight', dpi=300)
        plt.close()

    def illustrate_morphological_operations(self):
        """Illustrate erosion and dilation for object detection."""
        # First get the video subtraction result
        video_path = f"{self.base_path}/MOT16-04-raw.webm"
        cap = cv2.VideoCapture(video_path)
        
        # Read two frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
        ret, frame2 = cap.read()
        
        # Convert to grayscale and get difference mask
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        _, binary_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Create structuring element (kernel)
        kernel = np.ones((5, 5), np.uint8)
        
        # Apply morphological operations
        dilated = cv2.dilate(binary_mask, kernel, iterations=1)  # First dilate
        eroded = cv2.erode(dilated, kernel, iterations=1)  # Then erode
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # Original binary mask
        ax1.imshow(binary_mask, cmap='gray')
        ax1.set_title("Masque binaire original", pad=10)
        ax1.axis('off')
        
        # Dilated image
        ax2.imshow(dilated, cmap='gray')
        ax2.set_title("Après dilatation", pad=10)
        ax2.axis('off')
        
        # Eroded image (after dilation)
        ax3.imshow(eroded, cmap='gray')
        ax3.set_title("Après dilatation + érosion", pad=10)
        ax3.axis('off')
         
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/morphological_operations.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        cap.release()

    def illustrate_image_subtraction(self):
        """Illustrate image subtraction for change detection."""
        # Create two similar images with a difference
        img1 = np.zeros((200, 200))
        img2 = np.zeros((200, 200))
        
        # Common elements in both images
        # Rectangle
        img1[50:100, 50:150] = 1
        img2[50:100, 50:150] = 1
        
        # Element only in second image (circle)
        center = (150, 100)
        for i in range(200):
            for j in range(200):
                if np.sqrt((i-center[0])**2 + (j-center[1])**2) < 30:
                    img2[i,j] = 1
        
        # Compute difference
        diff = np.abs(img2 - img1)
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # First image
        ax1.imshow(img1, cmap='gray')
        ax1.set_title("Image 1", pad=10)
        ax1.axis('off')
        
        # Second image
        ax2.imshow(img2, cmap='gray')
        ax2.set_title("Image 2", pad=10)
        ax2.axis('off')
        
        # Difference
        ax3.imshow(diff, cmap='gray')
        ax3.set_title("Différence", pad=10)
        ax3.axis('off')
        
        plt.savefig(f"{self.save_path}/image_subtraction.png", bbox_inches='tight', dpi=300)
        plt.close()

    def illustrate_real_video_subtraction(self):
        """Illustrate background subtraction with real video frames and its limitations."""
        # Read video
        video_path = f"{self.base_path}/MOT16-04-raw.webm"
        cap = cv2.VideoCapture(video_path)
        
        # Read two frames with some temporal distance
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # First frame
        ret, frame1 = cap.read()
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 5)  # 6th frame (some movement)
        ret, frame2 = cap.read()
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Simple difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Apply threshold to get binary image
        threshold = 30
        _, diff_thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Create figure with 2x2 layout
        fig = plt.figure(figsize=(10, 8))
        gs = plt.GridSpec(2, 2, figure=fig)
        
        # Original frames side by side on top
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        ax1.set_title("Image 1 (I₁)", pad=10)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
        ax2.set_title("Image 2 (I₂)", pad=10)
        ax2.axis('off')
        
        # Difference images on bottom
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(diff, cmap='gray')
        ax3.set_title("Différence brute D(x,y)", pad=10)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.imshow(diff_thresh, cmap='gray')
        ax4.set_title(f"Masque binaire M(x,y)\n(seuil = {threshold})", pad=10)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/real_video_subtraction.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        cap.release()

    def manual_conv2d(self, image, kernel):
        """Manually compute 2D convolution."""
        # Get dimensions
        i_height, i_width = image.shape
        k_height, k_width = kernel.shape
        
        # Calculate padding needed
        pad_h = k_height // 2
        pad_w = k_width // 2
        
        # Create output array
        output = np.zeros_like(image, dtype=float)
        
        # Pad the input image
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        
        # Perform convolution
        for i in range(i_height):
            for j in range(i_width):
                # Extract the region
                region = padded_image[i:i+k_height, j:j+k_width]
                # Compute convolution for this position
                output[i, j] = np.sum(region * kernel)
        
        return output

    def illustrate_convolution_operation(self):
        """Illustrate 2D convolution operation with a simple example."""
        # Create a simple 6x6 input
        input_matrix = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Define a 3x3 kernel
        kernel = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ], dtype=np.float32)
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # Show input matrix
        ax1.imshow(input_matrix, cmap='gray')
        ax1.set_title("Image d'entrée", pad=10)
        for i in range(input_matrix.shape[0]):
            for j in range(input_matrix.shape[1]):
                ax1.text(j, i, str(int(input_matrix[i, j])), ha='center', va='center')
        ax1.grid(True)
        
        # Show kernel
        ax2.imshow(kernel, cmap='RdBu')
        ax2.set_title("Kernel (filtre vertical)", pad=10)
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                ax2.text(j, i, str(int(kernel[i, j])), ha='center', va='center')
        ax2.grid(True)
        
        # Compute and show convolution result
        result = self.manual_conv2d(input_matrix, kernel)
        ax3.imshow(result, cmap='gray')
        ax3.set_title("Résultat de la convolution", pad=10)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                ax3.text(j, i, f"{result[i, j]:.1f}", ha='center', va='center')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/convolution_operation.png", bbox_inches='tight', dpi=300)
        plt.close()

    def illustrate_convolution_animation(self):
        """
        Create an animated GIF visualizing the convolution process on an MNIST digit.
        
        This function performs the following steps:
        1. Loads an MNIST digit from the dataset.
        2. Defines a Sobel vertical kernel.
        3. Iterates over the image with a sliding 3x3 window.
        4. For each window position, it computes the convolution sum and updates an output image.
        5. Generates frames showing:
            - The original image with a red rectangle indicating the current kernel position.
            - A zoomed-in view of the current 3x3 region with multiplication details.
            - An inset showing the accumulating convolution output.
        6. Saves each frame as a PNG and compiles them into an animated GIF.
        """
        # --- Imports ---
        import numpy as np
        import matplotlib.pyplot as plt
        import imageio
        import os
        import torchvision

        # --- Load MNIST Data ---
        mnist_data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        digit = mnist_data.data[0].numpy().astype(np.float32)

        # --- Define the Sobel Vertical Kernel ---
        kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)

        # --- Prepare for Animation ---
        output = np.zeros_like(digit)  # To store the convolution results
        frames = []
        # Define valid positions to apply a 3x3 kernel (avoiding borders)
        height, width = digit.shape
        positions = [(i, j) for i in range(1, height-1) for j in range(1, width-1)]

        # Create figure and subplots for original image and detail view
        fig, (ax_orig, ax_detail) = plt.subplots(1, 2, figsize=(12, 5))

        # --- Process Each Position ---
        # Use slicing to reduce the number of frames for a smoother GIF
        for pos in positions[::2]:
            i, j = pos

            # Clear previous drawings
            ax_orig.clear()
            ax_detail.clear()

            # Display the original image and highlight the current kernel region
            ax_orig.imshow(digit, cmap='gray')
            rect = plt.Rectangle((j - 1, i - 1), 3, 3, edgecolor='red', facecolor='none', linewidth=2)
            ax_orig.add_patch(rect)
            ax_orig.set_title("Kernel Position", pad=10)
            ax_orig.axis('off')

            # Extract the 3x3 region for convolution and compute the result
            region = digit[i-1:i+2, j-1:j+2]
            conv_value = np.sum(region * kernel)
            output[i, j] = conv_value  # Update output image

            # Show detailed computation for the current region
            ax_detail.imshow(region, cmap='gray')
            ax_detail.set_title(f"Local Region\nSum = {conv_value:.1f}", pad=10)
            for r in range(3):
                for c in range(3):
                    ax_detail.text(c, r, f"{region[r, c]:.0f}×{kernel[r, c]:.0f}",
                                ha='center', va='center', color='red', fontsize=10)
            ax_detail.grid(True)

            # Add an inset on the original image showing the accumulating output
            inset_ax = ax_orig.inset_axes([0.65, 0.65, 0.3, 0.3])
            inset_ax.imshow(np.abs(output), cmap='gray')
            inset_ax.set_title("Output", fontsize=8)
            inset_ax.axis('off')

            plt.tight_layout()
            fig.canvas.draw()

            # Capture the frame from the canvas
            frame = np.array(fig.canvas.buffer_rgba())[:, :, :3]
            frames.append(frame)

        plt.close(fig)

        # --- Save Frames and Create GIF ---
        # Create directory for saving individual frames
        gif_dir = os.path.join(self.save_path, "gif")
        os.makedirs(gif_dir, exist_ok=True)
        for idx, frame in enumerate(frames):
            filename = os.path.join(gif_dir, f"convolution_animation{idx:03d}.png")
            imageio.imwrite(filename, frame)

        # Save the final animated GIF
        gif_filename = os.path.join(self.save_path, "convolution_animation.gif")
        imageio.mimsave(gif_filename, frames, fps=10)

    def illustrate_conv_params(self):
        """
        Crée une figure illustrant la différence entre padding, stride et dilation
        dans une convolution. On utilise ici une grille (de taille 10x10) pour représenter
        une image, et on superpose sur cette grille la fenêtre de convolution (noyau de taille 3)
        avec :
        - Un ajout de bordures (padding) pour montrer la zone originale par rapport à la zone étendue.
        - Un déplacement de la fenêtre (stride) pour montrer les positions espacées de la convolution.
        - Un espacement des éléments du noyau (dilation) qui élargit le champ effectif.
        
        L'illustration est sauvegardée au format PNG sous le nom "conv_params_illustration.png".
        """
        grid_size = 10  # Taille de la grille d'entrée

        # Création de la figure avec 3 sous-figures
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # ============================
        # 1. Illustration du Padding
        # ============================
        ax = axes[0]
        ax.set_title("Padding\n(ajout de bordures)", fontsize=14)
        padding = 1  # On ajoute une bordure de 1 pixel
        # Pour la zone paddée, les axes vont de -padding à grid_size+padding
        ax.set_xlim(-padding, grid_size + padding)
        ax.set_ylim(grid_size + padding, -padding)
        ax.set_aspect('equal')
        # Tracer la grille complète (zone avec padding)
        for x in range(-padding, grid_size + padding + 1):
            ax.axvline(x, color='gray', linestyle='--', linewidth=1)
        for y in range(-padding, grid_size + padding + 1):
            ax.axhline(y, color='gray', linestyle='--', linewidth=1)
        # Encadrer la zone originale (sans padding)
        rect = plt.Rectangle((0, 0), grid_size, grid_size, edgecolor='red', facecolor='none', lw=2)
        ax.add_patch(rect)
        ax.text(grid_size/2, -0.5, "Zone originale", ha='center', color='red', fontsize=12)

        # ============================
        # 2. Illustration du Stride
        # ============================
        ax = axes[1]
        ax.set_title("Stride\n(déplacement de la fenêtre)", fontsize=14)
        ax.set_xlim(0, grid_size)
        ax.set_ylim(grid_size, 0)
        ax.set_aspect('equal')
        # Tracer la grille d'origine
        for x in range(grid_size + 1):
            ax.axvline(x, color='gray', linestyle='--', linewidth=1)
        for y in range(grid_size + 1):
            ax.axhline(y, color='gray', linestyle='--', linewidth=1)
        # Définir la taille du noyau et le stride
        kernel_size = 3
        stride = 4
        # Afficher toutes les positions possibles avec stride=2
        for i in range(0, grid_size - kernel_size + 1, stride):
            for j in range(0, grid_size - kernel_size + 1, stride):
                rect = plt.Rectangle((j, i), kernel_size, kernel_size,
                                    edgecolor='green', facecolor='none', lw=2)
                ax.add_patch(rect)
        ax.text(grid_size/2, grid_size + 0.5, "Fenêtres (stride = 2)", ha='center', color='green', fontsize=12)

        # ============================
        # 3. Illustration de la Dilation
        # ============================
        ax = axes[2]
        ax.set_title("Dilation\n(espacement du noyau)", fontsize=14)
        ax.set_xlim(0, grid_size)
        ax.set_ylim(grid_size, 0)
        ax.set_aspect('equal')
        # Tracer la grille d'origine
        for x in range(grid_size + 1):
            ax.axvline(x, color='gray', linestyle='--', linewidth=1)
        for y in range(grid_size + 1):
            ax.axhline(y, color='gray', linestyle='--', linewidth=1)
        # Paramètres pour le noyau dilaté
        dilation = 2
        kernel_size = 3
        # Choix d'un point de départ pour la fenêtre de convolution
        start_x, start_y = 2, 2
        # Afficher les éléments du noyau en tenant compte de la dilation
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = start_x + j * dilation
                y = start_y + i * dilation
                # Chaque élément est représenté par un petit carré
                rect = plt.Rectangle((x, y), 1, 1, edgecolor='blue', facecolor='none', lw=2)
                ax.add_patch(rect)
        # Tracer la frontière effective de la fenêtre dilatée
        effective_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        rect_eff = plt.Rectangle((start_x, start_y), effective_size, effective_size,
                                edgecolor='black', facecolor='none', linestyle='--', lw=2)
        ax.add_patch(rect_eff)
        ax.text(start_x + effective_size/2, start_y - 0.5, "Noyau dilaté (dilation = 2)",
                ha='center', color='black', fontsize=12)

        plt.tight_layout()
        # Sauvegarder la figure en tant que PNG
        plt.savefig(f"{self.save_path}/convolution_parameters.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def illustrate_convolution_filters(self):
        """Illustrate different convolution filters and their effects."""
        # Load MNIST data for a clean example
        mnist_data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        digit = mnist_data.data[0].numpy().astype(np.float32)
        
        # Define different kernels
        vertical_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        horizontal_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        blur_kernel = np.ones((3, 3), dtype=np.float32) / 9
        
        # Apply filters using manual convolution
        vertical = self.manual_conv2d(digit, vertical_kernel)
        horizontal = self.manual_conv2d(digit, horizontal_kernel)
        blur = self.manual_conv2d(digit, blur_kernel)
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original image
        ax1.imshow(digit, cmap='gray')
        ax1.set_title("Image originale", pad=10)
        ax1.axis('off')
        
        # Vertical edges
        ax2.imshow(np.abs(vertical), cmap='gray')
        ax2.set_title("Filtre vertical (Sobel)", pad=10)
        ax2.axis('off')
        
        # Horizontal edges
        ax3.imshow(np.abs(horizontal), cmap='gray')
        ax3.set_title("Filtre horizontal (Sobel)", pad=10)
        ax3.axis('off')
        
        # Blur
        ax4.imshow(blur, cmap='gray')
        ax4.set_title("Filtre de flou (moyenne)", pad=10)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/convolution_filters.png", bbox_inches='tight', dpi=300)
        plt.close()

    def illustrate_simple_cnn_vertical(self):
        """Illustrate a simple CNN learning to detect vertical lines."""
        # Create figure with multiple components
        fig = plt.figure(figsize=(15, 8))
        gs = plt.GridSpec(2, 3)

        # 1. Input image with horizontal line
        ax1 = fig.add_subplot(gs[0, 0])
        input_img = np.zeros((28, 28))
        input_img[13:15, :] = 1  # Horizontal line
        ax1.imshow(input_img, cmap='gray')
        ax1.set_title("Image d'entrée\n(ligne horizontale)", pad=10)
        ax1.axis('off')

        # 2. Learned filter visualization (keeping the same vertical filter)
        ax2 = fig.add_subplot(gs[0, 1])
        vertical_filter = np.array([
            [-1, 2, -1],
            [-1, 2, -1],
            [-1, 2, -1]
        ])
        ax2.imshow(vertical_filter, cmap='RdBu')
        ax2.set_title("Filtre appris", pad=10)
        for i in range(3):
            for j in range(3):
                ax2.text(j, i, f"{vertical_filter[i,j]:.0f}", 
                        ha='center', va='center', color='black')
        ax2.grid(True)

        # 3. Feature map after convolution
        ax3 = fig.add_subplot(gs[0, 2])
        feature_map = self.manual_conv2d(input_img, vertical_filter)
        # Use absolute value on feature map for better visualization
        feature_map_abs = np.abs(feature_map)
        # Normalize feature map for better visualization
        feature_map_normalized = (feature_map_abs - feature_map_abs.min()) / (feature_map_abs.max() - feature_map_abs.min() + 1e-8)
        ax3.imshow(feature_map_normalized, cmap='gray')
        ax3.set_title("Feature Map\n(après convolution)", pad=10)
        ax3.axis('off')

        # 4. Second input image with vertical line
        ax4 = fig.add_subplot(gs[1, 0])
        input_img2 = np.zeros((28, 28))
        input_img2[:, 13:15] = 1  # Vertical line
        ax4.imshow(input_img2, cmap='gray')
        ax4.set_title("Image d'entrée\n(ligne verticale)", pad=10)
        ax4.axis('off')

        # 5. Same filter visualization
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(vertical_filter, cmap='RdBu')
        ax5.set_title("Même filtre", pad=10)
        for i in range(3):
            for j in range(3):
                ax5.text(j, i, f"{vertical_filter[i,j]:.0f}", 
                        ha='center', va='center', color='black')
        ax5.grid(True)

        # 6. Feature map for vertical line
        ax6 = fig.add_subplot(gs[1, 2])
        feature_map2 = self.manual_conv2d(input_img2, vertical_filter)
        # Use absolute value on feature map for better visualization
        feature_map2_abs = np.abs(feature_map2)
        # Normalize feature map for better visualization
        feature_map2_normalized = (feature_map2_abs - feature_map2_abs.min()) / (feature_map2_abs.max() - feature_map2_abs.min() + 1e-8)
        ax6.imshow(feature_map2_normalized, cmap='gray')
        ax6.set_title("Feature Map\n(après convolution)", pad=10)
        ax6.axis('off')

        plt.tight_layout()
        plt.savefig(f"{self.save_path}/simple_cnn_vertical.png", bbox_inches='tight', dpi=300)
        plt.close()

    def draw_simple_cnn_architecture(self, ax):
        """Helper function to draw CNN architecture."""
        def add_box(x, y, w, h, label, color='lightblue'):
            rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, label, ha='center', va='center')

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        
        # Input image
        add_box(0.5, 1, 1.5, 2, "Image\n28x28")
        
        # Convolution layer
        add_box(3, 1, 1.5, 2, "Conv2D\n3x3")
        
        # Feature map
        add_box(5.5, 1, 1.5, 2, "Feature Map\n26x26")
        
        # Fully connected
        add_box(8, 1, 1.5, 2, "Dense\n(2)")

        # Add arrows
        ax.arrow(2, 2, 0.9, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(4.5, 2, 0.9, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(7, 2, 0.9, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

        ax.set_title("Architecture du Réseau", pad=10)
        ax.axis('off')


    def illustrate_resnet_block(self):
        """Illustrate a ResNet block with skip connection."""
        fig = plt.figure(figsize=(12, 8))
        
        # Main subplot for the ResNet block
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)

        def add_box(x, y, w, h, label, color='lightblue'):
            rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2, label, ha='center', va='center')

        # Input
        add_box(0.5, 2, 1.5, 2, "Input\nFeature Map")

        # First convolution
        add_box(3, 2, 1.5, 2, "Conv\n3x3")

        # Second convolution
        add_box(5.5, 2, 1.5, 2, "Conv\n3x3")

        # Output
        add_box(8, 2, 1.5, 2, "Output\nFeature Map")

        # Skip connection (curved arrow)
        ax.annotate("", xy=(8, 4), xytext=(2, 4),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3",
                                 color='red', lw=2))
        ax.text(5, 4.5, "Skip Connection", color='red', ha='center')

        # Add straight arrows
        ax.arrow(2, 3, 0.9, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(4.5, 3, 0.9, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(7, 3, 0.9, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

        # Add plus symbol at the merge point
        circle = plt.Circle((7.7, 3), 0.2, color='yellow', ec='black')
        ax.add_patch(circle)
        ax.text(7.7, 3, "+", ha='center', va='center', fontsize=15, weight='bold')

        ax.set_title("Bloc ResNet avec Skip Connection", pad=20)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"{self.save_path}/resnet_block.png", bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    illustrator = VisionIllustrator()
    illustrator.generate_all()
