import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Download MNIST dataset
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)

# Select a few random samples
plt.figure(figsize=(15, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    idx = np.random.randint(0, len(mnist_train))
    img, label = mnist_train[idx]
    plt.imshow(img, cmap='gray')
    plt.title(f'Digit: {label}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('Lesson_1/images/mnist_samples.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a visualization of LeNet-5 architecture
def plot_lenet():
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    
    # Input
    plt.text(0.5, 2, 'Input\n32x32', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
    
    # Conv1
    plt.text(2, 2, 'Conv1\n6@28x28', ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black'))
    
    # Pool1
    plt.text(3.5, 2, 'Pool1\n6@14x14', ha='center', va='center', bbox=dict(facecolor='lightgreen', edgecolor='black'))
    
    # Conv2
    plt.text(5, 2, 'Conv2\n16@10x10', ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black'))
    
    # Pool2
    plt.text(6.5, 2, 'Pool2\n16@5x5', ha='center', va='center', bbox=dict(facecolor='lightgreen', edgecolor='black'))
    
    # FC1
    plt.text(8, 2, 'FC\n120', ha='center', va='center', bbox=dict(facecolor='orange', edgecolor='black'))
    
    # Output
    plt.text(9.5, 2, 'Output\n10', ha='center', va='center', bbox=dict(facecolor='yellow', edgecolor='black'))
    
    # Add arrows
    for x in [1.2, 2.8, 4.2, 5.8, 7.2, 8.8]:
        ax.arrow(x, 2, 0.3, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    plt.axis('off')
    plt.savefig('Lesson_1/images/lenet.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_lenet()

# Create a timeline visualization
def plot_cv_timeline():
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlim(1985, 2025)
    ax.set_ylim(0, 4)
    
    events = [
        (1989, "LeNet - First CNN\nYann LeCun"),
        (1998, "MNIST Dataset\nPostal Service"),
        (2012, "AlexNet\nDeep Learning Revolution"),
        (2015, "ResNet\nDeeper Networks"),
        (2020, "Transformers in CV")
    ]
    
    for year, event in events:
        plt.plot([year, year], [1, 2], 'k-', linewidth=2)
        plt.text(year, 2.2, event, ha='center', va='bottom', rotation=0)
    
    plt.axhline(y=1.5, color='k', linestyle='-', linewidth=2)
    plt.title('Timeline of Computer Vision Evolution', pad=20)
    plt.axis('off')
    plt.savefig('Lesson_1/images/timeline_cv.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_cv_timeline()

# Create a visualization of postal recognition system
def plot_postal_recognition():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    
    # Draw envelope
    envelope = plt.Rectangle((1, 1), 4, 2, facecolor='white', edgecolor='black')
    ax.add_patch(envelope)
    plt.text(3, 2, 'ZIP Code\n12345', ha='center', va='center')
    
    # Draw arrow
    ax.arrow(5.5, 2, 1, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    # Draw recognition system
    system = plt.Rectangle((7, 1), 2, 2, facecolor='lightblue', edgecolor='black')
    ax.add_patch(system)
    plt.text(8, 2, 'LeNet\nRecognition\nSystem', ha='center', va='center')
    
    plt.title('Postal Service Recognition System')
    plt.axis('off')
    plt.savefig('Lesson_1/images/postal_recognition.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_postal_recognition() 