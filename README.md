# Computer Vision Course Materials

This repository contains materials for a comprehensive Computer Vision course, including lecture slides and associated code for generating illustrations.

## Repository Structure

The repository is organized by lessons, with each lesson containing:
- A `slides.tex` file for building presentation slides
- Supporting code and resources specific to that lesson

Example structure:
```
├── Lesson_1/
│   ├── slides.tex               # Lecture slides in LaTeX
│   ├── generate_illustrations.py # Script to generate images for slides
│   └── images/                  # Directory for images used in slides
├── Lesson_2/
│   ├── slides.tex
│   └── ...
└── ...
```

## Setup Instructions

### 1. Install Requirements

Before using the materials, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Generate Illustrations

For each lesson, you need to run the `generate_illustrations.py` script to create the necessary images for the slides:

```bash
python Lesson_1/generate_illustrations.py
```

This script will generate all the required illustrations in the appropriate directories for use in the slides.

### 3. Build LaTeX Slides

After generating the illustrations, you can build the slides using a LaTeX compiler:

```bash
cd Lesson_1
pdflatex slides.tex
```

This will create a PDF file containing the presentation slides for the lesson.

## Automated PDF Builds

This repository uses GitHub Actions to automatically build the PDF slides whenever changes are pushed to the repository. The workflow:

1. Runs when changes are made to any `slides.tex` or `generate_illustrations.py` file
2. Generates all necessary illustrations
3. Compiles the LaTeX slides into PDFs
4. Creates a new release with the compiled PDFs

You can find the latest PDF versions of all slides in the [Releases](../../releases) section of this repository, without needing to build them locally.

## Course Content

The course covers various topics in computer vision, including:

- Fundamentals of image processing
- Classical computer vision methods
- Convolution operations and filters
- Convolutional Neural Networks (CNNs)
- Advanced architectures like ResNet
- Practical applications of computer vision

## Requirements

The code requires Python 3.6+ and several packages including:
- numpy
- matplotlib
- torch/torchvision
- OpenCV
- scikit-image

A complete list of dependencies can be found in the `requirements.txt` file.

## Notes

- Make sure to run `generate_illustrations.py` before attempting to build the slides, as the LaTeX files depend on the generated images.
- Some lessons may have additional requirements or setup steps; check the specific lesson directories for details. 