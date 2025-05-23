# Operation Final Front++

Arman Singhal - 240184

This repository contains Python scripts to predict and analyze a graph structure from an input image. It uses a PyTorch model (`model.pth`) to process image features and returns the minimum cost path based on the graph's adjacency matrix.

## Requirements

- Python 3.8+
- pip
- Virtualenv (recommended)

## Installation

### 1. Clone or Download the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
```

### 3. Install Python Dependencies

```bash
pip install torch torchvision opencv-python numpy pytesseract
```

### 4. Install Tesseract OCR

You need to install Tesseract OCR on your system and make sure it is in your `PATH`.

#### On macOS:

```bash
brew install tesseract
```

#### On Ubuntu/Linux:

```bash
sudo apt update
sudo apt install tesseract-ocr
```

You can verify installation by running:

```bash
tesseract --version
```

### 5. Place the Model File

Make sure `model.pth` is in the same directory as `Solution.py` and `pred.py`.

## Usage

To run the solver on an input image:

```bash
python Solution.py
```

You will be prompted to enter the image path:

```
Enter the path to the image:
```

Example:

```bash
Enter the path to the image: sample_graph.png
```

### Output

- If a valid path is found through the graph, it prints the **minimum cost**.
- If no valid path exists, it prints `-1`.

## File Descriptions

- `pred.py`: Extracts circles, lines, and arrowheads from an image, normalizes features, and predicts adjacency matrix using the PyTorch model.
- `Solution.py`: Runs the graph solver using the adjacency matrix to compute the minimum cost to reach the last node.
- `model.pth`: Pre-trained PyTorch model for predicting the graph's structure.

## Troubleshooting

- If `cv2.imread()` returns `None`, verify the image path is correct.
- If Tesseract OCR errors occur, check that it's installed and accessible from your terminal (`tesseract --version`).
