# Model Training Pipeline

This section describes the end-to-end training pipeline used to generate `model.pth` — a PyTorch model trained to predict the adjacency matrix of a graph from image features (nodes, edges, arrowheads).

---

## 📊 Dataset Representation

Each input image is processed to extract geometric features:

- **Nodes (Circles)**: up to 7 circles, each with:
  - Coordinates $\(x, y\)$
  - Radius \(r\)
  - Node number (OCR-processed digit)

- **Edges (Lines)**: up to 42 line segments, each with endpoints $\((x_1, y_1), (x_2, y_2)\)$

- **Arrowheads (Directionality)**: up to 32 centroids of small circles detected near lines.

Each image is converted into a normalized feature tensor:

$$\[
X \in \mathbb{R}^{81 \times 4}
\]$$

Where:

- Rows 0–6: Node features $\([x, y, r, \text{number}]\)$
- Rows 7–48: Edge features $\([x_1, y_1, x_2, y_2]\)$
- Rows 49–80: Arrowhead centroids padded to shape $\([x, y, -1, -1]\)$

All coordinates are normalized by image dimensions, and padded entries use the value \(-1\).

---

## 🎯 Target Output

For each image, the target is a **flattened 7×7 adjacency matrix** $\( A \in \{0, 1\}^{49} \)$, indicating connections between nodes.

---

## 🧠 Model Architecture

The model is a simple fully connected neural network:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Flatten(start_dim=1),
    nn.Linear(64 * 81, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 49)
)
```

---

## ⚙️ Training Configuration

- **Loss Function**: Binary Cross-Entropy (BCE) with logits  
- **Optimizer**: Adam  
- **Learning Rate**: $\(1 \times 10^{-3}\)$
- **Epochs**: 200
- **Batch Size**: 32

---

## 🏋️‍♀️ Training Loop

```python
from torch.utils.data import Subset
import numpy as np

for epoch in range(200):
    indices = np.random.permutation(len(X_tensor))
    val_size = int(0.2 * len(X_tensor))
    
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_loader = DataLoader(Subset(TensorDataset(X_tensor, Y_tensor), train_idx), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(TensorDataset(X_tensor, Y_tensor), val_idx), batch_size=32)

    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_outputs = model(val_x)
            v_loss = criterion(val_outputs, val_y)
            val_loss += v_loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader):.4f}, "
          f"Val Loss: {val_loss / len(val_loader):.4f}")
```

---

## 💾 Saving the Model

After training:

```python
torch.save(model, "model.pth")
```

This file is used in `pred.py` to make predictions on unseen images.

---

## 🧪 Inference Pipeline

```python
model = torch.load('model.pth')
X_tensor = preprocess(img)  # shape (1, 81, 4)
output = model(X_tensor)
n = len(allCirclesarr)
final_pred = (torch.sigmoid(output) > 0.5).int()
adj = final_pred[:n*n].reshape(n, n)
```

---
