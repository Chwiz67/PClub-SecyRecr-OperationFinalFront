import torch
import cv2
import numpy as np

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def pred(img_path):

    model = torch.load('model.pth', weights_only=False)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img)
    img = img.astype(np.uint8)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=20, maxRadius=30)

    blurred = cv2.GaussianBlur(img, (9, 9), 50)
    heads = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=5, minRadius=2, maxRadius=5)

    edges = cv2.Canny(img,20, 80)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=10, maxLineGap=10)

    allCircles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            x, y, r = int(x), int(y), int(r)
            x1 = max(0, x - r)
            x2 = min(img.shape[1], x + r)
            y1 = max(0, y - r)
            y2 = min(img.shape[0], y + r)
            roi = img[y1:y2, x1:x2]
            _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            config = '--psm 10 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(thresh, config=config)
            number = ''.join(filter(str.isdigit, text))
            if number:
                allCircles.append([x, y, r, int(number[0])])
            else:
                allCircles.append([x, y, r, -1])
    else:
        print("No circles found")

    allHeads = []
    if heads is not None:
        heads = np.uint16(np.around(heads))
        for (x, y, r) in heads[0, :]:
            allHeads.append([int(x), int(y)])
    else:
        print("No heads found")

    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        keep = True
        for f_line in filtered_lines:
            fx1, fy1, fx2, fy2 = f_line[0]
            dist_start = ((x1 - fx1)**2 + (y1 - fy1)**2)**0.5
            dist_end = ((x2 - fx2)**2 + (y2 - fy2)**2)**0.5
            if dist_start < 10 or dist_end < 10:
                keep = False
                break
        if keep:
            filtered_lines.append(line)
    allLines = []
    if filtered_lines is not None:
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            allLines.append([int(x1), int(y1), int(x2), int(y2)])
    else:
        print("No lines found")

    n = len(allCircles)
    for i in range(n):
        if(allCircles[i][3] == -1):
            s = 0
            for j in range(n):
                if(allCircles[j][3] != -1):
                    s += allCircles[j][3]
            allCircles[i][3] = n*(n-1)/2 - s

    for _ in range(7 - len(allCircles)):
        allCircles.append([-757, -757, -757, -757])
    for _ in range(42 - len(allLines)):
        allLines.append([-757, -757, -757, -757])
    for _ in range(32 - len(allHeads)):
        allHeads.append([-757, -757])

    allCirclesarr = np.array(allCircles, dtype=np.float32)
    allLinesarr = np.array(allLines, dtype=np.float32)
    allHeadsarr = np.array(allHeads, dtype=np.float32)

    for j in range(7):
        allCirclesarr[j][0] = allCirclesarr[j][0] / 757
        allCircles[j][1] = allCircles[j][1] / 757
        allCircles[j][2] = allCircles[j][2] / 727
    for j in range(42):
        allLinesarr[j] = allLinesarr[j] / 757
    for j in range(32):
        allHeadsarr[j] = allHeadsarr[j] / 757

    X = np.concatenate((allCirclesarr, allLinesarr, np.pad(allHeadsarr, ((0, 0), (0, 2)), constant_values=-1)), axis=0)
    X_tensor = torch.tensor(np.expand_dims(X, axis=0), dtype=torch.float32)

    with torch.no_grad():
        preds = model(X_tensor)
        preds_binary = (torch.sigmoid(preds) > 0.5).int()
        preds_binary = preds_binary.numpy()
        final_pred = preds_binary[0][:n*n]
        final_pred = final_pred.reshape(n, n)

    return (n, final_pred.tolist())

if __name__ == "__main__":
    img_path = input("Enter the path to the image: ")
    predictions = pred(img_path)
    print(predictions)