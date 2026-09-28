import os
import cv2
import numpy as np

def check_folder(path, name):
    print(f"Checking {name} at {path}:")
    classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print(f"Found {len(classes)} class folders.")
    for c in classes[:3]:
        cp = os.path.join(path, c)
        files = os.listdir(cp)
        print(f"  Class '{c}': {len(files)} files. Sample files: {files[:3]}")
        if files:
            img_path = os.path.join(cp, files[0])
            img = cv2.imread(img_path)
            if img is not None:
                print(f"    Sample image shape: {img.shape}, min: {img.min()}, max: {img.max()}, mean: {img.mean():.2f}")
            else:
                print(f"    Failed to read image: {img_path}")

with open('inspect_out2.txt', 'w', encoding='utf-8') as f:
    import sys
    sys.stdout = f
    check_folder('data/kannada', 'Kannada')
    check_folder('data/hindi', 'Hindi')
    check_folder('data/tamil', 'Tamil')
    sys.stdout = sys.__stdout__

print("Inspection 2 complete. Wrote inspect_out2.txt")
