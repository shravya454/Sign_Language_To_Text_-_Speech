import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

data_dir = 'data'
for lang in ['kannada', 'hindi', 'tamil']:
    lang_path = os.path.join(data_dir, lang)
    if os.path.exists(lang_path):
        classes = sorted(os.listdir(lang_path))
        total_imgs = 0
        class_info = []
        for c in classes:
            cp = os.path.join(lang_path, c)
            if os.path.isdir(cp):
                imgs = [f for f in os.listdir(cp) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_imgs += len(imgs)
                class_info.append((c, len(imgs)))
        print(f"Language: {lang}")
        print(f"  Total classes: {len(class_info)}")
        print(f"  Total images: {total_imgs}")
        print(f"  First 10 classes: {[c[0] for c in class_info[:10]]}")
        print("-" * 50)
