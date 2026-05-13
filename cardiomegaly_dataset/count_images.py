import os

base = "c:/Users/94775/Desktop/Component_1/cardio_image_384"
for split in ["train", "val", "test"]:
    for cls in ["positive", "negative"]:
        p = os.path.join(base, split, cls)
        if os.path.isdir(p):
            files = os.listdir(p)
            print(f"{split}/{cls}: {len(files)} images")
            if files:
                print(f"  Sample file: {files[0]}")
