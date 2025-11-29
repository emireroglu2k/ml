import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ------------------------------------------------------
# 1. TRANSFORMS (Advanced Augmentation for Training)
# ------------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),

    # strong + safe augmentations
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
        shear=3
    ),

    # color augmentation
    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25,
        hue=0.05
    ),

    transforms.ToTensor(),

    # random erasing (helps avoid dependence on small features)
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),

    # normalization for pretrained models
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ------------------------------------------------------
# 2. VALIDATION & TEST TRANSFORMS
# ------------------------------------------------------

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ------------------------------------------------------
# 3. LOAD DATASETS (Already Split)
# ------------------------------------------------------

base_dir = r"C:\Users\user\Desktop\ml\dataset"   # <-- CHANGE THIS

train_dataset = datasets.ImageFolder(os.path.join(base_dir, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(base_dir, "validation"), transform=eval_transform)
test_dataset  = datasets.ImageFolder(os.path.join(base_dir, "test"), transform=eval_transform)

print("Classes:", train_dataset.classes)
print("Number of breeds:", len(train_dataset.classes))
print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

# ------------------------------------------------------
# 4. DATALOADERS
# ------------------------------------------------------

batch_size = 32
num_workers = 4  # adjust based on CPU power

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True
)

val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True
)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True
)

# ------------------------------------------------------
# 5. QUICK CHECK (Optional)
# ------------------------------------------------------

if __name__ == "__main__":
    import torch
    from torchvision.utils import save_image
    from PIL import Image

    out_dir = r"C:\Users\user\Desktop\ml\preprocessed_train"
    os.makedirs(out_dir, exist_ok=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    total = len(train_dataset.samples)
    print(f"Processing {total} train images -> saving 3 preprocessed + 1 original each to {out_dir}")

    for idx, (img_path, label) in enumerate(train_dataset.samples, 1):
        class_name = train_dataset.classes[label]
        class_dir = os.path.join(out_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # save original (converted to RGB and saved as PNG)
        pil_img = Image.open(img_path).convert("RGB")
        orig_save_path = os.path.join(class_dir, f"{base_name}_original.png")
        pil_img.save(orig_save_path)

        # save 3 preprocessed variants using the training transform (randomized)
        for v in range(1, 4):
            tensor_img = train_transform(pil_img)  # returns normalized tensor
            # denormalize for saving
            img_denorm = tensor_img * std + mean
            img_denorm = img_denorm.clamp(0.0, 1.0)
            save_path = os.path.join(class_dir, f"{base_name}_preprocessed_{v}.png")
            save_image(img_denorm, save_path)

        if idx % 100 == 0 or idx == total:
            print(f"Processed {idx}/{total} images")

    print("Done.")