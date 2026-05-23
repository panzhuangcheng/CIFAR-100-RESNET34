import argparse
import csv
import pickle
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_resnet34_cifar100 import build_model


class Cifar100SubmissionDataset(Dataset):
    def __init__(self, pickle_path, transform=None):
        self.pickle_path = Path(pickle_path)
        self.transform = transform

        with self.pickle_path.open("rb") as f:
            item = pickle.load(f, encoding="latin1")

        self.images = item["data"].reshape(-1, 3, 32, 32)
        self.filenames = item.get("filenames", [str(i) for i in range(len(self.images))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].transpose(1, 2, 0)
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, index, self.filenames[index]


def main():
    parser = argparse.ArgumentParser(description="Generate a CIFAR-100 submission CSV.")
    parser.add_argument("--checkpoint", default="checkpoints_low_lr/best_resnet34_cifar100.pt")
    parser.add_argument("--test-file", default="data/test/test")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--include-filename", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = Cifar100SubmissionDataset(args.test_file, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(num_classes=100, pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for images, indices, filenames in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().tolist()

            for index, filename, prediction in zip(indices.tolist(), filenames, predictions):
                row = {"id": index, "label": prediction}
                if args.include_filename:
                    row["filename"] = filename
                rows.append(row)

    fieldnames = ["id", "label"]
    if args.include_filename:
        fieldnames.append("filename")

    output_path = Path(args.output)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved submission: {output_path}")
    print(f"rows: {len(rows)}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}")


if __name__ == "__main__":
    main()
