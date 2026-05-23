import argparse
import csv
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class Cifar100PickleDataset(Dataset):
    def __init__(self, pickle_path, transform=None):
        self.pickle_path = Path(pickle_path)
        self.transform = transform

        # CIFAR-100 pickle files store flattened RGB images in the "data" field.
        with self.pickle_path.open("rb") as f:
            item = pickle.load(f, encoding="latin1")

        # Restore the original (N, 3072) array to PyTorch's common (N, C, H, W) layout.
        self.images = item["data"].reshape(-1, 3, 32, 32)
        self.labels = item["fine_labels"]
        self.filenames = item.get("filenames")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # torchvision transforms operate naturally on PIL images.
        image = self.images[index].transpose(1, 2, 0)
        image = Image.fromarray(image)
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_model(num_classes=100, pretrained=True, freeze_backbone=False):
    # Transfer learning starts from an ImageNet-pretrained ResNet34.
    weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet34(weights=weights)

    if freeze_backbone:
        # Stage 1 freezes the backbone and trains only the classifier head.
        for parameter in model.parameters():
            parameter.requires_grad = False

    # CIFAR-100 has 100 classes, so replace the original ImageNet 1000-way head.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_optimizer(model, args):
    # The new head can use a larger LR; the backbone uses a smaller LR to preserve pretrained features.
    head_lr = args.head_lr if args.head_lr is not None else args.lr
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.lr * 0.1

    if args.freeze_backbone:
        # In the frozen stage, only fc parameters are passed to the optimizer.
        trainable_parameters = [p for p in model.fc.parameters() if p.requires_grad]
        print(f"optimizer: frozen backbone, head_lr={head_lr}")
        return torch.optim.AdamW(
            trainable_parameters,
            lr=head_lr,
            weight_decay=args.weight_decay,
        )

    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # ResNet's final classifier is named fc, so keep it in a separate parameter group.
        if name.startswith("fc."):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": backbone_lr})
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": head_lr})

    print(f"optimizer: backbone_lr={backbone_lr}, head_lr={head_lr}")
    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)


def build_scheduler(optimizer, args):
    if not args.use_scheduler:
        return None

    # Reduce LR based on validation loss when later training stops improving.
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )


def get_learning_rates(optimizer):
    return [group["lr"] for group in optimizer.param_groups]


def accuracy(logits, targets):
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train=True, max_batches=None):
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Enable gradients for training and disable them for evaluation to save memory and compute.
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits.detach(), labels) * batch_size
        total_samples += batch_size

        if train and batch_index % 50 == 0:
            print(
                f"batch {batch_index:04d} "
                f"loss={total_loss / total_samples:.4f} "
                f"acc={total_acc / total_samples:.4f}"
            )

    return total_loss / total_samples, total_acc / total_samples


def save_history(history, output_dir):
    # Rewrite the full history after each epoch so partial runs remain inspectable.
    csv_path = output_dir / "training_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "test_loss",
                "test_acc",
                "learning_rates",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    epochs = [item["epoch"] for item in history]

    # Save loss and accuracy curves to make overfitting and generalization trends visible.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="Train Loss")
    axes[0].plot(epochs, [item["test_loss"] for item in history], label="Test Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [item["train_acc"] for item in history], label="Train Acc")
    axes[1].plot(epochs, [item["test_acc"] for item in history], label="Test Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    figure_path = output_dir / "training_curves.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"saved training history: {csv_path}")
    print(f"saved training curves: {figure_path}")


def load_history(output_dir):
    # When resuming, load the existing CSV so curves continue in the same figure.
    csv_path = output_dir / "training_history.csv"
    if not csv_path.exists():
        return []

    history = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_acc": float(row["train_acc"]),
                    "test_loss": float(row["test_loss"]),
                    "test_acc": float(row["test_acc"]),
                    "learning_rates": row.get("learning_rates", ""),
                }
            )
    return history


def main():
    parser = argparse.ArgumentParser(description="Train ImageNet-pretrained ResNet34 on CIFAR-100.")
    parser.add_argument("--train-file", default="data/train/train")
    parser.add_argument("--test-file", default="data/test/test")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--torch-cache", default=".cache/torch")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--use-scheduler", action="store_true")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # Keep downloaded pretrained weights inside the project instead of the global torch cache.
    torch.hub.set_dir(str(Path(args.torch_cache) / "hub"))

    # ImageNet-pretrained ResNet34 expects inputs normalized with ImageNet RGB statistics.
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            # Common CIFAR augmentation: pad and random-crop to simulate small translations.
            transforms.RandomCrop(32, padding=4),
            # ImageNet-pretrained ResNet34 uses 224x224 inputs.
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.92, 1.08),
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std),
            # Random erasing runs on normalized tensors and improves occlusion robustness.
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            # The test set uses deterministic preprocessing only, keeping evaluation reproducible.
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std),
        ]
    )

    train_dataset = Cifar100PickleDataset(args.train_file, transform=train_transform)
    test_dataset = Cifar100PickleDataset(args.test_file, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        num_classes=100,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    start_epoch = 1
    best_acc = float("-inf")
    if args.resume is not None:
        # Stage 2 or follow-up experiments resume model weights from an existing checkpoint.
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        resumed_acc = float(checkpoint.get("test_acc", 0.0))
        print(
            f"resumed checkpoint: {args.resume}, "
            f"epoch={start_epoch - 1}, test_acc={resumed_acc:.4f}"
        )

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    if args.resume is not None and args.resume_optimizer:
        # Optimizer state is not restored by default so Stage 2 can reset smaller differential LRs.
        checkpoint = torch.load(args.resume, map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("resumed optimizer state")

    criterion = nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(output_dir) if args.resume is not None else []
    if history:
        best_acc = max(best_acc, max(item["test_acc"] for item in history))

    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        print(f"epoch {epoch}/{end_epoch}")
        # Run one training epoch, then evaluate on the test set.
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            max_batches=args.max_batches,
        )
        test_loss, test_acc = run_epoch(
            model,
            test_loader,
            criterion,
            optimizer,
            device,
            train=False,
            max_batches=args.max_batches,
        )

        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )
        if scheduler is not None:
            # ReduceLROnPlateau steps after the validation metric is available.
            scheduler.step(test_loss)

        learning_rates = get_learning_rates(optimizer)
        print("learning_rates:", ", ".join(f"{lr:.8f}" for lr in learning_rates))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "learning_rates": ";".join(str(lr) for lr in learning_rates),
            }
        )
        save_history(history, output_dir)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "test_acc": test_acc,
            "args": vars(args),
        }
        # last always stores the latest epoch; best updates only when test_acc improves.
        torch.save(checkpoint, output_dir / "last_resnet34_cifar100.pt")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(checkpoint, output_dir / "best_resnet34_cifar100.pt")
            print(f"saved best checkpoint: test_acc={best_acc:.4f}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
