"""诊断模型训练脚本

使用模拟数据训练 MLP 诊断预测模型，保存最优权重。
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from app.ml.diagnosis_model import DiagnosisPredictor, DISEASE_LABELS, LAB_FEATURE_NAMES
from app.ml.data_generator import generate_mock_data


def prepare_data(patients: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """将病例数据转为训练特征 X 和标签 y"""
    X = np.array(
        [[p["lab_data"][feat] for feat in LAB_FEATURE_NAMES] for p in patients],
        dtype=np.float32,
    )
    y = np.array(
        [DISEASE_LABELS.index(p["diagnosis"]) for p in patients],
        dtype=np.int64,
    )
    return torch.from_numpy(X), torch.from_numpy(y)


def train(
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    n_per_disease: int = 50,
):
    """训练诊断预测模型并保存最优权重

    Args:
        num_epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        n_per_disease: 每种疾病生成的模拟病例数
    """
    print(f"生成模拟数据（{len(DISEASE_LABELS)} 种疾病 × {n_per_disease} 条 = "
          f"{len(DISEASE_LABELS) * n_per_disease} 条）...")
    patients = generate_mock_data(n_per_disease=n_per_disease)
    X, y = prepare_data(patients)

    # 8:2 划分训练/验证集
    n = len(patients)
    n_train = int(n * 0.8)
    indices = torch.randperm(n)
    X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
    X_val, y_val = X[indices[n_train:]], y[indices[n_train:]]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size
    )

    model = DiagnosisPredictor(
        input_dim=len(LAB_FEATURE_NAMES),
        num_diseases=len(DISEASE_LABELS),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    print(f"开始训练（{num_epochs} epochs, batch_size={batch_size}, lr={lr}）...")
    for epoch in range(num_epochs):
        # ── 训练阶段 ──
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            # CrossEntropyLoss 内置 softmax，不需要 model 再输出 softmax
            # 但这里 model.forward 已含 softmax，训练时需取 log
            loss = criterion(torch.log(outputs + 1e-8), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── 验证阶段 ──
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        acc = correct / total

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss / len(train_loader):.4f} | "
                f"Val Acc: {acc:.4f}"
            )

        # 保存最优模型
        if acc > best_acc:
            best_acc = acc
            model_dir = Path("data/models")
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / "diagnosis_model.pt")

    print(f"\n训练完成！最佳验证准确率: {best_acc:.4f}")
    print(f"模型已保存至 data/models/diagnosis_model.pt")

    # 保存模拟数据 JSON（后续导入数据库用）
    mock_file = Path("data/mock_patients.json")
    with open(mock_file, "w", encoding="utf-8") as f:
        json.dump(patients, f, ensure_ascii=False, indent=2)
    print(f"模拟数据已保存至 {mock_file} ({len(patients)} 条)")


if __name__ == "__main__":
    train()
