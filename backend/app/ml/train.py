"""诊断模型训练脚本

62 种疾病, XGBoost 主力 + MLP 基线, 8:1:1 划分
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
from pathlib import Path

from app.ml.data_generator import generate_mock_data, LAB_FEATURES, DISEASE_LABELS_V2
from app.ml.diagnosis_model import DiagnosisPredictor


def prepare_data(patients: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """提取 lab_data 为特征矩阵, 填充缺失值"""
    X_list = []
    y_list = []
    for p in patients:
        ld = p["lab_data"]
        feat_vec = []
        for feat in LAB_FEATURES:
            val = ld.get(feat)
            feat_vec.append(val if val is not None else float("nan"))
        X_list.append(feat_vec)
        y_list.append(p["diagnosis"])
    X = np.array(X_list, dtype=np.float32)
    # 中位数填充缺失
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            col[mask] = np.nanmedian(col)
    return X, np.array(y_list)


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    input_dim: int, num_classes: int,
    epochs: int = 80, batch_size: int = 64, lr: float = 0.001,
) -> tuple[DiagnosisPredictor, float]:
    """训练 MLP 模型"""
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    train_ds = TensorDataset(torch.FloatTensor(X_train_s), torch.LongTensor(y_train_enc))
    val_ds = TensorDataset(torch.FloatTensor(X_val_s), torch.LongTensor(y_val_enc))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = DiagnosisPredictor(input_dim=input_dim, num_diseases=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                outputs = model(bx)
                _, pred = torch.max(outputs, 1)
                total += by.size(0)
                correct += (pred == by).sum().item()
        acc = correct / total

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  MLP Epoch {epoch+1:3d}/{epochs} | Loss {train_loss/len(train_loader):.4f} | Val Acc {acc:.4f}")

    model.load_state_dict(best_state)
    return model, best_acc, scaler, le


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    num_classes: int,
) -> tuple[xgb.XGBClassifier, float]:
    """训练 XGBoost 基线模型"""
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(
        X_train_s, y_train_enc,
        eval_set=[(X_val_s, y_val_enc)],
        verbose=False,
    )
    y_pred = model.predict(X_val_s)
    acc = accuracy_score(y_val_enc, y_pred)
    return model, acc, scaler, le


def evaluate(model, X: np.ndarray, y: np.ndarray, scaler, le, name: str, is_torch: bool = True):
    """评估模型"""
    X_s = scaler.transform(X)
    if is_torch:
        model.eval()
        tensor = torch.FloatTensor(X_s)
        with torch.no_grad():
            outputs = model(tensor)
            _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.numpy()
    else:
        y_pred = model.predict(X_s)

    y_enc = le.transform(y)
    acc = accuracy_score(y_enc, y_pred)
    f1 = f1_score(y_enc, y_pred, average="weighted")
    print(f"  {name}: Acc={acc:.4f} F1={f1:.4f}")


def main():
    print("=" * 50)
    print("生成训练数据...")
    patients = generate_mock_data(n_per_disease=200, comorbidity_rate=0.20, missing_rate=0.08)
    print(f"  总样本: {len(patients)}, 疾病种类: {len(DISEASE_LABELS_V2)}")

    X, y = prepare_data(patients)
    n = len(patients)
    indices = np.random.permutation(n)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    X_train, y_train = X[indices[:n_train]], y[indices[:n_train]]
    X_val, y_val = X[indices[n_train:n_train+n_val]], y[indices[n_train:n_train+n_val]]
    X_test, y_test = X[indices[n_train+n_val:]], y[indices[n_train+n_val:]]
    print(f"  训练: {n_train}  验证: {n_val}  测试: {n - n_train - n_val}")

    num_classes = len(DISEASE_LABELS_V2)
    input_dim = X.shape[1]

    # ── MLP ─────────────────────────────────────
    print("\n[MLP 训练]")
    mlp_model, mlp_acc, mlp_scaler, mlp_le = train_mlp(
        X_train, y_train, X_val, y_val, input_dim, num_classes,
        epochs=150, batch_size=128, lr=0.002,
    )

    # ── XGBoost ─────────────────────────────────
    print("\n[XGBoost 训练]")
    xgb_model, xgb_acc, xgb_scaler, xgb_le = train_xgboost(
        X_train, y_train, X_val, y_val, num_classes,
    )

    # ── 评估 ────────────────────────────────────
    print("\n[测试集评估]")
    evaluate(mlp_model, X_test, y_test, mlp_scaler, mlp_le, "MLP", is_torch=True)
    evaluate(xgb_model, X_test, y_test, xgb_scaler, xgb_le, "XGBoost", is_torch=False)

    # ── 保存 ────────────────────────────────────
    model_path = Path("data/models/diagnosis_model.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # 主力: XGBoost (在 60 类上明显优于 MLP)
    import pickle
    with open("data/models/xgboost_baseline.pkl", "wb") as f:
        pickle.dump((xgb_model, xgb_scaler, xgb_le), f)
    print(f"\n>>> XGBoost 主力模型已保存")

    # 基线对照: MLP
    torch.save(mlp_model.state_dict(), model_path)
    print(f">>> MLP 基线已保存 (对比用)")

    # 保存标签映射
    meta = {
        "disease_labels": DISEASE_LABELS_V2,
        "lab_features": LAB_FEATURES,
        "num_classes": num_classes,
        "input_dim": input_dim,
        "primary_model": "xgboost",
        "mlp_acc": round(mlp_acc, 4),
        "xgb_acc": round(xgb_acc, 4),
    }
    with open("data/models/model_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  元数据已保存: data/models/model_meta.json")


if __name__ == "__main__":
    main()
