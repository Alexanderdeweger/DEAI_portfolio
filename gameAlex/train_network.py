import argparse
import csv
import os
import random
import numpy as np

INPUT_COLUMNS = [
    "player_x",
    "obs1_x",
    "obs1_y",
    "obs2_x",
    "obs2_y",
    "obs3_x",
    "obs3_y",
]


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    rows = []
    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                x = [float(row[col]) for col in INPUT_COLUMNS]
            except (KeyError, ValueError):
                continue
            rows.append(x)

    if len(rows) < 2:
        raise ValueError("Need at least two valid rows to build training targets.")

    x = np.array(rows[:-1], dtype=np.float32)
    y = np.array([row[0] for row in rows[1:]], dtype=np.float32).reshape(-1, 1)
    return x, y


def train_test_split(x, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    split = int(len(x) * (1.0 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


class MLP:
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.3, size=(input_size, hidden_size)).astype(np.float32)
        self.b1 = np.zeros((1, hidden_size), dtype=np.float32)
        self.W2 = rng.normal(0.0, 0.3, size=(hidden_size, output_size)).astype(np.float32)
        self.b2 = np.zeros((1, output_size), dtype=np.float32)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = np.tanh(z1)
        output = a1 @ self.W2 + self.b2
        return a1, output

    def predict(self, x):
        _, output = self.forward(x)
        return output


def train(model, x_train, y_train, x_test, y_test, epochs, batch_size, lr, l2):
    num_samples = len(x_train)
    num_batches = max(1, int(np.ceil(num_samples / batch_size)))

    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(num_samples)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0.0
        for batch in range(num_batches):
            start = batch * batch_size
            end = min(start + batch_size, num_samples)
            xb = x_shuffled[start:end]
            yb = y_shuffled[start:end]
            if len(xb) == 0:
                continue

            a1, preds = model.forward(xb)
            error = preds - yb
            data_loss = np.mean(error ** 2)
            reg_loss = 0.5 * l2 * (np.sum(model.W1 ** 2) + np.sum(model.W2 ** 2))
            loss = data_loss + reg_loss
            epoch_loss += loss

            dpreds = (2.0 / len(yb)) * error
            dW2 = a1.T @ dpreds + l2 * model.W2
            db2 = np.sum(dpreds, axis=0, keepdims=True)
            da1 = dpreds @ model.W2.T
            dz1 = da1 * (1.0 - a1 ** 2)
            dW1 = xb.T @ dz1 + l2 * model.W1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            model.W1 -= lr * dW1
            model.b1 -= lr * db1
            model.W2 -= lr * dW2
            model.b2 -= lr * db2

        train_rmse = rmse(model, x_train, y_train)
        test_rmse = rmse(model, x_test, y_test) if len(x_test) else 0.0
        avg_loss = epoch_loss / max(1, num_batches)
        print(
            f"Epoch {epoch:02d} | loss {avg_loss:.4f} | train rmse {train_rmse:.4f} | test rmse {test_rmse:.4f}"
        )


def rmse(model, x, y):
    if len(x) == 0:
        return 0.0
    preds = model.predict(x)
    return float(np.sqrt(np.mean((preds - y) ** 2)))


def save_model(model, path):
    np.savez(
        path,
        W1=model.W1,
        b1=model.b1,
        W2=model.W2,
        b2=model.b2,
        input_columns=np.array(INPUT_COLUMNS),
        output_kind=np.array(["next_player_x"]),
    )


def main():
    parser = argparse.ArgumentParser(description="Train a small MLP on training_data.csv.")
    parser.add_argument("--data", default="training_data.csv", help="Path to CSV data file.")
    parser.add_argument("--model", default="model.npz", help="Output path for trained model.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    x, y = load_data(args.data)
    x_train, y_train, x_test, y_test = train_test_split(x, y, seed=args.seed)

    model = MLP(len(INPUT_COLUMNS), args.hidden, 1, seed=args.seed)
    train(model, x_train, y_train, x_test, y_test, args.epochs, args.batch_size, args.lr, args.l2)
    save_model(model, args.model)
    print(f"Saved model to {args.model}")


if __name__ == "__main__":
    main()
#python train_network.py --data training_data.csv --model model.npz