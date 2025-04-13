import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models.autoencoder import CustomAE
from utils.dominance import find_dominated_pairs
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import Counter

df = pd.read_csv('data/dataset.csv')
molecule_ids = df['molecule_id'].values
scaler = StandardScaler()
X_np = scaler.fit_transform(df.drop(columns=['molecule_id']).values.astype(np.float32))
X = torch.tensor(X_np)

model = CustomAE(input_dim=X.shape[1], latent_dim=5)
model.load_state_dict(torch.load("model.pth"))
model.eval()

with torch.no_grad():
    x_hat, z = model(X)

x_hat_np = x_hat.numpy()
z_np = z.numpy()

X_orig = scaler.inverse_transform(X_np)
x_hat_orig = scaler.inverse_transform(x_hat_np)

mae = mean_absolute_error(X_orig, x_hat_orig)
mse = mean_squared_error(X_orig, x_hat_orig)

dom_orig = set(find_dominated_pairs(X_orig, epsilon=1e-2))
dom_hat = set(find_dominated_pairs(x_hat_orig, epsilon=1e-2))

lost_dom = dom_orig - dom_hat
new_dom = dom_hat - dom_orig

def translate_pairs(pairs, ids):
    return [(int(ids[i]), int(ids[j])) for (i, j) in pairs]

lost_ids = translate_pairs(lost_dom, molecule_ids)
new_ids = translate_pairs(new_dom, molecule_ids)

pd.DataFrame(lost_ids, columns=["dominated_id", "dominating_id"]).to_csv("lost_dominance_pairs.csv", index=False)
pd.DataFrame(new_ids, columns=["dominated_id", "dominating_id"]).to_csv("new_dominance_pairs.csv", index=False)

print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")
print(f"Исходных доминирующих пар: {len(dom_orig)}")
print(f"Потеряно доминирующих пар: {len(lost_dom)}")
print(f"Появилось новых (ложных) пар: {len(new_dom)}")
print("Логи сохранены в:")
print("- lost_dominance_pairs.csv")
print("- new_dominance_pairs.csv")

dominated_counter = Counter(i for i, _ in new_ids)
dominating_counter = Counter(j for _, j in new_ids)

np.save("data/latent_z.npy", z_np)
np.save("data/x_hat.npy", x_hat_orig)

delta = 0.1
total_elements = X_orig.size
correct_elements = np.sum(np.abs(X_orig - x_hat_orig) < delta)
accuracy = (correct_elements / total_elements) * 100

print(f"\nПроцент точно восстановленных признаков (<{delta} отклонение): {accuracy:.2f}%")

preserved_pairs = len(dom_orig & dom_hat)
preserved_percent = preserved_pairs / len(dom_orig) * 100

print(f"\nСохранено доминирующих пар: {preserved_pairs} / {len(dom_orig)}")
print(f"Процент сохранённых доминирований: {preserved_percent:.2f}%")