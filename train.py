import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models.autoencoder import CustomAE
from models.losses import domination_penalty_loss
from utils.dominance import find_dominated_pairs

df = pd.read_csv('data/dataset.csv')
scaler = StandardScaler()
X_np = scaler.fit_transform(df.drop(columns=['molecule_id']).values.astype(np.float32))
X = torch.tensor(X_np)

dominated_pairs = find_dominated_pairs(X_np, epsilon=1e-2)

model = CustomAE(input_dim=X.shape[1], latent_dim=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
mse_loss = torch.nn.MSELoss()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    x_hat, z = model(X)

    loss_mse = mse_loss(x_hat, X)
    loss_mae = torch.mean(torch.abs(x_hat - X))
    loss_rec = 0.7 * loss_mse + 0.3 * loss_mae

    loss_dom = domination_penalty_loss(x_hat, dominated_pairs)
    loss = loss_rec + 2.0 * loss_dom

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"[Epoch {epoch}] MSE: {loss_mse.item():.4f}, MAE: {loss_mae.item():.4f}, DomLoss: {loss_dom.item():.4f}")

torch.save(model.state_dict(), "model.pth")
