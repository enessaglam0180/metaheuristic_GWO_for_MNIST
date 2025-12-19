import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


N_WOLVES = 8
MAX_ITER = 50
K_NEIGHBORS = 5

print(">> Görselleştirme Modülü Başlatıldı...")
print(">> Veri okunuyor...")

try:
    df = pd.read_csv('mnist_test.csv', header=None)
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1:].values
except FileNotFoundError:
    print("HATA: 'mnist_test.csv' dosyası bulunamadı.")
    exit()

sample_size = 1000
x = x[:sample_size] / 255.0
y = y[:sample_size]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
dim = X_train.shape[1]

def calculate_fitness(mask):
    if np.sum(mask) == 0: return 1.0
    selected_indices = [i for i, val in enumerate(mask) if val == 1]
    clf = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
    clf.fit(X_train[:, selected_indices], y_train)
    pred = clf.predict(X_test[:, selected_indices])
    acc = accuracy_score(y_test, pred)
    return 0.99 * (1 - acc) + 0.01 * (len(selected_indices) / dim)

# GWO Motoru
print(f">> GWO {MAX_ITER} tur boyunca maskeyi oluşturuyor...")
wolves = np.random.randint(0, 2, (N_WOLVES, dim))
alpha_pos, alpha_score = np.zeros(dim), float("inf")
beta_pos, beta_score = np.zeros(dim), float("inf")
delta_pos, delta_score = np.zeros(dim), float("inf")

for t in range(MAX_ITER):
    a = 2 - t * (2 / MAX_ITER)
    for i in range(N_WOLVES):
        fit = calculate_fitness(wolves[i])
        if fit < alpha_score: alpha_score, alpha_pos = fit, wolves[i].copy()
        if fit > alpha_score and fit < beta_score: beta_score, beta_pos = fit, wolves[i].copy()
        if fit > alpha_score and fit > beta_score and fit < delta_score: delta_score, delta_pos = fit, wolves[i].copy()
    
    # Konum Güncelleme
    for i in range(N_WOLVES):
        for j in range(dim):
            r1, r2 = np.random.rand(), np.random.rand()
            X1 = alpha_pos[j] - (2*a*r1 - a) * abs(2*r2*alpha_pos[j] - wolves[i,j])
            r1, r2 = np.random.rand(), np.random.rand()
            X2 = beta_pos[j] - (2*a*r1 - a) * abs(2*r2*beta_pos[j] - wolves[i,j])
            r1, r2 = np.random.rand(), np.random.rand()
            X3 = delta_pos[j] - (2*a*r1 - a) * abs(2*r2*delta_pos[j] - wolves[i,j])
            
            wolves[i, j] = 1 if np.random.rand() < (1 / (1 + np.exp(-10 * ((X1+X2+X3)/3 - 0.5)))) else 0

selected_count = np.sum(alpha_pos)
print(f">> Bitti! Seçilen Özellik Sayısı: {int(selected_count)}")

mask_gorsel = alpha_pos.reshape(28, 28)
plt.figure(figsize=(6, 6))
plt.imshow(mask_gorsel, cmap='gray', interpolation='nearest')
plt.title(f"GWO Result: {int(selected_count)} Critical Pixels Selected")
plt.colorbar(label="0: Eliminated, 1: Selected")
plt.axis('off')
plt.show()