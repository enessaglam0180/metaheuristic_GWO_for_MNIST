import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

N_WOLVES = 6 
MAX_ITER = 10
K_NEIGHBORS = 5


try:
    df = pd.read_csv('mnist_test.csv', header=None)
except FileNotFoundError:
    print("HATA: 'mnist_test.csv' dosyası bulunamadı!")
    exit()

# İlk sütun (0. sütun) = Label
# Diğer 784 sütun = Pikseller
y_raw = df.iloc[:, 0].values
X_raw = df.iloc[:, 1:].values

sample_size = 1000
X = X_raw[:sample_size]
y = y_raw[:sample_size]

# Normalizasyon
X = X / 255.0

print(f"Veri Hazır! Kullanılan Örnek Sayısı: {sample_size}")
print(f"Toplam Özellik (Piksel) Sayısı: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dim = X_train.shape[1]
print("-" * 40)

print("Referans skor hesaplanıyor...")
knn_base = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
knn_base.fit(X_train, y_train)
y_pred_base = knn_base.predict(X_test)
base_acc = accuracy_score(y_test, y_pred_base)
print(f">> TÜM PİKSELLERLE ({dim} tane) BAŞARI: %{base_acc * 100:.2f}")
print("-" * 40)

# GWO SINIFI
class BinaryGWO:
    def __init__(self, n_wolves, max_iter, dim):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.dim = dim
        self.positions = np.random.randint(0, 2, (n_wolves, dim))
        
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float("inf")
        self.beta_pos = np.zeros(dim)
        self.beta_score = float("inf")
        self.delta_pos = np.zeros(dim)
        self.delta_score = float("inf")
        self.history = []

    def fitness(self, mask):
        if np.sum(mask) == 0: return 1.0 
        
        selected_indices = [i for i, val in enumerate(mask) if val == 1]
        
        X_tr_sub = X_train[:, selected_indices]
        X_te_sub = X_test[:, selected_indices]
        
        clf = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
        clf.fit(X_tr_sub, y_train)
        pred = clf.predict(X_te_sub)
        acc = accuracy_score(y_test, pred)
        
        # Ceza katsayısı daha agresif bir eleme için artırıldı
        error = 1 - acc
        selection_ratio = len(selected_indices) / self.dim
        cost = 0.99 * error + 0.01 * selection_ratio
        return cost

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-10 * (x - 0.5)))

    def optimize(self):
        print(f"Kurtlar {dim} boyutlu alanda avlanıyor...")
        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)
            
            for i in range(self.n_wolves):
                fit = self.fitness(self.positions[i])
                if fit < self.alpha_score:
                    self.alpha_score = fit
                    self.alpha_pos = self.positions[i].copy()
                if fit > self.alpha_score and fit < self.beta_score:
                    self.beta_score = fit
                    self.beta_pos = self.positions[i].copy()
                if fit > self.alpha_score and fit > self.beta_score and fit < self.delta_score:
                    self.delta_score = fit
                    self.delta_pos = self.positions[i].copy()
            
            self.history.append(self.alpha_score)
            print(f"Iterasyon {t+1}/{self.max_iter} -> En İyi Cost: {self.alpha_score:.4f}")
            
            # Hareket (Update)
            for i in range(self.n_wolves):
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    X1 = self.alpha_pos[j] - (2*a*r1 - a) * abs(2*r2*self.alpha_pos[j] - self.positions[i,j])
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    X2 = self.beta_pos[j] - (2*a*r1 - a) * abs(2*r2*self.beta_pos[j] - self.positions[i,j])
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    X3 = self.delta_pos[j] - (2*a*r1 - a) * abs(2*r2*self.delta_pos[j] - self.positions[i,j])
                    
                    X_avg = (X1 + X2 + X3) / 3
                    if np.random.rand() < self.sigmoid(X_avg):
                        self.positions[i, j] = 1
                    else:
                        self.positions[i, j] = 0
                        
        return self.alpha_pos, self.history

gwo = BinaryGWO(N_WOLVES, MAX_ITER, dim)
best_mask, history = gwo.optimize()

# SONUÇLAR
selected_indices = [i for i, val in enumerate(best_mask) if val == 1]
X_tr_final = X_train[:, selected_indices]
X_te_final = X_test[:, selected_indices]

clf_final = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
clf_final.fit(X_tr_final, y_train)
pred_final = clf_final.predict(X_te_final)
final_acc = accuracy_score(y_test, pred_final)

print("\n" + "="*40)
print(f"MNIST SONUÇ RAPORU (Sample: {sample_size})")
print("="*40)
print(f"Başlangıç Özellik : 784")
print(f"SEÇİLEN Özellik   : {len(selected_indices)}")
print(f"Referans Başarı   : %{base_acc * 100:.2f}")
print(f"GWO Sonrası Başarı: %{final_acc * 100:.2f}")
print("="*40)

mask_gorsel = best_mask.reshape(28, 28)

plt.figure(figsize=(6, 6))
plt.imshow(mask_gorsel, cmap='gray', interpolation='nearest')
plt.title(f"MNIST İçin Seçilen {len(selected_indices)} Kritik Piksel")
plt.axis('off')
plt.show()