import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

N_WOLVES = 8       
MAX_ITER = 20      
K_NEIGHBORS = 5    

print("Veri yükleniyor ve hazırlanıyor...")
digits = load_digits()
X = digits.data
y = digits.target

# Veriyi %80 Eğitim, %20 Test olarak bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dim = X_train.shape[1]
print(f"Toplam Özellik (Piksel) Sayısı: {dim}")
print("-" * 40)

print("Referans skor hesaplanıyor (Tüm piksellerle)...")
knn_base = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
knn_base.fit(X_train, y_train)
y_pred_base = knn_base.predict(X_test)
base_acc = accuracy_score(y_test, y_pred_base)
print(f">> REFERANS BAŞARISI: %{base_acc * 100:.2f}")
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
        
        if np.sum(mask) == 0:
            return 1.0 
        
        selected_indices = [i for i, val in enumerate(mask) if val == 1]
        
        X_tr_sub = X_train[:, selected_indices]
        X_te_sub = X_test[:, selected_indices]
        
        clf = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
        clf.fit(X_tr_sub, y_train)
        pred = clf.predict(X_te_sub)
        acc = accuracy_score(y_test, pred)
        
        # Fitness Formülü: Hata Oranı + (Seçilen Özellik Oranı * 0.01)
        error = 1 - acc
        selection_ratio = len(selected_indices) / self.dim
        cost = 0.99 * error + 0.01 * selection_ratio
        
        return cost

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-10 * (x - 0.5)))

    def optimize(self):
        print("Kurtlar avlanmaya başladı...")
        
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
            
            # Pozisyonları Güncelle
            for i in range(self.n_wolves):
                for j in range(self.dim):
                    # Alpha'ya göre
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    D_alpha = abs(2 * r2 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha
                    
                    # Beta'ya göre
                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    D_beta = abs(2 * r2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta
                    
                    # Delta'ya göre
                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    D_delta = abs(2 * r2 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta
                    
                    # Ortalama ve Sigmoid
                    X_avg = (X1 + X2 + X3) / 3
                    prob = self.sigmoid(X_avg)
                    
                    if np.random.rand() < prob:
                        self.positions[i, j] = 1
                    else:
                        self.positions[i, j] = 0
                        
        return self.alpha_pos, self.history

# Çalıştırdık
gwo = BinaryGWO(N_WOLVES, MAX_ITER, dim)
best_mask, history = gwo.optimize()

# Sonuçlar
selected_indices = [i for i, val in enumerate(best_mask) if val == 1]
X_tr_final = X_train[:, selected_indices]
X_te_final = X_test[:, selected_indices]

clf_final = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
clf_final.fit(X_tr_final, y_train)
pred_final = clf_final.predict(X_te_final)
final_acc = accuracy_score(y_test, pred_final)

print("\n" + "="*40)
print("             SONUÇ RAPORU")
print("="*40)
print(f"Başlangıç Özellik Sayısı : {dim}")
print(f"SEÇİLEN Özellik Sayısı   : {len(selected_indices)}  (Daha Az Veri!)")
print("-" * 40)
print(f"Referans Başarısı        : %{base_acc * 100:.2f}")
print(f"GWO Sonrası Başarı       : %{final_acc * 100:.2f}")
print("="*40)

plt.plot(history, marker='o', linestyle='-')
plt.title('GWO Maliyet Analizi (Düşük = İyi)')
plt.xlabel('İterasyon')
plt.ylabel('Cost')
plt.grid(True)
plt.show()

# Görselleştirme
mask_gorsel = best_mask.reshape(8, 8)

plt.figure(figsize=(6, 6))
plt.imshow(mask_gorsel, cmap='gray', interpolation='nearest')
plt.title(f"Kurtların Seçtiği {len(selected_indices)} Kritik Piksel (Beyazlar)")
plt.colorbar(label="0: Elendi, 1: Seçildi")
plt.axis('off')
plt.show()