import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# --- 1. AYARLAR ---
# İterasyon sayılarını bir liste yaptık. Kod sırayla hepsini deneyecek.
ITERATION_SCENARIOS = [10, 30, 50] 
N_AGENTS = 8       
K_NEIGHBORS = 5    

# --- 2. VERİ YÜKLEME ---
try:
    df = pd.read_csv('mnist_test.csv', header=None)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
except FileNotFoundError:
    print("HATA: 'mnist_test.csv' bulunamadı.")
    exit()

# Hız için 800 örnek (Maraton koşacağımız için bunu artırma)
sample_size = 800
X = X[:sample_size] / 255.0
y = y[:sample_size]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dim = X_train.shape[1]

# --- 3. FITNESS FONKSİYONU ---
def calculate_fitness(mask):
    if np.sum(mask) == 0: return 1.0
    selected_indices = [i for i, val in enumerate(mask) if val == 1]
    
    clf = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
    clf.fit(X_train[:, selected_indices], y_train)
    pred = clf.predict(X_test[:, selected_indices])
    acc = accuracy_score(y_test, pred)
    
    error = 1 - acc
    ratio = len(selected_indices) / dim
    cost = 0.99 * error + 0.01 * ratio 
    return cost

# --- 4. ALGORITMALAR (Class yapıları özetlendi) ---
class GWO:
    def __init__(self, max_iter):
        self.name = "GWO"
        self.max_iter = max_iter
    def run(self):
        wolves = np.random.randint(0, 2, (N_AGENTS, dim))
        alpha_pos, alpha_score = np.zeros(dim), float("inf")
        beta_pos, beta_score = np.zeros(dim), float("inf")
        delta_pos, delta_score = np.zeros(dim), float("inf")
        
        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)
            for i in range(N_AGENTS):
                fit = calculate_fitness(wolves[i])
                if fit < alpha_score: alpha_score, alpha_pos = fit, wolves[i].copy()
                if fit > alpha_score and fit < beta_score: beta_score, beta_pos = fit, wolves[i].copy()
                if fit > alpha_score and fit > beta_score and fit < delta_score: delta_score, delta_pos = fit, wolves[i].copy()
            
            for i in range(N_AGENTS):
                for j in range(dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    X1 = alpha_pos[j] - (2*a*r1 - a) * abs(2*r2*alpha_pos[j] - wolves[i,j])
                    r1, r2 = np.random.rand(), np.random.rand()
                    X2 = beta_pos[j] - (2*a*r1 - a) * abs(2*r2*beta_pos[j] - wolves[i,j])
                    r1, r2 = np.random.rand(), np.random.rand()
                    X3 = delta_pos[j] - (2*a*r1 - a) * abs(2*r2*delta_pos[j] - wolves[i,j])
                    
                    wolves[i, j] = 1 if np.random.rand() < (1 / (1 + np.exp(-10 * ((X1+X2+X3)/3 - 0.5)))) else 0
        return alpha_pos, alpha_score

class GA:
    def __init__(self, max_iter):
        self.name = "GA"
        self.max_iter = max_iter
    def run(self):
        pop = np.random.randint(0, 2, (N_AGENTS, dim))
        best_sol, best_score = np.zeros(dim), float("inf")
        for t in range(self.max_iter):
            scores = [calculate_fitness(p) for p in pop]
            for i, s in enumerate(scores):
                if s < best_score: best_score, best_sol = s, pop[i].copy()
            
            new_pop = [best_sol]
            while len(new_pop) < N_AGENTS:
                # Basit tournament selection ve crossover
                p1 = pop[np.random.randint(0, N_AGENTS)]
                p2 = pop[np.random.randint(0, N_AGENTS)]
                cut = np.random.randint(1, dim-1)
                child = np.concatenate((p1[:cut], p2[cut:]))
                if np.random.rand() < 0.01: # Mutasyon
                     idx = np.random.randint(0, dim)
                     child[idx] = 1 - child[idx]
                new_pop.append(child)
            pop = np.array(new_pop)
        return best_sol, best_score

class PSO:
    def __init__(self, max_iter):
        self.name = "PSO"
        self.max_iter = max_iter
    def run(self):
        particles = np.random.randint(0, 2, (N_AGENTS, dim))
        velocities = np.zeros((N_AGENTS, dim))
        pbest_pos = particles.copy()
        pbest_score = [float("inf")] * N_AGENTS
        gbest_pos, gbest_score = np.zeros(dim), float("inf")
        
        for t in range(self.max_iter):
            for i in range(N_AGENTS):
                fit = calculate_fitness(particles[i])
                if fit < pbest_score[i]: pbest_score[i], pbest_pos[i] = fit, particles[i].copy()
                if fit < gbest_score: gbest_score, gbest_pos = fit, particles[i].copy()
                
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = 0.5 * velocities[i] + 1.5 * r1 * (pbest_pos[i] - particles[i]) + 1.5 * r2 * (gbest_pos - particles[i])
                particles[i] = (np.random.rand(dim) < (1 / (1 + np.exp(-velocities[i])))).astype(int)
        return gbest_pos, gbest_score

# --- 5. TOPLU TEST DÖNGÜSÜ ---
final_results = []

print(f"ANALİZ BAŞLIYOR... Senaryolar: {ITERATION_SCENARIOS} İterasyon")
print("="*60)

for iter_count in ITERATION_SCENARIOS:
    print(f"\n>> SENARYO: {iter_count} İterasyon Test Ediliyor...")
    
    # Her senaryo için algoritmaları yeniden oluştur
    current_algos = [GWO(iter_count), GA(iter_count), PSO(iter_count)]
    
    for algo in current_algos:
        start_t = time.time()
        best_mask, best_cost = algo.run()
        elapsed = time.time() - start_t
        feat_count = np.sum(best_mask)
        
        final_results.append({
            "İterasyon": iter_count,
            "Algoritma": algo.name,
            "Maliyet (Cost)": round(best_cost, 4),
            "Seçilen Özellik": feat_count,
            "Süre (sn)": round(elapsed, 2)
        })
        print(f"   -> {algo.name} bitti. (Cost: {best_cost:.4f})")

# --- 6. SONUÇ TABLOSU ---
df_results = pd.DataFrame(final_results)
print("\n" + "="*60)
print("             DUYARLILIK ANALİZİ SONUÇLARI")
print("="*60)
print(df_results.to_string(index=False))
print("="*60)

df_results.to_csv("sonuc_tablosu.csv", index=False)