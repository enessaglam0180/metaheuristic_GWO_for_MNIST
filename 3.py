import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import os 

N_AGENTS = 8       
K_NEIGHBORS = 5    
current_iter = 10  

print(">> Veri hazırlanıyor...")

try:
    
    df = pd.read_csv('mnist_test.csv', header=None)
    y = df.iloc[:, 0].values     
    x = df.iloc[:, 1:].values    
except FileNotFoundError:
    print("HATA: 'mnist_test.csv' dosyası bulunamadı! Lütfen proje klasörüne ekleyin.")
    exit()

# Runtime efficiency için subsampling
sample_size = 800
if len(x) > sample_size:
    x = x[:sample_size]
    y = y[:sample_size]

x = x / 255.0  

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
dim = X_train.shape[1]

print(f">> Veri Hazır: {dim} Özellik, {len(x)} Örnek işleniyor.")


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

# ALGORİTMALAR

class GWO:
    def __init__(self, iterations):
        self.name = "GWO (Kurtlar)"
        self.max_iter = iterations
        
    def run(self):
        wolves = np.random.randint(0, 2, (N_AGENTS, dim))
        alpha_pos, beta_pos, delta_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim)
        alpha_score, beta_score, delta_score = float("inf"), float("inf"), float("inf")
        history = []
        
        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)
            
            for i in range(N_AGENTS):
                fit = calculate_fitness(wolves[i])
                if fit < alpha_score: alpha_score, alpha_pos = fit, wolves[i].copy()
                if fit > alpha_score and fit < beta_score: beta_score, beta_pos = fit, wolves[i].copy()
                if fit > alpha_score and fit > beta_score and fit < delta_score: delta_score, delta_pos = fit, wolves[i].copy()
            
            history.append(alpha_score)
            
            for i in range(N_AGENTS):
                for j in range(dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    X1 = alpha_pos[j] - (2*a*r1 - a) * abs(2*r2*alpha_pos[j] - wolves[i,j])
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    X2 = beta_pos[j] - (2*a*r1 - a) * abs(2*r2*beta_pos[j] - wolves[i,j])
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    X3 = delta_pos[j] - (2*a*r1 - a) * abs(2*r2*delta_pos[j] - wolves[i,j])
                    
                    X_avg = (X1 + X2 + X3) / 3
                    prob = 1 / (1 + np.exp(-10 * (X_avg - 0.5)))
                    wolves[i, j] = 1 if np.random.rand() < prob else 0
                    
        return history, int(np.sum(alpha_pos))

class GA:
    def __init__(self, iterations):
        self.name = "GA (Genetik)"
        self.max_iter = iterations
        self.mutation_rate = 0.01
        
    def run(self):
        population = np.random.randint(0, 2, (N_AGENTS, dim))
        best_sol = np.zeros(dim)
        best_score = float("inf")
        history = []
        
        for t in range(self.max_iter):
            scores = []
            for i in range(N_AGENTS):
                fit = calculate_fitness(population[i])
                scores.append(fit)
                if fit < best_score:
                    best_score = fit
                    best_sol = population[i].copy()
            
            history.append(best_score)
            
            new_pop = [best_sol] 
            while len(new_pop) < N_AGENTS:
                idx1, idx2 = np.random.randint(0, N_AGENTS), np.random.randint(0, N_AGENTS)
                parent1 = population[idx1] if scores[idx1] < scores[idx2] else population[idx2]
                
                idx1, idx2 = np.random.randint(0, N_AGENTS), np.random.randint(0, N_AGENTS)
                parent2 = population[idx1] if scores[idx1] < scores[idx2] else population[idx2]
                
                cut = np.random.randint(1, dim-1)
                child = np.concatenate((parent1[:cut], parent2[cut:]))
                
                for j in range(dim):
                    if np.random.rand() < self.mutation_rate:
                        child[j] = 1 - child[j]
                new_pop.append(child)
            
            population = np.array(new_pop)
            
        return history, int(np.sum(best_sol))

class PSO:
    def __init__(self, iterations):
        self.name = "PSO (Kuşlar)"
        self.max_iter = iterations
        
    def run(self):
        particles = np.random.randint(0, 2, (N_AGENTS, dim))
        velocities = np.zeros((N_AGENTS, dim))
        pbest_pos = particles.copy()
        pbest_score = np.full(N_AGENTS, float("inf"))
        gbest_pos = np.zeros(dim)
        gbest_score = float("inf")
        history = []
        
        for t in range(self.max_iter):
            w, c1, c2 = 0.5, 1.5, 1.5
            
            for i in range(N_AGENTS):
                fit = calculate_fitness(particles[i])
                if fit < pbest_score[i]:
                    pbest_score[i] = fit
                    pbest_pos[i] = particles[i].copy()
                if fit < gbest_score:
                    gbest_score = fit
                    gbest_pos = particles[i].copy()
            
            history.append(gbest_score)
            
            for i in range(N_AGENTS):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = w * velocities[i] + c1 * r1 * (pbest_pos[i] - particles[i]) + c2 * r2 * (gbest_pos - particles[i])
                prob = 1 / (1 + np.exp(-velocities[i]))
                particles[i] = (np.random.rand(dim) < prob).astype(int)
                
        return history, int(np.sum(gbest_pos))

# GUI & Simülasyon
fig, ax = plt.subplots(figsize=(12, 7)) 
plt.subplots_adjust(bottom=0.25)

def run_simulation(val):
    global current_iter
    
    try:
        if isinstance(val, str): 
             iters = int(text_box.text)
        elif hasattr(val, 'text'):
             iters = int(val)
        else: 
             iters = int(text_box.text)
            
        if iters <= 0: raise ValueError
        current_iter = iters
    except ValueError:
        print("Lütfen geçerli bir tam sayı gir!")
        return

    print(f"\n--- Simülasyon Başlıyor (İterasyon: {current_iter}) ---")
    ax.clear()
    ax.set_title(f'Yarış Sürüyor... İterasyon: {current_iter} (Lütfen Bekleyiniz)', color='red')
    plt.draw()
    plt.pause(0.01) 
    
    algos = [GWO(current_iter), GA(current_iter), PSO(current_iter)]
    colors = ['r', 'g', 'b']
    
    best_algo_name = ""
    best_cost = float("inf")
    summary_text = "SKOR TABLOSU:\n" + "-"*30 + "\n"
    
    
    run_results = []
    
    for algo, col in zip(algos, colors):
        print(f"> {algo.name} Çalışıyor...")
        start_time = time.time()
        history, feat_cnt = algo.run()
        elapsed = time.time() - start_time
        
        final_cost = history[-1]
        
        if final_cost < best_cost:
            best_cost = final_cost
            best_algo_name = algo.name
            
        label_txt = f"{algo.name}"
        print(f"  -> Bitti. Cost={final_cost:.4f} | Süre={elapsed:.2f}s | Özellik={feat_cnt}")
        
        summary_text += f"{algo.name:<15} : {final_cost:.4f} (Seçim: {feat_cnt})\n"
        
        ax.plot(history, label=label_txt, color=col, marker='o', markersize=4, linewidth=2)
        
    
        run_results.append({
            "Tarih": time.strftime("%Y-%m-%d %H:%M:%S"),
            "İterasyon": current_iter,
            "Algoritma": algo.name,
            "Cost (Hata+Oran)": final_cost,
            "Seçilen Özellik": feat_cnt,
            "Süre (sn)": round(elapsed, 2)
        })

    print(f"\n*** KAZANAN: {best_algo_name} ***")
    
    ax.set_title(f'SONUÇ: Kazanan {best_algo_name} (Cost: {best_cost:.4f})', fontsize=12, fontweight='bold', color='darkblue')
    
    ax.set_xlabel('İterasyon')
    ax.set_ylabel('Cost')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, fontfamily='monospace')
    
    plt.draw()


    excel_filename = "simulasyon_sonuclari.xlsx"
    df_new = pd.DataFrame(run_results)
    
    if os.path.exists(excel_filename):
        try:
            
            df_existing = pd.read_excel(excel_filename)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(excel_filename, index=False)
            print(f">> Sonuçlar başarıyla '{excel_filename}' dosyasına EKLENDİ.")
        except Exception as e:
            print(f"!! Excel'e ekleme hatası: {e}. Dosya açık olabilir mi?")
    else:
   
        df_new.to_excel(excel_filename, index=False)
        print(f">> Yeni dosya oluşturuldu ve sonuçlar '{excel_filename}' dosyasına kaydedildi.")


axbox = plt.axes([0.15, 0.1, 0.2, 0.05])
text_box = TextBox(axbox, 'Max İter:', initial=str(current_iter))
text_box.on_submit(run_simulation) 

axbtn = plt.axes([0.4, 0.1, 0.2, 0.05])
btn = Button(axbtn, 'Yarışı Başlat', color='lightblue', hovercolor='0.975')
btn.on_clicked(run_simulation) 


plt.show()