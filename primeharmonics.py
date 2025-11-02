import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.stats import linregress
import mpmath
import matplotlib.pyplot as plt
from scipy import sparse

# Config
N = 24000  # Proven; set 48000 in Colab
P = 48049
NUM_ZEROS = 100
s = 0.5

# Totient
def totient_sieve(max_n):
    phi = np.arange(max_n + 1)
    for i in range(2, max_n + 1):
        if phi[i] == i:
            for j in range(i, max_n + 1, i):
                phi[j] = phi[j] // i * (i - 1)
    return phi

phi = totient_sieve(N)

# Exact block kernel (dot for cos, no FFT approx)
def build_kernel_blocked(N, P, phi, s, block_size=3000):
    K_blocks = []
    print("Building kernel blocks...")
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        i_block = np.arange(start, end)
        j_arr = np.arange(N)
        cos_block = np.cos(2 * np.pi * i_block[:, np.newaxis] * j_arr[np.newaxis, :] / P)
        phi_j = phi[j_arr + 1] / (j_arr + 1)**s
        kernel_block = cos_block * phi_j[np.newaxis, :]
        K_blocks.append(kernel_block)
        print(f"Block {start//block_size +1} done ({end-start} rows)")
    return np.vstack(K_blocks)

K = build_kernel_blocked(N, P, phi, s)

print("Kernel built—sparse Dirac...")

D_dense = np.block([[np.zeros((N, N)), K], [K.T, np.zeros((N, N))]]) + 1e-4 * np.eye(2 * N)
D_sparse = sparse.csr_matrix(D_dense)

print("Sparse ready—eigs...")
evals, _ = eigsh(D_sparse, k=400, which='LM', tol=1e-6, maxiter=3000, ncv=500)
lambdas_pos = np.sort(np.abs(evals[evals > 1e-4]))[:NUM_ZEROS]

gamma_n = np.array([float(mpmath.zetazero(k).imag) for k in range(1, NUM_ZEROS + 1)])
num_use = min(len(lambdas_pos), len(gamma_n))
lambdas_pos = lambdas_pos[:num_use]
gamma_n = gamma_n[:num_use]
slope, intercept, r_value, _, _ = linregress(lambdas_pos, gamma_n)
lambdas_scaled = slope * lambdas_pos + intercept
corr = r_value**2
deltas = np.diff(lambdas_scaled)
mean_delta = np.mean(deltas)
std_delta = np.std(deltas)

print(f"N={N}, P={P}, s={s}: Corr={corr:.4f} ({corr*100:.2f}%), Mean Δ={mean_delta:.2f}, Std Δ={std_delta:.2f}")
print(f"Used {num_use} eigs for fit (target {NUM_ZEROS})")
print("First 5 γ_n:", gamma_n[:5])
print("First 5 scaled λ_n:", lambdas_scaled[:5])
print("First 5 Δ:", deltas[:5])

plt.figure(figsize=(10, 6))
plt.scatter(gamma_n, lambdas_scaled, alpha=0.7, label=f'λ_n vs γ_n (R²={corr:.4f})')
plt.plot(gamma_n, slope * lambdas_pos + intercept, 'r--', label='Fit line')
plt.xlabel('True Zeros γ_n'); plt.ylabel('Scaled Eigs λ_n')
plt.legend(); plt.title('Modular Resonance Alignment')
plt.savefig('rh_alignment.png', dpi=300); plt.show()

np.savetxt('lambdas_vs_zeros.csv', np.column_stack([gamma_n, lambdas_scaled]), 
           header='gamma_n,lambda_scaled', delimiter=',', comments='')

print("Done! Check rh_alignment.png and lambdas_vs_zeros.csv")
