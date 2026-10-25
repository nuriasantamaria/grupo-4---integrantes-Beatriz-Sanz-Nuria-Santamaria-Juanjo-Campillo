def analytic_eigenvalues(I, twoT):
    # en ω = Ω_k e_k (Ω_k² = 2T/I_k): λ = 0 (a lo largo de e_k) y λ² = Ω_k² (I_k − I_i)(I_j − I_k) / (I_i I_j)
    I = np.asarray(I, dtype=float)
    lam = np.zeros((3, 3), dtype=complex)
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        mu = np.sqrt(complex(twoT / I[k] * (I[k] - I[i]) * (I[j] - I[k]) / (I[i] * I[j])))
        lam[k] = [0, mu, -mu]
    return lam
