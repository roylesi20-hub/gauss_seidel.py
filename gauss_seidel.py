
import numpy as np

def gauss_seidel(A, b, toleransi=1e-6, maks_iterasi=100):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = len(b)

    if A.shape != (n, n):
        raise ValueError("Matriks A harus berbentuk persegi.")

    if np.any(np.diag(A) == 0):
        raise ValueError("Elemen diagonal tidak boleh nol.")

    x = np.zeros(n)

    print("=== METODE GAUSS-SEIDEL ===")

    for iterasi in range(1, maks_iterasi + 1):
        x_lama = x.copy()

        for i in range(n):
            jumlah = sum(
                A[i, j] * x[j]
                for j in range(n) if j != i
            )
            x[i] = (b[i] - jumlah) / A[i, i]

        error = np.max(np.abs(x - x_lama))

        print(f"Iterasi {iterasi}: {x}, Error: {error:.8f}")

        if error < toleransi:
            print("\nKonvergen!")
            break
    else:
        print("\nBatas iterasi tercapai.")

    print("\n=== HASIL AKHIR ===")
    for i, nilai in enumerate(x, start=1):
        print(f"x{i} = {nilai:.6f}")

    print("Jumlah iterasi:", iterasi)
    print("Pemeriksaan A @ x:", A @ x)
    print("Konstanta b:", b)

    return x


# Contoh sistem persamaan linear
A = [
    [10, -1, 2],
    [-1, 11, -1],
    [2, -1, 10]
]

b = [6, 25, -11]

# Menjalankan algoritma
solusi = gauss_seidel(A, b)
