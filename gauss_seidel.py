


import numpy as np

# PROGRAM PENYELESAIAN SPL METODE GAUSS-SEIDEL

# 1. Memasukkan matriks koefisien dan konstanta
A = np.array([
    [10, -1, 2],
    [-1, 11, -1],
    [2, -1, 10]
], dtype=float)

b = np.array([6, 25, -11], dtype=float)

# 2. Menentukan parameter
n = len(b)
x = np.zeros(n)
toleransi = 0.000001
maks_iterasi = 100

# 3. Memeriksa matriks
if A.shape != (n, n):
    raise ValueError("Matriks A harus berbentuk persegi.")

if np.any(np.diag(A) == 0):
    raise ValueError("Elemen diagonal tidak boleh nol.")

# 4. Menampilkan data awal
print("=== METODE GAUSS-SEIDEL ===")
print("\nMatriks A:")
print(A)
print("\nVektor b:")
print(b)
print("\nNilai awal:", x)
print("Toleransi:", toleransi)
print("Maksimum iterasi:", maks_iterasi)

# 5. Proses iterasi
konvergen = False

for iterasi in range(1, maks_iterasi + 1):
    x_lama = x.copy()

    for i in range(n):
        jumlah = 0.0

        for j in range(n):
            if j != i:
                jumlah += A[i, j] * x[j]

        x[i] = (b[i] - jumlah) / A[i, i]

    # 6. Menghitung error
    error = np.max(np.abs(x - x_lama))

    print(f"\nIterasi ke-{iterasi}")
    for i in range(n):
        print(f"x{i+1} = {x[i]:.8f}")

    print(f"Error maksimum = {error:.8f}")

    # 7. Memeriksa toleransi
    if error < toleransi:
        konvergen = True
        break

# 8. Menampilkan hasil akhir
print("\n=== HASIL AKHIR ===")
print("Status:", "Konvergen" if konvergen else
      "Belum konvergen")

for i in range(n):
    print(f"Nilai x{i+1} = {x[i]:.6f}")

print("Jumlah iterasi:", iterasi)

# 9. Pemeriksaan hasil
print("\n=== PEMERIKSAAN SPL ===")
hasil = A @ x
print("Hasil A @ x:", hasil)
print("Konstanta b:", b)
print("Residual:", hasil - b)

# 10. Solusi pembanding
print("\n=== SOLUSI PEMBANDING ===")
solusi = np.linalg.solve(A, b)

for i in range(n):
    print(f"x{i+1} = {solusi[i]:.6f}")
