

import numpy as np

# PROGRAM PENYELESAIAN SPL METODE GAUSS-SEIDEL

# 1. Input matriks koefisien dan konstanta
A = np.array([
    [10, -1,  2],
    [-1, 11, -1],
    [ 2, -1, 10]
], dtype=float)

b = np.array([6, 25, -11], dtype=float)

# 2. Menentukan parameter
n = len(b)
x = np.zeros(n)
toleransi = 0.000001
maks_iterasi = 100

# 3. Validasi matriks
if A.shape != (n, n):
    raise ValueError("Matriks A harus berbentuk persegi.")

if np.any(np.diag(A) == 0):
    raise ValueError("Elemen diagonal matriks tidak boleh nol.")

# 4. Menampilkan data awal
print("=== METODE GAUSS-SEIDEL ===")
print("\nMatriks koefisien A:")
print(A)
print("\nVektor konstanta b:")
print(b)
print("\nNilai awal variabel:", x)
print("Toleransi:", toleransi)
print("Maksimum iterasi:", maks_iterasi)

# 5. Proses iterasi Gauss-Seidel
konvergen = False

for iterasi in range(1, maks_iterasi + 1):
    x_lama = x.copy()

    for i in range(n):
        jumlah = 0.0

        for j in range(n):
            if j != i:
                jumlah += A[i, j] * x[j]

        x[i] = (b[i] - jumlah) / A[i, i]

    # 6. Menghitung error maksimum
    error = np.max(np.abs(x - x_lama))

    # 7. Menampilkan hasil setiap iterasi
    print(f"\nIterasi ke-{iterasi}")
    for i in range(n):
        print(f"x{i+1} = {x[i]:.8f}")
    print(f"Error maksimum = {error:.8f}")

    # 8. Memeriksa toleransi
    if error < toleransi:
        konvergen = True
        break

# 9. Menampilkan hasil akhir
print("\n=== HASIL AKHIR ===")

if konvergen:
    print("Status: Konvergen")
else:
    print("Status: Belum konvergen dalam batas iterasi")

for i in range(n):
    print(f"Nilai x{i+1} = {x[i]:.6f}")

print("Jumlah iterasi:", iterasi)

# 10. Memeriksa hasil dengan substitusi ke SPL
print("\n=== PEMERIKSAAN SPL ===")
hasil = A @ x

for i in range(n):
    print(
        f"Persamaan {i+1}: "
        f"hasil = {hasil[i]:.6f}, "
        f"konstanta = {b[i]:.6f}"
    )
