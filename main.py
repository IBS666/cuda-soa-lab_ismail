from fastapi import FastAPI, UploadFile, HTTPException
import numpy as np
from numba import cuda
import time

app = FastAPI()

# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# CUDA kernel pour addition de matrices
@cuda.jit
def matrix_add_kernel(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        C[i, j] = A[i, j] + B[i, j]

# Endpoint /add
@app.post("/add")
async def add_matrices(file_a: UploadFile, file_b: UploadFile):
    # Charger matrices
    try:
        matrix_a = np.load(file_a.file)['arr_0'].astype(np.float32)
        matrix_b = np.load(file_b.file)['arr_0'].astype(np.float32)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid .npz file")

    # Vérifier dimensions
    if matrix_a.shape != matrix_b.shape:
        raise HTTPException(status_code=400, detail="Matrices must have same shape")

    # Copier sur GPU
    A_gpu = cuda.to_device(matrix_a)
    B_gpu = cuda.to_device(matrix_b)
    C_gpu = cuda.device_array_like(matrix_a)

    # Config threads et blocks
    threadsperblock = (16, 16)
    blockspergrid_x = (matrix_a.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (matrix_a.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Lancer kernel et mesurer temps
    start = time.perf_counter()
    matrix_add_kernel[blockspergrid, threadsperblock](A_gpu, B_gpu, C_gpu)
    cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "matrix_shape": list(matrix_a.shape),
        "elapsed_time": elapsed,
        "device": "GPU"
    }


