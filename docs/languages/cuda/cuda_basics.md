---

marp: true
theme: gaia
paginate: true
highlight: nord



---


<style>
  section {
    background-color: lightblue;
  }
</style>


<style>
section.center {
  display: flex;
  justify-content: center;
  align-items: center;
  text-align: center;
  flex-direction: column;
}
</style>


<style>
section.small {
  font-size: 32px;
}
</style>


<!-- _class: center -->

# CUDA Fundamentals

--- 

# Part 1: GPU/CPU

 

---



 

<!-- _class: center -->

 




# 1.1 GPU/CPU

---

## CPU vs GPU

![w:700](./img/image.png)

GPUs rely more on latency hiding than CPUs to achieve high performance.



---

<!-- _class: small -->

## CPU and GPU architectural difference
 
| Feature          | CPU               | GPU                   |
| ---------------- | ----------------- | --------------------- |
| Threads          | Few               | Thousands             |
| Caches           | Large             | Small                 |
| Latency strategy | Minimize          | Hide                  |
| Context switch   | Expensive         | Cheap                 |
| Best for         | Low-latency tasks | High-throughput tasks |
| Memory bandwidth        | Low | High|

--- 

## Heterogeneous programming model 

```
┌──────────────┐    kernels / memcpy    ┌──────────────┐
│   CPU (Host) │ ───────────────────▶  │ GPU (Device) │
│  Control &   │                       │ Parallel     │
│  Launch      │ ◀───────────────────  │ Execution    │
└──────────────┘        results         └──────────────┘


```

CPU to manage and launch work while offloading massively parallel computations to the GPU for execution
 
---

```

void main(){
    float *a, *b, *out;
    float *d_a;
    a = (float*)malloc(sizeof(float) * N);
    // Allocate device memory for a
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);  
    vector_add<<<1,1>>>(out, d_a, b, N);
    // Cleanup after kernel execution
    cudaFree(d_a);
    free(a);
}
```

---

nvcc

- ``` __global__```

- ``` __device__ ```


gcc, cl.exe

- ``` __host__ ```


--- 

Triple angle brackets maker

---

## GPU properities

```
#include <stdio.h> 

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}
```
--- 

<!-- _class: small -->

## Compute capabilites
| Feature                     | Tesla C870 | Tesla C1060 | Tesla C2050 | Tesla K10 | Tesla K20 |
|----------------------------|------------|-------------|-------------|-----------|-----------|
| Compute Capability         | 1.0        | 1.3         | 2.0         | 3.0       | 3.5       |
| Max Threads per Thread Block | 512        | 512         | 1024        | 1024      | 1024      |
| Max Threads per SM         | 768        | 1024        | 1536        | 2048      | 2048      |
| Max Thread Blocks per SM   | 8          | 8           | 8           | 16        | 16        |



---

<!-- _class: center -->

# 1.2 Thread

---

# IDs

- blockIdx, threadIdx   <3D>
- blockDim, gridDim  <3D>

---

## Global Thread ID 

![w:800](https://developer-blogs.nvidia.com/wp-content/uploads/2017/01/Even-easier-intro-to-CUDA-image.png)

--- 

## Kernel execution time

---
<!-- _class: center -->

# 1.3 Memory

---

## Host and device memory transfer 

```
cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
```
- data transfer between CPU and GPU 

- We need to call `cudaDeviceSynchronize()` to ensure all GPU operations are completed before copying data from the GPU to the CPU; however, synchronization is not required when copying data from the CPU to the GPU.

- We use `cudaMalloc` and `cudaFree` to manage GPU memories.

---

## Memory brandwidth 


--- 

## Computational throughput


---

<!-- _class: center -->

# 1.4 Toolkit

---

## nvcc

`nvcc vector_add.cu -o vector_add`

---

## nvprof

`nvprof ./vector_add`

```
==6326== Profiling application: ./vector_add
==6326== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
 97.55%  1.42529s         1  1.42529s  1.42529s  1.42529s  vector_add(float*, float*, float*, int)
  1.39%  20.318ms         2  10.159ms  10.126ms  10.192ms  [CUDA memcpy HtoD]
  1.06%  15.549ms         1  15.549ms  15.549ms  15.549ms  [CUDA memcpy DtoH]

```

---

# Part 2: CUDA Libraries

---

## Thrust

Thrust is a powerful library of parallel algorithms and data structures.

---

## cub

CUB provides state-of-the-art, reusable software components for every layer of the CUDA programming model.


 

