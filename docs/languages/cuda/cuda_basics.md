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

## GPU/CPU
## GPU

---

 



 

<!-- _class: center -->

 




# GPU/CPU

---

## What makes GPU different from CPU?

![w:700](./img/image.png)

GPUs rely more on latency hiding than CPUs to achieve high performance.



---

<!-- _class: small -->

## GPU and CPU architectural difference
 
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

__cuda demo__: the GPU kenerl is marked with the `__global__` qualifier.

```

__global__ void vector_add(const float *A, const float *B, float *C, int ds){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx < ds)
    C[idx] = A[idx] + B[idx];
}

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

GPU kernel is compiled with __nvcc__

- ``` __global__```: running kernels from the host

- ``` __device__ ```: running kernels from the device


CPU codes are compiled with __gcc__, __cl.exe__

- ``` __host__ ```: running codes from the host

 
---

## Other examples

- [matrix multiplication.cu](./code/matrix_mul.cu)

---


<!-- _class: center -->

# GPU


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

- [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus?utm_source=chatgpt.com)

- [Legacy CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus/legacy)
 

- [GPU Comparison Guides](https://www.rightnowai.co/guides/gpu-comparison?utm_source=chatgpt.com)
 
 

