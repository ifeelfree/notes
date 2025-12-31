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



# CUDA Extensions



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

