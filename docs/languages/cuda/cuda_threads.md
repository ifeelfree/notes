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

# CUDA Threads



---

# IDs

- blockIdx, threadIdx   <3D>
- blockDim, gridDim  <3D>

---

## Global Thread ID 

![w:800](https://developer-blogs.nvidia.com/wp-content/uploads/2017/01/Even-easier-intro-to-CUDA-image.png)


--- 
# Warps

- A warp is excuted physically in parallel (SIMD) on a multiprocessor.

- A thread block consists of warps.


---

# Launch Configuration

- instructions are issued in order

- a thread stalls when one of the operands isn't ready

- latency is hiden by switching threads
  - GMEM latency
  - arithmetic latency 
---

