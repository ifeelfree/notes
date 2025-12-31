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



# CUDA Threads

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

