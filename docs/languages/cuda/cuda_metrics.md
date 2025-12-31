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

# CUDA Metrics

 
---

# Throughput

# Bandwidth 
 

---

# Memory Throughput

Refers to how fast data can be read from or written to memory (global, shared, or texture memory)


---

# Compute Throughput

Refers to the rate at which the GPU can execute operations (like floating-point operations, matrix multiplications, etc.).

---

# Throughput in Data Transfer

When moving data between different parts of the system (like CPU to GPU or between different levels of GPU memory), data transfer throughput measures the efficiency and speed of that movement.

---

# Throughput vs Latency

Latency is the time it takes to complete a single operation, while throughput is the rate at which operations are completed.

Ideally, you want high throughput and low latency for maximum performance. However, there’s often a tradeoff.

---

# Bandwidth 

---

