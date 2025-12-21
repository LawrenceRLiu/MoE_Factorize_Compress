# **RESEARCH SPEC: Shared-Core Compression for MoE LLMs**

## **1\. Project Overview**

Goal: Develop a new compressed architecture for MoE models that significantly reduce the parameter count by exploriting expert redundancy. We expect that the active parameter count will slightly increase, however we are willing to take that “hit”.  
Core Hypothesis: Experts in a given layer share a significant underlying subspace. We can factorize expert weights into a single Shared Core (common to all experts) and lightweight Identity \+ Low-Rank Wrappers (specialized per expert).  
**Methodology:**

1. **Formulation:** Decompose $W\_{expert} \\approx (I \+ L\_{out}) W\_{core} (I \+ L\_{in})$.  
2. **Initialization:** Parallelized "Zero-Shot" reconstruction using Adam to minimize the L2 norm  
3. **Refinement:** Knowledge Distillation (Teacher-Student) on high-quality tokens.

As a proof of concept, we will compress Qwen-3-30B-A3B. We shall target \~20-30% compression ratio.

## **2\. Mathematical Formulation**

For a specific layer $l$ and projection $p \\in \\{\\text{Gate, Up, Down}\\}$, let the original expert weights be $W\_e \\in \\mathbb{R}^{d\_{out} \\times d\_{in}}$ for $e \\in \\{1 \\dots E\\}$.  
We approximate $W\_e$ as:  
$$\\hat{W}\_{e} \= (I \+ U\_e^{out} V\_e^{out^T}) \\cdot C \\cdot (I \+ U\_e^{in} V\_e^{in^T})$$  
Where:

* **$C \\in \\mathbb{R}^{d\_{out} \\times d\_{in}}$**: The **Shared Core**, learned but identical for all experts in the layer.  
* **$U\_e^{in}, V\_e^{in} \\in \\mathbb{R}^{d\_{in} \\times r}$**: Input low-rank adapter (rank $r \\ll d\_{in}$).  
* **$U\_e^{out}, V\_e^{out} \\in \\mathbb{R}^{d\_{out} \\times r}$**: Output low-rank adapter (rank $r \\ll d\_{out}$).

**Why this works:** The formulation $I \+ UV^T$ acts as a "residual rotation," allowing each expert to slightly shift the input/output manifold of the shared core without learning a full dense matrix.

## **3\. Infrastructure Strategy**

We utilize a heterogeneous cluster: **2x A100 (80GB)** and **6x A6000 (48GB)**.

Our algorithms will work in two phases:

**Phase 1:  Zero-Shot Init,** All 8 GPUs will independently (parallel layer/projection wise) work to compress the model based on the algorithm detailed in 4.1. 

**Phase 2\. Recovery Distillation with Async Evaluatio**n. To recover performance, we will conduct knowledge distillation between the original and compressed model. To evaluate the checkpoints, we will save some of the GPUs to asynchronously evaluate the checkpoints.

## **4\. Implementation Specification**

Implementation will be done according to best research code design practices, all code shall be commented and readable. As much of the implementation should rely on open-source libraries as possible, i.e. minimal original code should be written.

A conda python 3.14 venv \`MoE\_Compress\` has already been created with \`transformers, torch, torchvision, accelerate, deepspeed, lm\_eval, datasets, hydra, wandb\` installed. Additional packages can be installed as needed. Hydra will be used for configuration management

All source core should be written in the \`src/\` directory, scripts in the \`scripts/\` directory, configuration yamls/jsons should be saved in the \`conf/\`, and the model artifacts/checkpoints should be saved in the \`models\` directory, with subdirectories for each specific base model used as the starting point.

### **4.1. Algorithm: Zero-Shot Initialization**

We will parallelize execution on the 8 GPUs. Each GPU should handle one projection class for one layer.  
**Algorithm Steps:**

1. **Load Layer:** Load original weights $W\_e$ for the assigned layer.  
2. **Compute Mean:** $C\_{init} \= \\frac{1}{E} \\sum\_e W\_e$.  
3. **Intialize the low rank matrices** LoRA like: $U\_e^{in}=0$, $U\_e^{out}=0$, $V\_e^{in}\\sim \\mathcal{N}(0,1)$, $V\_e^{out}}\\sim \\mathcal{N}(0,1)$.  
4. **Refinement:**  
   * Run Adam/AdamW on the wrapper and core to minimize $\\sum\_e \\|W\_e \- \\hat{W}\_e\\|\_F^2$ until convergence or for a certain number of steps (hyperparameter specified in config)  
5. **Save:** Serialize the compressed state dict.

### **4.2. Algorithm: Distillation**

**Trainer Setup:**

* Use transformers.Trainer with a custom compute\_loss.  
* **Teacher:** Load the original MoE (Frozen) in 4-bit or 8-bit if needed to save VRAM, or bfloat16 if space permits.  
* **Student:** Compressed  initialized with compressed weights from Phase 1\.  
* **Loss:** KL Divergence on logits.   
* **Data:** Use \`HuggingFaceFW/fineweb-edu\` or other high quality pretraining dataset (streaming=True).

### **4.3. Algorithm: Async Evaluation**

**Logic:**

* Save a certain number of GPUs to asynchronously evaluate the model. Should start by evaluating the original model to form a baseline.  
* Infinite loop: while True: sleep(60)  
* Check output\_dir for new subfolders (e.g., checkpoint-1000).  
* If found and not yet evaluated:  
  * Load model onto the Async GPUs  
  * Run lm\_eval (Harness) tasks: wikitext, mmlu, gsm8k, and other common pretraining evaluation tasks (this should also be something which can be specified by the config)  
  * Log results to WandB project moe-compression.  
  * Mark checkpoint as evaluated.

