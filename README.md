## Tensors 

We use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.
Tensors are similar to NumPy’s ndarrays, except that tensors can run on GPUs or other hardware accelerators.

CPU → general-purpose brain - 
GPU → parallel math engine (accelerator) (5 to 10x than CPU's) - GPUs don’t make things faster by default. They make parallel math faster.
TPU / NPU / ANE → ML-specialized accelerators

### A PyTorch Tensor:

Can live on CPU or GPU
Has a .device attribute

Dispatches operations to the right hardware

Over 1200 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing)

By default, tensors are created on the CPU

### Datasets & DataLoaders
2 important Library 
torch.utils.data.Dataset that allow you to use pre-loaded datasets as well as your own data. 
Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

torch.utils.data.DataLoader
wraps an iterable around the Dataset to enable easy access to the samples.

### What DataLoader actually does ?

DataLoader is not about access.
It’s about efficient training.

It adds four critical capabilities.

### 1️⃣ Batching (this is the big one)

Instead of this (slow and noisy):

for i in range(len(training_data)):
    img, label = training_data[i]
    model(img)
You get this:

for imgs, labels in train_dataloader:
    model(imgs)
Now:

imgs.shape = (64, 1, 28, 28)

GPU gets 64 samples at once

Massive speedup

Stable gradients

Neural networks expect batches.

### 2️⃣ Shuffling (avoids learning garbage patterns)

shuffle=True
Means:

Data order is randomized every epoch

Model doesn’t learn “position-based” bias

Improves generalization

Manual indexing almost always forgets this.

### 3️⃣ Epoch abstraction (clean training loops)

With DataLoader:

for epoch in range(epochs):
    for batch in train_dataloader:
        ...
This is:

Clean

Idiomatic

Correct

Without DataLoader, you reimplement half of ML engineering badly.

### 4️⃣ Performance & scaling (hidden superpower)

DataLoader can:

Load data in parallel (num_workers)

Pin memory

Stream data efficiently

Scale to datasets that don’t fit in memory

Indexing cannot do this.

Why GPUs need DataLoader

GPUs want:

Large contiguous tensors

Few Python calls

Minimal overhead

DataLoader provides exactly that.


Transforms
Data does not always come in its final processed form that is required for training machine learning algorithms. We use transforms to perform some manipulation of the data and make it suitable for training.
