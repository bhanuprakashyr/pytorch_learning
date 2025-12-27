# PyTorch Learning

## Tensors

We use tensors to encode the inputs and outputs of a model, as well as the model's parameters.

Tensors are similar to NumPy's ndarrays, except that tensors can run on GPUs or other hardware accelerators.

- CPU → general-purpose brain
- GPU → parallel math engine (accelerator) (5 to 10x than CPU's) - GPUs don't make things faster by default. They make parallel math faster.
- TPU / NPU / ANE → ML-specialized accelerators

### A PyTorch Tensor:

- Can live on CPU or GPU
- Has a `.device` attribute
- Dispatches operations to the right hardware
- Over 1200 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing)
- By default, tensors are created on the CPU

## Datasets & DataLoaders

Two important libraries:

- `torch.utils.data.Dataset` that allow you to use pre-loaded datasets as well as your own data. Dataset stores the samples and their corresponding labels.
- `torch.utils.data.DataLoader` wraps an iterable around the Dataset to enable easy access to the samples.

### What DataLoader actually does?

DataLoader is not about access. It's about efficient training.

It adds four critical capabilities.

#### 1. Batching (this is the big one)

Instead of this (slow and noisy):

```python
for i in range(len(training_data)):
    img, label = training_data[i]
    model(img)
```

You get this:

```python
for imgs, labels in train_dataloader:
    model(imgs)
```

Now:

- `imgs.shape = (64, 1, 28, 28)`
- GPU gets 64 samples at once
- Massive speedup
- Stable gradients
- Neural networks expect batches.

#### 2. Shuffling (avoids learning garbage patterns)

`shuffle=True` means:

- Data order is randomized every epoch
- Model doesn't learn "position-based" bias
- Improves generalization
- Manual indexing almost always forgets this.

#### 3. Epoch abstraction (clean training loops)

With DataLoader:

```python
for epoch in range(epochs):
    for batch in train_dataloader:
        ...
```

This is:

- Clean
- Idiomatic
- Correct

Without DataLoader, you reimplement half of ML engineering badly.

#### 4. Performance & scaling (hidden superpower)

DataLoader can:

- Load data in parallel (`num_workers`)
- Pin memory
- Stream data efficiently
- Scale to datasets that don't fit in memory

Indexing cannot do this.

Why GPUs need DataLoader

GPUs want:

- Large contiguous tensors
- Few Python calls
- Minimal overhead

DataLoader provides exactly that.

## Transforms

Data does not always come in its final processed form that is required for training machine learning algorithms. We use transforms to perform some manipulation of the data and make it suitable for training.

## Important Pytorch modules

| Priority | Package / Module        | Status            | Primary Purpose |
|---------:|-------------------------|-------------------|-----------------|
| 1        | torch                   | Core / Mandatory  | Tensors, autograd, NN training, GPU |
| 2        | torchvision             | Actively Used     | Computer vision datasets, transforms, models |
| 3        | torchaudio              | Actively Used     | Audio & speech processing |
| 4        | torch.utils.data        | Core / Mandatory  | Data loading, batching, pipelines |
| 5        | torch.nn                | Core / Mandatory  | Layers, losses, model definitions |
| 6        | torch.optim             | Core / Mandatory  | Optimizers (SGD, Adam, etc.) |
| 7        | torch_geometric         | Actively Used     | Graph Neural Networks |
| 8        | torchmetrics            | Actively Used     | Standard ML evaluation metrics |
| 9        | pytorch-lightning       | Common (Optional) | Training loop abstraction |
| 10       | functorch               | Advanced / Niche  | vmap, higher-order autodiff |
| 11       | torchdata               | Limited Adoption  | Data pipelines (inconsistent use) |
| 12       | torch_sparse            | Niche             | Sparse tensor ops (mostly GNNs) |
| 13       | torch_scatter           | Niche             | Scatter / reduce ops (GNN internals) |
| 14       | torch_cluster           | Niche             | Graph clustering & sampling |
| 15       | torchdiffeq             | Research-only     | Neural ODEs |
| 16       | torchtext               | Legacy            | NLP preprocessing (largely replaced) |

