# CM-IBQ (Cross-Modality Information Bottleneck Quantization)

This document summarises how to integrate and train the CM-IBQ components that
ship with the LLaVA repository.

## 1. Overview

The CM-IBQ framework quantises the multimodal fusion bottleneck of a
vision-language model (VLM) by directly optimising a rate–distortion objective
via the information bottleneck (IB) principle.  The Python implementation lives
under [`llava/model/quantization/cm_ibq.py`](../llava/model/quantization/cm_ibq.py)
and exposes the following modules:

- `ImportanceEstimationNetwork` – learns task-aware importance scores that
  approximate the contribution of each latent channel to the downstream task.
- `BitAllocationNetwork` – predicts a differentiable per-channel bit allocation
  that respects a global bit budget.
- `DifferentiableQuantizer` – performs straight-through differentiable
  quantisation with per-channel precision.
- `IBQuantizedLayer` – a drop-in information bottleneck layer wrapping the three
  components above.
- `stage_one_pretrain` and `stage_two_finetune` – helper routines that implement
  the two-stage progressive training recipe.

All components are configured via a single `CMIBQConfig` dataclass.

## 2. Wiring CM-IBQ into a VLM

1. **Instantiate the layer** in the model file that constructs your multimodal
   projector or fusion module:

   ```python
   from llava.model.quantization import CMIBQConfig, IBQuantizedLayer

   config = CMIBQConfig(
       feature_dim=fused_dim,
       task_dim=task_embedding_dim,
       hidden_dim=1024,
       latent_dim=fused_dim,
       min_bits=2,
       max_bits=8,
       stage1_bits=8,
       stage2_bits=4,
   )
   self.cm_ibq = IBQuantizedLayer(config)
   ```

2. **Call the layer in the forward pass**.  The layer requires the fused
   features and a task embedding.  It returns the quantised latent, the
   reconstruction and diagnostic tensors that must be fed back to the training
   helpers:

   ```python
   def forward(self, vision_features, text_features, task_embedding, target_bits=None):
       quantised, recon, mu, logvar, bit_allocation = self.cm_ibq(
           vision_features,
           task_embedding,
           target_bits=target_bits,
       )
        return {
            "quantised": quantised,
            "reconstructed": recon,
            "mu": mu,
            "logvar": logvar,
            "bit_allocation": bit_allocation,
            "text_features": text_features,
            "features": vision_features,
        }
    ```

    During stage two the training helper expects the host model to surface these
    diagnostics under the `cm_ibq` key in its output dictionary.

## 3. Training Pipeline

### Stage 1 – Bottleneck Shaping & Importance Pre-training

```python
from torch.optim import AdamW

layer = model.cm_ibq  # the IBQuantizedLayer
optimiser = AdamW(layer.parameters(), lr=1e-4)

stage_one_pretrain(
    layer,
    dataloader=ib_pretrain_loader,
    optimiser=optimiser,
    device=torch.device("cuda"),
    gamma=1.0,
)
```

During this stage the rest of the VLM should remain frozen.  The dataloader must
yield dictionaries containing `features` (the fused modality features before
quantisation) and `task_embedding`.

### Stage 2 – End-to-End Task-Aware Joint Optimisation

```python
optimiser = AdamW(model.parameters(), lr=5e-5)

stage_two_finetune(
    layer,
    model,
    dataloader=task_loader,
    optimiser=optimiser,
    device=torch.device("cuda"),
    anneal=True,
)
```

The host model's forward method should return a dictionary containing a scalar
`loss` tensor for the downstream task and a nested `cm_ibq` dictionary with the
diagnostics produced above:

```python
def forward(self, batch, target_bits=None):
    cm_ibq_outputs = self.quantised_bridge(
        batch["vision_features"],
        batch["text_features"],
        batch["task_embedding"],
        target_bits=target_bits,
    )
    logits = self.head(cm_ibq_outputs["quantised"], batch)
    task_loss = self.task_loss_fn(logits, batch["labels"])
    return {"loss": task_loss, "cm_ibq": cm_ibq_outputs}
```

The stage-two dataloader should emit the usual training batches for the main
VLM objective (e.g. VQA, captioning).  The host model must accept a
`target_bits` argument and return a dictionary with a `loss` tensor and a
`cm_ibq` diagnostics dictionary as described above.  The helper automatically
applies progressive bit-width annealing according to the schedule specified in
`CMIBQConfig`.

## 4. Loss Terms

The helper functions internally optimise the composite loss described in the
specification:

- **Information Bottleneck Loss** – reconstruction and KL divergence.
- **Bitrate Regularisation** – ensures the learned bit allocation respects the
  target average bit budget.
- **Cross-Modal Alignment** – InfoNCE-style loss between quantised vision and
  text representations to prevent modality drift.

The relative influence of each term can be adjusted via the fields in
`CMIBQConfig`.

## 5. Practical Tips

- Start the progressive quantisation with a relaxed bit budget (e.g. 8-bit) and
  let the helper anneal towards the final target (e.g. 4-bit) over several
  thousand steps.
- The task embedding can encode the downstream task in various ways: a learned
  parameter vector, a language prompt embedding, or metadata describing the
  dataset split.
- The contrastive alignment loss expects that the batch contains aligned visual
  and textual pairs (same ordering).  When this is not the case, disable the
  alignment term by setting `alignment_weight=0.0`.
- Monitor the diagnostics returned in `cm_ibq` (bit allocation statistics,
  reconstruction error) to verify that the quantiser converges to a stable
  solution.

## 6. Extending the Framework

The modular design allows researchers to plug in alternative importance metrics,
bit allocation strategies or quantiser formulations.  For example, the
`BitAllocationNetwork` can be replaced with a reinforcement learning agent, or
`DifferentiableQuantizer` can be swapped for a log-uniform quantiser if desired.
