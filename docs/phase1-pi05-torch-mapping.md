# Phase 1 — Current PI0Pytorch O2/P2 Mapping

**Status:** `FORMAL COMPLETE / PASS`

**Completion date:** 2026-09-12

**Materialization ID:** `phase1_o2_p2_pi05_torch_v1`

## Scope

Phase 1 re-established the frozen Pilot v0.2 `O2 ↔ P2` shared space using the
current `pi05_torch` / `PI0Pytorch` backend. It stopped after held-out validation
and publication of a new reusable mapping. It did not define a shared-feature
loss, run texture optimization, or evaluate attack transfer.

The frozen protocol remained unchanged:

```text
200 paired Pilot v0.2 observations
→ group-aware TRAIN 40 groups / 160 observations
→ HELD-OUT 10 groups / 40 observations
→ [N,256,D] flattened observation-major, token rows 0..255
→ TRAIN-only 99%-variance PCA
→ ordinary linear CCA
→ HELD-OUT evaluation without refit
```

## Runtime identity

```text
shared-feature-tex3d commit:
  a67b51c1f09021e9ae057276f384ff73e3a2546e

openpi commit:
  15a9616a00943ada6c20a0f158e3adb39df2ccac

pi05_libero model.safetensors SHA-256:
  feeedaf6abe1601f8fb24041e21ae8c022b91141ebf1616678cfd2ea8640a09e

server run root:
  /data/xiaomengqi/logs/shared-feature-phase1/
  phase1-o2-p2-pi05-torch-v1/

synchronized local artifact root:
  experiment_inbox/shared-feature-phase1/
  phase1-o2-p2-pi05-torch-v1/
```

P2 was extracted as the base-camera output of
`paligemma_with_expert.embed_image`, after the official policy input transform and
`PI0Pytorch._preprocess_observation(train=False)`. Extraction verified the
base/left/right image-slot order, `true/true/false` masks, model-native token
order, BF16 native features, `[256,2048]` shape, and bitwise equality to the base
image slice of `embed_prefix`. Archives serialize P2 as float32 under
`pi05_torch_features_v1`.

## Formal results

```text
C3 current PI0Pytorch extraction       COMPLETE (200/200)
C4 paired materialization              COMPLETE (200/200)
C5-A representation geometry           GO
C5-B explicit shared-space alignment   PASS
Phase 1 mapping materialization         PASS
```

Primary 99%-PCA `O2 ↔ P2` result:

```text
O2 retained PCA dimensions       1793
P2 retained PCA dimensions        262
canonical components              262

TRAIN Top5Mean              0.977269488584
HELD-OUT Top1               0.984868435781
HELD-OUT Top5Mean           0.970537564615
HELD-OUT Top10Mean          0.957619462990
TRAIN→HELD-OUT Top5 gap     0.006731923968

HELD-OUT Top5 null median   0.941040551584
HELD-OUT Top5 null q95      0.942210134704
true − null median          0.029497013032
empirical p                 0.00497512437811
```

The current Torch held-out Top5Mean differs from the historical JAX/NNX result
`0.970583518852` by `-0.000045954237`. Both backends retain 262 P2 PCA dimensions.
Across all 200 observations, raw current and historical P2 tensors have global
cosine `0.999978480563` and relative L2 difference `0.006569676069`; they are
numerically close, not bitwise identical.

## Artifact validation and authority

The synchronized result passed a separate read-only audit:

- all 200 P2 archives passed schema, shape, dtype, finiteness, and metadata checks;
- all 200 paired sample identities, source-image hashes, and O2/P2 content hashes
  matched the mapping provenance;
- C5-A and C5-B null archives validated;
- all nine mapping arrays passed the authoritative array validator and their
  serialized hashes matched metadata;
- an independent mapping refit reproduced all four recorded C5-B scalars with
  absolute difference `0.0`;
- the historical C5-BM four-file artifact remained byte-identical.

The current authoritative mapping is:

```text
experiment_inbox/shared-feature-phase1/
  phase1-o2-p2-pi05-torch-v1/mapping/
    mapping.npz
    metadata.json
    validation.json
    summary.md
```

```text
mapping.npz SHA-256:
  572d4772432025f130ecf0403562bab20a20d4bec008c778985b2b9aee28caec
```

The historical `experiment_inbox/c5bm-formal-output/` mapping remains immutable
scientific provenance for the JAX/NNX P2 path. It is not the current PI0Pytorch
mapping.

Some paths inside the synchronized C3/C4 manifests record the original server
layout. Reusing the frozen mapping does not depend on those paths. A local C4/C5
rerun must recreate the original layout or materialize a new local manifest; the
authoritative synchronized files must not be edited in place.

## Scientific conclusion and boundary

For the recorded dataset, commits, checkpoint, preprocessing, and frozen protocol,
the current PI0Pytorch P2 representation supports an explicit O2/P2 linear shared
space that generalizes to held-out trajectory groups. This closes Phase 1 and the
previous backend-compatibility uncertainty.

It does not establish shared-feature vulnerability, action relevance, policy
degradation, texture optimization, or held-out VLA attack transfer. Those questions
belong to later phases.
