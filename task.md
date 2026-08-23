- - # C2 Contract — OpenVLA Representation Extractor

    ## 0. Status

    **Current coding milestone:** C2 — OpenVLA Representation Extractor

    Prerequisites:

    - C0 `PilotObservation` schema: **PASS**
    - C1 LIBERO observation collector: **PASS**
    - C1 real OpenVLA/LIBERO smoke integration: **PASS**

    C2 operates only on the observations produced by C1.

    C2 does **not** implement π0 extraction, shared-space discovery, PCA, CCA, CKA, regression, attack optimization, or Tex3D modifications.

    ---

    # 1. Goal

    Implement a minimal OpenVLA representation extractor that:

    1. loads existing `PilotObservation` records;
    2. reconstructs the exact OpenVLA policy-query visual preprocessing used by C1;
    3. obtains the checkpoint processor's fused DINOv2 + SigLIP `pixel_values`;
    4. directly calls existing OpenVLA vision submodules;
    5. extracts and serializes three full token-level representations:
       - O1-S: OpenVLA SigLIP branch feature;
       - O1-F: fused DINOv2 + SigLIP feature;
       - O2: OpenVLA multimodal projector output;
    6. preserves one-to-one identity with the original Pilot `sample_id`.

    The extractor must not modify OpenVLA source code and must not use forward hooks.

    ---

    # 2. Scientific scope

    C2 answers only:

    > For each C1 policy-query observation, what visual representations does the real OpenVLA visual pipeline produce at the selected internal nodes?

    C2 does **not** yet answer:

    - whether these features are shared with π0;
    - whether they are transferable;
    - whether they are policy-relevant;
    - whether mean pooling is appropriate;
    - whether a shared subspace exists.

    Those are later milestones.

    ---

    # 3. Frozen OpenVLA representation nodes

    For the current fused OpenVLA visual backbone:

    ```text
    policy image
        ↓
    PrismaticProcessor
        ↓
    pixel_values [B, 6, 224, 224]
        ↓
    split channels
     ┌────────────────┬─────────────────┐
     │ first 3 ch     │ last 3 ch       │
     │ DINO input     │ SigLIP input    │
     ↓                ↓
    DINOv2          SigLIP
                       │
                       └──────────── O1-S
            │           │
            └─ concat ──┘
                  ↓
                 O1-F
                  ↓
              projector
                  ↓
                  O2
    ```

    ## 3.1 O1-S

    Definition:

    ```text
    O1-S = output of OpenVLA's actual SigLIP branch featurizer
    ```

    Expected Pilot shape per sample:

    ```text
    [256, 1152]
    ```

    This is the representation actually used by the OpenVLA fused visual backbone.

    Do not replace it with a separately instantiated SigLIP model.

    Do not change its layer.

    The OpenVLA featurizer is already configured so its forward path returns the actual second-to-last transformer-layer patch representation used by OpenVLA.

    ---

    ## 3.2 O1-F

    Definition:

    ```text
    O1-F = concat(DINOv2 feature, SigLIP feature, dim=-1)
    ```

    Expected Pilot shape per sample:

    ```text
    [256, 2176]
    ```

    This is exactly the full visual feature passed into the OpenVLA multimodal projector.

    O1-F is the primary OpenVLA **pre-projector visual representation**.

    ---

    ## 3.3 O2

    Definition:

    ```text
    O2 = model.projector(O1-F)
    ```

    Expected Pilot shape per sample:

    ```text
    [256, 4096]
    ```

    O2 is OpenVLA's VLA-adapted visual representation immediately after the multimodal projector and before insertion into the language-model input sequence.

    ---

    # 4. Extraction method — FROZEN

    Use **direct calls to existing OpenVLA submodules**.

    Do not use hooks.

    Do not modify OpenVLA model source.

    Do not run the full language model or `predict_action()` merely to obtain features.

    Conceptually:

    ```python
    dino_pixels, siglip_pixels = torch.split(
        pixel_values,
        [3, 3],
        dim=1,
    )

    dino_feature = model.vision_backbone.featurizer(
        dino_pixels
    )

    siglip_feature = model.vision_backbone.fused_featurizer(
        siglip_pixels
    )

    fused_feature = torch.cat(
        [dino_feature, siglip_feature],
        dim=-1,
    )

    projected_feature = model.projector(
        fused_feature
    )
    ```

    The implementation must reuse these already-loaded checkpoint modules.

    Each DINO/SigLIP branch should be evaluated only once per batch.

    Do not call:

    ```python
    model.vision_backbone(pixel_values)
    ```

    after separately calling both branch featurizers, because that would recompute the same features.

    ---

    # 5. Input source — FROZEN

    C2 consumes serialized C1 `PilotObservation` records.

    For OpenVLA C2:

    ```text
    use:
        PilotObservation.base_rgb_raw

    do not use:
        PilotObservation.wrist_rgb_raw
    ```

    Classic OpenVLA LIBERO policy uses the base/agent-view image only.

    `state` is not an OpenVLA visual backbone input and must not affect O1-S/O1-F/O2 extraction.

    The saved state and wrist image remain part of the Pilot observation dataset for later π0 and policy-level work.

    ---

    # 6. OpenVLA preprocessing semantics — FROZEN

    C2 must reproduce the **actual C1 OpenVLA policy-query preprocessing**, not invent a new preprocessing path.

    The scientific requirement is:

    ```text
    C2 feature input
    ==
    C1 rollout OpenVLA policy input
    ```

    C2 must not feed `base_rgb_raw` directly into the Hugging Face processor.

    ---

    ## 6.1 C1-compatible policy-image construction

    Starting from:

    ```python
    record.base_rgb_raw
    ```

    reproduce the C1 path:

    ```text
    raw 512×512 agent-view observation
        ↓
    official get_libero_image(..., 512)
        ↓
    180° LIBERO rotation
        ↓
    JPEG encode/decode
        ↓
    C1 resize behavior
        ↓
    PIL resize to OpenVLA input size 224×224
        ↓
    OpenVLA center crop behavior
        ↓
    checkpoint processor
    ```

    C2 must preserve the currently validated C1 behavior even though upstream OpenVLA's standalone LIBERO evaluation path has a slightly different resize sequence.

    Do not silently replace the C1 path with upstream:

    ```text
    get_libero_image(..., 224)
    ```

    during C2.

    That baseline difference is a separate provenance/correctness note and is not a C2 change.

    ---

    ## 6.2 Center crop

    For the current Pilot checkpoint:

    ```text
    center_crop = True
    crop_scale = 0.9
    ```

    Use the same OpenVLA center-crop semantics already used by the real C1 rollout.

    Reuse existing official OpenVLA preprocessing helpers where possible.

    Do not independently invent a numerically different crop implementation.

    ---

    ## 6.3 Prompt / processor path

    Use the checkpoint's existing `PrismaticProcessor`.

    The processor must be the processor corresponding to the same OpenVLA checkpoint used during C1.

    Use the Pilot record's exact:

    ```python
    record.prompt
    ```

    when reconstructing the OpenVLA action prompt.

    Prompt formatting must follow the same OpenVLA checkpoint-dependent logic used by the official OpenVLA action path.

    C2 needs only:

    ```python
    inputs["pixel_values"]
    ```

    and must not run language-model inference.

    ---

    # 7. Fused processor semantics — FROZEN

    The checkpoint processor produces:

    ```text
    pixel_values: [B, 6, 224, 224]
    ```

    For the frozen `dinosiglip` backbone ordering:

    ```text
    pixel_values[:, 0:3]
        = DINOv2-preprocessed RGB tensor

    pixel_values[:, 3:6]
        = SigLIP-preprocessed RGB tensor
    ```

    These two 3-channel tensors may use different backbone-specific normalization / transform parameters.

    Therefore do **not** reuse the first three channels for both models.

    Do not manually reproduce DINO/SigLIP normalization.

    Use the checkpoint processor output directly.

    Explicitly validate:

    ```text
    pixel_values.ndim == 4
    pixel_values.shape[1] == 6
    ```

    before feature extraction.

    ---

    # 8. Model compatibility validation

    C2 is Pilot-specific and should fail explicitly if the loaded model is incompatible with the frozen OpenVLA architecture.

    At minimum verify that the model exposes:

    ```text
    model.vision_backbone
    model.vision_backbone.featurizer
    model.vision_backbone.fused_featurizer
    model.projector
    ```

    and that the processor yields a six-channel fused visual input.

    Do not silently adapt to a single-backbone OpenVLA checkpoint.

    Do not introduce a generic VLA/backbone registry.

    ---

    # 9. Compute dtype and device — FROZEN

    The current OpenVLA Pilot checkpoint is loaded for normal BF16 inference.

    Feature extraction must preserve the loaded OpenVLA inference semantics.

    Do not convert the entire model to FP32 for C2.

    Do not reload the backbone separately.

    Use:

    ```text
    existing loaded model
    existing model device
    existing vision-model dtype
    ```

    for computation.

    The extractor may determine the appropriate device/dtype from the loaded vision backbone parameters.

    Use inference-only execution:

    ```python
    model.eval()

    with torch.inference_mode():
        ...
    ```

    Do not enable gradients.

    ---

    # 10. Persisted feature dtype — FROZEN

    Before serialization, every extracted representation must be:

    ```text
    detach
    → float32
    → CPU
    → remove batch dimension
    → NumPy array
    ```

    Persisted dtype:

    ```text
    float32
    ```

    Expected per-sample arrays:

    ```text
    o1_siglip:
        shape = [256, 1152]
        dtype = float32

    o1_fused:
        shape = [256, 2176]
        dtype = float32

    o2_projected:
        shape = [256, 4096]
        dtype = float32
    ```

    The FP32 serialization format is for stable downstream numerical analysis.

    It does not imply that the OpenVLA forward itself was FP32.

    ---

    # 11. Token preservation — FROZEN

    C2 must save the complete patch-token tensors.

    Do not mean-pool during extraction.

    Do not:

    - average the 256 tokens;
    - select a subset of patches;
    - perform PCA;
    - perform SVD;
    - quantize features;
    - compress feature dimensions;
    - compute attention-weighted features.

    Later analysis will derive frame-level vectors such as:

    ```python
    frame_feature = token_feature.mean(axis=0)
    ```

    from the saved full tensors.

    ---

    # 12. Batching semantics — FROZEN

    C2 may process multiple Pilot observations in one GPU batch.

    Batching is only an implementation/runtime optimization.

    It does not change the scientific sample unit.

    Conceptually:

    ```text
    sample A
    sample B
    sample C
        ↓
    processor batch
        ↓
    [B, 6, 224, 224]
        ↓
    feature extraction
        ↓
    split by batch index
        ↓
    one feature record per original sample
    ```

    The mapping must remain exact:

    ```text
    feature[0] ↔ sample_id[0]
    feature[1] ↔ sample_id[1]
    ...
    ```

    Do not depend on unordered filesystem iteration.

    ---

    ## 12.1 batch_size

    Expose only a minimal runtime parameter:

    ```python
    batch_size: int = 1
    ```

    `batch_size` is not a scientific parameter.

    It must not affect feature values except for normal deterministic floating-point execution behavior.

    Do not introduce a dataloader framework or generalized batching abstraction for C2.

    ---

    # 13. Public API

    Introduce a minimal public API conceptually equivalent to:

    ```python
    class OpenVLAFeatureExtractionError(RuntimeError):
        ...


    def extract_openvla_features(
        *,
        model,
        processor,
        pretrained_checkpoint: str | Path,
        observation_paths: Sequence[str | Path],
        output_dir: str | Path,
        center_crop: bool = True,
        batch_size: int = 1,
    ) -> tuple[Path, ...]:
        ...
    ```

    The exact spelling may change only if there is a concrete implementation reason identified during planning.

    Do not expose parameters for:

    - task suite;
    - task ID;
    - camera resolution;
    - token pooling;
    - selected layers;
    - selected backbone;
    - projector choice;
    - dtype selection;
    - feature dimensions;
    - π0;
    - CCA;
    - attack configuration.

    Those are frozen or outside C2.

    ---

    # 14. Observation identity

    Each input record must be loaded using:

    ```python
    PilotObservation.load(...)
    ```

    The exact `PilotObservation.sample_id` is the canonical feature identity.

    Do not infer identity from input file ordering.

    Do not generate a new UUID.

    Do not renumber observations.

    ---

    # 15. Feature serialization — FROZEN

    Serialize exactly one feature `.npz` per Pilot sample.

    Recommended output filename:

    ```text
    {sample_id}.npz
    ```

    One file contains all three representations.

    Required array keys:

    ```text
    o1_siglip
    o1_fused
    o2_projected
    ```

    Required metadata must contain at least:

    ```text
    sample_id
    source_model = "openvla"
    checkpoint
    feature_schema_version
    ```

    Keep metadata minimal.

    Do not duplicate the complete `PilotObservation` metadata.

    The original observation remains the source of truth for:

    - task;
    - episode;
    - step;
    - prompt;
    - success;
    - raw images;
    - state.

    Feature ↔ observation pairing is:

    ```text
    feature.sample_id == PilotObservation.sample_id
    ```

    ---

    # 16. Serialization safety

    Use a non-pickle NumPy serialization format.

    Feature files must be loadable with:

    ```python
    np.load(path, allow_pickle=False)
    ```

    Do not store Python objects.

    Do not silently overwrite an existing feature file.

    Before writing a batch, validate all intended output paths for that batch.

    A serialization/runtime error must be explicit.

    Do not silently skip failed samples.

    Do not substitute another Pilot observation.

    ---

    # 17. Input validation

    Fail explicitly for at least:

    - empty observation list;
    - duplicate input paths if they resolve to the same Pilot sample;
    - duplicate `sample_id`;
    - missing observation file;
    - invalid `PilotObservation`;
    - invalid `base_rgb_raw`;
    - nonpositive/noninteger `batch_size`;
    - incompatible model architecture;
    - processor output without six visual channels;
    - unexpected branch feature rank;
    - unexpected token count;
    - unexpected feature dimensions;
    - output path collision;
    - runtime preprocessing error;
    - model forward error;
    - serialization error.

    For this frozen Pilot, expected token count is:

    ```text
    256
    ```

    Unexpected shapes should be treated as architecture/checkpoint mismatch, not silently accepted.

    ---

    # 18. Output ordering

    The returned:

    ```python
    tuple[Path, ...]
    ```

    must correspond to the exact order of the input `observation_paths`.

    Batching must not change output ordering.

    ---

    # 19. Files allowed

    Prefer:

    ```text
    new:
        shared_feature/openvla_features.py

    new:
        tests/test_openvla_features.py

    minimal optional update:
        shared_feature/__init__.py
    ```

    Do not modify:

    ```text
    shared_feature/pilot_observation.py
    shared_feature/libero_collector.py
    ```

    unless planning identifies a genuine blocker.

    If C2 appears to require modifying C1 production code, stop and explain why before implementation.

    Do not create a generic feature framework.

    ---

    # 20. Official/reference source boundaries

    Relevant OpenVLA references include:

    ```text
    ../openvla/prismatic/extern/hf/modeling_prismatic.py
    ../openvla/prismatic/extern/hf/processing_prismatic.py
    ../openvla/experiments/robot/openvla_utils.py
    ../openvla/experiments/robot/libero/libero_utils.py
    ```

    Relevant current-project references:

    ```text
    shared_feature/pilot_observation.py
    shared_feature/libero_collector.py
    ```

    Use the loaded checkpoint's actual model/processor attributes as runtime truth.

    Do not inspect or copy unrelated OpenVLA training code.

    `../modified-tex3d` is not needed for normal C2 implementation unless a specific already-known preprocessing correctness question requires targeted confirmation.

    ---

    # 21. Unit tests

    Tests must be CPU-only and use minimal fakes/stubs.

    Do not require a real 7B checkpoint for unit tests.

    At minimum cover:

    1. deterministic one-to-one `sample_id` mapping;

    2. input observation order is preserved through batching;

    3. duplicate `sample_id` is rejected;

    4. existing output collision is rejected without overwrite;

    5. C1-compatible preprocessing path is invoked with the frozen 512 → 224 behavior;

    6. center-crop behavior is invoked when `center_crop=True`;

    7. checkpoint processor output is used for visual tensors;

    8. six-channel processor output is split exactly:
       - channels 0:3 → DINO;
       - channels 3:6 → SigLIP;

    9. DINO branch is evaluated exactly once per batch;

    10. SigLIP branch is evaluated exactly once per batch;

    11. fused representation equals:
        ```python
        torch.cat([dino_feature, siglip_feature], dim=-1)
        ```

    12. projector receives exactly O1-F;

    13. full token dimension is retained;

    14. no mean pooling occurs;

    15. batch dimension is removed during per-sample serialization;

    16. persisted arrays are FP32;

    17. expected output shapes are enforced:
        ```text
        O1-S = [256, 1152]
        O1-F = [256, 2176]
        O2   = [256, 4096]
        ```

    18. malformed six-channel processor output is rejected;

    19. malformed feature shapes are rejected;

    20. runtime/model exception is converted to
        `OpenVLAFeatureExtractionError`
        with the original exception preserved as the cause;

    21. every generated `.npz` loads with:
        ```python
        np.load(..., allow_pickle=False)
        ```

    22. metadata contains the correct `sample_id`, source model, checkpoint, and schema version.

    Keep tests focused on the frozen C2 contract.

    Do not create a generalized fake VLA framework.

    ---

    # 22. Real C2 smoke test — NOT part of the first implementation turn

    After unit implementation is reviewed, a separate real smoke integration will verify the extractor using:

    - the real C1 smoke observations;
    - the same OpenVLA checkpoint used by C1;
    - real BF16 OpenVLA on GPU.

    Do not run this real smoke until explicitly authorized.

    The real smoke must eventually verify:

    ```text
    C2 reconstructed pixel_values
    ==
    normal C1/OpenVLA policy preprocessing pixel_values
    ```

    for at least one known Pilot sample, ideally with exact equality or a reported zero/max absolute difference.

    It must also verify real feature shapes:

    ```text
    O1-S = [256,1152]
    O1-F = [256,2176]
    O2   = [256,4096]
    ```

    and confirm all serialized arrays are FP32 CPU NumPy arrays.

    ---

    # 23. Explicitly forbidden in C2

    Do not implement:

    - π0 feature extraction;
    - π0 preprocessing;
    - DINO/SigLIP hooks;
    - model-source modifications;
    - language-model hidden-state extraction;
    - action-token extraction;
    - mean pooling;
    - PCA/SVD;
    - CCA;
    - CKA;
    - regression;
    - shuffled baselines;
    - train/held-out split logic;
    - feature-space attack loss;
    - Tex3D attack integration;
    - differentiable rendering;
    - model ensembles;
    - manifest/index framework;
    - multiprocessing;
    - distributed extraction;
    - generalized experiment configuration.

    Stop when C2 is complete.

    ---

    # 24. First Codex response — planning only

    For the first turn after this contract is installed:

    **Do not modify files.**

    Read:

    ```text
    AGENTS.md
    docs/research-map.md
    docs/pilot-v0.1-spec.md
    task.md
    shared_feature/pilot_observation.py
    shared_feature/libero_collector.py
    ```

    Then inspect only the OpenVLA source files necessary to verify this contract.

    Return a minimal read-only implementation plan containing:

    1. exact files to create or modify;

    2. proposed public API;

    3. exact OpenVLA model attributes to call for:
       - DINO;
       - SigLIP;
       - fused representation;
       - projector;

    4. exact preprocessing path from:
       ```text
       PilotObservation.base_rgb_raw
       ```
       to:
       ```text
       pixel_values
       ```

    5. how C1 preprocessing equivalence will be preserved without modifying C1;

    6. how the checkpoint processor's six-channel tensor will be split;

    7. how device/dtype will be inferred from the loaded model;

    8. how batching preserves exact `sample_id` ordering;

    9. exact NPZ layout and metadata representation;

    10. focused CPU tests;

    11. every assumption not explicitly frozen by this contract;

    12. every conflict found between:
        - this contract;
        - current C1 implementation;
        - OpenVLA source behavior.

    Do not implement anything in that first response.

    Do not start π0/C3.

    Stop after the plan.
