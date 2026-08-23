- - # C3 Coding Contract — π0.5 Representation Extractor

    ## Goal

    Implement the Pilot v0.1 π0.5 representation extractor for the official OpenPI LIBERO model:

        config = "pi05_libero"
        checkpoint = "gs://openpi-assets/checkpoints/pi05_libero"
    
    The extractor must:
    
    1. load existing PilotObservation records;
    2. reproduce the official OpenPI LIBERO client-side image preprocessing;
    3. reconstruct the official pi05_libero policy input;
    4. reuse the official OpenPI server-side inference transforms;
    5. construct the batched model Observation using official semantics;
    6. extract the base-camera π0.5 visual representations P1 and P2;
    7. preserve sample identity and provenance;
    8. serialize one feature artifact per observation.

    This task is C3 extractor implementation only.

    Do not implement the real π0.5 integration smoke in this turn.
    
    ---
    
    ## Scientific Contract
    
    Pilot v0.1 compares:
    
        OpenVLA O1-S ↔ π0.5 P1
        OpenVLA O2   ↔ π0.5 P2
    
    The frozen Physical Intelligence target is:
    
        config = "pi05_libero"
    
    with official released checkpoint identifier:
    
        checkpoint = "gs://openpi-assets/checkpoints/pi05_libero"
    
    The relevant model semantics are:
    
        Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
        )
    
    Do not substitute:
    
    - classic π0;
    - π0-FAST;
    - another OpenPI checkpoint;
    - another standalone SigLIP implementation.

    The scientific preprocessing design has already been audited against OpenPI commit:

        15a9616a00943ada6c20a0f158e3adb39df2ccac
    
    Do not reopen or modify the frozen scientific design unless implementation reveals a genuine source contradiction.
    
    ---
    
    ## Allowed Production Files
    
    Prefer no more than:
    
        shared_feature/pi05_features.py
        shared_feature/__init__.py
    
    Tests may add:
    
        tests/test_pi05_features.py
    
    Do not modify unrelated production files.
    
    Do not modify:
    
        ../openpi
        ../openvla
        ../LIBERO
        ../tex3d
        ../modified-tex3d

    If implementation cannot satisfy this contract without modifying upstream OpenPI or changing the frozen scientific semantics, STOP and report the blocker.

    ---
    
    ## Forbidden Scope

    Do not implement or modify:

    - C3 real π0.5 integration smoke;
    - checkpoint model downloading/loading inside the core extractor;
    - paired-feature validator;
    - PCA / SVD;
    - CCA / SVCCA;
    - CKA;
    - linear regression;
    - mean pooling;
    - token-level analysis;
    - object-region analysis;
    - Tex3D integration;
    - shared-feature attack loss;
    - adversarial texture optimization;
    - Gemma hidden-state extraction;
    - Action Expert feature extraction;
    - wrist-camera feature serialization;
    - OpenVLA C2 behavior;
    - OpenPI source code;
    - forward hooks;
    - duplicated SigLIP forward implementations;
    - standalone replacement vision encoders;
    - unrelated refactors.
    
    Do not mark full C3 PASS in this task.
    
    ---
    
    ## Public API
    
    Implement:
    
        class Pi05FeatureExtractionError(RuntimeError):
            ...

    and:

        def extract_pi05_features(
            *,
            model,
            train_config,
            checkpoint: str | Path,
            norm_stats,
            observation_paths: Sequence[str | Path],
            output_dir: str | Path,
            batch_size: int = 1,
        ) -> tuple[Path, ...]:
            ...

    `norm_stats` is a required dependency.

    It must represent the checkpoint-associated normalization statistics loaded by the caller from the same released checkpoint used for the model.

    Do not provide a silent fallback to:

        data_config.norm_stats

    or unrelated local assets.

    The `checkpoint` argument remains the stable provenance identifier used in output metadata.

    The core extractor receives:

    - an already loaded JAX/NNX-compatible π0.5 model;
    - the corresponding pi05_libero TrainConfig;
    - checkpoint-associated norm_stats.
    
    Do not make checkpoint downloading or complete policy construction part of the core extractor.
    
    Small private helpers are allowed.
    
    Do not expose unnecessary implementation details as public API.
    
    ---
    
    ## Norm-Stats Ownership
    
    The caller is responsible for loading checkpoint-associated norm stats using the official OpenPI checkpoint-loading path.
    
    The core extractor must only consume the supplied:
    
        norm_stats
    
    and must not trigger download of the full checkpoint.

    The extractor must validate that supplied stats are compatible with the pi05_libero inference path.
    
    At minimum:
    
    - a `state` entry must be available;
    - required quantile statistics for PI05 normalization must be present;
    - malformed or incompatible stats must fail loudly.
    
    Do not silently replace missing checkpoint stats with configuration-default or stale local stats.
    
    ---
    
    ## Input Records
    
    Each input path must load through the existing:
    
        PilotObservation
    
    schema.
    
    The extractor requires:
    
        sample_id
        base_rgb_raw
        wrist_rgb_raw
        state
        prompt
    
    The returned tuple must preserve the exact input observation order.
    
    Reject:
    
    - missing input files;
    - invalid PilotObservation archives;
    - duplicate input paths;
    - duplicate sample_id values;
    - unsafe sample_id values;
    - output collisions.

    Do not depend on lexical filename ordering for sample identity.

    ---

    ## Canonical State Validation

    C3 must explicitly validate the frozen C1 LIBERO state representation.

    Require:
    
        state.shape == (8,)
    
    with:

        real numeric dtype
        all finite values
    
    The expected semantics are:
    
        eef_pos            3
        axis-angle quat    3
        gripper_qpos       2
                           —
                           8
    
    Do not silently reshape, truncate, pad, or reinterpret malformed source state before the official OpenPI transforms.
    
    The later official:
    
        PadStatesAndActions(32)
    
    is responsible for model-side state padding.
    
    ---
    
    ## Safe sample_id
    
    Because the output filename is:
    
        {sample_id}.npz
    
    `sample_id` must be a safe single filesystem path component.
    
    Reject values that would allow:
    
    - absolute paths;
    - parent traversal;
    - embedded path separators;
    - `"."`;
    - `".."`;
    - escaping output_dir.
    
    Do not reject otherwise valid identifiers merely because they contain ordinary punctuation such as a period.

    C1-generated Pilot sample IDs should pass unchanged.

    ---

    ## Raw Observation Semantics

    PilotObservation stores raw LIBERO observations before π0.5-specific preprocessing.

    In particular:

        base_rgb_raw
        wrist_rgb_raw
    
    must remain unchanged as the common cross-model observation source.

    The extractor must not mutate these arrays in place.

    Before preprocessing, validate both images:

        image.ndim == 3
        image.shape[-1] == 3
        image.dtype == uint8

    Do not compute provenance from any preprocessed image.

    ---

    ## Source Image Hash

    Compute:

        source_image_hash
    
    from the ORIGINAL:

        PilotObservation.base_rgb_raw

    before:

    - 180° rotation;
    - resize_with_pad;
    - convert_to_uint8;
    - LiberoInputs;
    - normalization;
    - Observation.from_dict;
    - any other model-specific preprocessing.

    Use:

        hashlib.sha256(
            np.ascontiguousarray(
                record.base_rgb_raw
            ).tobytes()
        ).hexdigest()
    
    Serialize as:

        sha256:<hex_digest>

    Do not compute the pairing hash from:
    
    - rotated images;
    - resized images;
    - client policy images;
    - base_0_rgb;
    - normalized images;
    - SigLIP input arrays;
    - wrist images.

    This hash must remain directly comparable with the existing OpenVLA C2 source_image_hash.

    ---

    ## Official LIBERO Client-Side Image Preprocessing

    This step is REQUIRED.

    The official OpenPI LIBERO client performs model-specific image preprocessing before constructing the policy input dictionary.
    
    For both base and wrist images reproduce:
    
        raw LIBERO image
            ↓
        180° rotation
            ↓
        np.ascontiguousarray
            ↓
        openpi_client.image_tools.resize_with_pad(224, 224)
            ↓
        openpi_client.image_tools.convert_to_uint8
    
    The 180° rotation semantics are:
    
        image[::-1, ::-1]
    
    Equivalent behavior:
    
        base_image = np.ascontiguousarray(
            record.base_rgb_raw[::-1, ::-1]
        )
        
        wrist_image = np.ascontiguousarray(
            record.wrist_rgb_raw[::-1, ::-1]
        )
        
        base_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                base_image,
                224,
                224,
            )
        )
        
        wrist_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                wrist_image,
                224,
                224,
            )
        )
    
    Use the official OpenPI/OpenPI-client utilities corresponding to the audited path.
    
    Do not substitute:

    - PIL.Image.resize;
    - OpenCV resize;
    - torchvision resize;
    - custom image normalization;
    - OpenVLA preprocessing;
    - ImageNet normalization;
    - custom orientation handling.
    
    The raw PilotObservation arrays must remain unmodified.
    
    ---
    
    ## Official LIBERO Policy Input Reconstruction
    
    Only after client-side preprocessing construct:
    
        {
            "observation/image": base_image,
            "observation/wrist_image": wrist_image,
            "observation/state": record.state,
            "prompt": record.prompt,
        }
    
    Do NOT construct the π0.5 policy dictionary directly from:
    
        record.base_rgb_raw
        record.wrist_rgb_raw
    
    without the required official client preprocessing.
    
    ---
    
    ## DataConfig Construction
    
    Construct the pi05_libero data configuration using the existing TrainConfig boundary:
    
        data_config = train_config.data.create(
            train_config.assets_dirs,
            train_config.model,
        )
    
    Do not reimplement the semantics of the data configuration.
    
    The resulting configuration should provide the official:
    
    - LIBERO input transforms;
    - normalization mode;
    - model transforms;
    - asset identity.
    
    Validate that the resulting configuration is compatible with the frozen pi05_libero semantics.
    
    ---
    
    ## Official Server-Side Transform Orchestration
    
    OpenPI does not expose a dedicated standalone helper whose sole purpose is C3 preprocessing.
    
    Therefore C3 is explicitly allowed to perform minimal transform orchestration using the existing official OpenPI transform objects.
    
    This is allowed:
    
        compose official transform objects
        in create_trained_policy-compatible order

    This is forbidden:
    
        reimplement individual transform algorithms
    
    The implementation should preserve the inference ordering semantics of the official policy construction for the current no-external-default-prompt case:
    
        InjectDefaultPrompt(None)
            ↓
        data_config.data_transforms.inputs
            ↓
        Normalize(
            norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        )
            ↓
        data_config.model_transforms.inputs
    
    Use existing OpenPI transform classes and composition utilities.
    
    Do not duplicate the implementations of:
    
    - LiberoInputs;
    - Normalize;
    - ResizeImages;
    - TokenizePrompt;
    - PadStatesAndActions;
    - InjectDefaultPrompt.

    Do not invoke full create_trained_policy merely to obtain preprocessing, because that would introduce inappropriate model/checkpoint loading into C3.
    
    ---
    
    ## Normalization Semantics
    
    PI05 uses:
    
        use_quantiles = True
    
    for the official pi05_libero configuration.
    
    The extractor must preserve the official:
    
        Normalize(
            norm_stats,
            use_quantiles=True,
        )
    
    semantics.
    
    Normalize affects only input fields with matching norm-stat entries.
    
    Do not apply checkpoint normalization indiscriminately to all fields.
    
    Do not manually normalize RGB arrays here.
    
    The image model-range conversion occurs later through:
    
        Observation.from_dict(...)
    
    which converts uint8 RGB values from:
    
        [0, 255]
    
    to:
    
        float32 [-1, 1]
    
    ---
    
    ## Official Image-Slot Semantics
    
    After LiberoInputs for PI05, preserve exactly:
    
        base_0_rgb
        left_wrist_0_rgb
        right_wrist_0_rgb
    
    with:
    
        base_0_rgb        = real client-preprocessed base image
        left_wrist_0_rgb  = real client-preprocessed wrist image
        right_wrist_0_rgb = np.zeros_like(base_image)
    
    and masks:
    
        base_0_rgb        = True
        left_wrist_0_rgb  = True
        right_wrist_0_rgb = False
    
    The extractor must construct and validate the complete official image-slot semantics.
    
    Pilot v0.1 feature extraction uses only:
    
        observation.images["base_0_rgb"]
    
    for P1 and P2.
    
    Do not serialize wrist-camera representations.
    
    Do not use the dummy right-wrist slot for feature extraction.
    
    ---
    
    ## Per-Record Transform Boundary
    
    Official OpenPI transforms operate on unbatched per-record dictionaries.
    
    Therefore for every PilotObservation:
    
    1. load and validate the record;
    2. compute provenance from raw base_rgb_raw;
    3. perform official client preprocessing;
    4. construct an independent policy dictionary;
    5. apply the composed official server transform pipeline;
    6. store the independently transformed result.
    
    Do not pass a pre-stacked batch through transforms that expect unbatched dictionaries.
    
    Do not reuse mutable nested dictionaries across records.
    
    Some transforms mutate input structures.
    
    ---
    
    ## Batching Boundary
    
    Only after all records in the current batch have independently passed through the official transform pipeline:
    
    1. stack equivalent leaves in original input order;
    2. convert stacked values to JAX-compatible arrays;
    3. construct the batched Observation.
    
    Batching must preserve:
    
    - sample order;
    - sample identity;
    - one-to-one provenance;
    - one-to-one P1/P2 correspondence.
    
    Support:
    
        batch_size >= 1
    
    Default:
    
        batch_size = 1
    
    Do not introduce multiprocessing, distributed inference, or asynchronous extraction.
    
    ---
    
    ## Observation Construction
    
    After stacking transformed records:
    
        batched_dict
            ↓
        convert leaves to jax.Array-compatible values
            ↓
        Observation.from_dict(batched_dict)
    
    Preserve the OpenPI inference semantics where image uint8 arrays become:
    
        float32 [-1, 1]
    
    Do not bypass Observation.from_dict with custom image conversion.
    
    Because Observation.from_dict may mutate nested image structures, each batch tree must be independently constructed.
    
    ---
    
    ## preprocess_observation
    
    After constructing the batched Observation, call:
    
        observation = preprocess_observation(
            None,
            observation,
            train=False,
        )
    
    Call this once on the batched Observation.
    
    Do not apply it independently to each record before batching.
    
    Do not introduce training-time random augmentation.
    
    For valid official inference images already at 224×224, image geometry is expected to remain unchanged, but this call remains part of the frozen inference path.
    
    ---
    
    ## Representation Extraction
    
    Use the real JAX/NNX π0.5 visual module directly.
    
    For:
    
        observation.images["base_0_rgb"]
    
    call:
    
        p2, aux = model.PaliGemma.img(
            observation.images["base_0_rgb"],
            train=False,
        )
    
    then:
    
        p1 = aux["encoded"]
    
    No hooks.
    
    No OpenPI source modification.
    
    No duplicated SigLIP forward implementation.
    
    No standalone replacement vision encoder.
    
    No deeper Gemma or Action Expert extraction.
    
    ---
    
    ## P1 Definition
    
    P1 is:
    
        aux["encoded"]
    
    Semantics:
    
        π0.5 SigLIP So400m/14 encoder output
        after final encoder normalization
        before the 1152 → 2048 image projection
    
    Expected batched shape:
    
        [B, 256, 1152]
    
    Serialized per-sample shape:
    
        [256, 1152]
    
    Preserve all 256 tokens.
    
    No pooling.
    
    ---
    
    ## P2 Definition
    
    P2 is the first return value of:
    
        model.PaliGemma.img(...)
    
    Semantics:
    
        π0.5 PaliGemma-ready projected image tokens
        after the SigLIP 1152 → 2048 projection
        before deeper Gemma processing
    
    Expected batched shape:
    
        [B, 256, 2048]
    
    Serialized per-sample shape:
    
        [256, 2048]
    
    Preserve all 256 tokens.
    
    No pooling.
    
    ---
    
    ## JAX → NumPy Serialization Boundary
    
    For extracted JAX arrays use explicit device-to-host conversion semantics equivalent to:
    
        np.asarray(
            jax.device_get(value),
            dtype=np.float32,
        )
    
    Do not rely on implicit device conversion during np.savez.
    
    Before writing each sample:
    
    - remove the batch dimension by indexing the correct sample;
    - convert P1 to float32 NumPy;
    - convert P2 to float32 NumPy;
    - validate shapes and values.
    
    ---
    
    ## Serialization
    
    Write exactly one π0.5 feature archive per observation:
    
        {sample_id}.npz
    
    Each archive must contain exactly:
    
        p1_siglip
        p2_projected
        metadata_json
    
    Required arrays:
    
        p1_siglip
            shape = (256, 1152)
            dtype = float32
        
        p2_projected
            shape = (256, 2048)
            dtype = float32
    
    Before writing validate:
    
    - exact expected shape;
    - finite values;
    - non-empty arrays;
    - not trivially all-zero.
    
    Do not serialize the complete SigLIP auxiliary dictionary.
    
    Use compressed NPZ consistent with the existing project convention.
    
    ---
    
    ## Provenance Metadata
    
    metadata_json must contain exactly:
    
        sample_id
        source_model
        checkpoint
        feature_schema_version
        source_image_hash
    
    Use:
    
        source_model = "pi05"
    
    and:
    
        feature_schema_version = "pi05_features_v1"
    
    `checkpoint` must preserve the caller-provided checkpoint identifier deterministically.
    
    Do not add:
    
        node_name
    
    Do not duplicate representation shape or dtype metadata in metadata_json.
    
    ---
    
    ## Error Handling
    
    Raise:
    
        Pi05FeatureExtractionError
    
    for extractor-level external-boundary failures.
    
    Appropriate wrapped boundaries include:
    
    - PilotObservation loading;
    - OpenPI dependency loading;
    - client preprocessing;
    - transform construction;
    - transform execution;
    - model invocation;
    - JAX device transfer;
    - serialization.
    
    If a Pi05FeatureExtractionError is already raised internally, propagate it unchanged.
    
    For wrapped external exceptions preserve:
    
        __cause__
    
    using normal exception chaining.
    
    Fail loudly for:
    
    - invalid PilotObservation;
    - malformed raw images;
    - invalid canonical state;
    - unsafe sample_id;
    - duplicate input path;
    - duplicate sample_id;
    - output collision;
    - incompatible TrainConfig;
    - malformed norm_stats;
    - missing required state norm stats;
    - missing OpenPI preprocessing dependency;
    - malformed image slots;
    - unexpected image masks;
    - batching mismatch;
    - missing PaliGemma.img;
    - missing aux["encoded"];
    - unexpected P1 shape;
    - unexpected P2 shape;
    - non-finite output features.
    
    Do not silently skip bad samples.
    
    Do not broadly wrap obvious internal programmer errors if doing so would destroy useful diagnostics.
    
    ---
    
    ## Unit-Test Strategy
    
    Do not create one test function for every checklist item.
    
    Use a small number of focused tests that collectively validate the required behavior.
    
    Mocks/fakes should model only the necessary public behavior.
    
    Do not reimplement OpenPI in the tests.
    
    ### Unit-Level Behaviors to Verify
    
    Tests should collectively cover:
    
    1. PilotObservation loading and input-order preservation;
    2. raw base/wrist images are not mutated;
    3. canonical state validation:
           shape == (8,)
           numeric real dtype
           finite values;
    4. safe sample_id validation;
    5. exact raw source_image_hash semantics;
    6. 180° rotation for base and wrist images;
    7. official resize_with_pad utility usage;
    8. official convert_to_uint8 utility usage;
    9. client preprocessing occurs before policy-dict construction;
    10. policy dict receives processed base/wrist images;
    11. correct DataConfig construction from TrainConfig;
    12. explicit supplied norm_stats are used;
    13. missing/malformed norm_stats fail;
    14. official transform objects are composed in correct inference ordering;
    15. transforms operate per record before batching;
    16. mutable transformed dictionaries are independent;
    17. PI05 camera slots and masks are correct;
    18. right-wrist slot uses zero padding;
    19. batching preserves order and identity;
    20. Observation.from_dict is used after batching;
    21. preprocess_observation(..., train=False) is applied to batched Observation;
    22. only base_0_rgb is sent to the visual module;
    23. model.PaliGemma.img is the extraction boundary;
    24. P1 comes from aux["encoded"];
    25. P2 comes from the first return value;
    26. expected batch shape validation;
    27. per-sample serialization shapes;
    28. float32 NumPy output;
    29. full 256-token preservation;
    30. exact NPZ keys;
    31. exact metadata fields;
    32. duplicate path/sample rejection;
    33. missing/invalid input rejection;
    34. output collision rejection;
    35. malformed model output rejection;
    36. non-finite output rejection;
    37. Pi05FeatureExtractionError chaining.
    
    ---
    
    ## Explicitly Deferred to Real C3 Smoke
    
    Unit tests must NOT claim to establish:
    
    - real pi05_libero checkpoint loading;
    - actual released checkpoint backend;
    - actual checkpoint norm-stat contents;
    - actual checkpoint asset paths;
    - real tokenizer numerical behavior;
    - tokenizer asset downloading;
    - real JAX/NNX runtime return structure;
    - real model dtype behavior;
    - real device placement;
    - real sharding behavior;
    - real P1/P2 numerical values;
    - real P1/P2 shape confirmation against the released model;
    - serialized-vs-direct numerical equality on the real model;
    - complete real-policy transform numerical equivalence.
    
    These belong to the separate real C3 integration smoke.
    
    ---
    
    ## Integration Boundary
    
    Successful implementation and unit tests establish only:
    
        π0.5 representation extractor — unit-level PASS
    
    They do NOT establish:
    
        C3 fully closed
    
    A later real integration smoke must verify the released pi05_libero path end-to-end.
    
    Do not implement that smoke in this turn.
    
    ---
    
    ## Required Verification Before Completion
    
    Run at minimum:
    
        tests/test_pi05_features.py
        tests/test_pilot_observation.py
    
    Also run existing relevant project tests if:
    
        shared_feature/__init__.py
    
    or other public exports are changed.
    
    Run Ruff on all changed Python files.
    
    Report:
    
    - files changed;
    - implementation summary;
    - exact test commands;
    - test results;
    - Ruff result;
    - any OpenPI API assumptions discovered;
    - any difference between contract expectations and actual local API;
    - whether any file outside the allowed scope changed.
    
    ---
    
    ## Documentation
    
    Do not modify:
    
        docs/pilot-v0.1-spec.md
    
    during normal implementation.
    
    The preprocessing scientific design has already been corrected and frozen.
    
    Only report a documentation blocker if the implementation discovers a genuine contradiction with the actual OpenPI source.
    
    Do not alter research decisions during coding.
    
    Do not mark full C3 PASS.
    
    Maximum allowed status after successful completion:
    
        π0.5 representation extractor — unit-level PASS
        C3 real π0.5 integration smoke — PENDING
    
    ---
    
    ## Stop Condition
    
    Stop when:
    
    1. raw PilotObservation inputs are validated;
    2. official client-side LIBERO preprocessing is reproduced;
    3. checkpoint-associated explicit norm_stats are used;
    4. official pi05_libero server-side transforms are orchestrated from existing OpenPI objects;
    5. transforms run per record before batching;
    6. batched Observation construction matches official inference semantics;
    7. batched preprocess_observation(train=False) is preserved;
    8. P1/P2 are extracted through model.PaliGemma.img;
    9. provenance and serialization match the frozen schema;
    10. focused unit tests pass;
    11. relevant existing tests remain green;
    12. Ruff passes;
    13. no forbidden scope was touched.
    
    If any requirement cannot be satisfied without:
    
    - modifying upstream OpenPI;
    - changing the frozen scientific semantics;
    - downloading the full checkpoint inside the extractor;
    - inventing or silently substituting normalization statistics;
    - bypassing required official preprocessing;
    - duplicating OpenPI transform algorithms;
    - duplicating SigLIP internals;
    
    STOP and report the blocker rather than expanding scope.
