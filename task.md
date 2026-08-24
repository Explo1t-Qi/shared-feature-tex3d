# C4 Coding Contract — Paired Feature Dataset Manifest

## Goal

Implement the Pilot v0.1 paired-feature dataset validator and manifest builder.

C4 must establish a scientifically valid one-to-one correspondence between the already extracted:

    OpenVLA feature artifacts

and:

    π0.5 feature artifacts

before any cross-model representation analysis begins.

The purpose of C4 is to answer:

    Do the OpenVLA and π0.5 feature archives refer to exactly
    the same Pilot observations, with valid feature schemas and
    matching raw-image provenance?

C4 must:

1. discover OpenVLA feature archives;
2. discover π0.5 feature archives;
3. validate both feature schemas;
4. require exact equality of the two sample sets;
5. pair samples by metadata sample_id;
6. verify source_image_hash equality for every pair;
7. freeze a deterministic sample order;
8. write a lightweight paired-feature manifest.

C4 must NOT perform any representation-similarity analysis.

---

## Current Project Status

Already closed:

    C0 PilotObservation schema                   PASS
    C1 LIBERO observation collector              PASS
    C1 real OpenVLA/LIBERO smoke                 PASS
    C2 OpenVLA feature extractor                 PASS
    C2 real OpenVLA/GPU smoke                    PASS
    C3 π0.5 feature extractor                    PASS
    C3 real π0.5 integration smoke               PASS

C3 canonical scientific extraction is frozen to:

    batch_size = 1

for Pilot v0.1 π0.5 feature generation.

The optional batch-size>1 numerical mismatch observed during C3 smoke is an engineering note and is not part of C4.

---

## Frozen C4 Decisions

### DECISION 1 — Exact Sample-Set Equality

C4 requires:

    set(OpenVLA sample_ids)
    ==
    set(π0.5 sample_ids)

Do not silently use only the intersection.

If either side contains unmatched samples, validation must fail and report the missing IDs.

---

### DECISION 2 — Pair Identity

A valid cross-model pair requires BOTH:

    OpenVLA.sample_id == π0.5.sample_id

and:

    OpenVLA.source_image_hash
    ==
    π0.5.source_image_hash

Matching filenames alone are insufficient.

Matching sample_id alone is insufficient.

The source_image_hash is the cross-model proof that both representations originate from the same raw:

    PilotObservation.base_rgb_raw

---

### DECISION 3 — Metadata sample_id Is Authoritative

Do not require:

    filename stem == metadata.sample_id

Archive filenames are storage details.

The metadata field:

    sample_id

is the authoritative identity used for:

- duplicate detection;
- sample-set equality;
- cross-model pairing;
- deterministic manifest ordering.

A valid archive may have a filename unrelated to its sample_id.

Do not silently rewrite or infer sample_id from the filename.

---

### DECISION 4 — No Representation Analysis

C4 must not perform:

- mean pooling;
- max pooling;
- CLS-style pooling;
- token pooling;
- feature flattening for analysis;
- PCA;
- SVD;
- CCA;
- SVCCA;
- CKA;
- cosine similarity;
- Euclidean similarity analysis;
- linear regression;
- R²;
- canonical correlation;
- shuffled-null analysis;
- train / held-out splitting.

Those belong to later milestones.

---

### DECISION 5 — Manifest Only

C4 produces a lightweight manifest.

Do not duplicate all feature tensors into a consolidated NPZ or another large artifact.

The existing C2 and C3 feature archives remain the source feature artifacts.

---

### DECISION 6 — Deterministic Frozen Ordering

The manifest pair order must be:

    sorted(sample_id)

using deterministic Python string ordering.

Do not depend on:

- filesystem glob order;
- directory enumeration order;
- file modification time;
- archive filename order.

Once written, the manifest is the canonical sample order for downstream Pilot analysis.

---

### DECISION 7 — Deterministic Relative Path Semantics

All source archive paths and the manifest parent must be resolved before manifest construction.

For every validated source archive:

1. resolve the archive path;
2. resolve the manifest parent directory;
3. compute the archive path relative to the resolved manifest parent;
4. serialize the result as a POSIX-style path string.

Example:

    resolved manifest:
        /data/xiaomengqi/project/manifests/paired_features_manifest.json
    
    resolved archive:
        /data/xiaomengqi/project/features/openvla/sample.npz
    
    manifest value:
        ../features/openvla/sample.npz

Do not serialize environment-specific absolute archive paths.

Do not use an absolute-path fallback.

If a deterministic relative path cannot be represented safely, fail loudly rather than changing path semantics.

---

### DECISION 8 — Output Creation Semantics

All input archives and cross-model pairing constraints must be validated BEFORE creating output directories or writing the manifest.

After validation succeeds:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

The final manifest must never overwrite an existing file.

Use exclusive-create semantics for the final manifest.

If writing fails after creating an incomplete file, clean up only the incomplete file created by this invocation.

Do not delete unrelated files or directories.

---

### DECISION 9 — Checkpoint Provenance Validation

Both OpenVLA and π0.5 metadata must contain:

    checkpoint

as a non-empty string.

For OpenVLA:

    any non-empty checkpoint provenance string is accepted.

Do not require one exact local checkpoint path because C2 may record an environment-specific local checkpoint identifier.

For π0.5 Pilot v0.1:

    checkpoint must equal exactly:
    
        gs://openpi-assets/checkpoints/pi05_libero

This C4 contract validates the frozen Pilot v0.1 target identity, not arbitrary π0.5 checkpoints.

---

### DECISION 10 — Feature Content Validation Scope

C4 validates feature arrays only for:

    exact expected shape
    exact float32 dtype
    finite values

Do NOT add an all-zero feature rejection in C4.

C2/C3 own feature-generation correctness.

C4 owns:

    schema
    identity
    provenance
    pairing

Do not expand C4 into feature-quality analysis.

---

## Allowed Production Files

Prefer adding only:

    shared_feature/paired_features.py

and updating:

    shared_feature/__init__.py

Tests may add:

    tests/test_paired_features.py

Do not modify unrelated production files.

Do not modify:

    shared_feature/openvla_features.py
    shared_feature/pi05_features.py
    shared_feature/pilot_observation.py
    docs/pilot-v0.1-spec.md
    docs/research-map.md
    ../openpi
    ../openvla
    ../LIBERO
    ../tex3d
    ../modified-tex3d

unless a genuine contract contradiction is first reported.

Documentation synchronization is a separate later task.

---

## Forbidden Scope

Do not implement:

- feature extraction;
- model loading;
- OpenVLA preprocessing;
- π0.5 preprocessing;
- LIBERO collection;
- paired tensor stacking;
- consolidated feature archives;
- PCA / SVD;
- CCA / SVCCA;
- CKA;
- regression;
- shuffled-null experiments;
- policy relevance;
- attack loss;
- Tex3D integration;
- feature optimization;
- train/test splitting;
- data sampling for analysis;
- plotting;
- representation statistics;
- model inference;
- GPU-dependent validation.

Do not reopen C2 or C3 scientific decisions.

---

## Public API

Implement:

    class PairedFeatureValidationError(RuntimeError):
        ...

and:

    def build_paired_feature_manifest(
        *,
        openvla_feature_dir: str | Path,
        pi05_feature_dir: str | Path,
        output_path: str | Path,
    ) -> Path:
        ...

Small private helpers are allowed.

Do not expose unnecessary implementation internals.

Do not add additional public APIs unless implementation reveals a genuine need.

---

## Input Directory Semantics

The two inputs are directories containing already-produced feature archives:

    openvla_feature_dir
    pi05_feature_dir

C4 must not recursively search arbitrary nested trees.

Discover:

    *.npz

directly under each supplied directory.

Reject:

- missing directories;
- non-directory paths;
- empty feature directories;
- malformed NPZ files;
- duplicate sample IDs within one feature directory.

Ignore unrelated non-NPZ files.

Do not infer sample identity from filename.

Resolve each source archive path before recording it.

---

## OpenVLA Archive Contract

Each OpenVLA archive must contain exactly:

    o1_siglip
    o1_fused
    o2_projected
    metadata_json

Expected arrays:

    o1_siglip
        shape = (256, 1152)
        dtype = float32
    
    o1_fused
        shape = (256, 2176)
        dtype = float32
    
    o2_projected
        shape = (256, 4096)
        dtype = float32

All feature arrays must be:

    finite
    non-empty by virtue of the required exact shapes

Do not reject all-zero arrays solely because they are all zero.

Do not pool, reshape, transpose, normalize, or otherwise alter feature arrays.

---

## OpenVLA Metadata Contract

OpenVLA metadata_json must contain exactly:

    sample_id
    source_model
    checkpoint
    feature_schema_version
    source_image_hash

Require:

    source_model == "openvla"

Require:

    feature_schema_version == "openvla_features_v1"

Require:

    sample_id

to be a non-empty string.

Require:

    checkpoint

to be a non-empty string.

Do not require a specific OpenVLA checkpoint string.

Require:

    source_image_hash

to match exactly:

    sha256:<64 lowercase hexadecimal characters>

The metadata sample_id is authoritative.

Do not compare it with the filename stem.

---

## π0.5 Archive Contract

Each π0.5 archive must contain exactly:

    p1_siglip
    p2_projected
    metadata_json

Expected arrays:

    p1_siglip
        shape = (256, 1152)
        dtype = float32
    
    p2_projected
        shape = (256, 2048)
        dtype = float32

All feature arrays must be:

    finite
    non-empty by virtue of the required exact shapes

Do not reject all-zero arrays solely because they are all zero.

Do not pool, reshape, transpose, normalize, or otherwise alter feature arrays.

---

## π0.5 Metadata Contract

π0.5 metadata_json must contain exactly:

    sample_id
    source_model
    checkpoint
    feature_schema_version
    source_image_hash

Require:

    source_model == "pi05"

Require:

    feature_schema_version == "pi05_features_v1"

Require:

    sample_id

to be a non-empty string.

Require:

    checkpoint

to equal exactly:

    gs://openpi-assets/checkpoints/pi05_libero

Require:

    source_image_hash

to match exactly:

    sha256:<64 lowercase hexadecimal characters>

The metadata sample_id is authoritative.

Do not compare it with the filename stem.

---

## metadata_json Parsing

Load all NPZ files using:

    allow_pickle=False

metadata_json must be:

    a NumPy scalar
    Unicode JSON text
    valid JSON object

Reject:

- object arrays;
- non-scalar metadata_json;
- non-Unicode metadata_json;
- malformed JSON;
- non-object JSON;
- missing metadata keys;
- additional unexpected metadata keys;
- checkpoint values that violate the model-specific rules.

Do not mutate metadata.

---

## Sample Discovery

For each directory:

1. enumerate direct-child `*.npz` archives;
2. resolve each archive path;
3. validate each archive;
4. extract metadata sample_id;
5. construct:

       sample_id -> validated archive record

6. reject duplicate sample_id values.

Do not assume:

    archive filename stem == sample_id

Do not validate filename/sample_id equality.

Metadata sample_id is the only authoritative archive identity.

---

## Exact Sample-Set Validation

After validating both directories, compute:

    openvla_ids = set(...)
    pi05_ids = set(...)

Require:

    openvla_ids == pi05_ids

If not equal, raise:

    PairedFeatureValidationError

and report both:

    missing_in_openvla
    missing_in_pi05

where:

    missing_in_openvla = sorted(pi05_ids - openvla_ids)
    missing_in_pi05    = sorted(openvla_ids - pi05_ids)

Do not proceed with the intersection.

Do not silently skip unmatched samples.

---

## Cross-Model Pair Validation

For every sample_id in:

    sorted(openvla_ids)

retrieve:

    openvla_record
    pi05_record

Require:

    openvla_record.sample_id
    ==
    pi05_record.sample_id

and:

    openvla_record.source_image_hash
    ==
    pi05_record.source_image_hash

If hashes differ, fail loudly and include:

    sample_id
    OpenVLA hash
    π0.5 hash

Do not recompute the raw-image hash in C4.

C4 validates the provenance carried forward by C2 and C3.

Raw PilotObservation archives are not required inputs to C4.

---

## Representation Pairing Semantics

C4 validates the existence of the following scientific representation pairings:

    O1-S ↔ P1

where:

    OpenVLA o1_siglip    shape (256,1152)
    π0.5   p1_siglip     shape (256,1152)

and:

    O2 ↔ P2

where:

    OpenVLA o2_projected shape (256,4096)
    π0.5   p2_projected  shape (256,2048)

Also preserve availability of:

    OpenVLA o1_fused     shape (256,2176)

as a supplementary representation.

C4 must not numerically compare these representations.

Different feature dimensions are expected.

---

## Manifest Output

Write exactly one manifest file to:

    output_path

Recommended filename:

    paired_features_manifest.json

The manifest must be UTF-8 JSON.

Use deterministic serialization:

    ensure_ascii=False
    sort_keys=True
    indent=2

Do not use pickle.

Do not write binary tensor data into the manifest.

---

## Manifest Schema

Use:

    schema_version = "paired_features_v1"

Top-level structure:

    {
        "schema_version": "paired_features_v1",
        "num_samples": N,
        "pairs": [...]
    }

The top-level object must contain exactly:

    schema_version
    num_samples
    pairs

Each pair entry must contain exactly:

    sample_id
    source_image_hash
    openvla_feature_path
    pi05_feature_path

Example:

    {
        "sample_id":
            "libero_spatial__task02__state00__step0000",
    
        "source_image_hash":
            "sha256:...",
    
        "openvla_feature_path":
            "../features/openvla/example.npz",
    
        "pi05_feature_path":
            "../features/pi05/example.npz"
    }

Do not duplicate:

- feature tensors;
- feature shapes;
- feature dtypes;
- checkpoint strings;
- source_model;
- feature schema versions;

inside every pair.

Those remain available in the validated source archives.

---

## Manifest Path Construction

Before generating path strings:

    resolved_manifest_parent = output_path.resolve(strict=False).parent

For every validated source archive:

    resolved_archive = archive_path.resolve(strict=True)

Construct the manifest path using a deterministic relative-path operation from:

    resolved_manifest_parent

to:

    resolved_archive

Serialize the result using POSIX separators.

The manifest must therefore remain independent of the original absolute environment prefix when the directory tree is moved as a unit.

Do not store absolute archive paths.

Do not provide an absolute fallback.

---

## Output Safety

Before any filesystem mutation:

1. validate both input directories;
2. validate all OpenVLA archives;
3. validate all π0.5 archives;
4. validate exact sample-set equality;
5. validate all cross-model source-image hashes;
6. construct the complete manifest object in memory.

Only after all validation succeeds:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

Reject if:

    output_path.exists()

Do not overwrite.

Write the final manifest using exclusive-create semantics.

If writing creates an incomplete final file and then fails:

    remove only that incomplete final file

if it can be identified safely as having been created by the current invocation.

Do not delete pre-existing files.

---

## Deterministic Ordering

The manifest pairs must appear in exactly:

    sorted(sample_id)

order.

Require:

    num_samples == len(pairs)

Every sample_id must appear exactly once.

This manifest ordering becomes the canonical downstream Pilot ordering.

C5 and later stages must consume this order rather than rediscovering archives independently.

---

## Validation Summary

The function does not need to print extensively.

However, successful completion must imply:

    OpenVLA samples: N
    π0.5 samples: N
    matched sample IDs: N
    matched source hashes: N
    hash mismatches: 0
    missing in OpenVLA: 0
    missing in π0.5: 0

Do not add a second report artifact in C4.

The paired manifest is the required output artifact.

---

## Error Handling

Raise:

    PairedFeatureValidationError

for C4 validation and external-boundary failures.

Appropriate failures include:

- missing input directory;
- input path is not a directory;
- empty feature directory;
- malformed NPZ;
- unexpected NPZ keys;
- malformed metadata_json;
- metadata schema mismatch;
- wrong source_model;
- wrong feature_schema_version;
- malformed sample_id;
- malformed checkpoint provenance;
- wrong π0.5 checkpoint;
- malformed source_image_hash;
- duplicate sample_id;
- wrong feature shape;
- wrong feature dtype;
- non-finite features;
- unequal cross-model sample sets;
- cross-model source-image hash mismatch;
- relative-path construction failure;
- output collision;
- manifest serialization failure.

If a PairedFeatureValidationError already exists internally, propagate it unchanged.

Wrap external I/O, NumPy, JSON, and filesystem failures with exception chaining where appropriate.

Do not silently skip bad archives.

---

## Unit-Test Strategy

Use focused tests rather than one test function per checklist item.

Mocks are not necessary for most C4 behavior.

Tests should construct small temporary NPZ archives using real NumPy serialization.

Do not depend on:

- OpenPI;
- OpenVLA;
- JAX;
- PyTorch;
- CUDA;
- LIBERO.

---

## Unit-Level Behaviors to Verify

Tests should collectively cover:

1. successful pairing of multiple valid OpenVLA and π0.5 archives;
2. deterministic sorted metadata sample_id ordering;
3. filename stem is NOT required to equal sample_id;
4. exact sample-set equality;
5. matching source_image_hash requirement;
6. OpenVLA exact archive keys;
7. π0.5 exact archive keys;
8. OpenVLA expected feature shapes;
9. π0.5 expected feature shapes;
10. float32 dtype requirement;
11. finite feature requirement;
12. all-zero finite arrays are not rejected solely for being zero;
13. OpenVLA metadata schema;
14. π0.5 metadata schema;
15. source_model validation;
16. feature_schema_version validation;
17. OpenVLA non-empty checkpoint requirement;
18. π0.5 exact checkpoint requirement;
19. malformed metadata JSON rejection;
20. non-scalar metadata_json rejection;
21. non-Unicode metadata_json rejection;
22. malformed source-image hash rejection;
23. duplicate sample_id rejection within OpenVLA;
24. duplicate sample_id rejection within π0.5;
25. missing sample on either side;
26. explicit reporting of missing_in_openvla;
27. explicit reporting of missing_in_pi05;
28. hash mismatch rejection;
29. non-NPZ files are ignored;
30. empty feature directory rejection;
31. missing directory rejection;
32. output collision rejection;
33. exact manifest top-level schema;
34. exact per-pair manifest fields;
35. manifest archive paths are relative;
36. manifest archive paths use POSIX separators;
37. manifest paths resolve back to the exact validated source archives;
38. num_samples consistency;
39. output file is valid UTF-8 JSON;
40. output parent is created only after successful input validation;
41. source feature archives remain unchanged.

---

## Required Happy-Path Test

Construct at least three samples with deliberately unrelated archive filenames, for example:

    z_file.npz -> metadata sample-b
    a_file.npz -> metadata sample-c
    m_file.npz -> metadata sample-a

on each model side.

The test must succeed.

Verify manifest order:

    sample-a
    sample-b
    sample-c

based only on metadata sample_id.

Verify every pair carries the expected shared source_image_hash.

Verify stored source paths are relative POSIX paths from the manifest parent.

Verify those paths resolve back to the exact source archives.

---

## Required Sample-Set Failure Test

Construct:

    OpenVLA:
        sample-a
        sample-b
        sample-c
    
    π0.5:
        sample-a
        sample-c
        sample-d

Require failure reporting:

    missing_in_openvla = ["sample-d"]
    missing_in_pi05    = ["sample-b"]

Do not allow a two-sample intersection manifest to be created.

---

## Required Hash-Mismatch Test

Construct valid archives with:

    sample_id == "sample-a"

on both sides but:

    OpenVLA source_image_hash = hash_A
    π0.5   source_image_hash = hash_B

Require:

    PairedFeatureValidationError

and include:

    sample-a
    hash_A
    hash_B

in the error context.

---

## Required Checkpoint Tests

Verify:

OpenVLA:

    checkpoint = "/some/local/openvla/checkpoint"

is accepted if non-empty.

Reject:

    checkpoint = ""
    checkpoint = null
    checkpoint = numeric value

For π0.5 accept only:

    gs://openpi-assets/checkpoints/pi05_libero

Reject:

    ""
    null
    numeric values
    gs://openpi-assets/checkpoints/pi05_base
    arbitrary local checkpoint strings

---

## Integration Boundary

C4 does not require a GPU/model smoke.

A successful implementation plus unit tests establishes:

    C4 paired-feature manifest builder — UNIT-LEVEL PASS

Before using the manifest for scientific C5 analysis, perform one lightweight real-artifact validation using existing C2 and C3 feature directories.

That validation must:

1. run the production C4 manifest builder;
2. use real C2 OpenVLA feature artifacts;
3. use real C3 π0.5 feature artifacts;
4. confirm full sample-set equality;
5. confirm all source hashes match;
6. inspect the resulting manifest;
7. resolve every manifest path back to its source archive.

This real-artifact validation does not require model inference.

Do not rerun OpenVLA or π0.5 models for C4.

---

## Required Verification Before Completion

Run at minimum:

    tests/test_paired_features.py

Also run:

    tests/test_openvla_features.py
    tests/test_pi05_features.py

if public package exports are changed.

Run the complete project test suite if practical.

Run Ruff on:

    shared_feature/paired_features.py
    shared_feature/__init__.py
    tests/test_paired_features.py

Report:

- files changed;
- implementation summary;
- public API;
- test commands;
- test results;
- Ruff result;
- manifest schema;
- exact path-storage semantics;
- checkpoint-validation semantics;
- any contract mismatch discovered;
- whether any forbidden file changed.

---

## Documentation

Do not modify:

    docs/pilot-v0.1-spec.md
    docs/research-map.md

during this coding task.

Their stale status / historical π0 wording is a known non-blocking documentation issue.

Documentation synchronization will be handled in a separate task.

If implementation reveals a genuine contradiction with C2/C3 artifact schemas, STOP and report it before changing documentation or source extractors.

---

## Maximum Status After Coding

Successful implementation and tests may establish:

    C4 paired-feature manifest builder — UNIT-LEVEL PASS

Do not yet claim:

    C4 Paired Feature Dataset — PASS

until the real-artifact validation is completed.

Do not claim:

    shared representation discovered
    transferable representation discovered
    policy-relevant representation discovered

Those are later scientific conclusions.

---

## Real-Artifact Follow-Up

After the unit-level implementation is accepted, run a separate lightweight real-artifact C4 validation.

Expected successful result:

    OpenVLA sample count == π0.5 sample count
    sample sets identical
    all source_image_hash values identical pairwise
    manifest written successfully
    all relative manifest paths resolve correctly

After that validation:

    C4 Paired Feature Dataset — PASS

Then C5 may begin.

---

## Stop Condition

Stop when:

1. both feature directories are strictly validated;
2. exact cross-model sample-set equality is enforced;
3. every pair is validated by metadata sample_id and source_image_hash;
4. all expected feature schemas are validated;
5. checkpoint provenance rules are enforced;
6. deterministic sample ordering is frozen;
7. deterministic relative manifest paths are written;
8. the lightweight manifest is written;
9. focused unit tests pass;
10. relevant existing tests remain green;
11. Ruff passes;
12. no representation analysis has been added.

If satisfying C4 would require:

- changing C2 or C3 feature semantics;
- modifying source extractors;
- recomputing representations;
- silently dropping unmatched samples;
- ignoring provenance mismatches;
- accepting arbitrary π0.5 checkpoint provenance;
- introducing PCA / CCA / CKA / regression;
- copying all feature tensors into a new dataset;

STOP and report the blocker rather than expanding scope.
