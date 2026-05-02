# Restructuring plan: fold sub-crates into root `opus` crate

Goal: convert the multi-crate workspace into a single `opus` crate with sub-crate code as inline modules, while keeping `opus-ffi` as a sibling crate (because it requires `unsafe`, has `links = "opus_c_ref"`, and exists only for testing/validation).

## Current state (verified)

```
/Cargo.toml                 ← duplicates crates/opus/Cargo.toml; ../opus-* paths are broken
/crates/opus/               ← still intact: src/, benches/, tests/, COPYING, README.md
/crates/opus-range-coder/   ← 7 src files + tests/roundtrip.rs        (no deps)
/crates/opus-silk/          ← 26 src files (incl. encoder_flp/)        (deps: range-coder)
/crates/opus-celt/          ← 13 src files + tests/encoder_roundtrip.rs (deps: range-coder)
/crates/opus-dnn/           ← 35 src files + build.rs + benches/ + tests/ + tools/ + model-data/  (deps: range-coder, celt; features: dred/osce/deep-plc)
/crates/opus-ffi/           ← unsafe FFI + cmake build (links = opus_c_ref, reads opus-dnn/model-data/)
```

Cross-crate use-site counts (informs the rewrite scope):
- `opus_range_coder`: ~22 sites (mostly silk, some celt)
- `opus_silk`: ~42 sites (in opus, opus tests)
- `opus_celt`: ~39 sites (in opus, opus-dnn nndsp, opus tests/benches)
- `opus_dnn`: ~156 sites (mostly internal to opus-dnn; some opus tests/benches)

## Final target layout

```
/Cargo.toml          ← [workspace] + [package] for opus crate
/build.rs            ← (only if `dnn` feature is enabled) downloads model weights
/src/lib.rs          ← declares pub mod range_coder; silk; celt; #[cfg(feature="dnn")] dnn;
/src/range_coder/    ← was opus-range-coder/src/
/src/silk/           ← was opus-silk/src/
/src/celt/           ← was opus-celt/src/
/src/dnn/            ← was opus-dnn/src/ (gated by feature "dnn")
/src/<existing opus modules: encoder.rs, decoder.rs, ...>
/benches/            ← was crates/opus/benches/
/tests/              ← was crates/opus/tests/  (Rust integration tests)
/examples/           ← unchanged
/model-data/         ← was crates/opus-dnn/model-data/ (gitignored, populated by build.rs)
/tools/              ← was crates/opus-dnn/tools/ (parse_c_weights.rs include!()'d by build.rs)
/crates/opus-ffi/    ← kept as sibling; workspace member
```

Renames (drop the `opus_` prefix since we are now inside the `opus` crate):
- `use opus_range_coder::EcCtx`  →  `use crate::range_coder::EcCtx`
- `use opus_silk::SilkEncoder`   →  `use crate::silk::SilkEncoder`  (or `use opus::silk::...` from tests/benches)
- `use opus_celt::CeltEncoder`   →  `use crate::celt::CeltEncoder`
- `use opus_dnn::nnet::...`      →  `use crate::dnn::nnet::...`

## Strategy for `opus-ffi`

Keep it as a separate crate at `crates/opus-ffi/`. Reasons:
1. **Safety policy** — root `opus` crate forbids `unsafe`. FFI requires `unsafe`. Folding it would force the safety policy off the whole library.
2. **`links` uniqueness** — `links = "opus_c_ref"` must belong to one crate; folding it would couple every release-build of `opus` to cmake + the C submodule.
3. **Build cost** — current setup means `cargo build` (no dev deps) doesn't compile C libopus. That stays true.
4. **Dev-dep relationship is already correct** — `opus-ffi` is a dev-dependency of `opus`; integration tests under `tests/` use it for cross-validation.

Convert root `Cargo.toml` into a Cargo workspace **with the root package** (Cargo allows both `[workspace]` and `[package]` in one manifest). `opus-ffi` becomes a workspace member:

```toml
# /Cargo.toml
[workspace]
members = ["crates/opus-ffi"]

[package]
name = "opus"
# ... existing metadata ...

[features]
default = []
dnn          = ["dnn-dred", "dnn-osce", "dnn-deep-plc"]   # umbrella: turn on all three (matches old opus-dnn defaults)
dnn-deep-plc = []
dnn-dred     = ["dnn-deep-plc"]                            # dred depends on deep-plc (mirrors C build)
dnn-osce     = []

[dependencies]
# (no inter-crate path deps — they're now inline modules)

[dev-dependencies]
opus-ffi = { path = "crates/opus-ffi" }
criterion = { version = "0.5", features = ["html_reports"] }
```

The one cross-cutting concern: `opus-ffi/build.rs` currently looks for DNN weight files at `crates/opus-ffi/opus-c/dnn/*_data.c` (placed there by `opus-dnn`'s build.rs). After folding, the root `build.rs` (running under `CARGO_FEATURE_DNN`) takes over the download and is responsible for placing those `*_data.c` files where `opus-ffi/build.rs` expects. Two options:

- **Option A (preferred):** Keep the same destination path — root `build.rs` writes weight C files into `crates/opus-ffi/opus-c/dnn/`. `opus-ffi/build.rs` is unchanged.
- **Option B:** Move the destination to a workspace-level `target/dnn-data/` and update `opus-ffi/build.rs` to look there. Cleaner, but more code churn.

Recommend Option A — minimum disruption.

## Phased migration

Each phase ends with `cargo check` + `cargo test` + a commit. Order is leaves-first so each step has working downstream consumers.

### Phase 0 — Move `crates/opus/*` to root (preparation)

The root `Cargo.toml` already exists but `src/`, `tests/`, `benches/` are still inside `crates/opus/`. Move them so the root crate is actually buildable before any fold work begins.

```
git mv crates/opus/src .            # src/lib.rs, encoder.rs, decoder.rs, ...
git mv crates/opus/benches .
git mv crates/opus/tests .
rm -rf crates/opus                  # only Cargo.toml + COPYING + README.md left, all duplicated at root
```

Adjust paths in `Cargo.toml` deps from `../opus-range-coder` → `crates/opus-range-coder` (etc.) so the project compiles **before** any folding. This gives a known-green baseline.

Commit: "Move opus crate contents to repo root."

### Phase 1 — Fold `opus-range-coder` → `src/range_coder/`

Leaf module (no inter-crate deps). Lowest risk, highest leverage (silk and celt both depend on it; once it's folded, the next phases only need to update one prefix, not two).

1. `git mv crates/opus-range-coder/src src/range_coder` then rename `src/range_coder/lib.rs` → `src/range_coder/mod.rs` (or keep the modern `src/range_coder.rs` + dir layout — pick `mod.rs` to avoid splitting one file into two).
2. Add `pub mod range_coder;` to `src/lib.rs`.
3. Move `crates/opus-range-coder/tests/roundtrip.rs` → `tests/range_coder_roundtrip.rs`.
4. Search/replace across the workspace (silk, celt, dnn, opus, tests, benches):
   - `opus_range_coder::` → `crate::range_coder::` for lib code in the same crate
   - `opus_range_coder::` → `opus::range_coder::` for tests/benches (they're external consumers)
   - Inside still-external crates (opus-silk/celt/dnn during this transitional phase): leave the `path = "../opus-range-coder"` redirected via `[patch]`, OR keep them building by adding a thin `crates/opus-range-coder/src/lib.rs` re-export shim until they're folded too. **Simpler: do Phases 1–4 without committing intermediate states; one commit at the end.** But that's risky.
   - **Cleaner alternative: do all four folds in one phase**, with a single `cargo check` at the end. Documented below as Phase 1b.

Decision point: **do we fold all four at once, or incrementally with shim crates?** Recommendation: **fold all four in one go**, because the inter-crate dependencies make incremental folding require shim crates that double the work. The folded code is mechanically derivable; the risk is in the search/replace + path updates, not in the algorithm.

### Phase 1b (chosen) — Fold range-coder + silk + celt together

Sequence inside one phase:

1. Move sources:
   ```
   git mv crates/opus-range-coder/src src/range_coder    # then rename lib.rs → mod.rs
   git mv crates/opus-silk/src         src/silk
   git mv crates/opus-celt/src         src/celt
   ```
2. Move tests into root `tests/`:
   ```
   git mv crates/opus-range-coder/tests/roundtrip.rs       tests/range_coder_roundtrip.rs
   git mv crates/opus-celt/tests/encoder_roundtrip.rs      tests/celt_encoder_roundtrip.rs
   ```
3. Edit `src/lib.rs`:
   ```rust
   pub mod range_coder;
   pub mod silk;
   pub mod celt;
   ```
4. Search/replace across `src/`, `tests/`, `benches/`:
   - In `src/silk/**` and `src/celt/**`: `use opus_range_coder::` → `use crate::range_coder::`
   - In `src/encoder.rs`, `src/decoder.rs`, etc.: `use opus_silk::` → `use crate::silk::`, `use opus_celt::` → `use crate::celt::`, `use opus_range_coder::` → `use crate::range_coder::`
   - In `tests/**` and `benches/**`: `use opus_silk::` → `use opus::silk::`, etc.
5. Edit `Cargo.toml`: drop `opus-range-coder`, `opus-silk`, `opus-celt` from `[dependencies]`.
6. `rm -rf crates/opus-range-coder crates/opus-silk crates/opus-celt`.
7. `cargo check && cargo test`.

Commit: "Fold opus-range-coder, opus-silk, opus-celt into root crate as modules."

### Phase 2 — Fold `opus-dnn` → `src/dnn/` (feature-gated)

DNN is heavier: build script, model-data download, three sub-features.

1. Move sources:
   ```
   git mv crates/opus-dnn/src       src/dnn        # rename lib.rs → mod.rs
   git mv crates/opus-dnn/build.rs  build.rs       # root-level build script
   git mv crates/opus-dnn/tools     tools          # parse_c_weights.rs
   git mv crates/opus-dnn/tests/blob_loading.rs    tests/dnn_blob_loading.rs
   git mv crates/opus-dnn/benches/dnn_bench.rs     benches/dnn_bench.rs
   # crates/opus-dnn/COPYING is a duplicate of the root COPYING — discard it
   # (rm -rf handles this when crates/opus-dnn/ is removed in step 6).
   ```
   The `model-data/` directory is build-output (gitignored); leave it to be regenerated.
2. Edit `src/lib.rs`:
   ```rust
   #[cfg(any(feature = "dnn-dred", feature = "dnn-osce", feature = "dnn-deep-plc"))]
   pub mod dnn;
   ```
3. Edit `build.rs` so the download/extract logic runs only when at least one DNN sub-feature is enabled:
   ```rust
   fn main() {
       let dnn_on = env::var_os("CARGO_FEATURE_DNN_DRED").is_some()
           || env::var_os("CARGO_FEATURE_DNN_OSCE").is_some()
           || env::var_os("CARGO_FEATURE_DNN_DEEP_PLC").is_some();
       if !dnn_on {
           return;
       }
       // ... existing download/extract logic ...
   }
   ```
   Adjust paths inside `build.rs`: `manifest_dir` is now the workspace root, so `model-data/` is `<root>/model-data/`.  The destination for `*_data.c` files (consumed by `opus-ffi/build.rs`) stays `<root>/crates/opus-ffi/opus-c/dnn/`. (Option A above.)
4. Edit `Cargo.toml`:
   ```toml
   [features]
   default = []
   dnn = []
   dnn-dred  = ["dnn"]
   dnn-osce  = ["dnn"]
   dnn-deep-plc = ["dnn"]
   # default DNN sub-features when `dnn` is on:
   # match opus-dnn's old default = ["dred", "osce", "deep-plc"]
   ```
   Decision needed: do we want DNN sub-features to remain user-selectable, or always enable all three when `dnn` is on? Recommend the latter (matches old `opus` crate behavior — `dnn = ["dep:opus-dnn"]` pulled in opus-dnn with its default features). Keep one feature flag: `dnn`.
5. Search/replace across `src/`, `tests/`, `benches/`:
   - `use opus_dnn::` → `use crate::dnn::` (in `src/`)
   - `use opus_dnn::` → `use opus::dnn::` (in `tests/`, `benches/`)
   - Inside `src/dnn/**`: `use opus_celt::` → `use crate::celt::`, `use opus_range_coder::` → `use crate::range_coder::`
   - **Rewrite the sub-feature `cfg` attributes** to match the new feature names:
     - `#[cfg(feature = "dred")]`     → `#[cfg(feature = "dnn-dred")]`
     - `#[cfg(feature = "osce")]`     → `#[cfg(feature = "dnn-osce")]`
     - `#[cfg(feature = "deep-plc")]` → `#[cfg(feature = "dnn-deep-plc")]` (note: hyphenated)
     - Any `not(feature = "...")` and combined `any(...)`/`all(...)` forms get the same prefix bump.
   - Verify: `cargo build --features dnn-deep-plc` (should compile a deep-plc-only subset), `--features dnn-dred` (should also pull in deep-plc via the dep), `--features dnn` (everything on).
6. `rm -rf crates/opus-dnn`.
7. Verify `cargo build` (no features), `cargo build --features dnn`, `cargo test`, `cargo test --features dnn`.

Commit: "Fold opus-dnn into root crate as feature-gated module."

### Phase 3 — Workspace setup + opus-ffi adjustments

1. Add to root `Cargo.toml`:
   ```toml
   [workspace]
   members = ["crates/opus-ffi"]
   ```
2. Verify `opus-ffi/build.rs` still finds the DNN data. After Phase 2, root `build.rs` writes the same files to the same path, so this should be a no-op.
3. `opus-ffi/Cargo.toml` is unchanged (no path deps to update — it never depended on the now-folded crates).
4. Run `cargo test --workspace --features dnn` end-to-end.

Commit: "Convert to single-crate layout with opus-ffi as workspace member."

### Phase 4 — Documentation cleanup

Update `CLAUDE.md`:
- Remove the "Project structure" `crates/` listing; replace with "single-crate layout".
- Update "Where to put new tests" — all integration tests live in root `tests/`.
- Note that `opus-ffi` is the only sub-crate, kept for cross-validation.
- Mention `cargo build --features dnn` flag explicitly.

Update root `README.md` if it references the workspace structure.

Commit: "Update docs to reflect single-crate restructuring."

## Risk register

| Risk | Mitigation |
|---|---|
| Search/replace misses a `use opus_silk::...` or string reference | Run `grep -r "opus_silk\|opus_celt\|opus_range_coder\|opus_dnn"` after each phase; only `crates/opus-ffi/` (via its C build) and possibly comments should match. |
| `[cfg(feature = "dred")]` etc. inside opus-dnn become silent dead code | Phase 2 step 5 handles this explicitly — choose unconditional or rename. |
| `opus-ffi/build.rs` looks for DNN data at wrong path | Manual verify: after Phase 2, `ls crates/opus-ffi/opus-c/dnn/*_data.c` after `cargo build --features dnn`. |
| Duplicate `COPYING` license files | Confirmed: `crates/opus-dnn/COPYING` is a copy of the root `COPYING`. Discard the duplicate. |
| `links = "opus_c_ref"` collision | Non-issue: it stays inside `opus-ffi` and is unique. |
| Public API break for downstream users of `opus_silk`, `opus_celt`, etc. | These are `publish = false` already — no external users. The published `opus` crate's surface is what matters; preserve `pub use` re-exports in `src/lib.rs`. |
| Bench `criterion` benchmarks miss internal items now they go through `opus::celt::...` | Make sure `pub mod celt;` etc. so bench access works. |
| `cargo test` between phases fails because intermediate state is broken | Phase 1b folds all three at once; Phase 2 is a clean second commit. No half-states. |

## What stays unchanged

- The `opus-ffi` crate (kept as sibling for tests/benches that need C libopus).
- Public API of `opus` crate — `OpusEncoder`, `OpusDecoder`, etc. all re-exported from `src/lib.rs` as today.
- Examples directory.
- The `forbid(unsafe_code)` lint on root `opus` crate (because all four folded crates already had it).

## Cross-checking against `graphify-out/`

The existing `graphify-out/GRAPH_REPORT.md` (2060 nodes, 4076 edges, 109 communities, dated 2026-05-02) was used to sanity-check this plan:

- **All sub-crate communities map to single subsystems**, not split across crates. "SILK Encoder/Decoder Core", "CELT Codec Core", "Range Coder Decode + CWRS", "DNN Layer Benchmarks", "DRED & DNN Architecture Concepts" — each stays intact as one Rust module after folding. Nothing in the community structure resists the move.
- **Cross-community edges I should not miss:**
  - `opus-dnn/src/freq.rs::forward_transform` → `opus-celt/src/fft.rs::opus_fft` (inferred edge, confidence 0.8). Not visible from a `use opus_celt::` grep — uses the symbol indirectly. Phase 2's full-text search/replace catches it.
  - "DRED & DNN Architecture Concepts" community spans both opus-dnn and opus modules (`dnn_decoder.rs`, `dnn_silk_bridge.rs`) — currently routed via the `dnn` feature dependency. After folding, both sides live in the same crate; the DRED extension parsing in `src/extensions.rs` should keep working without changes.
  - "C-vs-Rust Benchmarks" community is centered on `opus-ffi` consumers — confirms `opus-ffi` belongs as a separate dev-only crate.
- **Hub functions that will become `crate::`-prefixed callees:** `silk_smlawb()` (31 edges), `silk_smulwb()`, `assert_f32_slice_close()` (35 edges, lives in test common). High edge count means a lot of call sites to update, but these are all intra-crate after folding — no surface area change.

### Regenerating the graph

The graph is keyed on file paths (`opus-dnn/src/freq.rs`) and crate names. After restructuring:
- All 852 indexed files at `crates/opus-{range-coder,silk,celt,dnn}/src/**` move to `src/{range_coder,silk,celt,dnn}/**` — every node in the graph's source-location metadata becomes stale.
- Community detection should produce a near-identical clustering (the algorithm clusters by edge density, not file paths), but node identifiers change.

**Recommendation:** delete `graphify-out/` after the restructuring is complete and the test suite is green, then rerun `/graphify`. Don't try to migrate it incrementally — cheaper to regenerate against the new tree.

Add to the cleanup commit:
```
rm -rf graphify-out/
```
…then run `/graphify` after the dust settles.

## Decisions (locked in)

1. **DNN feature granularity:** expose `dnn-dred`, `dnn-osce`, `dnn-deep-plc` as individually selectable. Keep `dnn` as an umbrella that turns on all three. Mirror the C build's `dred → deep-plc` dependency so `dnn-dred` implies `dnn-deep-plc`. The root `dnn` module is gated on `any(feature = "dnn-dred", "dnn-osce", "dnn-deep-plc")`.
2. **Fold ordering:** all four crates fold in one go. Phase 1b bundles range-coder + silk + celt. Phase 2 folds dnn. No intermediate shim crates.
3. **License files:** root `COPYING` is canonical. `crates/opus-dnn/COPYING` is a duplicate and is discarded along with the rest of the directory in Phase 2 step 6.
