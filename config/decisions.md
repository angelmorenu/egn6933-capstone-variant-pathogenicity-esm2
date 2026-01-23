# Decisions

## Week 2 pilot: defer missense-only (VEP) validation

- For the Week 2 pilot build, I do **not** enforce missense-only consequence filtering.
- I need to build a strict-label training table by mapping ClinVar `VariationID -> chr_pos_ref_alt` and filtering clinical significance to `{Benign, Likely benign}` vs `{Pathogenic, Likely pathogenic}`.
- Missense-only filtering/validation (e.g., via Ensembl VEP consequences) is planned for a later week and will be recorded in the training-table metadata when enabled.

## Missense definition (Dr. Fan)

- Define missense strictly as VEP `missense_variant` on the **canonical transcript**.
- Exclude all other coding variants/consequences (e.g., `stop_gained/stop_lost`, `start_lost`, `inframe_insertion/deletion`, splice-related terms including `splice_region_variant`).
- Rationale: those are typically loss-of-function and are out of scope for this predictor.

## Label policy (Dr. Fan)

- Base labels on ClinVar clinical significance:
	- `Pathogenic` / `Likely pathogenic` (with evidence) are treated as **Pathogenic**.
	- `Benign` / `Likely benign` are treated as **Benign**.
- VUS handling has two categories:
	- “Unknown/insufficient evidence” VUS are **excluded**.
	- “Conflicting interpretations” can be **rescued** if the number of `P/LP` annotations is at least **2×** the number of `VUS` annotations; otherwise they are excluded.

## Provenance note (Dr. Fan)

- The dataset I am using is **post-quality control** (those criteria have already been applied).
- Dr. Fan indicated that the downloaded ClinVar release/version **20240805**.
- The `P/LP` vs `VUS` annotation counts were computed from a **ClinVar VCF** that represents a combined view of SCV submissions.

## Implementation prerequisites for the rescue rule

- To implement the rescue rule reproducibly, I need per-variant counts by category (at minimum: `P/LP` count and `VUS` count).
- ClinVar `variant_summary.txt.gz` typically has aggregate fields (e.g., review status, number of submitters) but does not directly provide `P/LP` vs `VUS` submission counts.
- Based on Dr. Fan’s clarification, the most direct source to reproduce counts is the **ClinVar VCF** (combined SCV submissions), by parsing INFO fields that summarize clinical significance and (for conflicts) per-significance counts.
- Once counts are available, the label-building step should be centralized (e.g., a “build labels” artifact keyed by `VariationID`), then joined into the Week-2/Week-3 training table build.
