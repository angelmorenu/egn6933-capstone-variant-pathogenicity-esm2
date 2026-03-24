"""Week 11 sequence-level homology confirmation for flagged split pairs.

Consumes flagged pairs from `results/homology_leakage_audit.json` and runs
Smith-Waterman local alignment to confirm sequence identity leakage.

Pipeline:
1) Read flagged pairs (default scope: train_vs_test)
2) Resolve protein sequences via:
   - local sequence table (`--sequence-table`) and/or
   - UniProt fetch (`--fetch-uniprot`)
3) Align each pair with Smith-Waterman
4) Confirm leakage if identity >= threshold and coverage >= threshold
5) Compute confirmed test leakage rate
6) If material (> threshold), re-evaluate RF/XGB on filtered test genes

Example:
  python scripts/homology_sequence_followup.py \
    --data data/processed/week4_curated_dataset.parquet \
    --audit-json results/homology_leakage_audit.json \
    --pair-scope train_vs_test \
    --identity-threshold 0.90 \
    --min-coverage 0.50 \
    --out-json results/homology_sequence_followup.json \
    --out-csv results/homology_sequence_pair_results.csv \
    --out-report results/homology_sequence_followup_report.txt
"""

from __future__ import annotations

import argparse
import json
import re
import ssl
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import xgboost as xgb


_REPO_ROOT = Path(__file__).resolve().parents[1]
_AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


def _resolve_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p


def _clean_sequence(seq: str | None) -> str | None:
    if seq is None:
        return None
    s = str(seq).strip().upper()
    if not s:
        return None
    s = _AA_RE.sub("", s)
    return s or None


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _smith_waterman(
    seq_a: str,
    seq_b: str,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_penalty: int = -2,
) -> dict[str, Any]:
    a = seq_a
    b = seq_b
    m, n = len(a), len(b)

    if m == 0 or n == 0:
        return {
            "score": 0.0,
            "identity": 0.0,
            "matches": 0,
            "aligned_ungapped_len": 0,
            "alignment_len": 0,
            "coverage_a": 0.0,
            "coverage_b": 0.0,
            "start_a": 0,
            "end_a": 0,
            "start_b": 0,
            "end_b": 0,
        }

    score = np.zeros((m + 1, n + 1), dtype=np.int32)
    trace = np.zeros((m + 1, n + 1), dtype=np.int8)

    best = 0
    best_i, best_j = 0, 0

    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            bj = b[j - 1]
            diag = score[i - 1, j - 1] + (match_score if ai == bj else mismatch_score)
            up = score[i - 1, j] + gap_penalty
            left = score[i, j - 1] + gap_penalty
            cur = max(0, diag, up, left)
            score[i, j] = cur

            if cur == 0:
                trace[i, j] = 0
            elif cur == diag:
                trace[i, j] = 1
            elif cur == up:
                trace[i, j] = 2
            else:
                trace[i, j] = 3

            if cur > best:
                best = int(cur)
                best_i, best_j = i, j

    aligned_a: list[str] = []
    aligned_b: list[str] = []

    i, j = best_i, best_j
    end_a = i
    end_b = j

    while i > 0 and j > 0 and score[i, j] > 0:
        t = trace[i, j]
        if t == 1:
            aligned_a.append(a[i - 1])
            aligned_b.append(b[j - 1])
            i -= 1
            j -= 1
        elif t == 2:
            aligned_a.append(a[i - 1])
            aligned_b.append("-")
            i -= 1
        elif t == 3:
            aligned_a.append("-")
            aligned_b.append(b[j - 1])
            j -= 1
        else:
            break

    start_a = i
    start_b = j

    aligned_a.reverse()
    aligned_b.reverse()

    aln_len = len(aligned_a)
    matches = 0
    ungapped = 0
    for ca, cb in zip(aligned_a, aligned_b):
        if ca != "-" and cb != "-":
            ungapped += 1
            if ca == cb:
                matches += 1

    identity = float(matches / ungapped) if ungapped > 0 else 0.0
    cov_a = float(ungapped / m) if m > 0 else 0.0
    cov_b = float(ungapped / n) if n > 0 else 0.0

    return {
        "score": float(best),
        "identity": identity,
        "matches": int(matches),
        "aligned_ungapped_len": int(ungapped),
        "alignment_len": int(aln_len),
        "coverage_a": cov_a,
        "coverage_b": cov_b,
        "start_a": int(start_a),
        "end_a": int(end_a),
        "start_b": int(start_b),
        "end_b": int(end_b),
    }

# Follow-up to see if the SSL fallback resolves KMT2D/ARID1A
# Accessions: O14686 (KMT2D), O14497 (ARID1A)
def _fetch_uniprot_sequence(gene: str, timeout: float = 20.0) -> tuple[str | None, str | None]:
    queries = [
        f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
        f"gene:{gene} AND organism_id:9606 AND reviewed:true",
        f"{gene} AND organism_id:9606 AND reviewed:true",
    ]

    contexts: list[Any] = [None]
    try:
        contexts.append(ssl._create_unverified_context())
    except Exception:
        pass

    for query in queries:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "fields": "accession,gene_primary,protein_name,length,sequence",
                "format": "json",
                "size": 5,
            }
        )
        url = f"https://rest.uniprot.org/uniprotkb/search?{params}"

        for context in contexts:
            try:
                if context is None:
                    response = urllib.request.urlopen(url, timeout=timeout)
                else:
                    response = urllib.request.urlopen(url, timeout=timeout, context=context)

                with response as resp:
                    payload = json.loads(resp.read().decode("utf-8"))

                results = payload.get("results", [])
                for row in results:
                    accession = row.get("primaryAccession")
                    seq = _clean_sequence(row.get("sequence", {}).get("value"))
                    if seq:
                        return accession, seq
            except Exception:
                continue

    return None, None


def _write_missing_sequence_template(
    out_path: Path,
    missing_genes: list[str],
    pair_rows: list[dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    pair_map: dict[str, set[str]] = {gene: set() for gene in missing_genes}

    for row in pair_rows:
        left_group = str(row.get("left_group"))
        right_group = str(row.get("right_group"))
        if left_group in pair_map:
            pair_map[left_group].add(right_group)
        if right_group in pair_map:
            pair_map[right_group].add(left_group)

    for gene in sorted(missing_genes):
        rows.append(
            {
                "GeneSymbol": gene,
                "protein_sequence": "",
                "accession": "",
                "notes": f"Needed for flagged pair(s): {', '.join(sorted(pair_map.get(gene, [])))}",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=["GeneSymbol", "protein_sequence", "accession", "notes"],
    ).to_csv(out_path, sep="\t", index=False)


def _load_sequence_table(
    sequence_table: Path | None,
    gene_col: str,
    sequence_col: str,
) -> dict[str, dict[str, Any]]:
    if sequence_table is None or not sequence_table.exists():
        return {}

    suffix = sequence_table.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(sequence_table)
    elif suffix in {".tsv", ".txt"}:
        df = pd.read_csv(sequence_table, sep="\t")
    else:
        df = pd.read_csv(sequence_table)

    if gene_col not in df.columns or sequence_col not in df.columns:
        raise ValueError(
            f"Sequence table must contain columns `{gene_col}` and `{sequence_col}`"
        )

    records: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        gene = row.get(gene_col)
        seq = _clean_sequence(row.get(sequence_col))
        if pd.isna(gene) or seq is None:
            continue
        gene_name = str(gene)
        old = records.get(gene_name)
        if old is None or len(seq) > len(old["sequence"]):
            records[gene_name] = {
                "sequence": seq,
                "source": f"table:{sequence_table.name}",
                "accession": row.get("accession", None),
            }

    return records


def _collect_pairs(audit_json: dict[str, Any], pair_scope: str, max_pairs: int) -> list[dict[str, Any]]:
    top_pairs = (
        audit_json.get("cross_split_similarity", {})
        .get("top_pairs", {})
        .get(pair_scope, [])
    )
    if not isinstance(top_pairs, list):
        return []

    unique = []
    seen = set()
    for row in top_pairs:
        lg = str(row.get("left_group"))
        rg = str(row.get("right_group"))
        ls = str(row.get("left_split"))
        rs = str(row.get("right_split"))
        key = (lg, rg, ls, rs)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
        if len(unique) >= int(max_pairs):
            break
    return unique


def _evaluate_rf_xgb(
    df: pd.DataFrame,
    excluded_test_genes: set[str],
    rf_report_path: Path,
    xgb_report_path: Path,
) -> dict[str, Any]:
    required = {"embedding", "label", "split", "GeneSymbol"}
    missing = required - set(df.columns)
    if missing:
        return {"error": f"Dataset missing columns for re-evaluation: {sorted(missing)}"}

    x = np.asarray(df["embedding"].tolist(), dtype=np.float32)
    y = df["label"].to_numpy(dtype=int)
    split = df["split"].astype(str).to_numpy()

    idx_train = np.where(split == "train")[0]
    idx_test = np.where(split == "test")[0]

    if idx_train.size == 0 or idx_test.size == 0:
        return {"error": "Missing train/test split"}

    x_train, y_train = x[idx_train], y[idx_train]
    x_test, y_test = x[idx_test], y[idx_test]
    test_df = df.iloc[idx_test].reset_index(drop=True)

    mask_keep = ~test_df["GeneSymbol"].astype(str).isin(excluded_test_genes)
    keep_idx = np.where(mask_keep.to_numpy())[0]

    rf_report = json.loads(rf_report_path.read_text())
    rf_params = rf_report.get("model_params", {})
    rf_seed = int(rf_report.get("seed", 0))

    xgb_report = json.loads(xgb_report_path.read_text())
    xgb_params = xgb_report.get("bayesian_search", {}).get("best_params", {})
    xgb_seed = int(xgb_report.get("seed", 42))

    rf_model = RandomForestClassifier(
        n_estimators=int(rf_params.get("n_estimators", 200)),
        max_depth=int(rf_params.get("max_depth", 4)),
        min_samples_leaf=int(rf_params.get("min_samples_leaf", 5)),
        max_features=rf_params.get("max_features", "sqrt"),
        class_weight="balanced",
        random_state=rf_seed,
        n_jobs=-1,
    )
    rf_model.fit(x_train, y_train)
    rf_prob_all = rf_model.predict_proba(x_test)[:, 1]

    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    xgb_model = xgb.XGBClassifier(
        max_depth=int(float(xgb_params.get("max_depth", 6))),
        min_child_weight=float(xgb_params.get("min_child_weight", 1.0)),
        learning_rate=float(xgb_params.get("learning_rate", 0.08)),
        reg_lambda=float(xgb_params.get("lambda", 1.0)),
        subsample=float(xgb_params.get("subsample", 0.8)),
        colsample_bytree=float(xgb_params.get("colsample_bytree", 0.8)),
        scale_pos_weight=float(scale_pos_weight),
        n_estimators=100,
        random_state=xgb_seed,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(x_train, y_train)
    xgb_prob_all = xgb_model.predict_proba(x_test)[:, 1]

    def _metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
        return {
            "auroc": _safe_auc(y_true, prob),
            "auprc": _safe_auprc(y_true, prob),
            "n": int(len(y_true)),
            "pos_rate": float(np.mean(y_true)) if len(y_true) > 0 else None,
        }

    out = {
        "original": {
            "rf": _metrics(y_test, rf_prob_all),
            "xgb": _metrics(y_test, xgb_prob_all),
        },
        "filtered": {
            "excluded_test_genes": sorted(excluded_test_genes),
            "n_excluded_test_variants": int(len(y_test) - len(keep_idx)),
        },
    }

    if keep_idx.size == 0:
        out["filtered"]["rf"] = None
        out["filtered"]["xgb"] = None
        out["filtered"]["note"] = "No test variants remain after filtering"
    else:
        out["filtered"]["rf"] = _metrics(y_test[keep_idx], rf_prob_all[keep_idx])
        out["filtered"]["xgb"] = _metrics(y_test[keep_idx], xgb_prob_all[keep_idx])

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 11 sequence-level homology confirmation")
    parser.add_argument("--data", default="data/processed/week4_curated_dataset.parquet")
    parser.add_argument("--audit-json", default="results/homology_leakage_audit.json")
    parser.add_argument("--pair-scope", default="train_vs_test")
    parser.add_argument("--max-pairs", type=int, default=100)

    parser.add_argument("--sequence-table", default=None)
    parser.add_argument("--gene-col", default="GeneSymbol")
    parser.add_argument("--sequence-col", default="protein_sequence")
    parser.add_argument("--fetch-uniprot", action="store_true")
    parser.add_argument("--cache-json", default="results/homology_sequence_cache.json")
    parser.add_argument(
        "--missing-template",
        default="results/missing_sequences_template.tsv",
        help="TSV template listing unresolved genes that still need protein sequences",
    )

    parser.add_argument("--identity-threshold", type=float, default=0.90)
    parser.add_argument("--min-coverage", type=float, default=0.50)
    parser.add_argument("--min-seq-len", type=int, default=50)

    parser.add_argument("--match-score", type=int, default=2)
    parser.add_argument("--mismatch-score", type=int, default=-1)
    parser.add_argument("--gap-penalty", type=int, default=-2)

    parser.add_argument("--materiality-threshold", type=float, default=0.05)
    parser.add_argument("--rf-report", default="results/baseline_rf_seed37_bootstrap.json")
    parser.add_argument("--xgb-report", default="results/xgboost_train_eval_report.json")

    parser.add_argument("--out-json", default="results/homology_sequence_followup.json")
    parser.add_argument("--out-csv", default="results/homology_sequence_pair_results.csv")
    parser.add_argument("--out-report", default="results/homology_sequence_followup_report.txt")
    args = parser.parse_args()

    data_path = _resolve_path(args.data)
    audit_path = _resolve_path(args.audit_json)
    sequence_table_path = _resolve_path(args.sequence_table)
    cache_path = _resolve_path(args.cache_json)
    missing_template_path = _resolve_path(args.missing_template)
    rf_report_path = _resolve_path(args.rf_report)
    xgb_report_path = _resolve_path(args.xgb_report)
    out_json = _resolve_path(args.out_json)
    out_csv = _resolve_path(args.out_csv)
    out_report = _resolve_path(args.out_report)

    assert data_path is not None and audit_path is not None and cache_path is not None
    assert missing_template_path is not None
    assert out_json is not None and out_csv is not None and out_report is not None
    assert rf_report_path is not None and xgb_report_path is not None

    for p in [data_path, audit_path, rf_report_path, xgb_report_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    df = pd.read_parquet(data_path)
    audit_json = json.loads(audit_path.read_text())

    pairs = _collect_pairs(audit_json=audit_json, pair_scope=args.pair_scope, max_pairs=int(args.max_pairs))
    if not pairs:
        raise ValueError(
            f"No pairs found in audit json for pair scope `{args.pair_scope}`. "
            "Check `cross_split_similarity.top_pairs`."
        )

    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    seq_map = _load_sequence_table(
        sequence_table=sequence_table_path,
        gene_col=args.gene_col,
        sequence_col=args.sequence_col,
    )

    genes_needed = set()
    for row in pairs:
        genes_needed.add(str(row.get("left_group")))
        genes_needed.add(str(row.get("right_group")))

    for gene in genes_needed:
        if gene in seq_map:
            cache[gene] = {
                "sequence": seq_map[gene]["sequence"],
                "source": seq_map[gene]["source"],
                "accession": seq_map[gene].get("accession"),
            }
        elif gene in cache and _clean_sequence(cache[gene].get("sequence")):
            continue
        elif args.fetch_uniprot:
            accession, sequence = _fetch_uniprot_sequence(gene)
            if sequence is not None:
                cache[gene] = {
                    "sequence": sequence,
                    "source": "uniprot",
                    "accession": accession,
                }

    missing_genes = sorted(
        gene for gene in genes_needed if _clean_sequence(cache.get(gene, {}).get("sequence")) is None
    )
    _write_missing_sequence_template(
        out_path=missing_template_path,
        missing_genes=missing_genes,
        pair_rows=pairs,
    )

    results_rows: list[dict[str, Any]] = []
    confirmed_test_genes: set[str] = set()

    for row in pairs:
        left_group = str(row.get("left_group"))
        right_group = str(row.get("right_group"))
        left_split = str(row.get("left_split"))
        right_split = str(row.get("right_split"))

        left_seq = _clean_sequence(cache.get(left_group, {}).get("sequence"))
        right_seq = _clean_sequence(cache.get(right_group, {}).get("sequence"))

        out_row: dict[str, Any] = {
            "left_group": left_group,
            "left_split": left_split,
            "right_group": right_group,
            "right_split": right_split,
            "proxy_cosine_similarity": float(row.get("cosine_similarity", np.nan)),
            "left_sequence_found": left_seq is not None,
            "right_sequence_found": right_seq is not None,
            "left_sequence_source": cache.get(left_group, {}).get("source"),
            "right_sequence_source": cache.get(right_group, {}).get("source"),
            "left_accession": cache.get(left_group, {}).get("accession"),
            "right_accession": cache.get(right_group, {}).get("accession"),
        }

        if left_seq is None or right_seq is None:
            out_row.update(
                {
                    "alignment_status": "missing_sequence",
                    "identity": None,
                    "coverage_left": None,
                    "coverage_right": None,
                    "confirmed": False,
                }
            )
            results_rows.append(out_row)
            continue

        if len(left_seq) < int(args.min_seq_len) or len(right_seq) < int(args.min_seq_len):
            out_row.update(
                {
                    "alignment_status": "sequence_too_short",
                    "identity": None,
                    "coverage_left": None,
                    "coverage_right": None,
                    "confirmed": False,
                }
            )
            results_rows.append(out_row)
            continue

        aln = _smith_waterman(
            left_seq,
            right_seq,
            match_score=int(args.match_score),
            mismatch_score=int(args.mismatch_score),
            gap_penalty=int(args.gap_penalty),
        )

        identity = float(aln["identity"])
        cov_left = float(aln["coverage_a"])
        cov_right = float(aln["coverage_b"])
        confirmed = (
            identity >= float(args.identity_threshold)
            and min(cov_left, cov_right) >= float(args.min_coverage)
        )

        out_row.update(
            {
                "alignment_status": "aligned",
                "alignment_score": float(aln["score"]),
                "identity": identity,
                "coverage_left": cov_left,
                "coverage_right": cov_right,
                "aligned_ungapped_len": int(aln["aligned_ungapped_len"]),
                "confirmed": bool(confirmed),
            }
        )
        results_rows.append(out_row)

        if confirmed:
            if left_split == "test":
                confirmed_test_genes.add(left_group)
            if right_split == "test":
                confirmed_test_genes.add(right_group)

    result_df = pd.DataFrame(results_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(result_df.to_csv(index=False))

    test_df = df[df["split"].astype(str) == "test"]
    n_test_total = int(len(test_df))
    n_test_confirmed = int(test_df[test_df["GeneSymbol"].astype(str).isin(confirmed_test_genes)].shape[0])
    n_aligned = int((result_df["alignment_status"] == "aligned").sum()) if not result_df.empty else 0

    if n_aligned == 0:
        confirmation_status = "insufficient_sequence_data"
        confirmed_rate: float | None = None
        is_material: bool | None = None
    else:
        confirmation_status = "complete"
        confirmed_rate = float(n_test_confirmed / n_test_total) if n_test_total > 0 else 0.0
        is_material = confirmed_rate > float(args.materiality_threshold)

    reeval = _evaluate_rf_xgb(
        df=df,
        excluded_test_genes=confirmed_test_genes,
        rf_report_path=rf_report_path,
        xgb_report_path=xgb_report_path,
    )

    payload = {
        "data": str(data_path),
        "audit_json": str(audit_path),
        "pair_scope": args.pair_scope,
        "n_pairs_input": int(len(pairs)),
        "identity_threshold": float(args.identity_threshold),
        "min_coverage": float(args.min_coverage),
        "min_seq_len": int(args.min_seq_len),
        "materiality_threshold": float(args.materiality_threshold),
        "sequences": {
            "sequence_table": (str(sequence_table_path) if sequence_table_path is not None else None),
            "fetch_uniprot": bool(args.fetch_uniprot),
            "cache_json": str(cache_path),
            "n_genes_cached": int(len(cache)),
            "missing_genes": missing_genes,
            "missing_template": str(missing_template_path),
        },
        "pair_results": {
            "n_aligned": n_aligned,
            "n_missing_sequence": int((result_df["alignment_status"] == "missing_sequence").sum()) if not result_df.empty else 0,
            "n_confirmed_pairs": int(result_df["confirmed"].sum()) if (not result_df.empty and "confirmed" in result_df.columns) else 0,
        },
        "confirmed_test_leakage": {
            "status": confirmation_status,
            "n_test_total": n_test_total,
            "n_test_confirmed": n_test_confirmed,
            "confirmed_rate": confirmed_rate,
            "confirmed_test_genes": sorted(confirmed_test_genes),
            "is_material": is_material,
        },
        "model_re_evaluation": reeval,
        "outputs": {
            "out_csv": str(out_csv),
            "out_report": str(out_report),
            "missing_template": str(missing_template_path),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))

    lines = [
        "Week 11 Sequence-Level Homology Confirmation",
        "=" * 42,
        f"Pair scope: {args.pair_scope}",
        f"Pairs analyzed: {len(pairs)}",
        f"Identity threshold: {args.identity_threshold}",
        f"Coverage threshold: {args.min_coverage}",
        f"Materiality threshold: {args.materiality_threshold}",
        "",
        f"Aligned pairs: {payload['pair_results']['n_aligned']}",
        f"Pairs missing sequence: {payload['pair_results']['n_missing_sequence']}",
        f"Confirmed pairs: {payload['pair_results']['n_confirmed_pairs']}",
        f"Missing genes needing sequences: {missing_genes}",
        f"Missing-sequence template: {missing_template_path}",
        "",
        f"Confirmation status: {confirmation_status}",
        f"Confirmed test leakage rate: {('NA' if confirmed_rate is None else f'{confirmed_rate:.4f}')} ({n_test_confirmed}/{n_test_total})",
        f"Material leakage (> {args.materiality_threshold:.2f}): {is_material}",
        f"Confirmed test genes: {sorted(confirmed_test_genes)}",
    ]

    if isinstance(reeval, dict) and "original" in reeval and "filtered" in reeval:
        orig_rf = reeval["original"].get("rf")
        filt_rf = reeval["filtered"].get("rf")
        orig_xgb = reeval["original"].get("xgb")
        filt_xgb = reeval["filtered"].get("xgb")
        lines.extend(
            [
                "",
                "Model performance (original vs filtered test):",
                f"- RF AUROC: {orig_rf.get('auroc')} -> {None if filt_rf is None else filt_rf.get('auroc')}",
                f"- RF AUPRC: {orig_rf.get('auprc')} -> {None if filt_rf is None else filt_rf.get('auprc')}",
                f"- XGB AUROC: {orig_xgb.get('auroc')} -> {None if filt_xgb is None else filt_xgb.get('auroc')}",
                f"- XGB AUPRC: {orig_xgb.get('auprc')} -> {None if filt_xgb is None else filt_xgb.get('auprc')}",
            ]
        )

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines) + "\n")

    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_report}")
    if confirmed_rate is None:
        print("Confirmed test leakage rate: NA (insufficient sequence data)")
    else:
        print(f"Confirmed test leakage rate: {confirmed_rate:.4f}")
    print(f"Material leakage: {is_material}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
