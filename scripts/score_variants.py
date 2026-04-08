#!/usr/bin/env python3
"""
=========================================================
Command-Line Interface for Variant Pathogenicity Scoring
=========================================================

Batch scoring interface for classification of missense variants as pathogenic vs. benign 
using trained Random Forest classifier with ESM2 embeddings.

Supports both single-variant and batch-file scoring modes with CSV/JSON output options.

Usage:
    # Single variant scoring
    python scripts/score_variants.py --variant chr1_1022225_G_A --output app/demo_outputs/cli_real_model_output.csv

    # Batch CSV scoring
    python scripts/score_variants.py --input app/demo_outputs/demo_batch_upload_score_source.csv --output app/demo_outputs/cli_batch_output.csv --format csv

    # Batch TSV input with JSON output
    python scripts/score_variants.py --input variants.tsv --output app/demo_outputs/results.json --format json

Author: Angel Morenu
Date: March 2026
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass, asdict
from datetime import datetime


# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class VariantScore:
    """Container for variant scoring results."""
    variant_id: str
    pathogenicity_score: float
    prediction: str  # "PATHOGENIC" or "BENIGN"
    confidence: float
    model: str = "RandomForest"
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


# ============================================================================
# MODEL & DATA LOADING
# ============================================================================

def load_embeddings_and_metadata(strict: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load precomputed ESM2 embeddings and metadata.
    
    Args:
        strict: If True, loads strict-label dataset; otherwise loads default dataset
    
    Returns:
        Tuple of (embeddings array, metadata dictionary)
    
    Raises:
        FileNotFoundError: If data files cannot be located
    """
    dataset_type = "strict" if strict else "default"

    curated_parquet = DATA_DIR / "week4_curated_dataset.parquet"
    if curated_parquet.exists():
        curated_df = pd.read_parquet(curated_parquet, columns=["chr_pos_ref_alt", "label", "split", "embedding"])
        if "chr_pos_ref_alt" not in curated_df.columns or "embedding" not in curated_df.columns:
            raise ValueError(f"Curated dataset is missing required columns: {curated_parquet}")

        embeddings = np.stack(curated_df["embedding"].tolist(), axis=0)
        variant_id_map = {
            str(variant_id): index
            for index, variant_id in enumerate(curated_df["chr_pos_ref_alt"].astype(str).tolist())
        }
        metadata = {
            "source": "week4_curated_dataset.parquet",
            "variant_id_map": variant_id_map,
            "labels": curated_df["label"].astype(int).tolist() if "label" in curated_df.columns else None,
            "splits": curated_df["split"].astype(str).tolist() if "split" in curated_df.columns else None,
            "n_variants": len(curated_df),
        }
        return embeddings, metadata

    embeddings_file = DATA_DIR / f"week2_training_table_{dataset_type}_cleaned_smoke_embeddings.npy"
    metadata_file = DATA_DIR / f"week2_training_table_{dataset_type}_cleaned_smoke_meta.json"

    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    embeddings = np.load(embeddings_file)

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    return embeddings, metadata


def load_trained_model(model_path: Optional[Path] = None, n_features: int = 1280) -> RandomForestClassifier:
    """
    Load trained Random Forest model.
    
    For demonstration, returns a placeholder model if no saved model exists.
    In production, this would load a pickled trained model from disk.
    
    Args:
        model_path: Path to saved model pickle file
    
    Returns:
        Trained RandomForestClassifier instance
    """
    if model_path and model_path.exists():
        import pickle
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    # Placeholder: create a dummy trained model for demonstration
    print("[WARNING] No saved model found. Using placeholder Random Forest for demonstration.")
    print("[INFO] Train and save model using: pickle.dump(model, open('models/rf_model.pkl', 'wb'))")
    
    from sklearn.datasets import make_classification
    informative_features = max(10, n_features // 6)
    redundant_features = max(5, n_features // 12)
    X, y = make_classification(
        n_samples=5000,
        n_features=n_features,
        n_informative=min(informative_features, n_features - 1),
        n_redundant=min(redundant_features, max(0, n_features - informative_features - 1)),
        random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    return model


def validate_model_feature_compatibility(model: RandomForestClassifier, embeddings: np.ndarray) -> None:
    """Warn clearly when model feature count and embedding width are incompatible."""
    model_feature_count = getattr(model, "n_features_in_", None)
    embedding_feature_count = int(embeddings.shape[1])

    if model_feature_count is None:
        print("[WARNING] Loaded model does not expose `n_features_in_`; skipping feature compatibility check.")
        return

    if int(model_feature_count) != embedding_feature_count:
        print(
            "[WARNING] Model/embedding feature mismatch detected: "
            f"model expects {int(model_feature_count)} features but embeddings have {embedding_feature_count}."
        )
        raise ValueError(
            "Feature dimension mismatch between loaded model and embeddings. "
            "Train/save a model on the current embedding set or pass a compatible model via `--model-path`."
        )


# ============================================================================
# VARIANT SCORING
# ============================================================================

def parse_variant_identifier(variant_id: str) -> Tuple[str, int, str, str]:
    """
    Parse canonical variant identifier (chr_pos_ref_alt).
    
    Args:
        variant_id: Canonical format string, e.g., "chr1_100000_A_G"
    
    Returns:
        Tuple of (chromosome, position, ref_allele, alt_allele)
    
    Raises:
        ValueError: If variant ID format is invalid
    """
    normalized = normalize_variant_identifier(variant_id)
    parts = normalized.split('_')
    if len(parts) != 4:
        raise ValueError(
            f"Invalid variant ID format: {variant_id}. Expected: chr_pos_ref_alt"
        )
    
    chromosome, pos_str, ref, alt = parts

    try:
        position = int(pos_str)
    except ValueError:
        raise ValueError(f"Invalid position in variant ID: {pos_str}")
    
    if chromosome not in [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]:
        raise ValueError(f"Invalid chromosome: {chromosome}")
    
    if not all(allele in "ACGT" for allele in [ref, alt]):
        raise ValueError(f"Invalid alleles: {ref}, {alt}. Must be ACGT.")
    
    return f"chr{chromosome}", position, ref, alt


def normalize_variant_identifier(variant_id: str) -> str:
    """Normalize a user-supplied ID to the curated dataset's canonical `chrom_pos_ref_alt` form."""
    value = str(variant_id).strip()
    if not value:
        raise ValueError("Variant ID is empty.")

    parts = value.split("_")
    if len(parts) != 4:
        raise ValueError(
            f"Invalid variant ID format: {variant_id}. Expected: chr_pos_ref_alt"
        )

    chromosome, pos_str, ref, alt = parts
    chromosome = chromosome.strip().lower()
    if chromosome.startswith("chr"):
        chromosome = chromosome[3:]
    if chromosome == "m":
        chromosome = "MT"
    else:
        chromosome = chromosome.upper()

    if chromosome not in [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]:
        raise ValueError(f"Invalid chromosome: chr{chromosome}")

    try:
        position = int(pos_str)
    except ValueError:
        raise ValueError(f"Invalid position in variant ID: {pos_str}")

    if position <= 0:
        raise ValueError("Position must be a positive integer.")

    ref = ref.strip().upper()
    alt = alt.strip().upper()
    if not all(allele in "ACGT" for allele in [ref, alt]):
        raise ValueError(f"Invalid alleles: {ref}, {alt}. Must be ACGT.")

    return f"{chromosome}_{position}_{ref}_{alt}"


def lookup_embedding(
    variant_id: str,
    metadata: Dict[str, Any]
) -> Optional[np.ndarray]:
    """
    Look up embedding vector for a variant from metadata index.
    
    Args:
        variant_id: Canonical variant identifier
        metadata: Metadata dictionary with variant index
    
    Returns:
        Embedding vector if found, None otherwise
    """
    variant_map = metadata.get("variant_id_map", {})
    
    if variant_id not in variant_map:
        return None
    
    idx = variant_map[variant_id]
    return idx  # Return index; actual embedding lookup handled in batch processing


def score_single_variant(
    variant_id: str,
    model: RandomForestClassifier,
    embeddings: np.ndarray,
    metadata: Dict[str, Any],
    threshold: float = 0.5
) -> VariantScore:
    """
    Score a single variant.
    
    Args:
        variant_id: Canonical variant identifier
        model: Trained classifier
        embeddings: Embeddings array
        metadata: Metadata dictionary
        threshold: Decision threshold (default 0.5)
    
    Returns:
        VariantScore object with prediction and confidence
    
    Raises:
        ValueError: If variant not found in embedding data
    """
    # Validate variant ID format
    try:
        normalized_variant_id = normalize_variant_identifier(variant_id)
        parse_variant_identifier(variant_id)
    except ValueError as e:
        raise ValueError(f"Invalid variant ID '{variant_id}': {e}")
    
    # Lookup embedding
    variant_map = metadata.get("variant_id_map", {})
    if normalized_variant_id not in variant_map:
        raise ValueError(f"Variant '{variant_id}' not found in precomputed embeddings")
    
    idx = variant_map[normalized_variant_id]
    embedding = embeddings[idx:idx+1]  # Shape: (1, 1280)
    
    # Predict
    proba = model.predict_proba(embedding)[0]  # Shape: (2,)
    pathogenicity_prob = proba[1]  # Probability of pathogenic class
    
    prediction = "PATHOGENIC" if pathogenicity_prob >= threshold else "BENIGN"
    confidence = max(proba)
    
    return VariantScore(
        variant_id=normalized_variant_id,
        pathogenicity_score=float(pathogenicity_prob),
        prediction=prediction,
        confidence=float(confidence),
        model="RandomForest"
    )


def score_batch(
    variants: List[str],
    model: RandomForestClassifier,
    embeddings: np.ndarray,
    metadata: Dict[str, Any],
    threshold: float = 0.5
) -> List[VariantScore]:
    """
    Score multiple variants efficiently.
    
    Args:
        variants: List of canonical variant identifiers
        model: Trained classifier
        embeddings: Embeddings array
        metadata: Metadata dictionary
        threshold: Decision threshold
    
    Returns:
        List of VariantScore objects sorted by pathogenicity (descending)
    """
    variant_map = metadata.get("variant_id_map", {})
    scores = []
    failed_variants = []
    
    # Filter to variants with available embeddings
    available_variants = []
    indices = []
    
    for variant_id in variants:
        try:
            normalized_variant_id = normalize_variant_identifier(variant_id)
            parse_variant_identifier(variant_id)
            if normalized_variant_id in variant_map:
                available_variants.append(normalized_variant_id)
                indices.append(variant_map[normalized_variant_id])
            else:
                failed_variants.append((variant_id, "Not found in embedding data"))
        except ValueError as e:
            failed_variants.append((variant_id, str(e)))
    
    if not available_variants:
        print("[WARNING] No valid variants with embeddings found.")
        return scores
    
    # Batch predict
    embedding_batch = embeddings[indices]
    probas = model.predict_proba(embedding_batch)
    
    for variant_id, proba in zip(available_variants, probas):
        pathogenicity_prob = proba[1]
        prediction = "PATHOGENIC" if pathogenicity_prob >= threshold else "BENIGN"
        confidence = max(proba)
        
        scores.append(VariantScore(
            variant_id=variant_id,
            pathogenicity_score=float(pathogenicity_prob),
            prediction=prediction,
            confidence=float(confidence),
            model="RandomForest"
        ))
    
    # Report failures
    if failed_variants:
        print(f"[WARNING] {len(failed_variants)} variants could not be scored:")
        for variant_id, reason in failed_variants:
            print(f"  - {variant_id}: {reason}")
    
    # Sort by pathogenicity score (descending)
    scores.sort(key=lambda x: x.pathogenicity_score, reverse=True)
    
    return scores


# ============================================================================
# INPUT/OUTPUT HANDLING
# ============================================================================

def read_input_file(input_file: Path, format_hint: str = "auto") -> List[str]:
    """
    Read variant identifiers from input file.
    
    Supports CSV, TSV, and whitespace-separated formats.
    Expects column named 'variant_id' or 'variant' or uses first column.
    
    Args:
        input_file: Path to input file
        format_hint: File format hint ("csv", "tsv", "vcf", "auto")
    
    Returns:
        List of variant identifiers
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Detect format
    if format_hint == "auto":
        if input_file.suffix.lower() == ".csv":
            format_hint = "csv"
        elif input_file.suffix.lower() in [".tsv", ".vcf"]:
            format_hint = "tsv"
        else:
            format_hint = "csv"
    
    # Read variants
    if format_hint in ["csv", "tsv"]:
        sep = "," if format_hint == "csv" else "\t"
        df = pd.read_csv(input_file, sep=sep)
        
        # Find variant column
        variant_col = None
        for col in ["variant_id", "variant", "variant_name", "name"]:
            if col in df.columns:
                variant_col = col
                break
        
        if variant_col is None:
            # Use first column
            variant_col = df.columns[0]
            print(f"[INFO] Using first column '{variant_col}' for variant identifiers")
        
        variants = df[variant_col].astype(str).tolist()
    else:
        raise ValueError(f"Unsupported format: {format_hint}")
    
    return variants


def write_output_csv(scores: List[VariantScore], output_file: Path) -> None:
    """Write scoring results to CSV file."""
    df = pd.DataFrame([asdict(s) for s in scores])
    df.to_csv(output_file, index=False)
    print(f"[INFO] Results written to {output_file}")


def write_output_json(scores: List[VariantScore], output_file: Path) -> None:
    """Write scoring results to JSON file with metadata."""
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": "RandomForest",
            "n_variants_scored": len(scores),
            "n_pathogenic": sum(1 for s in scores if s.prediction == "PATHOGENIC"),
            "n_benign": sum(1 for s in scores if s.prediction == "BENIGN"),
        },
        "results": [asdict(s) for s in scores]
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"[INFO] Results written to {output_file}")


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Score missense variants for pathogenicity using ESM2 embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score a single variant
    python scripts/score_variants.py --variant chr1_1022225_G_A --output app/demo_outputs/cli_real_model_output.csv

  # Batch score from CSV file
    python scripts/score_variants.py --input app/demo_outputs/demo_batch_upload_score_source.csv --output app/demo_outputs/cli_batch_output.csv

  # Batch score with JSON output
    python scripts/score_variants.py --input variants.tsv --output app/demo_outputs/results.json --format json

  # Adjust decision threshold
    python scripts/score_variants.py --input app/demo_outputs/demo_batch_upload_score_source.csv --output app/demo_outputs/cli_batch_output_t06.csv --threshold 0.6
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--variant",
        type=str,
        help="Score a single variant (canonical ID: chr_pos_ref_alt, e.g., chr1_100000_A_G)"
    )
    input_group.add_argument(
        "--input", "-i",
        type=Path,
        help="Path to input CSV/TSV file with variant identifiers"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("variant_scores.csv"),
        help="Output file path (CSV or JSON based on extension, default: variant_scores.csv)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["csv", "json"],
        default=None,
        help="Output format (auto-detected from --output extension if not specified)"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Decision threshold for pathogenic classification (default: 0.5)"
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR / "rf_model.pkl",
        help="Path to trained model pickle file (default: models/rf_model.pkl)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with detailed information"
    )
    
    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_arguments()
    
    try:
        # Load model and data
        print("[INFO] Loading model and embeddings...")
        embeddings, metadata = load_embeddings_and_metadata(strict=True)
        model = load_trained_model(model_path=args.model_path, n_features=embeddings.shape[1])
        validate_model_feature_compatibility(model, embeddings)
        print(f"[INFO] Loaded embeddings: shape={embeddings.shape}")
        print(f"[INFO] Loaded metadata: {len(metadata.get('variant_id_map', {}))} variants indexed")
        
        # Determine output format
        output_format = args.format
        if output_format is None:
            if args.output.suffix.lower() == ".json":
                output_format = "json"
            else:
                output_format = "csv"
        
        # Process input
        if args.variant:
            # Single variant mode
            print(f"[INFO] Scoring single variant: {args.variant}")
            score = score_single_variant(
                args.variant,
                model,
                embeddings,
                metadata,
                threshold=args.threshold
            )
            scores = [score]
            
            print(f"\n[RESULT]")
            print(f"  Variant: {score.variant_id}")
            print(f"  Prediction: {score.prediction}")
            print(f"  Pathogenicity Score: {score.pathogenicity_score:.4f}")
            print(f"  Confidence: {score.confidence:.4f}")
        
        else:
            # Batch mode
            print(f"[INFO] Reading variants from {args.input}...")
            variants = read_input_file(args.input)
            print(f"[INFO] Found {len(variants)} variants to score")
            
            print(f"[INFO] Scoring variants (threshold={args.threshold})...")
            scores = score_batch(
                variants,
                model,
                embeddings,
                metadata,
                threshold=args.threshold
            )
            
            print(f"[INFO] Scored {len(scores)} variants successfully")
            print(f"  - Pathogenic: {sum(1 for s in scores if s.prediction == 'PATHOGENIC')}")
            print(f"  - Benign: {sum(1 for s in scores if s.prediction == 'BENIGN')}")
        
        # Write output
        print(f"[INFO] Writing results to {args.output} ({output_format.upper()})...")
        if output_format == "json":
            write_output_json(scores, args.output)
        else:
            write_output_csv(scores, args.output)
        
        print(f"\n[SUCCESS] Completed variant scoring")
        print(f"  Output: {args.output}")
        print(f"  Variants scored: {len(scores)}")
        
        return 0
    
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"[ERROR] Invalid input: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
