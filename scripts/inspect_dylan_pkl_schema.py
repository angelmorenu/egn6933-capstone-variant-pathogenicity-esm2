#!/usr/bin/env python3
"""Inspect Dylan's pickle file schema to identify predictor columns."""

import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

def inspect_pickle_file(pkl_path: str, max_records: int = 100):
    """Load pickle file record-by-record and inspect schema."""
    pkl_path = Path(pkl_path)
    print(f"Loading {pkl_path.name}...")
    print(f"File size: {pkl_path.stat().st_size / (1024**2):.1f} MB\n")
    
    data_records = []
    column_types = {}
    
    try:
        with open(pkl_path, 'rb') as f:
            count = 0
            while True:
                try:
                    record = pickle.load(f)
                    data_records.append(record)
                    count += 1
                    
                    # Track first record's schema
                    if count == 1 and isinstance(record, dict):
                        print(f"Record structure (sample):")
                        for k, v in record.items():
                            if isinstance(v, np.ndarray):
                                column_types[k] = f"ndarray shape={v.shape} dtype={v.dtype}"
                                print(f"  {k}: {column_types[k]}")
                            elif isinstance(v, (int, float)):
                                column_types[k] = type(v).__name__
                                print(f"  {k}: {type(v).__name__} = {v}")
                            elif isinstance(v, str):
                                column_types[k] = "str"
                                print(f"  {k}: str = '{v[:50]}...'")
                            else:
                                column_types[k] = type(v).__name__
                                print(f"  {k}: {type(v).__name__}")
                    
                    if count >= max_records:
                        print(f"\n[Stopping at {max_records} records for preview]")
                        break
                        
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error reading record {count}: {e}")
                    break
        
        print(f"\nTotal records found: {count}")
        
        # Convert all to DataFrame
        if data_records and isinstance(data_records[0], dict):
            df = pd.DataFrame(data_records)
            print(f"\nDataFrame shape: {df.shape}")
            print(f"\nAll columns ({len(df.columns)}):")
            for col in df.columns:
                non_null = df[col].notna().sum()
                print(f"  {col}: {non_null}/{len(df)} non-null")
                
                # Check for predictor-like names
                if any(pred in col.lower() for pred in ['polyphen', 'sift', 'primate', 'cadd', 'revel', 'fathmm', 'dann', 'score', 'prediction']):
                    print(f"    ^^^ PREDICTOR COLUMN ^^^")
            
            return df, column_types
        else:
            print("Records are not dictionaries, cannot convert to DataFrame")
            return None, column_types
            
    except Exception as e:
        print(f"Fatal error: {e}")
        return None, column_types

if __name__ == "__main__":
    pkl_file = "data/Dylan Tan/esm2_selected_features.pkl"
    
    df, schema = inspect_pickle_file(pkl_file)
    
    if df is not None:
        print("\n" + "="*60)
        print("SUMMARY: Expected predictor columns NOT FOUND" if not any(
            pred in col.lower() for col in df.columns 
            for pred in ['polyphen', 'sift', 'primate', 'cadd', 'revel', 'fathmm', 'dann', 'score']
        ) else "SUMMARY: Predictor columns found!")
