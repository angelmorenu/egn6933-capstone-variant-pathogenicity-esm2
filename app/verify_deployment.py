#!/usr/bin/env python3
"""
Deployment Interface Verification Script (Fixed for app/ directory structure)
============================================================================

Validates both Streamlit web application and CLI interface for functionality.
"""

import subprocess
import json
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION - FIXED PATHS
# ============================================================================

# Running from app/ directory
APP_ROOT = Path(__file__).parent  # app/
MAIN_ROOT = APP_ROOT.parent  # Main project directory

DATA_DIR = MAIN_ROOT / "data" / "processed"
RESULTS_DIR = MAIN_ROOT / "results"
SCRIPTS_DIR = MAIN_ROOT / "scripts"

APP_FILE = APP_ROOT / "app.py"
CLI_FILE = SCRIPTS_DIR / "score_variants.py"

print("=" * 70)
print("DEPLOYMENT INTERFACE VERIFICATION (Fixed)")
print("=" * 70)
print(f"App Root: {APP_ROOT}")
print(f"Main Root: {MAIN_ROOT}")
print(f"Data Dir: {DATA_DIR}")
print(f"CLI File: {CLI_FILE}")
print(f"Verification: {APP_FILE}")
print()

# ============================================================================
# CHECKS
# ============================================================================

results = []

# [1] Check dependencies
print("[1] Checking dependencies...")
required_packages = {
    "streamlit": "Streamlit web framework",
    "sklearn": "scikit-learn machine learning",
    "pandas": "Data processing",
    "numpy": "Numerical computing",
    "plotly": "Interactive visualizations",
}

missing = []
for package, description in required_packages.items():
    try:
        __import__(package)
        print(f"  ✅ {package}")
    except ImportError:
        print(f"  ❌ {package}")
        missing.append(package)

if not missing:
    results.append(("Dependencies", True))
    print("  ✅ All dependencies installed")
else:
    results.append(("Dependencies", False))
    print(f"  ❌ Missing: {', '.join(missing)}")

# [2] Check file structure
print("\n[2] Checking file structure...")
files_to_check = [
    ("Streamlit app", APP_FILE),
    ("CLI script", CLI_FILE),
    ("Embeddings", DATA_DIR / "week2_training_table_strict_embeddings.npy"),
    ("Metadata", DATA_DIR / "week2_training_table_strict_meta.json"),
    ("Performance metrics", RESULTS_DIR / "error_analysis_report.json"),
]

all_exist = True
for name, path in files_to_check:
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"  {status} {name}")
    all_exist = all_exist and exists

results.append(("File Structure", all_exist))

# [3] Check Streamlit structure
print("\n[3] Checking Streamlit app structure...")
try:
    with open(APP_FILE) as f:
        content = f.read()
    
    required_functions = [
        "main",
        "single_variant_section",
        "batch_upload_section",
        "performance_dashboard_section",
        "about_section"
    ]
    
    found_functions = [fn for fn in required_functions if f"def {fn}" in content]
    
    if len(found_functions) == len(required_functions):
        print(f"  ✅ All {len(required_functions)} required functions found")
        results.append(("Streamlit Structure", True))
    else:
        print(f"  ❌ Missing functions: {set(required_functions) - set(found_functions)}")
        results.append(("Streamlit Structure", False))
except Exception as e:
    print(f"  ❌ Error: {e}")
    results.append(("Streamlit Structure", False))

# [4] Check CLI syntax
print("\n[4] Checking CLI syntax...")
try:
    result = subprocess.run(
        ["python", str(CLI_FILE), "--help"],
        capture_output=True,
        timeout=5,
        text=True,
        cwd=str(MAIN_ROOT)
    )
    
    if result.returncode == 0 and "--variant" in result.stdout:
        print(f"  ✅ CLI help message works")
        results.append(("CLI Syntax", True))
    else:
        print(f"  ❌ CLI help failed")
        results.append(("CLI Syntax", False))

except subprocess.TimeoutExpired:
    print(f"  ❌ CLI check timed out")
    results.append(("CLI Syntax", False))
except Exception as e:
    print(f"  ❌ Error: {e}")
    results.append(("CLI Syntax", False))

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

passed_count = sum(1 for _, passed in results if passed)
total_count = len(results)

for check_name, passed in results:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {check_name}")

print("\n" + "=" * 70)
print(f"RESULT: {passed_count}/{total_count} checks passed")
print("=" * 70)

if passed_count == total_count:
    print("\n✅ **DEPLOYMENT READY**")
    print("\nTo start Streamlit:")
    print(f"  cd {APP_ROOT}")
    print(f"  streamlit run app.py")
    sys.exit(0)
else:
    print("\n⚠️  **INSTALLATION INCOMPLETE**")
    print(f"\n{total_count - passed_count} issue(s) need to be fixed:")
    for check_name, passed in results:
        if not passed:
            print(f"  - {check_name}")
    
    if missing:
        print(f"\nTo install missing packages:")
        print(f"  pip install {' '.join(missing)}")
    
    sys.exit(1)
