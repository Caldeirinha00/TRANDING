import json
import sys
import math
from pathlib import Path

TOLERANCE_REL = 1e-9
TOLERANCE_ABS_FLOOR = 1e-12

FLOAT_FIELDS = [
    "h1_adx", "m5_atr", "m5_atr_median", "m5_ema20",
    "trigger_open", "trigger_high", "trigger_low", "trigger_close"
]
DISCRETE_FIELDS = ["symbol", "direction", "setup_type", "reason_code", "h1_ctx_bar_time"]

def is_close(a: float, b: float) -> bool:
    if a is None or b is None:
        return a == b
    diff = abs(a - b)
    scale = max(abs(a), abs(b), TOLERANCE_ABS_FLOOR)
    return diff <= TOLERANCE_REL * scale

def load_jsonl(file_path: Path) -> dict:
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            bar_time = record["bar_time_m5"]
            if bar_time in data:
                raise ValueError(f"Duplicate bar_time_m5 {bar_time} at line {line_num} in {file_path}")
            data[bar_time] = record
    return data

def verify_parity(py_path: str, mql_path: str):
    py_data = load_jsonl(Path(py_path))
    mql_data = load_jsonl(Path(mql_path))
    
    # 1. Set Equality Check
    py_keys = set(py_data.keys())
    mql_keys = set(mql_data.keys())
    
    missing_in_mql = sorted(list(py_keys - mql_keys))
    missing_in_py = sorted(list(mql_keys - py_keys))
    
    if missing_in_mql or missing_in_py:
        print(f"[FAIL] Set Equality Violation!")
        print(f"Total Python bars: {len(py_keys)} | Total MQL5 bars: {len(mql_keys)}")
        if missing_in_mql:
            print(f"Sample missing in MQL5 (first 5): {missing_in_mql[:5]}")
        if missing_in_py:
            print(f"Sample missing in Python (first 5): {missing_in_py[:5]}")
        sys.exit(1)
        
    print(f"[PASS] Set Equality: {len(py_keys)} bars perfectly matched in time.")
    
    # 2. Field-by-Field Parity
    mismatches = 0
    total_comparisons = 0
    import collections
    reason_code_counts = collections.defaultdict(int)
    
    for bar_time in sorted(py_keys):
        py_row = py_data[bar_time]
        mql_row = mql_data[bar_time]
        
        # Tier 1: Discrete Parity (100% Exact)
        for field in DISCRETE_FIELDS:
            total_comparisons += 1
            if py_row.get(field) != mql_row.get(field):
                print(f"[FAIL] Tier 1 Mismatch at bar {bar_time} ({field}): "
                      f"PY='{py_row.get(field)}' != MQL='{mql_row.get(field)}'")
                mismatches += 1
                
        # LLM Veto Guard
        if py_row.get("vetoed_by_llm") is not False or mql_row.get("vetoed_by_llm") is not False:
            print(f"[FAIL] vetoed_by_llm is not false at bar {bar_time}. Py={py_row.get('vetoed_by_llm')}, MQL={mql_row.get('vetoed_by_llm')}")
            mismatches += 1

        # Track occurrences for quota check
        rc = py_row.get("reason_code")
        if rc:
            reason_code_counts[rc] += 1
                
        # Tier 2: Numerical Parity (Relative Tolerance < 1e-9)
        for field in FLOAT_FIELDS:
            total_comparisons += 1
            val_py = py_row.get(field)
            val_mql = mql_row.get(field)
            if not is_close(val_py, val_mql):
                diff = abs(val_py - val_mql) if (val_py is not None and val_mql is not None) else "N/A"
                py_str = f"{val_py:.12f}" if val_py is not None else "None"
                mql_str = f"{val_mql:.12f}" if val_mql is not None else "None"
                print(f"[FAIL] Tier 2 Float Mismatch at bar {bar_time} ({field}): "
                      f"PY={py_str} != MQL={mql_str} | Diff={diff}")
                mismatches += 1
                
        if mismatches >= 20:
            print(f"\n[ABORT] Exceeded 20 mismatches. Halting diff.")
            sys.exit(1)
            
    # Quotas verification
    missing_quotas = []
    expected_reasons = {"ADX_LOW", "ATR_LOW", "ATR_HIGH", "NO_BULL_PULLBACK", "NO_BEAR_PULLBACK", "VALID"}
    for reason in expected_reasons:
        if reason_code_counts[reason] == 0:
            missing_quotas.append(reason)
            
    if missing_quotas:
        print(f"[WARN] Quota coverage incomplete. Zero occurrences for: {missing_quotas}")
        mismatches += 1

    if mismatches == 0:
        print(f"[SUCCESS] 100% Parity Achieved across {total_comparisons} discrete and float assertions.")
    else:
        print(f"[FAIL] Completed with {mismatches} mismatches.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_parity.py <python_footprint.jsonl> <mql5_footprint.jsonl>")
        sys.exit(1)
    verify_parity(sys.argv[1], sys.argv[2])
