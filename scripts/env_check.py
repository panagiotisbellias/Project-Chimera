# scripts/env_check.py

import sys
import pandas_ta

print("\n--- ENVIRONMENT SANITY CHECK ---")
try:
    print(f"Python Executable Path: {sys.executable}")
    print(f"Pandas TA Library Version: {pandas_ta.version}")
    
    # Check if the 'Strategy' attribute actually exists
    if hasattr(pandas_ta, 'Strategy'):
        print("✅ 'Strategy' class FOUND in this environment.")
    else:
        print("❌ 'Strategy' class NOT FOUND in this environment.")

except Exception as e:
    print(f"An error occurred: {e}")

print("------------------------------\n")