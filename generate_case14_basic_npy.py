"""
Backward-compatible entrypoint.

Historically this repo used `generate_case14_basic_npy.py` for the case14 bank.
Keep the filename working, but delegate to the new case-agnostic generator so
both case14 and case39 follow the same code path.
"""

from generate_case_basic_npy import main


if __name__ == '__main__':
    main()
