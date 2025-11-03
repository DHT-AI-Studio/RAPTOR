fix(inference): Fix InferenceManager thread safety issues

## Problems Fixed

### 1. Thread Safety Issue in Singleton Pattern Initialization
- Problem: No atomicity guarantee between `hasattr()` check and attribute assignment in `__init__`
- Fix: Use class-level lock to protect the entire initialization process
- Impact: Prevents duplicate initialization in multi-threaded environments

### 2. Thread Safety Issue in Statistics Update
- Problem: `+=` operation on `stats` dictionary is non-atomic, causing data loss during concurrent updates
- Fix: Add `_stats_lock` to protect all statistics update operations
- Impact: Ensures accuracy of statistics data in high-concurrency environments

### 3. Consistency Issue in Statistics Reading
- Problem: Reading statistics without lock protection may result in inconsistent data
- Fix: Create data snapshot within lock, perform calculations outside lock
- Impact: Guarantees logically consistent statistics data

## Changes Made

### Core Modifications
- `Aigle/0.1/raptor/AiModelLifecycle/src/inference/manager.py`:
  - `__init__`: Add lock protection, rename `initialized` → `_initialized`
  - `infer`: Use `_stats_lock` to protect statistics updates
  - `get_stats`: Use lock to create consistent data snapshot

### New Files
- `Aigle/0.1/raptor/AiModelLifecycle/docs/THREAD_SAFETY_FIX.md`: Detailed fix documentation

## Backward Compatibility

✅ Fully backward compatible:
- API interface unchanged
- Return format unchanged
- Configuration files unchanged

## Related Issues

- Fixes: Code Review #3 (Singleton Pattern Thread Safety Issues)


