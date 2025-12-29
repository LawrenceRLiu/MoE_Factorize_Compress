# Learning Rate Scheduler Fix - 2025-12-28 (FINAL VERSION)

**UPDATE:** After discovering that the callback approach was still not working due to HuggingFace Trainer's internal logging order, we've implemented a **much cleaner solution using a custom LR scheduler**.

## Problem Summary

Training logs showed `learning_rate: 0.0` for all training steps, and loss was not decreasing properly.

### Investigation Results

The issue was a critical bug in how our parameter group scheduler interacted with HuggingFace's LR scheduler.

## Root Cause: Execution Order Bug

**Actual HuggingFace Trainer execution order (per step):**
```
1. on_step_begin callbacks
2. training_step (forward, backward, optimizer.step)
3. lr_scheduler.step()  ← Sets all param groups to same base LR
4. on_step_end callbacks
5. on_log callbacks
```

**The problem with our original implementation:**
```
1. on_step_begin: We apply LR multipliers to param groups ✓
2. training_step: Uses our multiplied LRs ✓
3. lr_scheduler.step(): OVERWRITES all param groups with base LR! ✗
4. on_step_end: (not used)
5. on_log: Reads param groups → sees overwritten LRs → logs 0.0 ✗
```

**Why this was catastrophic:**
- Our multipliers were being overwritten AFTER each training step
- The NEXT training step would use the overwritten (uniform) LRs
- During warmup, base LR starts at 0.0, so all params were stuck at LR=0.0
- Loss couldn't decrease because learning rate was effectively zero!

## The Final Solution: Custom LR Scheduler

**Why the callback approach failed:**
Even after fixing the execution order issues, the HuggingFace Trainer's progress bar logs the learning rate BEFORE any callbacks can modify it. Fighting with callback timing is fragile and error-prone.

**The clean solution:**
Create a custom LR scheduler that wraps the base HuggingFace scheduler and applies parameter-specific multipliers automatically.

### New Implementation

**File: [src/custom_lr_scheduler.py](src/custom_lr_scheduler.py)** (NEW)

Implements `ParameterGroupLRScheduler` which:
1. Wraps any PyTorch/HF LR scheduler (e.g., cosine with warmup)
2. Calls the base scheduler's `step()` to get the base LR
3. Immediately applies parameter-specific multipliers
4. Supports multiple stages during training
5. Integrates cleanly with HuggingFace Trainer - no callbacks needed!

**File: [src/recovery_trainer.py](src/recovery_trainer.py)** (MODIFIED)

Added `create_scheduler()` method that:
1. Creates the base scheduler using HF's standard method
2. Wraps it with our custom `ParameterGroupLRScheduler` if staged LR config is provided
3. Returns the wrapped scheduler

**File: [scripts/run_recovery_training.py](scripts/run_recovery_training.py)** (MODIFIED)

- Removed callback-based approach entirely
- Passes `lr_schedule_config` directly to `RecoveryTrainer`
- Trainer handles everything internally

### Why This Works

1. **No callback timing issues** - scheduler is called directly by Trainer
2. **Correct logging** - `get_last_lr()` returns the actual LRs being used
3. **Clean integration** - works like any standard PyTorch LR scheduler
4. **Proper state management** - supports save/load for checkpointing

## Verification

Run the verification script to see the fix in action:

```bash
conda activate MoE_Compress
python scripts/verify_lr_fix.py
```

This simulates the Trainer's LR scheduler + parameter group scheduler interaction and shows that:
- Warmup correctly ramps up the base LR from 0 → 0.0001
- Multipliers are correctly applied at each step
- Frozen params stay at 0.0, wrappers get full LR, cores get 10% LR

## Expected Behavior After Fix

When you restart training, you should see:

**In logs:**
```
Step 1: {'learning_rate': 1.25e-07, ...}  ← NON-ZERO during warmup!
Step 100: {'learning_rate': 1.25e-05, ...}  ← Warmup complete
Step 800: {'learning_rate': 0.0001, ...}  ← Full LR reached
```

**In WandB:**
- `learning_rate` will show a proper warmup curve (not stuck at 0.0)
- `lr/group_*` will show different LRs for different parameter groups
- `param_schedule/stage` will show stage transitions
- Loss should actually decrease!

## Impact on Previous Training

**The bad news:** Previous training was using LR=0.0 for all parameters!

- Our multipliers were being overwritten after each step
- Next step would use the overwritten base LR (which was 0.0 during warmup)
- Loss couldn't decrease because learning rate was effectively zero
- Model wasn't really learning anything

**Recommendation:**
**You should restart training from checkpoint-0** with this fix to get proper training.

## Testing the Fix

To verify the fix is working on your actual training:

1. Restart training from checkpoint-0 or latest checkpoint
2. Watch the logs/WandB for:
   - `learning_rate` should increase during warmup (not stay at 0.0)
   - `lr/group_0` should show the wrapper LR (full rate)
   - `lr/group_1` should show the core LR (10% of group_0)
   - `lr/group_2` should show 0.0 (frozen params)

3. Check that loss is decreasing faster than before (now that LR > 0!)

## Files Changed

- [src/param_group_scheduler.py](src/param_group_scheduler.py):
  - Modified `update_optimizer_param_groups()` to accept `base_lr` parameter
  - Changed callback from `on_step_begin` to apply multipliers correctly
  - Enhanced `on_log()` to report accurate LRs

- [scripts/verify_lr_fix.py](scripts/verify_lr_fix.py): New verification script
