# Self-Driving Policy Sweep

Synthetic policy sweep inspired by DriveLM / DriveBench-style perception failures and safety-critical local evidence tasks.
Generated with `120` scenes per task-condition-seed across seeds `[7, 11, 23]`.

## Key Findings

- `active_verify` is the strongest non-oracle policy overall with grounded accuracy `0.620`.
- Compared with `passive_cot`, `active_verify` improves grounded accuracy by `0.620` overall and `0.481` on `small_object` scenes.
- Under answer-only reward, trajectory selection still favors `passive_cot` `0.640` of the time. Under grounded reward, `passive_cot` drops to `0.000` while `oracle_roi` dominates overall at `0.668`.
- Pairwise grounded preference chooses `active_verify` over `passive_cot` `0.765` of the time, versus `0.258` under answer-only preference.
- Budgeted inspection shows diminishing returns past two crops: budget 2 grounded accuracy `0.616` vs budget 3 `0.726`.

## Overall Metrics

| Policy | Accuracy | Grounded Acc. | Evidence Hit | Unsafe Failure | Mean Crops | Grounded Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle_roi | 0.828 | 0.828 | 1.000 | 0.140 | 1.00 | 1.708 |
| active_verify | 0.763 | 0.620 | 0.715 | 0.190 | 2.00 | 1.141 |
| detector_first | 0.709 | 0.519 | 0.632 | 0.234 | 1.00 | 1.110 |
| task_prior_zoom | 0.663 | 0.336 | 0.366 | 0.286 | 1.00 | 0.759 |
| center_zoom | 0.575 | 0.116 | 0.128 | 0.332 | 1.00 | 0.381 |
| passive_cot | 0.640 | 0.000 | 0.000 | 0.294 | 0.00 | 0.352 |

## Reward-Induced Selection Rates

| Reward | passive_cot | center_zoom | task_prior_zoom | detector_first | active_verify | oracle_roi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| answer_only | 0.640 | 0.192 | 0.109 | 0.035 | 0.016 | 0.008 |
| grounded | 0.000 | 0.060 | 0.176 | 0.084 | 0.013 | 0.668 |

## Active Verify Budget Sweep

| Budget | Accuracy | Grounded Acc. | Evidence Hit | Mean Crops | Grounded Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| budget_1 | 0.662 | 0.494 | 0.632 | 1.00 | 1.073 |
| budget_2 | 0.756 | 0.616 | 0.715 | 2.00 | 1.135 |
| budget_3 | 0.803 | 0.726 | 0.823 | 2.34 | 1.270 |
