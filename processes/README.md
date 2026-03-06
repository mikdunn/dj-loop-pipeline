# Process-Oriented Workspace Map

This folder groups the project into execution phases so it is easier to follow.

## Recommended order

1. `01_data_prep` - build labels and prep loop/drumbreak datasets
2. `02_training` - train rankers/classifiers
3. `03_classification` - run inference/classification heads
4. `04_evaluation` - run sweeps, ablations, comparisons
5. `05_export` - export loop assets
6. `06_analysis` - analyze outcomes and diagnostics
7. `07_orchestration` - end-to-end and high-level pipeline entrypoints

> Note: Existing script paths remain unchanged (safe cleanup). This map organizes *how* to run them.
