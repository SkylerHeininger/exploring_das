# Training runs notes

Since we are doing LOOCV, we only want to see some decrease in loss between each fold - otherwise there is overfitting across runs. Using a lower learning rate and fewer epochs helps here.

## Hyperparameters that seem to work well for CNN approach

LRs: 5e-4 or 1e-4 (give both options as can differ run to run for grid search)

epochs: 30 at most

thresholds: 0.4

hidden dims: 32

embed dims: 64

Model shape:
Kernel size: 11
Number of layers: 11
These can differ some together, but seems as though a balanced model (if it turns wide, then make longer, visa versa).

epochs: 30 (20 if larger model)

negative decrease rate: 1.0 (dont decrease at all, account for in other ways)

positive weight scaling: 1.0

Adding context:
Allow some extra DAs before and after the important chunk, as these possible set up / follow up with context for an important chunk.



Does very poorly on patient stuff - not really much to do here, likely due to lack of labels across the board here.

