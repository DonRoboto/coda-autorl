CODA KL-UQ pilot
================

Files
-----
1. coda_scheduler_KL_UQ_PB2SPACE.py
2. CODA_KL_UQ_PB2SPACE_experiment.py

Place both files in the project root, next to:

    w_pb2_experimentos_config.py

Communication design
--------------------
I2O:
    PPO approximate policy KL -> bounded policy-update state -> EMA context.
    Value-function explained variance is logged, but it is not used by CODA.

O2I:
    completed-observation TV-GP posterior std
    / fitted-kernel prior std
    -> normalized uncertainty U in [0,1]
    -> entropy_coeff = 0.008 * U, capped at 0.008.

The O2I uncertainty is evaluated at the candidate outer configuration under
baseline entropy a0=0. Pending batch proposals do not enter the O2I signal;
they are used only by the auxiliary batch-adjusted UCB variance model.

Run complete CODA
-----------------

    python CODA_KL_UQ_PB2SPACE_experiment.py

The default output name is:

    CODA_KL_UQ_PB2SPACE

This prevents overwriting the earlier HIM-based CODA results.

Directional variants
--------------------
Linux/macOS:

    CODA_VARIANT=i2o python CODA_KL_UQ_PB2SPACE_experiment.py
    CODA_VARIANT=o2i python CODA_KL_UQ_PB2SPACE_experiment.py

Windows PowerShell:

    $env:CODA_VARIANT="i2o"; python CODA_KL_UQ_PB2SPACE_experiment.py
    $env:CODA_VARIANT="o2i"; python CODA_KL_UQ_PB2SPACE_experiment.py

Smoke test
----------
Set in CODA_KL_UQ_PB2SPACE_experiment.py:

    HOPPER_SMOKE_TEST = True

and choose the pilot seeds in HOPPER_TEST_SEEDS.

Essential audit columns
-----------------------
The generated metrics CSV should satisfy:

    custom_metrics/coda_effective_hp_mismatch_count == 0
    0 <= custom_metrics/coda_o2i_uncertainty_normalized <= 1
    0 <= config/entropy_coeff <= 0.008

For guided full/O2I updates, approximately:

    config/entropy_coeff
        = 0.008 * coda_o2i_uncertainty_normalized

unless clipping is active.

The KL-only I2O signal is stored as:

    custom_metrics/policy_update_state
    custom_metrics/policy_update_state_raw
    custom_metrics/policy_update_state_valid

Held-out test script
--------------------
Before testing this pilot, update the CODA entry in RUN_NAMES to:

    "CODA": "CODA_KL_UQ_PB2SPACE"

Do not pool these results with the earlier HIM-based CODA campaign. This is a
new CODA specification and must be evaluated with its own complete set of
training seeds and directional variants if selected as the final method.
