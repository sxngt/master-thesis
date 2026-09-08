## Report at step {step:,} ({progress:.0%} of budget), intervention #{k}

### Deterministic evaluation (KPI) — current vs previous report
{kpi_table}

### Objective decomposition (term = weight * KPI)
{objective_table}

### Objective trace (every evaluation, step: J)
{trace}
Best so far: {best}
Noise: {noise}

### Training statistics since last report
{stats_table}

### Learning dynamics (mean over the last {window} PPO updates)
{dynamics_table}

### Evidence (computed)
{phase}
{evidence}

### Intervention history (most recent last)
{history}

Propose the next parameter changes (or none).