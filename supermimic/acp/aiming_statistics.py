"""Distance summaries that distinguish invalid rays from finite misses."""
import numpy as np


def error_summary(errors):
    values=np.asarray(errors).reshape(-1)
    if not len(values):raise ValueError('Empty error sample')
    valid=np.isfinite(values)
    finite=values[valid]
    # Invalid rays rank worse than all finite misses. Never replace them with
    # float32 max, which is not a physical distance.
    ordered=np.sort(np.where(valid,values,np.inf))
    def quantile(q):
        # Empirical inverse CDF (nearest rank), including invalid outcomes.
        value=ordered[max(0,int(np.ceil(q*len(ordered)))-1)]
        return float(value) if np.isfinite(value) else None
    return dict(median_m=quantile(.5),p95_m=quantile(.95),
                invalid_count=int((~valid).sum()),sample_count=len(values),
                invalid_fraction=float((~valid).mean()),
                finite_only_p95_m=float(np.quantile(finite,.95)) if len(finite) else None,
                quantile_method='nearest_rank_invalid_as_worst; null means invalid ray')
