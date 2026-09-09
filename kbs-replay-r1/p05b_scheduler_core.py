"""P05B: prediction-only scheduling prototypes, NOT a production experiment runner.

No power-system, detector, recovery or perturbation-generating imports occur here.
The caller owns arrivals/completions and must construct the allowlisted DTOs below.
ATC is explicitly adapted to latest-START deadlines. Rolling planning is a
work-conserving, current-queue-only, finite-horizon MILP, not an online optimum.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
from math import ceil, isfinite, log
from typing import Mapping, Sequence, Any


class ContractError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def integer(x: Any) -> bool:
    return type(x) is int  # do not silently coerce floating indices or booleans


@dataclass(frozen=True, slots=True)
class VisibleJob:
    job_id: int
    arrival: int
    latest_start: int
    predicted_value: float
    predicted_duration: float

    def __post_init__(self) -> None:
        require(integer(self.job_id) and self.job_id >= 0, 'invalid job ID')
        require(integer(self.arrival) and self.arrival >= 0, 'invalid arrival')
        require(integer(self.latest_start) and self.latest_start >= self.arrival,
                'invalid latest-start deadline')
        require(isfinite(self.predicted_value) and self.predicted_value >= 0,
                'invalid predicted value')
        require(isfinite(self.predicted_duration) and self.predicted_duration > 0,
                'invalid predicted duration')


@dataclass(frozen=True, slots=True)
class VisibleActive:
    job_id: int
    started_at: int
    predicted_duration: float

    def __post_init__(self) -> None:
        require(integer(self.job_id) and self.job_id >= 0, 'invalid active ID')
        require(integer(self.started_at) and self.started_at >= 0, 'invalid start')
        require(isfinite(self.predicted_duration) and self.predicted_duration > 0,
                'invalid active duration forecast')


@dataclass(frozen=True, slots=True)
class Observation:
    now: int
    capacity: int
    waiting: tuple[VisibleJob, ...]
    active: tuple[VisibleActive, ...]

    def __post_init__(self) -> None:
        require(integer(self.now) and self.now >= 0, 'invalid clock')
        require(integer(self.capacity) and self.capacity >= 0, 'invalid capacity')
        require(type(self.waiting) is tuple and type(self.active) is tuple,
                'use immutable tuples, not original simulator containers')
        require(all(type(j) is VisibleJob for j in self.waiting), 'wrong job type')
        require(all(type(j) is VisibleActive for j in self.active), 'wrong active type')
        require(len(self.active) <= self.capacity, 'too many observed active jobs')
        require(all(j.arrival <= self.now for j in self.waiting), 'future arrival exposed')
        require(all(j.started_at <= self.now for j in self.active), 'future start exposed')
        ids = [j.job_id for j in self.waiting] + [j.job_id for j in self.active]
        require(len(ids) == len(set(ids)), 'duplicate active/waiting ID')

    @property
    def free_slots(self) -> int:
        return self.capacity - len(self.active)


def _dto(cls: type, row: Mapping[str, Any]):
    expected = {f.name for f in fields(cls)}
    require(type(row) is dict and set(row) == expected,
            f'{cls.__name__}: keys must be exactly {sorted(expected)}')
    return cls(**dict(row))


def observation_from_visible(*, now: int, capacity: int,
                             waiting: Sequence[dict], active: Sequence[dict]) -> Observation:
    """Reject extra keys (labels, actual_* fields, meta, busy_until, etc.).

    The simulator adapter must positively construct new dictionaries. This is
    an interface boundary and NOT a Python security sandbox against malicious code.
    """
    return Observation(now, capacity, tuple(_dto(VisibleJob, x) for x in waiting),
                       tuple(_dto(VisibleActive, x) for x in active))


def remaining_forecast(active: VisibleActive, now: int) -> int:
    require(integer(now) and now >= active.started_at, 'invalid elapsed time')
    # A task observed unfinished remains active even if its forecast has expired.
    return max(1, ceil(active.predicted_duration - (now - active.started_at)))


def _eligible(obs: Observation) -> list[VisibleJob]:
    return sorted((j for j in obs.waiting
                   if obs.now <= j.latest_start and j.predicted_value > 0),
                  key=lambda j: j.job_id)


def atc_order(obs: Observation, *, k: float = 2.0) -> tuple[int, ...]:
    """I_i = E_i / d_hat_i * exp[-(latest_start_i-now)/(k*mean(d_hat))].

    Completion-deadline ATC is NOT reproduced verbatim. An effective predicted
    completion deadline latest_start_i + d_hat_i gives this START-slack version.
    Caller performs the shared arrival gate; this function never sees rejected jobs.
    """
    require(isfinite(k) and k > 0, 'ATC k must be positive')
    jobs = _eligible(obs)
    if not jobs or obs.free_slots == 0:
        return ()
    mean_d = sum(j.predicted_duration for j in jobs) / len(jobs)
    def key(j: VisibleJob):
        log_index = (log(j.predicted_value) - log(j.predicted_duration)
                     - (j.latest_start - obs.now)/(k*mean_d))
        return (-log_index, j.arrival, j.job_id)
    return tuple(j.job_id for j in sorted(jobs, key=key)[:obs.free_slots])


@dataclass(frozen=True, slots=True)
class PlanningConfig:
    time_limit: float = 0.25
    lookahead_steps: int = 10
    max_jobs: int = 64
    max_columns: int = 4096
    max_span: int = 256
    fallback_k: float = 2.0

    def __post_init__(self) -> None:
        require(isfinite(self.time_limit) and self.time_limit > 0, 'bad time limit')
        for name in ('lookahead_steps', 'max_jobs', 'max_columns', 'max_span'):
            value = getattr(self, name)
            require(integer(value) and value >= (0 if name == 'lookahead_steps' else 1),
                    f'bad {name}')
        require(isfinite(self.fallback_k) and self.fallback_k > 0, 'bad fallback k')


def rolling_plan(obs: Observation, cfg: PlanningConfig = PlanningConfig(),
                 *, solver=None) -> dict:
    """One bounded MILP solve; no future-arrival or realized-duration input.

    All positive current jobs are considered, or the WHOLE decision falls back.
    Future starts are a tentative plan, not reservations. Actual completions and
    resource enforcement remain with the simulator. Tail occupancy is modeled.
    Work conservation is an explicit restriction, not a theorem about optimality.
    """
    fallback = atc_order(obs, k=cfg.fallback_k)
    def stop(reason: str, **extra):
        return dict(start_now=list(fallback), plan=[], fallback=True,
                    reason=reason, solver_called=False, **extra)
    jobs = _eligible(obs)
    if not jobs or obs.free_slots == 0:
        return dict(start_now=[], plan=[], fallback=False, reason='no_decision',
                    solver_called=False)
    if len(jobs) > cfg.max_jobs:
        return stop('queue_cap_no_truncation')
    durations = [max(1, ceil(j.predicted_duration)) for j in jobs]
    limits = [min(j.latest_start - obs.now, cfg.lookahead_steps) for j in jobs]
    ncol = sum(x+1 for x in limits)
    span = max(l+d for l, d in zip(limits, durations))
    if ncol > cfg.max_columns or span > cfg.max_span:
        return stop('model_cap_no_truncation')
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.optimize import milp, LinearConstraint, Bounds
    columns = [(i, s) for i, lim in enumerate(limits) for s in range(lim+1)]
    r, c, v = [], [], []
    n = len(jobs)
    for col, (i, s) in enumerate(columns):
        rr = [i] + [n+t for t in range(s, s+durations[i])]
        if s == 0:
            rr.append(n+span)
        r.extend(rr); c.extend([col]*len(rr)); v.extend([1.0]*len(rr))
    A = coo_matrix((v, (r, c)), shape=(n+span+1, len(columns))).tocsc()
    forecast_active = np.array([sum(t < remaining_forecast(a, obs.now)
                                    for a in obs.active) for t in range(span)])
    upper = np.r_[np.ones(n), obs.capacity-forecast_active,
                  min(obs.free_slots, len(jobs))].astype(float)
    lower = np.full_like(upper, -np.inf); lower[-1] = upper[-1]
    scale = max(j.predicted_value for j in jobs)
    objective = -np.array([jobs[i].predicted_value/scale for i, _ in columns])
    solve = milp if solver is None else solver
    try:
        result = solve(c=objective, integrality=np.ones(len(columns)),
                       bounds=Bounds(np.zeros(len(columns)), np.ones(len(columns))),
                       constraints=LinearConstraint(A, lower, upper),
                       options={'time_limit': cfg.time_limit, 'mip_rel_gap': 0.0,
                                'presolve': True, 'disp': False, 'threads': 1})
    except Exception as error:
        out = stop('solver_exception', exception_type=type(error).__name__)
        out['solver_called'] = True
        return out
    status = int(getattr(result, 'status', -1))
    x = getattr(result, 'x', None)
    def invalid(reason: str):
        out = stop(reason, solver_status=status)
        out['solver_called'] = True
        return out
    if x is None:
        return invalid('no_incumbent')
    x = np.asarray(x, dtype=float)
    if x.shape != (len(columns),) or not np.all(np.isfinite(x)):
        return invalid('invalid_incumbent')
    z = np.rint(x)
    if np.max(np.abs(x-z), initial=0.0) > 1e-6 or np.any(z < 0) or np.any(z > 1):
        return invalid('noninteger_incumbent')
    lhs = A @ z
    if np.any(lhs > upper+1e-7) or np.any(lhs < lower-1e-7):
        return invalid('infeasible_incumbent')
    chosen = [columns[j] for j in np.flatnonzero(z)]
    plan = sorted([(jobs[i].job_id, obs.now+s, durations[i]) for i, s in chosen],
                  key=lambda y: (y[1], y[0]))
    start = [job_id for job_id, s, _ in plan if s == obs.now]
    def finite_optional(name):
        q = getattr(result, name, None)
        return None if q is None or not isfinite(float(q)) else float(q)
    return dict(start_now=start, plan=plan, fallback=False,
                reason='checked_feasible_plan', solver_called=True,
                solver_status=status, solver_message=str(getattr(result, 'message', '')),
                solver_reported_gap=finite_optional('mip_gap'),
                predicted_reward=sum(jobs[i].predicted_value for i, _ in chosen),
                optimality_claim='none; see numerical solver status',
                work_conserving=True, forecast_only=True)

