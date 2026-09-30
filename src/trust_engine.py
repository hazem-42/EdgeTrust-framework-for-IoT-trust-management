"""
trust_engine.py -- Layer 1 (LightGBM) wrapper, Eqs. (3)-(6) trust engine
(bounded composite score, Holt forecast, sigmoid threshold, three-state
hysteresis with the 3-epoch promotion-confirmation rule), and a minimal
ABAC-lite mapping from trust state to an access decision.

This is the real-data counterpart of the synthetic EdgeTrust class used
in earlier mechanism-verification figures: same six equations, same
parameters (config.PARAMS), but q_t now comes from an actually-trained
LightGBM model and r_t comes from actually-normalized CICIoT2023 rows,
not synthetic input.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train_layer1(X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series,
                  seed: int = C.RANDOM_SEED) -> lgb.LGBMClassifier:
    """Trains the operational Layer-1 LightGBM model on the 14 selected features only."""
    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        random_state=seed, verbosity=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )
    return model


class EdgeTrustEngine:
    """
    One instance per monitored device. Call .step(r_t, q_t) once per epoch.
    r_t: length-m array of Eq.(1)-normalized behavioral evidence in [0,1].
    q_t: LightGBM attack probability in [0,1] (already excludes hard-blocked epochs).
    """

    def __init__(self, weights: np.ndarray, params: C.EdgeTrustParams = C.PARAMS):
        self.w = np.asarray(weights, dtype=float)
        self.p = params
        self.T_hist = 0.5
        self.l = 0.5
        self.b = 0.0
        self.s = 0          # 0=TRUSTED, 1=WARNING, 2=QUARANTINE
        self.c = 0          # promotion-confirmation counter

    @classmethod
    def new_first_join(cls, weights, params=C.PARAMS):
        return cls(weights, params)

    @classmethod
    def new_post_block(cls, weights, params=C.PARAMS):
        eng = cls(weights, params)
        eng.T_hist, eng.l, eng.s = 0.2, 0.2, 1
        return eng

    def step(self, r_t: np.ndarray, q_t: float):
        p = self.p
        if q_t >= p.hard_block_threshold:
            self.s = 2
            self.c = 0
            return dict(T_t=None, T_pred=None, tau_t=None, state=self.s, hard_blocked=True)

        # Eq. (3)
        T_t = (np.dot(self.w, r_t) + p.rho * self.T_hist + p.alpha * (1 - q_t)) / (1 + p.rho + p.alpha)

        # Eq. (4)
        l_prev, b_prev = self.l, self.b
        l_t = p.alpha_H * T_t + (1 - p.alpha_H) * (l_prev + b_prev)
        b_t = p.beta_H * (l_t - l_prev) + (1 - p.beta_H) * b_prev
        T_pred = float(np.clip(l_t + p.h * b_t, 0, 1))

        # Eq. (5)
        tau_t = p.tau_base + (p.tau_max - p.tau_base) * sigmoid(p.kappa * (T_pred - p.mu))

        # Eq. (6), extended with the promotion-confirmation counter
        if self.s < 2 and T_pred < (tau_t - p.delta):
            self.s += 1
            self.c = 0
        elif self.s > 0 and T_pred > (tau_t + p.delta):
            self.c += 1
            if self.c >= p.promote_confirm:
                self.s -= 1
                self.c = 0
        else:
            self.c = 0

        self.l, self.b, self.T_hist = l_t, b_t, T_t
        return dict(T_t=T_t, T_pred=T_pred, tau_t=tau_t, state=self.s, hard_blocked=False)


STATE_NAMES = {0: "TRUSTED", 1: "WARNING", 2: "QUARANTINE"}

# ----------------------------------------------------------------------
# ABAC-lite: maps trust state -> access decision, matching the lifecycle
# policy described in Section III.2 (read-only in WARNING, full block in
# QUARANTINE). This is a minimal stand-in, not a full ABAC policy engine;
# replace with your actual policy evaluator if you have one.
# ----------------------------------------------------------------------
def abac_decision(state: int) -> str:
    if state == 0:
        return "PERMIT"
    if state == 1:
        return "RESTRICT (read-only telemetry; firmware/actuation denied)"
    return "DENY"


def run_stream(engine: EdgeTrustEngine, R: np.ndarray, Q: np.ndarray) -> pd.DataFrame:
    """Runs a sequence of (r_t, q_t) pairs through one device's engine and
    returns a per-epoch record, including the ABAC decision at each step."""
    records = []
    for t in range(len(Q)):
        out = engine.step(R[t], Q[t])
        records.append({
            "epoch": t,
            "T_t": out["T_t"],
            "T_pred": out["T_pred"],
            "tau_t": out["tau_t"],
            "state": out["state"],
            "state_name": STATE_NAMES[out["state"]],
            "hard_blocked": out["hard_blocked"],
            "decision": "DENY" if out["hard_blocked"] else abac_decision(out["state"]),
        })
    return pd.DataFrame(records)
