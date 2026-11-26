"""Evaluate an unprivileged submitted policy on hidden continuous apartments."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ego_sim import Simulation
from robotics_grader import (
    Solver,
    SolverAborted,
    SolverMissing,
    ratio_rewards,
    read_anchor,
    seed_rng,
    stage_config,
    write_verdict,
    zero_verdict,
)


def main():
    cfg = stage_config()
    anchor, meta = read_anchor()
    rng = seed_rng()
    try:
        solver = Solver(step_timeout=cfg["step_timeout_s"])
    except (SolverMissing, SolverAborted) as err:
        write_verdict(zero_verdict(reason=str(err), anchored=anchor is not None))
        return
    rows = []
    aborted = 0
    try:
        for _ in range(cfg["episodes"]):
            sim = Simulation(rng.seed(bits=64), cfg["ticks"], cfg["flight_ticks"])
            try:
                solver.reset()
                while sim.t < sim.budget:
                    action = solver.act(sim.t, sim.observation())
                    try:
                        sim.advance(action)
                    except ValueError as err:
                        raise SolverAborted(str(err)) from err
                    if sim.result()["success"]:
                        break
                rows.append(sim.result())
            except SolverAborted:
                aborted += 1
                rows.append({})
            finally:
                sim.close()
    finally:
        solver.close()
    n = len(rows)

    def avg(k):
        return sum(float(row.get(k, 0)) for row in rows) / n

    write_verdict(
        ratio_rewards(
            avg("score"),
            anchor,
            higher_is_better=True,
            full_at=meta["full_at"],
            metric_name="mean_cleaning_score",
            extra={
                "episodes": n,
                "episodes_aborted": aborted,
                "full_success_rate": avg("success"),
                "mean_clean_fraction": avg("clean_fraction"),
                "carton_staged_rate": avg("staged"),
                "carton_parked_rate": avg("parked"),
                "mean_treatment_preserved": avg("treatment_preserved"),
            },
        )
    )


if __name__ == "__main__":
    main()
