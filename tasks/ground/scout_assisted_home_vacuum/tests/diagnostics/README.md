# Egocentric apartment: development and acceptance

This candidate replaces the overhead-cell protocol from `5a973995` with a forward-facing
continuous control task. Scores, anchor and public practice data from that revision do not transfer.

Run from the task directory with Python 3.12, the packages in `environment/requirements.txt`,
Mesa OSMesa, and `CUDA_VISIBLE_DEVICES="" LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

```sh
python tests/diagnostics/test_task.py -v
python tests/diagnostics/ego_reference.py --seeds 617 804 2110 5201 5202 5203
python tests/diagnostics/replay_ego.py
python tests/diagnostics/ego_gallery.py /tmp/ego-gallery
python tests/diagnostics/export_ego_practice.py /tmp/public-practice
```

The actual camera JPEGs, from one continuously flown public trajectory on seed 617, are in
[onboard/](onboard/). The side-by-side image is only an enlargement of these sensor frames.
No bird's-eye render or state map is provided to the participant.

**What ran.** [Per-home measurements](pilot-results.json): the privileged pilot completed
all six development homes and six additional layouts (9–12 rooms, 454–658 square metres).
The public seed 617 [action trace](onboard/success-617.jsonl) replays against fresh physics;
it contains only left/right wheel, suction, pitch, roll, lift and yaw commands. It is not
a solution to another apartment. The [flight triptych](onboard/egocentric-scout.jpg)
comes from ticks 34, 77 and 119 of a separate continuously flown survey.

A capped SOL closed-loop attempt exhausted its input budget without submitting a controller.
This is an inconclusive run, **not model failure on the task**. In a separate one-call,
three-image sanity check the model recognized the bed, sofa and blue work slip but
misidentified the bookcase. It correctly could not infer apartment-wide north from one
bedroom still. The cumulative/model-reported usage for both probes is in the measurements.
Public terminal practice now writes JPEGs to files by default, avoiding base64 transcripts.

**Keep three kinds of result separate.** The map-assisted pilot in `ego_reference.py` uses
hidden geometry and robot pose to steer; it reads the final swatch from real aerial pixels,
pushes the carton through physical contact, and earns the same raw outcome as any program.
It is a feasibility witness, not a participant and not the ALE anchor. The unprivileged
`solution/controller.py` is a minimal RGB/odometry wall-following bootstrap. The hidden
grader never imports or runs the privileged pilot for a participant. Until a competitive
unprivileged reference is measured, the anchor is deliberately null and normalized reward
is not admissible; use `mean_cleaning_score` and `full_success_rate` for pilot measurements.

The public practice MJB is a single fixed, baked apartment and includes its own local scene
metadata. It contains neither a hidden scene generator nor a hidden seed. The public runner
is generated from the same allowlisted camera and actuation methods; a parity test compares
the complete observation after applying actions. For a formal PR run the repository's
ordinary lint, domain checks, native validation and independent frontier calibration.
