# Scout-assisted home vacuum: review materials

[Task PR](https://github.com/Grounded-Intelligence-Initiative/agentic-robotics-benchmark/pull/210) · [Tested task revision](https://github.com/Waybaba/agentic-robotics-benchmark/tree/c0c9768194f73d7f106f39b34ec59dc6d21fe3ce/tasks/ground/scout_assisted_home_vacuum)

Find five requested spills, move a supply carton into a room and back out to its hall mat, protect the runners and dock. The scout has 36 ticks; the joint team has 200. No robot is given a global map. These review assets are not installed in the solver container.

![Daylight presentation view of the furnished apartment](room-daylight.jpg)

The daylight image is a CPU path trace of the simulator's furniture, wall geometry and material bitmaps. It replaces the lighting and uses a review camera. It is not a photograph and is not what a robot sees. [Native MuJoCo room view](engine-interior.jpg) · [Simultaneous onboard images](robot-cameras.png) · [9-second reference run](replay.mp4) · [Animated replay](replay.gif). Each uses the same public example home (617). The reference's decisions in the replay use only the onboard images and normal action feedback, never the extra reviewer information.

**Validation.** [Native engine record](validation.json) and [reference metrics](oracle.json). The reference completed all six hidden homes for a score of 1.0; the empty submission scored 0.0. The per-task tests also execute both physical deliveries on each of 100 generated homes. Only the two requested rooms and the hall count toward cleaning credit.

**Paired controls.** The same RGB ground planner was tested on six fixed diagnostic layouts (5201–5206). At the official 200-tick deadline, ground-only, fixed-scout and adaptive-scout completed 3/6, 6/6 and 6/6. The fixed and adaptive teams averaged 133.7 and 118.3 ticks on those same six successes. [Per-layout observations at 200 ticks](paired-200.jsonl) · [160-tick sensitivity check](paired-160.jsonl). These are author controls, not independent model submissions. Forward motion is discrete; the benchmark does not test flight stabilization or claim transfer to real homes.

**Model calibration.** On the earlier, single-stop revision (`2045bb06`), a fresh Astra submission scored [0.78, 3/6 complete](single-stop-astra.json) and parked all six cartons. That version had a much simpler delivery route.

**Reproduce the media.** Use the task's pinned Python requirements and Mesa OSMesa; add Mitsuba 3.7.1 for the optional path trace. From this branch, with `TASK` set to the task directory at the revision linked above:

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES="" LP_NUM_THREADS=1
python "$TASK/tests/diagnostics/gallery.py" --seed 617 --preview home.png
python replay.py --task "$TASK" --seed 617 --output replay.gif
python photo.py --task "$TASK" --seed 617 --room oak --width 1160 --spp 640 --output room-daylight.png
```

The mesh and surface credits are kept in the task's `tests/furnishings/CREDITS.md` and `tests/interior_materials/CREDITS.md`. [Suggested website card](suggested-card.toml) for a maintainer to incorporate; task assets and tests are not duplicated in this evidence branch.
