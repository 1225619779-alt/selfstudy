# Gate8 Release Cleanup Dry Run

No repo structure was changed. Release should split transfer, native, and sprint artifacts into explicit trees.

- `metric/case39_transfer/`: case14 train/val to case39 target evidence.
- `metric/case39_native/`: native case39 fit/eval and local-retune artifacts.
- `metric/case39_q1_sprint/`: Gate0-Gate8 evidence packs.
- Canonical case39 fit/eval symlinks must be removed or made unambiguous before release.
