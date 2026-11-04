# Project boundaries
- dataset/, imu_semg/, and rgb_d/ beside this repository are strictly read-only.
- Never create bytecode caches or other generated files in those dependencies.
- Keep all generated arrays, manifests, checkpoints, reports and caches under macro/runs/.
- Use repository-relative paths; do not hardcode the computer-specific parent path.
- Import model/read-only preprocessing code through macro_tools.sources; never launch upstream training entry points.
- Split complete macro trials before segment/window creation. Test data is never used for normalization or checkpoint selection.
- This project recognizes micro gestures in macro activity recordings; it does not classify macro activities.
