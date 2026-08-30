# Figure analyses

All manifests, code, and outputs are resolved relative to the repository by
`analysis.figures.paths`. The external imaging-data mount is deliberately not
hard-coded because its location differs across workstations and servers.

Set it once in the environment before opening Jupyter or running a figure CLI:

```bash
export ODYN_IMAGING_ROOT=/path/to/ImagingData
```

Command-line analyses may instead receive `--imaging-root /path/to/ImagingData`.
Notebook output remains under the corresponding repository-relative
`analysis/figures/figure*/outputs/` directory.
