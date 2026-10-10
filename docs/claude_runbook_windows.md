# Runbook for Claude: export the 9-view file and run the DS-Image M0 test (WSL + CUDA)

You are running inside **WSL (Ubuntu) on a Windows PC with an NVIDIA GPU**, in the repo
`~/sharp` (remote `https://github.com/proto-tech-macario/sharp`). SHARP and its web UI are already
installed in `~/sharp/.venv`. Use **bash** and Linux paths (`.venv/bin/...`). Your job, end to end:

1. get the `ds_m0` code without disturbing the current work,
2. turn the operator's photo into `spatial_photo.h5` using the SHARP web UI's own pipeline,
3. (if possible) make a MIV file from that same `.h5`,
4. run the DS-Image M0 evaluation on it,
5. report the results.

Work in order and **verify each step before moving on**. Do not skip a failing check or report a step
as done when it is not. If something is ambiguous or missing (photo, MIV tool), ask the operator
rather than guessing. Do not push, delete, or overwrite anything outside the run folder.
The checked-out branch may be `stage2-video-to-miv`, which has its own MIV work. Leave it alone.

---

## Step 0. Inputs you need from the operator

Ask for, or confirm, before doing anything else:

- **The photo path** (JPEG/PNG/HEIC), as a Linux path. Windows files are under `/mnt/c/Users/...`.
  If none is given, ask. Do not use the bundled sample for the real run.
- **Settings.** Defaults unless told otherwise: `angle=10`, `max_size=1280`, `precision=fp16`.
  Write them down. They go in the final report.
- **MIV:** ask how the MIV file is normally produced from a `spatial_photo.h5` on this machine. The
  `stage2-video-to-miv` branch may contain that pipeline: look at it (read-only, e.g.
  `git log origin/stage2-video-to-miv --oneline | head` and `git show origin/stage2-video-to-miv:README.md`)
  and propose the exact command to the operator before running it. If nobody knows, skip step 4 and
  say so in the report.

Create a run folder for everything you generate:

```bash
cd ~/sharp
RUN=~/ds_run_$(date +%Y%m%d_%H%M%S); mkdir -p "$RUN"; echo "$RUN"
```

## Step 1. Get the ds_m0 code (in a separate worktree)

The code is on the branch `feat/ds-m0`. Use a worktree so the current branch and any uncommitted
work stay untouched:

```bash
git status --short                       # just note it, do NOT discard anything
git fetch origin
git branch -r | grep feat/ds-m0          # must exist
git worktree add ~/sharp-dsm0 origin/feat/ds-m0 --detach
cd ~/sharp-dsm0
```

- If `origin/feat/ds-m0` does **not** exist, stop and tell the operator it has not been pushed yet.
- If `~/sharp-dsm0` already exists, `cd` into it and run `git fetch && git checkout --detach origin/feat/ds-m0`.

Use the existing environment (it already has numpy, pillow and h5py):

```bash
PY=~/sharp/.venv/bin/python
export PYTHONPATH=$PWD/src
$PY -m pytest tests/ds_m0 -q
```

All tests must pass (the macOS-only checks skip themselves on Linux). If tests fail, report the
failures and stop. If `pytest` is missing: `~/sharp/.venv/bin/python -m pip install pytest`
(or `uv pip install pytest`).

## Step 2. Export the 9-view `.h5` (same pipeline as the web UI)

Check the GPU:

```bash
nvidia-smi
```

If that fails, stop and report. Rendering requires CUDA. Then start the web UI server in the
background. This is the same server a human would use, and we drive it through its HTTP API, so no
browser clicking is needed. Run it from `~/sharp` (the repo with SHARP installed):

```bash
cd ~/sharp
.venv/bin/sharp-spatialize-webui --precision fp16 --port 8737 > "$RUN/webui.log" 2>&1 &
SRV=$!
sleep 8
curl -s http://127.0.0.1:8737/api/config
```

`device.cuda` in the output must be `true`. If false, stop and report `device.name`.

Save this script as `$RUN/export.py`:

```python
import json, sys, time, urllib.parse, urllib.request
from pathlib import Path

PHOTO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ANGLE, MAX_SIZE, PRECISION = sys.argv[3], sys.argv[4], sys.argv[5]
BASE = "http://127.0.0.1:8737"

q = urllib.parse.urlencode({"angle": ANGLE, "max_size": MAX_SIZE, "precision": PRECISION,
                            "filename": PHOTO.name})
req = urllib.request.Request(f"{BASE}/api/jobs?{q}", data=PHOTO.read_bytes(), method="POST")
job = json.load(urllib.request.urlopen(req))["job_id"]
print("job", job, flush=True)

last = None
while True:
    s = json.load(urllib.request.urlopen(f"{BASE}/api/jobs/{job}"))
    if (s["state"], s.get("stage")) != last:
        last = (s["state"], s.get("stage")); print(last, round(s.get("progress", 0), 2), flush=True)
    if s["state"] == "error":
        sys.exit("JOB FAILED:\n" + str(s.get("error")))
    if s["state"] == "done":
        break
    time.sleep(3)

OUT.write_bytes(urllib.request.urlopen(f"{BASE}/api/jobs/{job}/spatial_photo.h5").read())
print("metadata:", json.dumps(s.get("metadata"), indent=1))
print("saved", OUT, OUT.stat().st_size, "bytes")
```

Run it:

```bash
~/sharp/.venv/bin/python "$RUN/export.py" "/path/to/photo.jpg" "$RUN/spatial_photo.h5" 10 1280 fp16
```

Notes:

- **The first render after an install can take about 5-10 minutes with little output** (the renderer
  compiles its GPU code once). This is normal. Do not kill it. If your shell times out, run it in the
  background with output to `$RUN/export.log` and poll that file.
- If the job fails with out-of-memory, retry once with `max_size` 960 and record the change. For any
  other failure (for example `nvcc` not found), report the error text.

Verify the export, then stop only the server you started:

```bash
~/sharp/.venv/bin/sharp-spatialize-validate "$RUN/spatial_photo.h5"
ls -l "$RUN/spatial_photo.h5"
kill $SRV
```

The validator must report the file valid and the file must be well above a few KB. Do not kill other
processes. If `$SRV` is no longer set (new shell), ask the operator to stop the server.

## Step 3. Run the DS-Image M0 evaluation (no GPU needed)

```bash
cd ~/sharp-dsm0
PYTHONPATH=$PWD/src ~/sharp/.venv/bin/python -m ds_m0 evaluate --input "$RUN/spatial_photo.h5" --output-dir "$RUN/ds_out"
```

Expect `Round-trip: PASS`, `Presenter: PASS`, `Legacy JPEG: PASS`. Read `$RUN/ds_out/summary.txt`,
and look at the renders in `$RUN/ds_out/renders/` and `$RUN/ds_out/layers/`. If anything reports FAIL or
the command exits non-zero, do not hide it: copy the full output into the report. This is the first
run on real SHARP data, so a failure is valuable information. Do not "fix" code to make it pass
without telling the operator exactly what you changed and why.

Also record the structure of the result:

```bash
PYTHONPATH=$PWD/src ~/sharp/.venv/bin/python -m ds_m0 inspect --input "$RUN/ds_out/ds_image.jpg"
```

Note the Layer 1 occupancy and the size breakdown.

## Step 4. MIV comparison (only if the MIV procedure is known)

1. Make the MIV file from **`$RUN/spatial_photo.h5` itself**, using the agreed method. Source data and
   resolution must be identical. Show the operator the command before running it.
2. Re-run with the MIV file:

```bash
PYTHONPATH=$PWD/src ~/sharp/.venv/bin/python -m ds_m0 evaluate --input "$RUN/spatial_photo.h5" \
  --output-dir "$RUN/ds_out_miv" --miv-file /path/to/the.miv
```

3. `report_e_miv.json` has the DS/MIV size ratio. This is a **storage comparison only**. Make no claim
   about image quality.

If you cannot make the MIV file, say so clearly. Do not substitute another file.

## Step 5. Report back

Give the operator one short report with:

1. Settings used (photo, angle, max_size, precision) and the `Rendered` size from the job metadata.
2. Whether the code tests passed (count), and the `sharp-spatialize-validate` result.
3. The contents of `summary.txt`, verbatim, and the Layer 1 occupancy.
4. DS-Image size vs MIV size and the ratio, if available.
5. Any failure, warning, retry or workaround, stated plainly.
6. The full path of `$RUN` and its files. To send back: `spatial_photo.h5`, the MIV file,
   `ds_out/summary.txt`, `ds_out/report_*.json`, `ds_out/renders/`, `ds_out/layers/`, `ds_out/ds_image.jpg`.
   Windows can open the folder at `\\wsl$\Ubuntu\home\macario\ds_run_...`.

Do not upload or email anything yourself. The operator decides what to send.
