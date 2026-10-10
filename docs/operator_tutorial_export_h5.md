# Operator tutorial: exporting the 9-view file with the SHARP web UI

**Goal:** turn one photo into a `spatial_photo.h5` (the 9-view RGBD file), then hand it back together
with the MIV file made from that same `.h5`. The `.h5` is what the DS-Image test needs as input.

**You need:** the Windows PC with the NVIDIA GPU, this repo already installed (the SHARP web UI has
been run on it before), and the photo you were told to use.

---

## 0. Decide the settings first (and write them down)

Whoever asked you for this will tell you which photo to use. If they did not specify the rest, use:

| Setting | Value |
| --- | --- |
| Camera angle | **10°** (the default) |
| Preview resolution | **1280 px** (the default) |
| Precision | **fp16** (the default) |

> **Why this matters:** the resolution you pick sets the size of every view in the `.h5`. The MIV
> file must be made from *this exact* `.h5`, otherwise the file-size comparison is meaningless.
> Do not change settings between exporting and making the MIV file.

## 1. Start the web UI

Open a terminal (PowerShell or Command Prompt) in the repo folder, activate the project's Python
environment, and start the server:

```
.venv\Scripts\activate
sharp-spatialize-webui --open
```

- If your environment lives somewhere else or has another name, activate that one instead.
- If `sharp-spatialize-webui` is not found, this works too: `python -m sharp_spatialize.webui --open`.
- A browser tab opens at <http://127.0.0.1:8737/>. If it doesn't, type that address in yourself.
- **Leave the terminal window open** the whole time. Closing it stops the server.

Check the **top-right chip** of the page. It should show your GPU name and memory
(for example `NVIDIA ... · 12 GB`). If it says *no CUDA device* or *server unreachable*, stop and
see "If something goes wrong" below.

## 2. Load the photo

- Drag the photo onto the box that says **Drop an image**, or click it and browse.
  JPEG, PNG and HEIC work, up to 64 MB.
- (Optional dry run: the button **Use the bundled sample photo** runs a built-in test image. Use it
  to check everything works, but do **not** send that result back unless asked.)

## 3. Set the options (left panel)

1. **Camera angle** slider: set it to the value from step 0 (default 10°).
2. **Preview resolution**: pick the value from step 0 (default 1280 px).
3. **Precision**: pick `fp16` unless told otherwise.

## 4. Generate

Click **Generate spatial photo**. A progress panel walks through:
*Waiting for the GPU → Running SHARP → Rendering 9 views → Validating → Encoding previews → Done*.

> **First run after install can take about 5 minutes with no output.** The renderer compiles
> its GPU code once and caches it. Nothing is stuck, so wait for it. Later runs take seconds.
> The first run may also download the model once, which needs internet.

## 5. Check the result

When it finishes, the left panel shows a card with a **validated** badge, and the 3×3 views appear
on the right. Confirm:

- The badge says **validated** (not an error).
- Move the pointer over the image: the scene should shift smoothly as if you were looking around.
- Note the **Rendered** line, for example `1280×960 × 9`. Write it down and send it along.

## 6. Download the file, immediately

Click **Download spatial_photo.h5**. The browser saves it as `<photo name>_spatial_photo.h5`,
normally in your *Downloads* folder.

> **Download before generating anything else.** The server only keeps the data for the last
> couple of runs. If you generate another image first, the earlier download can fail with
> "results are no longer available", and you would have to re-run it.

Optional check that the saved file is good. In the same terminal, open a second one with the
environment activated and run:

```
sharp-spatialize-validate "C:\path\to\photo_spatial_photo.h5"
```

It should report the file as valid. Any failures are listed one per line. Also check the file is
not tiny: a file of 0 KB or a few KB means the download failed, so repeat step 6.

## 7. Make the MIV file

Run your usual MIV encoding on **that same `.h5`** (the 9 views it contains), with no changes to
resolution or angle. Keep the original filename of the `.h5` so they can be matched later.

## 8. Send back these files

1. `<name>_spatial_photo.h5`  (required)
2. The MIV file made from it  (needed for the size comparison)
3. A short note with: the photo name, **angle**, **resolution**, **precision**, and the
   **Rendered** size from step 5.

Do not send the SHARP model files, the `.ply` Gaussian file or the whole repo. They are not needed.

---

## If something goes wrong

| What you see | What to do |
| --- | --- |
| Top-right chip says *no CUDA device* | The GPU or driver is not available. Run `nvidia-smi` in a terminal. If it errors, the NVIDIA driver needs fixing or the PC needs a restart. |
| Chip says *server unreachable* | The terminal running the server was closed or crashed. Start it again (step 1) and refresh the page. |
| Page says **That run failed** with some text | Copy the whole text and send it along. Common causes: not enough GPU memory (try a lower *Preview resolution* such as 960, and keep `fp16`), or the CUDA toolkit (`nvcc`) missing so the renderer cannot compile. |
| It looks frozen for minutes on the first run | Normal on the very first render (GPU code is compiling). Wait about 5-10 minutes. |
| "Address already in use" in the terminal | A previous server is still running. Close it, or start with another port: `sharp-spatialize-webui --port 8738 --open`. |
| Download says results are no longer available | Generate the same photo again, then download straight away. |
| Photo rejected as too large | Maximum is 64 MB. Export a smaller JPEG of the same photo. |

You can also do steps 1-6 from the command line instead of the browser:

```
sharp-spatialize -i photo.jpg -o photo_spatial.h5 --angle 10 --precision fp16
```

but note the web UI **caps the long edge at the "Preview resolution"** (1280 px by default),
whereas the command line keeps the photo's full size unless `--width/--height` are given. Pick one
route and tell the person who asked which one you used.
