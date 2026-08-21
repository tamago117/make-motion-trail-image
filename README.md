# Motion Trail Image Creator

Create motion-trail composite images interactively using **SAM 3** (Segment Anything Model 3).

Click on each frame to select the object you want to extract, then generate a single composite image showing the object's motion across all frames.

<p align="center">
  <img src="media/gui.png" width="800">
</p>

## Setup

```bash
git clone git@github.com:kohonda/make-motion-trail-image.git
cd make-motion-trail-image
uv sync
```

The SAM 3 model checkpoint is automatically downloaded from HuggingFace on first run. A GPU with at least 8 GB VRAM is recommended.

> **Note:** The SAM 3 model weights are hosted on a gated HuggingFace repo. Once accepted, you need to be authenticated to download the checkpoints. You can do this by running the following steps:
>
> 1. Create or log in to your [Hugging Face](https://huggingface.co/) account.
> 2. Go to the SAM 3 model page and accept the license agreement.
> 3. Generate an access token at [Hugging Face Settings](https://huggingface.co/settings/tokens).
> 4. Run `huggingface-cli login` and paste your token when prompted.

## Usage

```bash
uv run app.py
```

Open http://127.0.0.1:7860 in your browser.

### Workflow

1. **Load frames** -- Frames are loaded by drag & drop into the active set:
   - **Image folder** -- Drop a folder onto the image drop zone; its images load immediately.
   - **Video** -- Drop a video onto the **Drop a video here** box (a browser-playable preview appears below it), set the **Start** / **End** time and sampling **Interval (sec)**, then click **Extract frames from video**. One frame is extracted every interval seconds across the chosen range (default `1.0`). Times accept plain seconds (`12.5`), `mm:ss` (`1:23.5`) or `hh:mm:ss` (`1:02:03`); set **End** to `0` to use the whole clip.
2. **Annotate each frame** -- Use the frame slider to navigate between frames. For each frame:
   - Select **Positive** mode and click on the object to segment (green dots).
   - Select **Negative** mode and click on areas to exclude (red dots).
   - The mask preview updates in real time after each click.
   - Use **Undo** to remove the last point or **Clear** to reset the current frame.
3. **Generate composite** -- Adjust the **Alpha** blending slider and click **Generate Motion Trail**. The result is saved to the specified output path.

### Output format

The extension of the **Output path** selects the format: `.png` (default), `.jpg` / `.jpeg`, `.webp`, `.bmp` and `.tiff` are supported. Anything else (or no extension at all) is saved as PNG, with a warning naming the file that was actually written.

The **Result** panel serves that exact file, so its download button gives you the format you asked for. TIFF is the one exception -- browsers cannot display it, so the panel shows a PNG preview while the file on disk stays TIFF.

### Saving and resuming work

Annotating many frames takes a while, so the work in progress can be saved and picked up later. Open the **Session -- save / restore work in progress** panel at the top of the page:

- **Save session** -- Writes every set (frames, masks, point prompts, colours), the chosen background and all the settings (times, interval, alpha, tint, emphasis, output path) to `sessions/<name>/`. Leave the name blank to get a timestamped one; saving again with the same name updates it in place.
- **Autosave on Generate Motion Trail** (on by default) -- Saves the session automatically every time a composite is generated, under the name in the box. A blank name gets a timestamped one that is filled back in, so later generations keep updating that same session instead of piling up. Untick it if you would rather only save by hand.
- **Restore session** -- Pick a saved session from the dropdown (newest first) and click **Restore session** to bring the whole workspace back, including which set and frame you were on. **Refresh list** re-reads the directory if sessions were added from elsewhere.

Sessions store the frames as PNG, so a restored session reproduces exactly the same composite. They do not need the original video or image folder, which means a session survives an app restart -- only the video *preview* is dropped, since the dropped file itself is not kept.

Re-saving is incremental: images whose contents haven't changed are left on disk untouched, so tweaking **Alpha** and regenerating re-saves in a fraction of the time of the first save (~0.5 s vs ~3 s for 60 frames at 1080p) and doesn't rewrite the frames. The first save of such a set is still a ~220 MB frame dump, so delete sessions you no longer need from `sessions/`.

### Preparing input

You can supply frames in two ways, both via drag & drop:

- **Image folder** -- A folder of images (`.png`, `.jpg`, `.jpeg`). The images are sorted by filename, so use zero-padded names (e.g. `frame_001.png`, `frame_002.png`, ...) to ensure the correct order.
- **Video file** -- A video (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.m4v`). One frame is extracted every **Interval (sec)** seconds across the **Start** / **End** range. The **Start** / **End** fields accept plain seconds (`12.5`), `mm:ss` (`1:23.5`) or `hh:mm:ss` (`1:02:03`).

> **Note:** Drag & drop uploads the files into the app's working area, so very large videos may take a moment to transfer.

## How it works

1. For each frame, SAM 3's interactive predictor segments the target object based on positive/negative point prompts.
2. A static background is estimated by computing the per-pixel median across all frames.
3. The segmented objects are composited onto the background: the first and last frames are pasted opaquely, while intermediate frames are alpha-blended to create the motion-trail effect.

## License

See [LICENSE](LICENSE).
**NOTE**: This project depends on Segment Anything Model 3 (SAM3) released by Meta under the SAM License.
