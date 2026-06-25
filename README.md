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
