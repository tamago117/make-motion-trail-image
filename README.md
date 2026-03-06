# Motion Trail Image Creator

Create motion-trail composite images interactively using **SAM 3** (Segment Anything Model 3).

Click on each frame to select the object you want to extract, then generate a single composite image showing the object's motion across all frames.

<p align="center">
  <img src="media/gui.png" width="800">
</p>

## Setup

```bash
git clone https://github.com/<your-repo>/make-motion-trail-image.git
cd make-motion-trail-image
uv sync
```

The SAM 3 model checkpoint is automatically downloaded from HuggingFace on first run. A GPU with at least 8 GB VRAM is recommended.

## Usage

```bash
uv run app.py
```

Open http://127.0.0.1:7860 in your browser.

### Workflow

1. **Load frames** -- Enter the directory path containing your image sequence and click **Load**.
2. **Annotate each frame** -- Use the frame slider to navigate between frames. For each frame:
   - Select **Positive** mode and click on the object to segment (green dots).
   - Select **Negative** mode and click on areas to exclude (red dots).
   - The mask preview updates in real time after each click.
   - Use **Undo** to remove the last point or **Clear** to reset the current frame.
3. **Generate composite** -- Adjust the **Alpha** blending slider and click **Generate Motion Trail**. The result is saved to the specified output path.

### Preparing input images

Place a sequence of images (`.png`, `.jpg`, `.jpeg`) in a directory. The images are sorted lexicographically, so use zero-padded filenames (e.g. `frame_001.png`, `frame_002.png`, ...) to ensure the correct order.

## How it works

1. For each frame, SAM 3's interactive predictor segments the target object based on positive/negative point prompts.
2. A static background is estimated by computing the per-pixel median across all frames.
3. The segmented objects are composited onto the background: the first and last frames are pasted opaquely, while intermediate frames are alpha-blended to create the motion-trail effect.

## License

See [LICENSE](LICENSE).
