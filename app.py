#!/usr/bin/env python3
"""
Interactive Gradio GUI for motion-trail image creation using SAM 3.

Workflow
-------
1. Load a directory of frames into a "set".
2. For each frame, click to place positive / negative point prompts.
3. SAM 3 segments the object in real time and shows a mask preview.
4. Navigate frames and annotate each independently.
5. Add more sets (+), each annotated separately and given its own colour;
   reorder them to choose which trail is drawn on top of which.
6. Choose one frame as the background, then generate a composite that
   overlays every set's motion trail in its own colour, either as a still
   image or as a video in which the trail grows one frame at a time.
7. Save the work in progress as a session at any point, and restore it later
   to continue from exactly where you left off.
"""

from __future__ import annotations

import gradio as gr

from motion_trail.frames import VIDEO_EXTS
from motion_trail.session import list_sessions
from motion_trail.ui.edit import (
    add_set,
    change_frame,
    clear_points,
    load_image_files,
    load_video_frames,
    move_set,
    on_image_click,
    on_video_drop,
    remove_set,
    select_set,
    set_background,
    set_color,
    toggle_no_color,
    undo_point,
)
from motion_trail.ui.render import (
    DEFAULT_SETTINGS,
    EMPHASIS_MODES,
    generate_and_autosave,
    generate_video_and_autosave,
    restore_session_cb,
    save_session_cb,
)
from motion_trail.ui.state import _new_set, _next_color, _rgb_to_hex


# Centre the small "or" label between the two browse buttons (passed to launch).
UI_CSS = (
    "#browse-or{flex:0 0 auto !important;min-width:0 !important;"
    "display:flex;align-items:center;justify-content:center;}"
    "#browse-or p{margin:0;}"
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Motion Trail – SAM 3") as demo:
        gr.Markdown("# Motion Trail Image Creator (SAM 3)")
        gr.Markdown(
            "Load a folder per set, annotate each set, give it a colour, "
            "pick a background frame, then overlay every trail."
        )

        init_color = _next_color(0)

        # ---- state ----
        st_sets = gr.State([_new_set(init_color)])  # list[set dict]
        st_active = gr.State(0)  # active set index
        st_idx = gr.State(0)  # current frame within active set
        st_bg = gr.State(None)  # chosen background frame (BGR)
        st_video = gr.State(None)  # original path of the dropped video

        # ---- session save / restore ----
        with gr.Accordion("Session – save / restore work in progress", open=False):
            with gr.Row():
                session_name = gr.Textbox(
                    label="Session name",
                    placeholder="blank = timestamp",
                    scale=3,
                )
                save_session_btn = gr.Button("Save session", scale=1)
            autosave_checkbox = gr.Checkbox(
                label="Autosave on Generate Motion Trail",
                value=True,
                info="Updates the session named above after every composite.",
            )
            with gr.Row():
                session_selector = gr.Dropdown(
                    choices=list_sessions(),
                    value=None,
                    label="Saved sessions (newest first)",
                    scale=3,
                )
                refresh_sessions_btn = gr.Button("Refresh list", scale=1)
                restore_session_btn = gr.Button("Restore session", scale=1)

        # ---- set management ----
        with gr.Row():
            set_selector = gr.Radio(
                choices=["Set 1"], value="Set 1", label="Active set", scale=4
            )
            add_btn = gr.Button("+ Add Set", scale=1)
            remove_btn = gr.Button("Remove Set", scale=1)
        with gr.Row():
            move_earlier_btn = gr.Button("◀ Move earlier (behind)", scale=1)
            move_later_btn = gr.Button("▶ Move later (on top)", scale=1)
            gr.Markdown(
                "Sets are drawn in order, so the **last set is on top** of the "
                "others where their trails overlap."
            )

        # ---- load (drag & drop) ----
        gr.Markdown(
            "**Load frames into the active set** — drop an image folder to load "
            "it immediately, or drop a video below, set the options and click "
            "**Extract frames from video**."
        )

        image_drop = gr.File(
            label="Drop an image folder here",
            file_count="directory",
            height=120,
        )

        # Drop target (a plain file box never tries to play the raw codec) and a
        # separate, read-only preview that shows the browser-playable version.
        video_drop = gr.File(
            label="Drop a video here",
            file_count="single",
            file_types=sorted(VIDEO_EXTS),
            height=120,
        )
        video_player = gr.Video(label="Video preview", interactive=False)

        with gr.Row():
            start_sec = gr.Textbox(
                label="Start (video)",
                value=DEFAULT_SETTINGS["start_sec"],
                placeholder="sec or mm:ss.s, e.g. 1:23.5",
            )
            end_sec = gr.Textbox(
                label="End (0 = until end)",
                value=DEFAULT_SETTINGS["end_sec"],
                placeholder="sec or mm:ss.s, e.g. 2:05",
            )
            interval_sec = gr.Number(
                label="Interval (sec, video)",
                value=DEFAULT_SETTINGS["interval_sec"],
                minimum=0.01,
            )
            extract_btn = gr.Button("Extract frames from video", scale=1)

        with gr.Row():
            color_picker = gr.ColorPicker(
                label="Set colour", value=_rgb_to_hex(init_color)
            )
            no_color_checkbox = gr.Checkbox(
                label="No colour (keep original)", value=False
            )

        # ---- images ----
        with gr.Row():
            input_image = gr.Image(label="Click to add points", interactive=False)
            preview_image = gr.Image(label="Mask preview", interactive=False)

        # ---- controls ----
        with gr.Row():
            mode_radio = gr.Radio(
                ["Positive", "Negative"],
                value="Positive",
                label="Point mode",
            )
            undo_btn = gr.Button("Undo")
            clear_btn = gr.Button("Clear")

        frame_slider = gr.Slider(
            minimum=0,
            maximum=0,
            step=1,
            value=0,
            label="Frame",
        )

        # ---- background ----
        with gr.Row():
            bg_btn = gr.Button("Use current frame as background")
            bg_preview = gr.Image(label="Background", interactive=False)

        # ---- composite ----
        with gr.Row():
            alpha_slider = gr.Slider(
                0.0, 1.0, value=DEFAULT_SETTINGS["alpha"], step=0.05, label="Alpha"
            )
            tint_slider = gr.Slider(
                0.0,
                1.0,
                value=DEFAULT_SETTINGS["tint_strength"],
                step=0.05,
                label="Tint strength",
            )
            emphasis_radio = gr.Radio(
                list(EMPHASIS_MODES),
                value=DEFAULT_SETTINGS["emphasis"],
                label="Emphasize (opaque) frames",
            )
            out_path = gr.Textbox(
                label="Output path (.png / .jpg / .webp / .bmp / .tiff)",
                value=DEFAULT_SETTINGS["output_path"],
            )
            gen_btn = gr.Button("Generate Motion Trail", variant="primary")
        # format=png only applies when a raw array is returned (a non-displayable
        # output format); a returned filepath is served untouched.
        result_image = gr.Image(label="Result", interactive=False, format="png")

        # ---- video ----
        with gr.Row():
            video_fps = gr.Number(
                label="Video FPS",
                value=DEFAULT_SETTINGS["video_fps"],
                minimum=0.1,
                info="Encoding rate only — the pace comes from Interval (sec).",
            )
            video_out_path = gr.Textbox(
                label="Video output path (.mp4 / .mov / .mkv / .avi)",
                value=DEFAULT_SETTINGS["video_output_path"],
            )
            gen_video_btn = gr.Button("Generate Trail Video", variant="primary")
        result_video = gr.Video(label="Trail video", interactive=False)

        # ---- wiring ----
        # User-only events (.input / .release) so programmatic updates from
        # add/remove/select/load do not re-trigger the same handlers.
        set_selector.input(
            select_set,
            inputs=[st_sets, set_selector],
            outputs=[
                st_active,
                st_idx,
                input_image,
                preview_image,
                frame_slider,
                color_picker,
                no_color_checkbox,
                start_sec,
                end_sec,
                interval_sec,
            ],
        )

        add_btn.click(
            add_set,
            inputs=[st_sets],
            outputs=[
                st_sets,
                st_active,
                st_idx,
                set_selector,
                input_image,
                preview_image,
                frame_slider,
                color_picker,
                no_color_checkbox,
            ],
        )

        remove_btn.click(
            remove_set,
            inputs=[st_sets, st_active],
            outputs=[
                st_sets,
                st_active,
                st_idx,
                set_selector,
                input_image,
                preview_image,
                frame_slider,
                color_picker,
                no_color_checkbox,
                start_sec,
                end_sec,
                interval_sec,
            ],
        )

        move_earlier_btn.click(
            lambda sets, active: move_set(sets, active, -1),
            inputs=[st_sets, st_active],
            outputs=[st_sets, st_active, set_selector, color_picker],
        )

        move_later_btn.click(
            lambda sets, active: move_set(sets, active, 1),
            inputs=[st_sets, st_active],
            outputs=[st_sets, st_active, set_selector, color_picker],
        )

        color_picker.input(
            set_color,
            inputs=[st_sets, st_active, color_picker],
            outputs=[st_sets, no_color_checkbox],
        )

        no_color_checkbox.input(
            toggle_no_color,
            inputs=[st_sets, st_active, no_color_checkbox, color_picker],
            outputs=[st_sets],
        )

        # Drag & drop: an image folder loads immediately; a dropped video is
        # stored (and shown playable), then Extract pulls frames from it using
        # the time / interval settings.
        image_drop.upload(
            load_image_files,
            inputs=[image_drop, st_sets, st_active],
            outputs=[
                input_image,
                preview_image,
                frame_slider,
                st_idx,
                st_sets,
                video_player,
            ],
        )

        video_drop.upload(
            on_video_drop,
            inputs=[video_drop],
            outputs=[st_video, video_player],
        )

        extract_btn.click(
            load_video_frames,
            inputs=[st_video, st_sets, st_active, start_sec, end_sec, interval_sec],
            outputs=[
                input_image,
                preview_image,
                frame_slider,
                st_idx,
                st_sets,
                video_player,
            ],
        )

        input_image.select(
            on_image_click,
            inputs=[st_sets, st_active, st_idx, mode_radio],
            outputs=[input_image, preview_image, st_sets],
        )

        undo_btn.click(
            undo_point,
            inputs=[st_sets, st_active, st_idx],
            outputs=[input_image, preview_image, st_sets],
        )

        clear_btn.click(
            clear_points,
            inputs=[st_sets, st_active, st_idx],
            outputs=[input_image, preview_image, st_sets],
        )

        frame_slider.release(
            change_frame,
            inputs=[st_sets, st_active, frame_slider],
            outputs=[input_image, preview_image, st_idx],
        )

        bg_btn.click(
            set_background,
            inputs=[st_sets, st_active, st_idx],
            outputs=[st_bg, bg_preview],
        )

        # Both render buttons feed the same arguments to the same autosave tail.
        render_inputs = [
            st_sets,
            st_bg,
            alpha_slider,
            tint_slider,
            emphasis_radio,
            out_path,
            video_out_path,
            video_fps,
            st_active,
            st_idx,
            st_video,
            session_name,
            start_sec,
            end_sec,
            interval_sec,
            autosave_checkbox,
        ]

        gen_btn.click(
            generate_and_autosave,
            inputs=render_inputs,
            outputs=[result_image, session_selector, session_name],
        )

        gen_video_btn.click(
            generate_video_and_autosave,
            inputs=render_inputs,
            outputs=[result_video, session_selector, session_name],
        )

        save_session_btn.click(
            save_session_cb,
            inputs=[
                st_sets,
                st_active,
                st_idx,
                st_bg,
                st_video,
                session_name,
                start_sec,
                end_sec,
                interval_sec,
                alpha_slider,
                tint_slider,
                emphasis_radio,
                out_path,
                video_out_path,
                video_fps,
            ],
            outputs=[session_selector, session_name],
        )

        refresh_sessions_btn.click(
            lambda: gr.update(choices=list_sessions()),
            outputs=[session_selector],
        )

        restore_session_btn.click(
            restore_session_cb,
            inputs=[session_selector],
            outputs=[
                st_sets,
                st_active,
                st_idx,
                st_bg,
                st_video,
                set_selector,
                input_image,
                preview_image,
                frame_slider,
                color_picker,
                no_color_checkbox,
                bg_preview,
                video_player,
                start_sec,
                end_sec,
                interval_sec,
                alpha_slider,
                tint_slider,
                emphasis_radio,
                out_path,
                video_out_path,
                video_fps,
                session_name,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    # Allow the movie-preview widget to serve videos the user browses to from
    # anywhere on the machine (this is a local, single-user tool on 127.0.0.1).
    demo.launch(allowed_paths=["/"], css=UI_CSS)
