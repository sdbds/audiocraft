"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from tempfile import NamedTemporaryFile
import argparse
import torch
import gradio as gr
import os
import time
import warnings
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import apply_fade
from audiocraft.utils.extend import generate_music_segments, add_settings_to_image, INTERRUPTING
import numpy as np
import random

MODEL = None
MODELS = None
IS_SHARED_SPACE = "musicgen/MusicGen" in os.environ.get('SPACE_ID', '')
INTERRUPTED = False
UNLOAD_MODEL = False
MOVE_TO_CPU = False

def interrupt_callback():
    return INTERRUPTED

def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out

def load_model(version):
    global MODEL, MODELS, UNLOAD_MODEL
    print("Loading model", version)
    if MODELS is None:
        return MusicGen.get_pretrained(version)
    else:
        t1 = time.monotonic()
        if MODEL is not None:
            MODEL.to('cpu') # move to cache
            print("Previous model moved to CPU in %.2fs" % (time.monotonic() - t1))
            t1 = time.monotonic()
        if MODELS.get(version) is None:
            print("Loading model %s from disk" % version)
            result = MusicGen.get_pretrained(version)
            MODELS[version] = result
            print("Model loaded in %.2fs" % (time.monotonic() - t1))
            return result
        result = MODELS[version].to('cuda')
        print("Cached model loaded in %.2fs" % (time.monotonic() - t1))
        return result


def predict(model, text, melody, duration, dimension, topk, topp, temperature, cfg_coef, background, title, include_settings, settings_font, settings_font_color, seed, overlap=1):
    global MODEL, INTERRUPTED, INTERRUPTING
    output_segments = None

    INTERRUPTED = False
    INTERRUPTING = False
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    if MODEL is None or MODEL.name != model:
        MODEL = load_model(model)
    else:
        if MOVE_TO_CPU:
            MODEL.to('cuda')

    output = None
    segment_duration = duration
    initial_duration = duration
    output_segments = []
    while duration > 0:
        if not output_segments: # first pass of long or short song
            if segment_duration > MODEL.lm.cfg.dataset.segment_duration:
                segment_duration = MODEL.lm.cfg.dataset.segment_duration
            else:
                segment_duration = duration
        else: # next pass of long song
            if duration + overlap < MODEL.lm.cfg.dataset.segment_duration:
                segment_duration = duration + overlap
            else:
                segment_duration = MODEL.lm.cfg.dataset.segment_duration
        # implement seed
        if seed < 0:
            seed = random.randint(0, 0xffff_ffff_ffff)
        torch.manual_seed(seed)


        print(f'Segment duration: {segment_duration}, duration: {duration}, overlap: {overlap}')
        MODEL.set_generation_params(
            use_sampling=True,
            top_k=topk,
            top_p=topp,
            temperature=temperature,
            cfg_coef=cfg_coef,
            duration=segment_duration,
            two_step_cfg=False,
            rep_penalty=0.5
        )

        if melody:
            # todo return excess duration, load next model and continue in loop structure building up output_segments
            if duration > MODEL.lm.cfg.dataset.segment_duration:
                output_segments, duration = generate_music_segments(text, melody, seed, MODEL, duration, overlap, MODEL.lm.cfg.dataset.segment_duration)
            else:
                # pure original code
                sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t().unsqueeze(0)
                print(melody.shape)
                if melody.dim() == 2:
                    melody = melody[None]
                melody = melody[..., :int(sr * MODEL.lm.cfg.dataset.segment_duration)]
                output = MODEL.generate_with_chroma(
                    descriptions=[text],
                    melody_wavs=melody,
                    melody_sample_rate=sr,
                    progress=True
                )
            # All output_segments are populated, so we can break the loop or set duration to 0
            break
        else:
            #output = MODEL.generate(descriptions=[text], progress=False)
            if not output_segments:
                next_segment = MODEL.generate(descriptions=[text], progress=True)
                duration -= segment_duration
            else:
                last_chunk = output_segments[-1][:, :, -overlap*MODEL.sample_rate:]
                next_segment = MODEL.generate_continuation(last_chunk, MODEL.sample_rate, descriptions=[text], progress=True)
                duration -= segment_duration - overlap
            output_segments.append(next_segment)

        if INTERRUPTING:
            INTERRUPTED = True
            INTERRUPTING = False
            print("Function execution interrupted!")
            raise gr.Error("Interrupted.")

    if output_segments:
        try:
            # Combine the output segments into one long audio file or stack tracks
            #output_segments = [segment.detach().cpu().float()[0] for segment in output_segments]
            #output = torch.cat(output_segments, dim=dimension)
            
            output = output_segments[0]
            for i in range(1, len(output_segments)):
                overlap_samples = overlap * MODEL.sample_rate
                #stack tracks and fade out/in
                overlapping_output_fadeout = output[:, :, -overlap_samples:]
                overlapping_output_fadeout = apply_fade(overlapping_output_fadeout,sample_rate=MODEL.sample_rate,duration=overlap,out=True,start=True, curve_end=0.9, current_device=MODEL.device)

                overlapping_output_fadein = output_segments[i][:, :, :overlap_samples]
                overlapping_output_fadein = apply_fade(overlapping_output_fadein,sample_rate=MODEL.sample_rate,duration=overlap,out=False,start=False, curve_start=0.1, current_device=MODEL.device)

                overlapping_output = (overlapping_output_fadeout + overlapping_output_fadein) / 2
                print(f" overlap size Fade:{overlapping_output.size()}\n output: {output.size()}\n segment: {output_segments[i].size()}")
                ##overlapping_output = torch.cat([output[:, :, -overlap_samples:], output_segments[i][:, :, :overlap_samples]], dim=1) #stack tracks
                ##print(f" overlap size stack:{overlapping_output.size()}\n output: {output.size()}\n segment: {output_segments[i].size()}")
                #overlapping_output = torch.cat([output[:, :, -overlap_samples:], output_segments[i][:, :, :overlap_samples]], dim=2) #stack tracks
                #print(f" overlap size cat:{overlapping_output.size()}\n output: {output.size()}\n segment: {output_segments[i].size()}")               
                output = torch.cat([output[:, :, :-overlap_samples], overlapping_output, output_segments[i][:, :, overlap_samples:]], dim=dimension)
            output = output.detach().cpu().float()[0]
        except Exception as e:
            print(f"Error combining segments: {e}. Using the first segment only.")
            output = output_segments[0].detach().cpu().float()[0]
    else:
        output = output.detach().cpu().float()[0]

    with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
        if include_settings:
            video_description = f"{text}\n Duration: {str(initial_duration)} Dimension: {dimension}\n Top-k:{topk} Top-p:{topp}\n Randomness:{temperature}\n cfg:{cfg_coef} overlap: {overlap}\n Seed: {seed}\n Model: {model}\n Melody File:#todo"
            background = add_settings_to_image(title, video_description, background_path=background, font=settings_font, font_color=settings_font_color)
        audio_write(
            file.name, output, MODEL.sample_rate, strategy="loudness",
            loudness_headroom_db=19, loudness_compressor=True, add_suffix=False, channels=2)
        waveform_video = make_waveform(file.name,bg_image=background, bar_count=45)
    if MOVE_TO_CPU:
        MODEL.to('cpu')
    if UNLOAD_MODEL:
        MODEL = None
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return waveform_video, seed


def ui(**kwargs):
    css="""
    #col-container {max-width: 910px; margin-left: auto; margin-right: auto;}
    a {text-decoration-line: underline; font-weight: 600;}
    """
    with gr.Blocks(title="UnlimitedMusicGen", css=css) as interface:
        gr.Markdown(
            """
            # MusicGen
            This is your private demo for [MusicGen](https://github.com/facebookresearch/audiocraft), a simple and controllable model for music generation

            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
            Todo: Working on improved Melody Conditioned Music Generation transitions and consistency.

            """
        )
        if IS_SHARED_SPACE and not torch.cuda.is_available():
            gr.Markdown("""
                ⚠ This Space doesn't work in this shared UI ⚠

                <a href="https://huggingface.co/spaces/musicgen/MusicGen?duplicate=true" style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
                <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
                to use it privately, or use the <a href="https://huggingface.co/spaces/facebook/MusicGen">public demo</a>
                """)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", interactive=True, value="4/4 100bpm 320kbps 48khz, Industrial/Electronic Soundtrack, Dark, Intense, Sci-Fi")
                    melody = gr.Audio(source="upload", type="numpy", label="Melody Condition (optional)", interactive=True)
                with gr.Row():
                    submit = gr.Button("Submit")
                    # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Row():
                    background= gr.Image(value="./assets/background.png", source="upload", label="Background", shape=(768,512), type="filepath", interactive=True)
                    include_settings = gr.Checkbox(label="Add Settings to background", value=True, interactive=True)
                with gr.Row():
                    title = gr.Textbox(label="Title", value="MusicGen", interactive=True)
                    settings_font = gr.Text(label="Settings Font", value="./assets/arial.ttf", interactive=True)
                    settings_font_color = gr.ColorPicker(label="Settings Font Color", value="#c87f05", interactive=True)
                with gr.Row():
                    model = gr.Radio(["melody", "medium", "small", "large"], label="Model", value="melody", interactive=True)
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=720, value=10, label="Duration", interactive=True)
                    overlap = gr.Slider(minimum=1, maximum=29, value=5, step=1, label="Overlap", interactive=True)
                    dimension = gr.Slider(minimum=-2, maximum=2, value=2, step=1, label="Dimension", info="determines which direction to add new segements of audio. (1 = stack tracks, 2 = lengthen, -2..0 = ?)", interactive=True)
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, precision=0, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, precision=0, interactive=True)
                    temperature = gr.Number(label="Randomness Temperature", value=0.75, precision=None, interactive=True)
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=5.5, precision=None, interactive=True)
                with gr.Row():
                    seed = gr.Number(label="Seed", value=-1, precision=0, interactive=True)
                    gr.Button('\U0001f3b2\ufe0f').style(full_width=False).click(fn=lambda: -1, outputs=[seed], queue=False)
                    reuse_seed = gr.Button('\u267b\ufe0f').style(full_width=False)
            with gr.Column() as c:
                output = gr.Video(label="Generated Music")
                seed_used = gr.Number(label='Seed used', value=-1, interactive=False)

        reuse_seed.click(fn=lambda x: x, inputs=[seed_used], outputs=[seed], queue=False)
        submit.click(predict, inputs=[model, text, melody, duration, dimension, topk, topp, temperature, cfg_coef, background, title, include_settings, settings_font, settings_font_color, seed, overlap], outputs=[output, seed_used])
        gr.Examples(
            fn=predict,
            examples=[
                [
                    "4/4 120bpm 320kbps 48khz, An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                    "melody"
                ],
                [
                    "4/4 120bpm 320kbps 48khz, A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                    "melody"
                ],
                [
                    "4/4 120bpm 320kbps 48khz, 90s rock song with electric guitar and heavy drums",
                    None,
                    "medium"
                ],
                [
                    "4/4 120bpm 320kbps 48khz, a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
                    "./assets/bach.mp3",
                    "melody"
                ],
                [
                    "4/4 320kbps 48khz, lofi slow bpm electro chill with organic samples",
                    None,
                    "medium",
                ],
            ],
            inputs=[text, melody, model],
            outputs=[output]
        )
        gr.Markdown(
            """
            ### More details

            The model will generate a short music extract based on the description you provided.
            You can generate up to 30 seconds of audio.

            We present 4 model variations:
            1. Melody -- a music generation model capable of generating music condition on text and melody inputs. **Note**, you can also use text only.
            2. Small -- a 300M transformer decoder conditioned on text only.
            3. Medium -- a 1.5B transformer decoder conditioned on text only.
            4. Large -- a 3.3B transformer decoder conditioned on text only (might OOM for the longest sequences.)

            When using `melody`, ou can optionaly provide a reference audio from
            which a broad melody will be extracted. The model will then try to follow both the description and melody provided.

            You can also use your own GPU or a Google Colab by following the instructions on our repo.
            See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
            for more details.
            """
        )

        # Show the interface
        launch_kwargs = {}
        username = kwargs.get('username')
        password = kwargs.get('password')
        server_port = kwargs.get('server_port', 0)
        inbrowser = kwargs.get('inbrowser', False)
        share = kwargs.get('share', False)
        server_name = kwargs.get('listen')

        launch_kwargs['server_name'] = server_name

        if username and password:
            launch_kwargs['auth'] = (username, password)
        if server_port > 0:
            launch_kwargs['server_port'] = server_port
        if inbrowser:
            launch_kwargs['inbrowser'] = inbrowser
        if share:
            launch_kwargs['share'] = share

        interface.queue().launch(**launch_kwargs, max_threads=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=7859,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    ui(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen
    )
