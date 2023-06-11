import torch
import math
from audiocraft.models import MusicGen
import numpy as np


def separate_audio_segments(audio, segment_duration=30):
    sr, audio_data = audio[0], audio[1]
    
    total_samples = len(audio_data)
    segment_samples = sr * segment_duration
    
    total_segments = math.ceil(total_samples / segment_samples)
    
    segments = []
    
    for segment_idx in range(total_segments):
        print(f"Audio Input segment {segment_idx + 1} / {total_segments + 1} \r")
        start_sample = segment_idx * segment_samples
        end_sample = (segment_idx + 1) * segment_samples
        
        segment = audio_data[start_sample:end_sample]
        segments.append((sr, segment))
    
    return segments

def generate_music_segments(text, melody, MODEL, duration:int=10, segment_duration:int=30):
    # generate audio segments
    melody_segments = separate_audio_segments(melody, segment_duration) 
    
    # Create a list to store the melody tensors for each segment
    melodys = []
    output_segments = []
    
    # Calculate the total number of segments
    total_segments = max(math.ceil(duration / segment_duration),1)
    print(f"total Segments to Generate: {total_segments} for {duration} seconds. Each segment is {segment_duration} seconds")

    # If melody_segments is shorter than total_segments, repeat the segments until the total number of segments is reached
    if len(melody_segments) < total_segments:
        for i in range(total_segments - len(melody_segments)):
            segment = melody_segments[i]
            melody_segments.append(segment)
        print(f"melody_segments: {len(melody_segments)} fixed")

    # Iterate over the segments to create list of Meldoy tensors
    for segment_idx in range(total_segments):
        print(f"segment {segment_idx} of {total_segments} \r")
        sr, verse = melody_segments[segment_idx][0], torch.from_numpy(melody_segments[segment_idx][1]).to(MODEL.device).float().t().unsqueeze(0)

        print(f"shape:{verse.shape} dim:{verse.dim()}")
        if verse.dim() == 2:
            verse = verse[None]
        verse = verse[..., :int(sr * MODEL.lm.cfg.dataset.segment_duration)]
        # Append the segment to the melodys list
        melodys.append(verse)

    for idx, verse in enumerate(melodys):
        print(f"Generating New Melody Segment {idx + 1}: {text}\r")
        output = MODEL.generate_with_chroma(
            descriptions=[text],
            melody_wavs=verse,
            melody_sample_rate=sr,
            progress=True
        )

        # Append the generated output to the list of segments
        #output_segments.append(output[:, :segment_duration])
        output_segments.append(output)
        print(f"output_segments: {len(output_segments)}: shape: {output.shape} dim {output.dim()}")
    return output_segments

#def generate_music_segments(text, melody, duration, MODEL, segment_duration=30):
#    sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t().unsqueeze(0)
    
#    # Create a list to store the melody tensors for each segment
#    melodys = []
    
#    # Calculate the total number of segments
#    total_segments = math.ceil(melody.shape[1] / (sr * segment_duration))

#    # Iterate over the segments
#    for segment_idx in range(total_segments):
#        print(f"segment {segment_idx + 1} / {total_segments + 1} \r")
#        start_frame = segment_idx * sr * segment_duration
#        end_frame = (segment_idx + 1) * sr * segment_duration

#        # Extract the segment from the melody tensor
#        segment = melody[:, start_frame:end_frame]

#        # Append the segment to the melodys list
#        melodys.append(segment)

#    output_segments = []

#    for segment in melodys:
#        output = MODEL.generate_with_chroma(
#            descriptions=[text],
#            melody_wavs=segment,
#            melody_sample_rate=sr,
#            progress=False
#        )

#        # Append the generated output to the list of segments
#        output_segments.append(output[:, :segment_duration])

#    return output_segments




