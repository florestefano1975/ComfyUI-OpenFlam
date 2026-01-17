# ComfyUI-OpenFlam Usage Examples

This guide provides practical examples for using OpenFLAM nodes in ComfyUI.

## Example 1: Simple Audio Classification

**Goal**: Determine which text description best matches an audio clip.

**Workflow**:
1. Add the `OpenFLAM Load Model` node
2. Add the `OpenFLAM Load Audio` node
   - Select your audio file
   - Set duration to 10.0 seconds
3. Add the `OpenFLAM Global Similarity` node
   - Connect model from Load Model node
   - Connect audio from Load Audio node
   - Enter your text prompts, for example:
     ```
     dog barking
     car engine
     human speech
     music
     rain
     ```
4. Run the workflow and view results

**Expected Output**:
```
Global Audio-Text Similarities:
========================================
dog barking: 0.8542
car engine: 0.2341
human speech: 0.4521
music: 0.3012
rain: 0.1854
========================================
Best Match: dog barking (0.8542)
```

## Example 2: Sound Event Detection Over Time

**Goal**: Identify when specific sounds occur in an audio clip.

**Workflow**:
1. `OpenFLAM Load Model` → load the model
2. `OpenFLAM Load Audio` → load 10-second audio
3. `OpenFLAM Local Similarity` with:
   - text_prompts: `"car horn, dog bark, door slam, bell ringing"`
   - method: `unbiased`
   - cross_product: `True`
   - median_filter_size: `3`

**Interpreting Results**:
- The node returns frame-wise similarity maps
- Each frame represents ~0.1 seconds
- Higher values indicate greater likelihood of event presence
- Statistics show max and mean for each prompt

## Example 3: Music Analysis

**Goal**: Identify instruments or elements in a music piece.

**Suggested Text Prompts**:
```
electric guitar
acoustic guitar
piano
drums
bass
violin
saxophone
synthesizer
vocal melody
background vocals
```

**Configuration**:
- Method: `unbiased` (more accurate for detailed analysis)
- Cross Product: `True`
- Duration: 10 seconds (take a representative sample)

## Example 4: Environmental Analysis

**Goal**: Identify sounds in an environmental recording.

**Suggested Text Prompts**:
```
bird chirping
wind blowing
water flowing
footsteps
traffic noise
children playing
construction sounds
nature sounds
```

**Tips**:
- Use `approximate` method for faster analysis
- Increase median_filter_size to 5 or 7 for smoother results
- Use longer audio to capture more variations

## Example 5: Speech Detection

**Goal**: Distinguish between different types of speech and other sounds.

**Suggested Text Prompts**:
```
male speaker
female speaker
child speaking
whispering
shouting
laughing
crying
silence
background music
```

**Best Practices**:
- Use 10-second clips for stable results
- Method `unbiased` for higher accuracy
- Compare global and local results for validation

## Example 6: Sound Design and Post-Production

**Goal**: Identify and catalog sound effects.

**Suggested Text Prompts**:
```
explosion
whoosh
impact
breaking glass
metal clang
wooden thud
sci-fi laser
magic spell
footstep concrete
footstep grass
```

## General Tips

### Audio Preparation
- **Sample Rate**: OpenFLAM works best at 48kHz (automatically converted)
- **Duration**: 10 seconds is optimal, but you can use 1 to 60 seconds
- **Quality**: Clean audio gives better results
- **Mono vs Stereo**: The node automatically converts to mono

### Writing Text Prompts
- **Specific is better**: "acoustic guitar melody" > "guitar"
- **Use descriptors**: "loud dog barking", "gentle rain"
- **Avoid contradictions**: Don't use "silence" with "loud noise" together
- **Test variations**: Try synonyms to see what works best

### Performance Optimization
- **GPU**: Enable CUDA if available (10x faster)
- **Batch Processing**: Load the model once and reuse it
- **Method Selection**:
  - `approximate`: 20-30% faster, great for previews
  - `unbiased`: More accurate, use for final analysis

### Interpreting Results

#### Global Similarity
- **>0.7**: High confidence - match very likely
- **0.4-0.7**: Medium confidence - possible match
- **<0.4**: Low confidence - probably not present

#### Local Similarity
- **Peaks**: Indicate specific moments where the event is present
- **Plateaus**: Continuous event over time
- **High mean + High max**: Event present and prominent
- **Low mean + High max**: Brief but distinctive event

## Troubleshooting

### Model Downloads Slowly
- The checkpoint is ~2GB, may take time
- Will be saved in `models/openflam/` and reused

### Unexpected Results
- Verify input audio quality
- Try different text prompts
- Compare with known examples

### Out of Memory
- Reduce audio duration
- Use CPU instead of GPU if necessary
- Close other memory-intensive applications

### Audio Won't Load
- Verify file is in ComfyUI's input folder
- Check that format is supported (WAV, MP3, etc.)
- Ensure file is not corrupted

## Advanced Use Cases

### Combining with Other ComfyUI Nodes
- Use results for conditional triggers
- Integrate with audio generation nodes
- Create automated analysis pipelines

### Post-Processing Similarity Maps
- Apply thresholding for binary detection
- Use temporal smoothing for continuous events
- Combine multiple predictions for higher accuracy

### Dataset Building
- Use global similarity to filter large datasets
- Use local similarity for temporal annotations
- Automate categorization of audio libraries
