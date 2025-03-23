# Enhanced ASCII Art Generator

A sophisticated Python-based ASCII art generator that converts images and videos into high-quality ASCII art with extensive customization options and advanced image processing techniques.

![Demo](test_ascii.gif)

## Features

- **Image & Video Support**: Convert both images and videos to ASCII art
- **Color Output**: Generate colored ASCII art with enhanced color mapping
- **Multiple Character Sets**: Choose from various ASCII character densities
- **Color Enhancement Modes**: 
  - Normal
  - Vibrant
  - Muted
  - Grayscale
  - High Contrast
- **Depth Enhancement**: Add depth perception through edge detection
- **Dynamic Range Adjustment**: Automatically optimize brightness distribution
- **Dithering**: Improve gradient representation
- **Custom Character Sets**: Define your own ASCII character mappings
- **Background Options**: Choose between black, white, or auto background

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Zapca/Ascii-Art-Generator.git
cd ascii-art-gen
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. For video processing, ensure ffmpeg is installed:
- Windows: `choco install ffmpeg`
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

## Usage

Basic usage:
```bash
python main.py --input ./input_image.jpg --output ./output
```

Advanced usage with all options:
```bash
python main.py --input ./input_image.jpg \
               --output ./output \
               --color \
               --resolution 720 \
               --charset ultra_light \
               --color_mode vibrant \
               --depth strong \
               --font_size 9 \
               --dithering \
               --dynamic_range \
               --background auto
```

## Examples

### Image Conversion Examples

<table>
  <tr>
    <td><b>Original</b></td>
    <td><b>ASCII Output</b></td>
  </tr>
  <tr>
    <td><img src="testing\test18.jpg" width="400"></td>
    <td><img src="output (default font_size = 9)\test18_ascii.png" width="400"></td>
  </tr>
  <tr>
    <td><img src="testing\test3.png" width="400"></td>
    <td><img src="output (default font_size = 9)\test3_ascii.png" width="400"></td>
  </tr>
</table>

### Different Font Sizes

<table>
  <tr>
    <th>Fontsize=10</th>
    <th>Fontsize=9</th>
    <th>Fontsize=5</th>
  </tr>
  <tr>
    <td><img src="predecessor output\test12_ascii.png" width="250"></td>
    <td><img src="output (default font_size = 9)\test12_ascii.png" width="250"></td>
    <td><img src="output (font_size = 5)\test12_ascii.png" width="250"></td>
  </tr>
</table>

## Parameters Explained

| Parameter | Description | Default | Options |
|-----------|-------------|---------|----------|
| --input | Input file/directory path | Required | Any image/video file |
| --output | Output directory path | Required | Directory path |
| --color | Enable color output | False | Flag |
| --resolution | Output height in pixels | 720 | Integer |
| --charset | ASCII character density | ultra_light | ultra_light, light, medium, dense, custom |
| --color_mode | Color enhancement mode | normal | normal, vibrant, muted, grayscale, high_contrast |
| --depth | Depth enhancement level | medium | none, light, medium, strong |
| --font_size | Font size for characters | 9 | Integer |
| --dithering | Enable dithering | False | Flag |
| --dynamic_range | Enable dynamic range | False | Flag |
| --background | Background color | black | black, white, auto |

## Character Sets

- **Ultra Light**: ` .,`'"^-_~:;=+!i1tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$`
- **Light**: ` .,:;+*?%S#@`
- **Medium**: ` .:-=+*#%@`
- **Dense**: ` .*#%@`
- **Custom**: Define your own set using `--custom_charset`

## Tips for Best Results

1. **Resolution**: Higher resolution gives more detail but takes longer to process
2. **Color Mode**: 
   - Use 'vibrant' for colorful images
   - 'high_contrast' for detailed black and white
   - 'muted' for subtle effects
3. **Depth**: 'strong' works best with architectural images
4. **Font Size**: Smaller fonts (5-9) give more detail
5. **Dynamic Range**: Enable for images with varying lighting

## Technical Details

### Font System and Resolution

The font system is a crucial component that determines the quality and detail of the ASCII output. The generator uses monospace fonts to ensure consistent character spacing.

#### Font Size Impact
- **Small (5-6)**: Highest detail, but may be hard to read
- **Medium (7-9)**: Best balance of detail and readability
- **Large (10+)**: More readable but less detailed

Example comparison with different font sizes:
```
Font Size 5:
@@@@##**!!||
@@##**!!||tt
##**!!||ttw_

Font Size 9:
@@@##*!|
@@#*!|tw
##*!|t_

Font Size 12:
@@#*!|
@#*|tw
#*|t_
```

### Architecture Overview

```
Input → Enhancement → ASCII Conversion → Rendering → Output
      ↓             ↓                 ↓           ↓
   Resizing     Color/Depth     Char Mapping   Font Rendering
   Filtering    Processing      Brightness      Anti-aliasing
```

#### Processing Pipeline

1. **Image Preprocessing**
   - Resolution calculation based on font metrics
   - Aspect ratio preservation
   - High-quality image resizing (LANCZOS)

2. **Enhancement Stage**
   ```python
   # Enhancement flow
   Image → Color Adjustment → Edge Detection → Dynamic Range → Dithering
   ```

3. **ASCII Conversion**
   - Brightness calculation: `0.299*R + 0.587*G + 0.114*B`
   - Character mapping using density-based lookup
   - Color information preservation for color output

4. **Rendering**
   - Font-based character rendering
   - Anti-aliasing and sharpening
   - Background color optimization

### Performance Comparisons

#### Resolution vs Processing Time
| Resolution | Font Size | Processing Time | File Size |
|------------|-----------|-----------------|-----------|
| 720p       | 5        | Fast            | Larger    |
| 720p       | 9        | Medium          | Medium    |
| 720p       | 12       | Slow            | Smaller   |

#### Memory Usage
- Font Size 5: ~1.5x base memory
- Font Size 9: ~1.0x base memory
- Font Size 12: ~0.7x base memory

### Output Quality Comparison

#### Architecture Impact
1. **Edge Detection + Small Font**
   - Highest detail preservation
   - Best for architectural images
   - Example: Building photographs

2. **Dithering + Medium Font**
   - Improved gradients
   - Better for portraits
   - Example: Human faces

3. **High Contrast + Large Font**
   - Bold, artistic look
   - Suitable for posters
   - Example: Logos, graphics

## Advanced Usage Examples

### Fine-tuned Settings for Different Scenarios

1. **Architectural Photography**
```bash
python main.py --input building.jpg \
               --color \
               --font_size 5 \
               --depth strong \
               --dynamic_range \
               --charset dense
```

2. **Portrait Photography**
```bash
python main.py --input portrait.jpg \
               --color \
               --font_size 9 \
               --depth light \
               --dithering \
               --color_mode vibrant
```

3. **Text Document Conversion**
```bash
python main.py --input document.png \
               --font_size 12 \
               --depth none \
               --background white \
               --color_mode grayscale
```

### Technical Considerations

#### Font Size Selection
The font size affects several aspects:

1. **Detail Level**
   - Font size 5: ~144 chars per 720p width
   - Font size 9: ~80 chars per 720p width
   - Font size 12: ~60 chars per 720p width

2. **Processing Impact**
   ```
   Processing Time ∝ (1/font_size)²
   Memory Usage ∝ (1/font_size)²
   ```

3. **Quality vs Performance**
   - Smaller font = Higher quality + Higher resource usage
   - Larger font = Lower quality + Better performance



## Font Size Examples

### Small Font (5)
- Pros: Incredible detail, precise edges
- Cons: Higher resource usage, may be hard to read
- Best for: High-resolution displays, detailed art

### Medium Font (9)
- Pros: Balanced detail and readability
- Cons: Moderate resource usage
- Best for: General purpose, most images

### Large Font (12)
- Pros: Very readable, efficient processing
- Cons: Less detail, more blocky appearance
- Best for: Quick previews, text-heavy images

## Requirements

- Python 3.7+
- Required packages listed in requirements.txt
- FFmpeg (for video processing)

## License

GNU General Public License v3.0 - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For technical contributions:

1. **Performance Improvements**
   - Font rendering optimization
   - Parallel processing implementation
   - Memory usage optimization

2. **Feature Additions**
   - New character sets
   - Additional color modes
   - Custom font support

3. **Documentation**
   - Technical deep-dives
   - Performance benchmarks
   - Example collections
