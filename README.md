# Enhanced ASCII Art Generator

A powerful Python-based ASCII art generator that converts images and videos into high-quality ASCII art with various enhancement options.

![Example Conversion](examples/comparison.png)

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
git clone https://github.com/yourusername/ascii-art-gen.git
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
    <td><img src="examples/landscape.jpg" width="400"></td>
    <td><img src="examples/landscape_ascii.png" width="400"></td>
  </tr>
  <tr>
    <td><img src="examples/portrait.jpg" width="400"></td>
    <td><img src="examples/portrait_ascii.png" width="400"></td>
  </tr>
</table>

### Different Enhancement Modes

<table>
  <tr>
    <th>Normal</th>
    <th>Vibrant</th>
    <th>High Contrast</th>
  </tr>
  <tr>
    <td><img src="examples/normal.png" width="250"></td>
    <td><img src="examples/vibrant.png" width="250"></td>
    <td><img src="examples/high_contrast.png" width="250"></td>
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

## Requirements

- Python 3.7+
- Required packages listed in requirements.txt
- FFmpeg (for video processing)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
