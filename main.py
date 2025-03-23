import argparse
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import cv2
from tqdm import tqdm
import subprocess
import shutil
import tempfile
import colorsys
from collections import defaultdict

# python main.py --input ./testing --output ./output --color --dithering --dynamic_range --font_size 5 --depth strong

# Dynamic ASCII character set based on visual density
ASCII_CHARS_DENSITY = {
    'ultra_light': ' .,`\'\"^-_~:;=+!i1tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$',
    'light': ' .,:;+*?%S#@',
    'medium': ' .:-=+*#%@',
    'dense': ' .*#%@',
    'custom': ' _.!|tw#@'  # Default custom set from original script
}

# Color enhancement settings
COLOR_ENHANCEMENTS = {
    'normal': {'contrast': 1.2, 'brightness': 1.0, 'saturation': 1.2},
    'vibrant': {'contrast': 1.5, 'brightness': 1.1, 'saturation': 1.5},
    'muted': {'contrast': 1.0, 'brightness': 0.9, 'saturation': 0.8},
    'grayscale': {'contrast': 1.3, 'brightness': 1.0, 'saturation': 0.0},
    'high_contrast': {'contrast': 2.0, 'brightness': 1.0, 'saturation': 1.3}
}

# Edge enhancement settings for depth perception
EDGE_ENHANCEMENT = {
    'none': 0,
    'light': 0.3,
    'medium': 0.6,
    'strong': 1.0
}

def main():
    parser = argparse.ArgumentParser(description='Convert image/video to enhanced ASCII art')
    parser.add_argument('--input', type=str, required=True, help='Input file or directory path')
    parser.add_argument('--output', type=str, required=True, help='Output directory path')
    parser.add_argument('--color', action='store_true', help='Output color ASCII art')
    parser.add_argument('--resolution', type=int, default=720,
                        help='Target output height in pixels (default: 720)')
    parser.add_argument('--charset', type=str, default='ultra_light', 
                        choices=['ultra_light', 'light', 'medium', 'dense', 'custom'],
                        help='ASCII character set density (default: ultra_light)')
    parser.add_argument('--color_mode', type=str, default='normal',
                        choices=['normal', 'vibrant', 'muted', 'grayscale', 'high_contrast'],
                        help='Color enhancement mode (default: normal)')
    parser.add_argument('--depth', type=str, default='medium',
                        choices=['none', 'light', 'medium', 'strong'],
                        help='Depth enhancement level (default: medium)')
    parser.add_argument('--invert', action='store_true', 
                        help='Invert brightness to darkness mapping')
    parser.add_argument('--font_size', type=int, default=9,
                        help='Font size for ASCII characters (default: 9)')
    parser.add_argument('--dithering', action='store_true',
                        help='Apply dithering for better gradients')
    parser.add_argument('--dynamic_range', action='store_true',
                        help='Dynamically adjust brightness range per image')
    parser.add_argument('--background', type=str, default='black',
                        choices=['black', 'white', 'auto'],
                        help='Background color (default: black)')
    parser.add_argument('--custom_charset', type=str,
                        help='Custom ASCII character set (from darkest to lightest)')
    
    args = parser.parse_args()

    supported_image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    supported_video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    all_supported_exts = supported_image_exts.union(supported_video_exts)

    os.makedirs(args.output, exist_ok=True)

    # Use custom charset if provided
    if args.custom_charset:
        ASCII_CHARS_DENSITY['custom'] = args.custom_charset
        args.charset = 'custom'

    # Create ASCII mapping
    ascii_chars = ASCII_CHARS_DENSITY[args.charset]
    if args.invert:
        ascii_chars = ascii_chars[::-1]

    # Create the mapping dictionary
    ascii_dict = create_ascii_dict(ascii_chars)
    
    font, char_width, char_height = get_font_metrics(font_size=args.font_size)
    
    # Process files
    if os.path.isdir(args.input):
        for filename in os.listdir(args.input):
            filepath = os.path.join(args.input, filename)
            if os.path.isfile(filepath):
                ext = os.path.splitext(filename)[1].lower()
                if ext in all_supported_exts:
                    try:
                        process_file(filepath, args.output, args, ascii_dict, font, char_width, char_height)
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
    else:
        process_file(args.input, args.output, args, ascii_dict, font, char_width, char_height)

def create_ascii_dict(ascii_chars):
    """Create a mapping of brightness values to ASCII characters"""
    # Create a dictionary mapping brightness values (0-255) to ASCII characters
    ascii_dict = {}
    char_count = len(ascii_chars)
    
    for i in range(256):
        # Map each brightness level to an appropriate character
        index = min(int(i / 256 * char_count), char_count - 1)
        ascii_dict[i] = ascii_chars[index]
    
    return ascii_dict

def process_file(input_path, output_dir, args, ascii_dict, font, char_width, char_height):
    """Process a single file (image or video)"""
    input_ext = os.path.splitext(input_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    if input_ext in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}:
        output_filename = f"{base_name}_ascii.png"
        output_path = os.path.join(output_dir, output_filename)
        img = Image.open(input_path)
        with tqdm(total=100, desc=f'Processing {os.path.basename(input_path)}') as pbar:
            ascii_art, output_width, output_height = image_to_ascii(
                img, args, ascii_dict, char_width, char_height, pbar
            )
            save_ascii_as_image(ascii_art, output_path, args, font, char_width, char_height)
            pbar.update(100 - pbar.n)
        print(f"Enhanced ASCII image saved to {output_path} ({output_width}x{output_height})")
    elif input_ext in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
        output_filename = f"{base_name}_ascii.mp4"
        output_path = os.path.join(output_dir, output_filename)
        video_to_ascii(input_path, output_path, args, ascii_dict, font, char_width, char_height)
        print(f"Enhanced ASCII video saved to {output_path}")
    else:
        raise ValueError(f"Unsupported file format: {input_ext}")

def get_font_metrics(font_path=None, font_size=10):
    """Get font metrics for character rendering"""
    # Try to load Courier or another monospace font
    font_options = ['Courier New.ttf', 'CourierNew.ttf', 'cour.ttf', 
                    'DejaVuSansMono.ttf', 'LiberationMono-Regular.ttf']
    
    font = None
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            pass
    
    if not font:
        for font_name in font_options:
            try:
                font = ImageFont.truetype(font_name, font_size)
                break
            except IOError:
                continue
    
    if not font:
        font = ImageFont.load_default()
        print("Using default font. For better results, install a monospace font.")
    
    # Get character dimensions using the new getbbox method
    bbox = font.getbbox('W')
    return font, bbox[2]-bbox[0], bbox[3]-bbox[1]

def enhance_image(img, args):
    """Apply enhancements to the image for better ASCII output"""
    # Make a copy to preserve original
    enhanced_img = img.copy()
    
    # Apply enhancements based on selected mode
    color_settings = COLOR_ENHANCEMENTS[args.color_mode]
    
    # Apply contrast
    enhancer = ImageEnhance.Contrast(enhanced_img)
    enhanced_img = enhancer.enhance(color_settings['contrast'])
    
    # Apply brightness
    enhancer = ImageEnhance.Brightness(enhanced_img)
    enhanced_img = enhancer.enhance(color_settings['brightness'])
    
    # Apply saturation if color output is requested
    if args.color:
        enhancer = ImageEnhance.Color(enhanced_img)
        enhanced_img = enhancer.enhance(color_settings['saturation'])
    
    # Apply edge enhancement for depth
    if args.depth != 'none':
        # Create edge mask
        edge_img = enhanced_img.filter(ImageFilter.FIND_EDGES)
        edge_img = edge_img.convert('L')
        
        # Apply edge overlay with appropriate strength
        edge_strength = EDGE_ENHANCEMENT[args.depth]
        enhanced_img = Image.blend(enhanced_img, 
                                   edge_img.convert('RGB'), 
                                   edge_strength * 0.3)
    
    # Apply optional dithering for better gradients
    if args.dithering:
        if enhanced_img.mode != 'RGB':
            enhanced_img = enhanced_img.convert('RGB')
        # Convert to palette mode with dithering
        enhanced_img = enhanced_img.convert('RGB').convert('P', 
                                                          palette=Image.ADAPTIVE, 
                                                          colors=256, 
                                                          dither=Image.FLOYDSTEINBERG)
        enhanced_img = enhanced_img.convert('RGB')
    
    return enhanced_img

def analyze_brightness_distribution(img):
    """Analyze brightness distribution for dynamic range adjustment"""
    img_gray = img.convert('L')
    pixels = np.array(img_gray).flatten()
    
    # Get 5th and 95th percentile for robust min/max
    min_brightness = np.percentile(pixels, 5)
    max_brightness = np.percentile(pixels, 95)
    
    return min_brightness, max_brightness

def image_to_ascii(img, args, ascii_dict, char_width, char_height, pbar):
    """Convert image to ASCII art with enhanced techniques"""
    # Ensure image is in RGB mode
    img = img.convert('RGB')
    original_width, original_height = img.size
    aspect_ratio = original_width / original_height
    
    # Calculate dimensions based on target height
    ascii_rows = int(args.resolution / char_height)
    ascii_cols = int(ascii_rows * aspect_ratio * (char_height / char_width))
    
    # Update progress
    pbar.update(10)
    
    # Resize with high-quality resampling
    img_resized = img.resize((ascii_cols, ascii_rows), resample=Image.LANCZOS)
    
    # Apply enhancements
    img_enhanced = enhance_image(img_resized, args)
    pbar.update(20)
    
    # Convert to numpy array for processing
    pixels = np.array(img_enhanced)
    
    # Analyze brightness distribution for dynamic range if enabled
    if args.dynamic_range:
        min_brightness, max_brightness = analyze_brightness_distribution(img_enhanced)
        brightness_range = max_brightness - min_brightness
        if brightness_range < 10:  # Avoid division by zero or very small ranges
            brightness_range = 255
            min_brightness = 0
    
    # Create ASCII art
    ascii_art = []
    for row in tqdm(pixels, desc='Converting pixels', leave=False):
        ascii_row = []
        for pixel in row:
            r, g, b = pixel
            
            # Calculate brightness using improved formula
            # This formula better matches human perception
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            
            # Apply dynamic range adjustment if enabled
            if args.dynamic_range and brightness_range > 10:
                # Scale brightness to use full range of ASCII characters
                brightness = ((brightness - min_brightness) / brightness_range) * 255
                brightness = max(0, min(255, brightness))
            
            # Get ASCII character for this brightness
            index = int(brightness)
            ascii_char = ascii_dict[index]
            
            # Store character with color or brightness info
            if args.color:
                # Enhance colors for more vibrant output
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                
                # Increase saturation for more vibrant colors
                if args.color_mode == 'vibrant':
                    s = min(1.0, s * 1.3)
                
                # Convert back to RGB
                r_adj, g_adj, b_adj = colorsys.hsv_to_rgb(h, s, v)
                color_rgb = (int(r_adj*255), int(g_adj*255), int(b_adj*255))
                
                ascii_row.append((ascii_char, color_rgb))
            else:
                ascii_row.append((ascii_char, int(brightness)))
        
        ascii_art.append(ascii_row)
    
    pbar.update(40)
    output_width = ascii_cols * char_width
    output_height = ascii_rows * char_height
    return ascii_art, output_width, output_height

def save_ascii_as_image(ascii_art, output_path, args, font, char_width, char_height):
    """Save ASCII art as an image with enhanced rendering"""
    cols = len(ascii_art[0]) if ascii_art else 0
    rows = len(ascii_art)
    
    # Determine background color
    bg_color = 'black'
    if args.background == 'white':
        bg_color = 'white'
    elif args.background == 'auto':
        # Use white background for light color modes, black for others
        bg_color = 'white' if args.color_mode in ['muted', 'grayscale'] else 'black'
    
    # Create image with appropriate background
    img = Image.new('RGB', (cols * char_width, rows * char_height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # For white background, we may need to adjust text color
    is_dark_bg = bg_color == 'black'
    
    for y, row in enumerate(ascii_art):
        for x, char_info in enumerate(row):
            if args.color:
                char, color_rgb = char_info
                
                # For white background with color, ensure text is visible
                if not is_dark_bg:
                    r, g, b = color_rgb
                    # If color is too light, darken it
                    if (r + g + b) / 3 > 200:
                        # Reduce brightness
                        r = int(r * 0.7)
                        g = int(g * 0.7)
                        b = int(b * 0.7)
                        color_rgb = (r, g, b)
                
                draw.text((x * char_width, y * char_height), char, fill=color_rgb, font=font)
            else:
                char, brightness = char_info
                
                # For grayscale on white background, invert the brightness
                if not is_dark_bg:
                    brightness = 255 - brightness
                
                gray_value = int(brightness)
                color_rgb = (gray_value, gray_value, gray_value)
                draw.text((x * char_width, y * char_height), char, fill=color_rgb, font=font)
    
    # Apply final image enhancement
    if args.color_mode in ['vibrant', 'high_contrast']:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
    
    # Apply a slight sharpening for more defined characters
    img = img.filter(ImageFilter.SHARPEN)
    
    # Save with high quality
    img.save(output_path, quality=95)

def video_to_ascii(input_path, output_path, args, ascii_dict, font, char_width, char_height):
    """Convert video to ASCII art with enhanced techniques"""
    if not shutil.which('ffmpeg'):
        raise EnvironmentError("ffmpeg is required for video processing.")

    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, 'temp_video.mp4')
    temp_audio_path = os.path.join(temp_dir, 'audio.aac')

    # Open video
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = original_width / original_height

    # Calculate ASCII dimensions
    ascii_rows = int(args.resolution / char_height)
    ascii_cols = int(ascii_rows * aspect_ratio * (char_height / char_width))
    output_width = ascii_cols * char_width
    output_height = ascii_rows * char_height

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (output_width, output_height))

    # Analyze first frame for dynamic range if enabled
    min_brightness, max_brightness = 0, 255
    brightness_range = 255
    
    if args.dynamic_range:
        ret, first_frame = cap.read()
        if ret:
            first_img = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
            min_brightness, max_brightness = analyze_brightness_distribution(first_img)
            brightness_range = max_brightness - min_brightness
            if brightness_range < 10:
                brightness_range = 255
                min_brightness = 0
            # Reset video capture
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Determine background color
    bg_color = 'black'
    if args.background == 'white':
        bg_color = 'white'
    elif args.background == 'auto':
        bg_color = 'white' if args.color_mode in ['muted', 'grayscale'] else 'black'
    
    # For OpenCV, convert color name to BGR tuple
    bg_color_bgr = (0, 0, 0) if bg_color == 'black' else (255, 255, 255)
    is_dark_bg = bg_color == 'black'

    with tqdm(total=frame_count, desc='Processing Video', unit='frame') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert from BGR to RGB (for PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Resize with high-quality resampling
            img_resized = img.resize((ascii_cols, ascii_rows), resample=Image.LANCZOS)
            
            # Apply enhancements
            img_enhanced = enhance_image(img_resized, args)
            pixels = np.array(img_enhanced)
            
            # Create ASCII art
            ascii_art = []
            for row in pixels:
                ascii_row = []
                for pixel in row:
                    r, g, b = pixel
                    
                    # Calculate brightness using improved formula
                    brightness = 0.299 * r + 0.587 * g + 0.114 * b
                    
                    # Apply dynamic range adjustment if enabled
                    if args.dynamic_range and brightness_range > 10:
                        brightness = ((brightness - min_brightness) / brightness_range) * 255
                        brightness = max(0, min(255, brightness))
                    
                    # Get ASCII character for this brightness
                    index = int(brightness)
                    ascii_char = ascii_dict[index]
                    
                    # Store character with color or brightness info
                    if args.color:
                        # Enhance colors for more vibrant output
                        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                        
                        # Adjust saturation based on color mode
                        if args.color_mode == 'vibrant':
                            s = min(1.0, s * 1.3)
                        
                        # Convert back to RGB
                        r_adj, g_adj, b_adj = colorsys.hsv_to_rgb(h, s, v)
                        color_rgb = (int(r_adj*255), int(g_adj*255), int(b_adj*255))
                        
                        ascii_row.append((ascii_char, color_rgb))
                    else:
                        ascii_row.append((ascii_char, int(brightness)))
                
                ascii_art.append(ascii_row)
            
            # Create frame image
            ascii_img = Image.new('RGB', (output_width, output_height), bg_color)
            draw = ImageDraw.Draw(ascii_img)
            
            for y, row in enumerate(ascii_art):
                for x, char_info in enumerate(row):
                    if args.color:
                        char, color_rgb = char_info
                        
                        # For white background with color, ensure text is visible
                        if not is_dark_bg:
                            r, g, b = color_rgb
                            # If color is too light, darken it
                            if (r + g + b) / 3 > 200:
                                r = int(r * 0.7)
                                g = int(g * 0.7)
                                b = int(b * 0.7)
                                color_rgb = (r, g, b)
                        
                        draw.text((x * char_width, y * char_height), char, fill=color_rgb, font=font)
                    else:
                        char, brightness = char_info
                        
                        # For grayscale on white background, invert the brightness
                        if not is_dark_bg:
                            brightness = 255 - brightness
                        
                        gray_value = int(brightness)
                        color_rgb = (gray_value, gray_value, gray_value)
                        draw.text((x * char_width, y * char_height), char, fill=color_rgb, font=font)
            
            # Apply final image enhancement
            if args.color_mode in ['vibrant', 'high_contrast']:
                enhancer = ImageEnhance.Contrast(ascii_img)
                ascii_img = enhancer.enhance(1.2)
            
            # Apply a slight sharpening for more defined characters
            ascii_img = ascii_img.filter(ImageFilter.SHARPEN)
            
            # Convert back to OpenCV format
            frame_np = np.array(ascii_img)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(frame_bgr)
            pbar.update(1)

    cap.release()
    out.release()

    # Process audio
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-vn', '-acodec', 'copy',
            temp_audio_path
        ], check=True, capture_output=True)

        subprocess.run([
            'ffmpeg', '-y',
            '-i', temp_video_path,
            '-i', temp_audio_path,
            '-c:v', 'h264', '-crf', '23', '-preset', 'medium',
            '-c:a', 'aac', '-b:a', '128k',
            '-map', '0:v:0', '-map', '1:a:0',
            output_path
        ], check=True, capture_output=True)

    except subprocess.CalledProcessError as e:
        print(f"Error processing audio: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
        # Fallback to video without audio
        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-i', temp_video_path,
                '-c:v', 'h264', '-crf', '23', '-preset', 'medium',
                output_path
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # Last resort - just copy the file
            shutil.copy(temp_video_path, output_path)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()