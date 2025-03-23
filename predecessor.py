import argparse
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm
import subprocess
import shutil
import tempfile

# Custom ASCII character mapping based on brightness ranges
ascii_dict = [None]  # Initialize with one element to enable slice assignments
ascii_dict[0:40] = [' '] * 40      # 0-39: Space
ascii_dict[40:50] = ['_'] * 10     # 40-49: Underscore
ascii_dict[50:75] = ['!'] * 25     # 50-74: Exclamation
ascii_dict[75:90] = ['|'] * 15     # 75-89: Vertical bar
ascii_dict[90:128] = ['t'] * 38    # 90-127: t
ascii_dict[128:170] = ['w'] * 42    # 128-169: w
ascii_dict[170:220] = ['#'] * 50    # 170-219: Hash
ascii_dict[220:256] = ['@'] * 36    # 220-255: At symbol

#ASCII_CHARS = " .'`^,:;Il!i><~+_-?}{1)(|\\/tfjrnxuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

def main():
    parser = argparse.ArgumentParser(description='Convert image/video to ASCII art')
    parser.add_argument('--input', type=str, required=True, help='Input file or directory path')
    parser.add_argument('--output', type=str, required=True, help='Output directory path')
    parser.add_argument('--color', action='store_true', help='Output color ASCII art')
    parser.add_argument('--resolution', type=int, choices=[144, 240, 360, 480, 720], default=720,
                        help='Target output height in pixels')
    args = parser.parse_args()

    supported_image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    supported_video_exts = {'.mp4', '.avi', '.mov'}
    all_supported_exts = supported_image_exts.union(supported_video_exts)

    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.input):
        for filename in os.listdir(args.input):
            filepath = os.path.join(args.input, filename)
            if os.path.isfile(filepath):
                ext = os.path.splitext(filename)[1].lower()
                if ext in all_supported_exts:
                    try:
                        process_file(filepath, args.output, args.color, args.resolution)
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
    else:
        process_file(args.input, args.output, args.color, args.resolution)

def process_file(input_path, output_dir, color, resolution):
    input_ext = os.path.splitext(input_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    font, char_width, char_height = get_font_metrics()

    if input_ext in {'.jpg', '.jpeg', '.png', '.bmp'}:
        output_filename = f"{base_name}_ascii.png"
        output_path = os.path.join(output_dir, output_filename)
        img = Image.open(input_path)
        with tqdm(total=100, desc=f'Processing {os.path.basename(input_path)}') as pbar:
            ascii_art, output_width, output_height = image_to_ascii(
                img, resolution, color, char_width, char_height, pbar
            )
            save_ascii_as_image(ascii_art, output_path, color, font, char_width, char_height)
            pbar.update(100 - pbar.n)
        print(f"ASCII image saved to {output_path} ({output_width}x{output_height})")
    elif input_ext in {'.mp4', '.avi', '.mov'}:
        output_filename = f"{base_name}_ascii.mp4"
        output_path = os.path.join(output_dir, output_filename)
        video_to_ascii(input_path, output_path, resolution, color, font, char_width, char_height)
        print(f"ASCII video saved to {output_path}")
    else:
        raise ValueError(f"Unsupported file format: {input_ext}")

def get_font_metrics(font_path='Courier New.ttf', font_size=10):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
    bbox = font.getbbox('A')
    return font, bbox[2]-bbox[0], bbox[3]-bbox[1]

def save_ascii_as_image(ascii_art, output_path, color, font, char_width, char_height):
    cols = len(ascii_art[0]) if ascii_art else 0
    rows = len(ascii_art)
    
    img = Image.new('RGB', (cols * char_width, rows * char_height), color='black')
    draw = ImageDraw.Draw(img)
    
    for y, row in enumerate(ascii_art):
        for x, char_info in enumerate(row):
            if color:
                char, color_rgb = char_info
            else:
                char, brightness = char_info
                gray_value = int(brightness)
                color_rgb = (gray_value, gray_value, gray_value)
            draw.text((x * char_width, y * char_height), char, fill=color_rgb, font=font)
    
    img.save(output_path)

def image_to_ascii(img, target_height_pixels, color, char_width, char_height, pbar):
    img = img.convert('RGB')
    original_width, original_height = img.size
    aspect_ratio = original_width / original_height

    ascii_rows = int(target_height_pixels / char_height)
    ascii_cols = int(ascii_rows * aspect_ratio * (char_height / char_width))
    
    pbar.update(20)
    
    img_resized = img.resize((ascii_cols, ascii_rows), resample=Image.Resampling.LANCZOS)
    pixels = np.array(img_resized)
    pbar.update(30)
    
    ascii_art = []
    for row in tqdm(pixels, desc='Converting pixels', leave=False):
        ascii_row = []
        for pixel in row:
            r, g, b = pixel
            brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b
            index = int(brightness)
            ascii_char = ascii_dict[index]
            if color:
                ascii_row.append((ascii_char, (r, g, b)))
            else:
                ascii_row.append((ascii_char, int(brightness)))
        ascii_art.append(ascii_row)
    
    pbar.update(30)
    output_width = ascii_cols * char_width
    output_height = ascii_rows * char_height
    return ascii_art, output_width, output_height

def video_to_ascii(input_path, output_path, target_height_pixels, color, font, char_width, char_height):
    if not shutil.which('ffmpeg'):
        raise EnvironmentError("ffmpeg is required for video processing.")

    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, 'temp_video.mp4')
    temp_audio_path = os.path.join(temp_dir, 'audio.aac')

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = original_width / original_height

    ascii_rows = int(target_height_pixels / char_height)
    ascii_cols = int(ascii_rows * aspect_ratio * (char_height / char_width))
    output_width = ascii_cols * char_width
    output_height = ascii_rows * char_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (output_width, output_height))

    with tqdm(total=frame_count, desc='Processing Video', unit='frame') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_resized = img.resize((ascii_cols, ascii_rows), resample=Image.Resampling.LANCZOS)
            pixels = np.array(img_resized)
            
            ascii_art = []
            for row in pixels:
                ascii_row = []
                for pixel in row:
                    r, g, b = pixel
                    brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    index = int(brightness)
                    ascii_char = ascii_dict[index]
                    if color:
                        ascii_row.append((ascii_char, (r, g, b)))
                    else:
                        ascii_row.append((ascii_char, int(brightness)))
                ascii_art.append(ascii_row)
            
            ascii_img = Image.new('RGB', (output_width, output_height), 'black')
            draw = ImageDraw.Draw(ascii_img)
            for y, row in enumerate(ascii_art):
                for x, char_info in enumerate(row):
                    if color:
                        char, color_rgb = char_info
                    else:
                        char, brightness = char_info
                        gray_value = int(brightness)
                        color_rgb = (gray_value, gray_value, gray_value)
                    draw.text((x * char_width, y * char_height), char, fill=color_rgb, font=font)
            
            frame_np = np.array(ascii_img)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            pbar.update(1)

    cap.release()
    out.release()

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
            '-c:v', 'copy', '-c:a', 'aac',
            '-map', '0:v:0', '-map', '1:a:0',
            output_path
        ], check=True, capture_output=True)

    except subprocess.CalledProcessError as e:
        print(f"Error processing audio: {e.stderr.decode()}")
        shutil.copy(temp_video_path, output_path)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()