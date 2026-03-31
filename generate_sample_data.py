"""
generate_sample_data.py - Generate synthetic sample images for quick testing
Run this to create a small dummy dataset when you don't have real images yet.

Usage:
    python generate_sample_data.py
    python generate_sample_data.py --count 50
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_healthy_leaf(size=(128, 128)):
    """Generate a synthetic healthy green leaf image."""
    img = Image.new('RGB', size, color=(34, 139, 34))  # forest green
    draw = ImageDraw.Draw(img)

    # Leaf body shape
    center_x, center_y = size[0] // 2, size[1] // 2
    leaf_color = (50 + np.random.randint(0, 40), 160 + np.random.randint(-20, 20), 50 + np.random.randint(0, 20))
    draw.ellipse([center_x - 45, center_y - 60, center_x + 45, center_y + 60], fill=leaf_color)

    # Central vein
    vein_color = (20, 100, 20)
    draw.line([(center_x, center_y - 60), (center_x, center_y + 60)], fill=vein_color, width=2)

    # Side veins
    for i in range(-3, 4):
        y = center_y + i * 15
        draw.line([(center_x, y), (center_x + 30 + np.random.randint(-5, 5), y + np.random.randint(-10, 10))],
                  fill=vein_color, width=1)
        draw.line([(center_x, y), (center_x - 30 - np.random.randint(-5, 5), y + np.random.randint(-10, 10))],
                  fill=vein_color, width=1)

    # Background variation
    bg_color = tuple([min(255, max(0, int(c + np.random.randint(-20, 20)))) for c in (240, 240, 220)])
    img_bg = Image.new('RGB', size, color=bg_color)
    img_bg.paste(img, (0, 0))
    img_bg = img_bg.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img_bg


def generate_diseased_leaf(size=(128, 128)):
    """Generate a synthetic diseased leaf with spots."""
    # Start with a yellowish/brownish leaf
    img = Image.new('RGB', size, color=(200, 200, 180))
    draw = ImageDraw.Draw(img)

    center_x, center_y = size[0] // 2, size[1] // 2
    # Sick leaf (pale/yellow-green)
    leaf_color = (
        150 + np.random.randint(0, 40),
        160 + np.random.randint(-30, 20),
        50 + np.random.randint(0, 20)
    )
    draw.ellipse([center_x - 45, center_y - 60, center_x + 45, center_y + 60], fill=leaf_color)

    # Disease spots (brown, dark spots)
    num_spots = np.random.randint(5, 15)
    for _ in range(num_spots):
        sx = center_x + np.random.randint(-35, 35)
        sy = center_y + np.random.randint(-50, 50)
        sr = np.random.randint(3, 12)
        spot_color = (
            np.random.randint(100, 160),
            np.random.randint(40, 80),
            np.random.randint(0, 30)
        )
        draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=spot_color)
        # Spot halo (yellowish ring)
        halo_color = (200, 180, 50)
        draw.ellipse([sx - sr - 2, sy - sr - 2, sx + sr + 2, sy + sr + 2],
                     outline=halo_color, width=1)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def create_dataset(output_dir='data', count_per_class=30):
    """Generate synthetic dataset."""
    healthy_dir = os.path.join(output_dir, 'healthy')
    diseased_dir = os.path.join(output_dir, 'diseased')
    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(diseased_dir, exist_ok=True)

    print(f"🌱 Generating {count_per_class} healthy leaf images...")
    for i in range(count_per_class):
        img = generate_healthy_leaf()
        img.save(os.path.join(healthy_dir, f'healthy_{i:04d}.jpg'))
        if (i + 1) % 10 == 0:
            print(f"   {i + 1}/{count_per_class} done")

    print(f"\n🍂 Generating {count_per_class} diseased leaf images...")
    for i in range(count_per_class):
        img = generate_diseased_leaf()
        img.save(os.path.join(diseased_dir, f'diseased_{i:04d}.jpg'))
        if (i + 1) % 10 == 0:
            print(f"   {i + 1}/{count_per_class} done")

    total = count_per_class * 2
    print(f"\n✅ Done! {total} sample images created in '{output_dir}/'")
    print(f"   healthy/  : {count_per_class} images")
    print(f"   diseased/ : {count_per_class} images")
    print("\n⚠️  NOTE: These are synthetic images for testing only.")
    print("   For real training, use a proper plant disease dataset from Kaggle.")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic leaf images for testing')
    parser.add_argument('--count', type=int, default=30,
                        help='Number of images per class (default: 30)')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory (default: data/)')
    args = parser.parse_args()

    print("=" * 50)
    print("🌿 Synthetic Dataset Generator")
    print("   (For testing purposes only)")
    print("=" * 50)
    create_dataset(output_dir=args.output, count_per_class=args.count)


if __name__ == '__main__':
    main()
