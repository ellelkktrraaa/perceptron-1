import os
from PIL import Image

dir = input("Enter directory path: ")

os.chdir(dir)

ppm_files = sorted([f for f in os.listdir('.') if f.endswith('.ppm')])

print(f"Found {len(ppm_files)} PPM files")

for i, fname in enumerate(ppm_files):
    img = Image.open(fname)
    img.show()
    print(f"[{i+1}/{len(ppm_files)}] {fname}")
    cmd = input("Press Enter for next (q to quit): ")
    if cmd.lower() == 'q':
        break
