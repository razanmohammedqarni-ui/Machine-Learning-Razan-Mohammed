import sys
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

ACCENT = "#0B8F7E"


def to_28(mask):
    ys, xs = np.where(mask)
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = mask.shape
    scale = 20 / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(size, Image.LANCZOS)) / 255
    small = np.clip(small, 0, 1)
    center_y, center_x = ndi.center_of_mass(small)
    top = min(max(int(round(14 - center_y)), 0), 28 - small.shape[0])
    side = min(max(int(round(14 - center_x)), 0), 28 - small.shape[1])
    canvas = np.zeros((28, 28))
    canvas[top:top + small.shape[0], side:side + small.shape[1]] = small
    return canvas.T.reshape(-1)


def find_digits(path):
    photo = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    scale = min(1, 2000 / max(photo.size))
    photo = photo.resize((round(photo.width * scale), round(photo.height * scale)))
    photo = np.asarray(photo)
    gray = photo.astype(float).mean(axis=2)
    background = ndi.median_filter(gray[::4, ::4], size=25)
    background = np.kron(background, np.ones((4, 4)))[:gray.shape[0], :gray.shape[1]]
    ink = ndi.binary_closing(background - gray > 38, iterations=2)
    regions, count = ndi.label(ndi.binary_dilation(ink, iterations=6))
    found = []
    for i, box in enumerate(ndi.find_objects(regions)):
        y0, y1, x0, x1 = box[0].start, box[0].stop, box[1].start, box[1].stop
        on_edge = y0 == 0 or x0 == 0 or y1 == gray.shape[0] or x1 == gray.shape[1]
        mask = ink[box] & (regions[box] == i + 1)
        if mask.sum() >= 60 and not on_edge:
            found.append((y0, x0, y1, x1, to_28(mask)))
    if not found:
        return [], photo
    tall = np.percentile([item[2] - item[0] for item in found], 75)
    found.sort(key=lambda item: item[0] + item[2])
    rows = [[found[0]]]
    for item in found[1:]:
        first = rows[-1][0]
        if (item[0] + item[2]) / 2 - (first[0] + first[2]) / 2 > tall:
            rows.append([item])
        else:
            rows[-1].append(item)
    return [sorted(row, key=lambda item: item[1]) for row in rows], photo


def predict(images, bases):
    total = np.sum(images ** 2, axis=1)
    columns = [total - np.sum((images @ U) ** 2, axis=1) for U in bases]
    return np.argmin(np.stack(columns, axis=1), axis=1)


def arabic(digit):
    return chr(0x0660 + digit)


path = sys.argv[1] if len(sys.argv) > 1 else "my_digits.png"
bases = np.load("bases.npy")
rows, photo = find_digits(path)
if not rows:
    print("No digits found. Use a dark pen on white paper and crop the photo to the paper.")
    sys.exit()

fig = plt.figure(figsize=(8, 8 * photo.shape[0] / photo.shape[1]), dpi=120)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(photo)
ax.axis("off")
for number, row in enumerate(rows):
    guesses = predict(np.array([item[4] for item in row]), bases)
    print("Row", number + 1, "|", " ".join(str(g) for g in guesses), "|", " ".join(arabic(g) for g in guesses))
    for item, guess in zip(row, guesses):
        y0, x0, y1, x1 = item[0], item[1], item[2], item[3]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=ACCENT, linewidth=1.5))
        ax.text((x0 + x1) / 2, y0 - 8, str(guess), color="white", fontsize=13, fontweight="bold", ha="center", va="bottom", bbox=dict(facecolor=ACCENT, edgecolor="none", boxstyle="round,pad=0.25"))
fig.savefig("reading.png")
print("Saved reading.png")
