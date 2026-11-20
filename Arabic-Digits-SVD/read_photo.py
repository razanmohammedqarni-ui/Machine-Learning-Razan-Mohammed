import sys
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

ACCENT = "#0B8F7E"
PAPER_WIDTH = 1200


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


def find_paper(photo):
    total = photo.sum(axis=2) + 1
    red, green = photo[:, :, 0] / total, photo[:, :, 1] / total
    bright = photo.mean(axis=2)
    height, width = bright.shape
    middle = (slice(int(height * 0.42), int(height * 0.58)), slice(int(width * 0.42), int(width * 0.58)))
    same_color = np.hypot(red - np.median(red[middle]), green - np.median(green[middle])) < 0.03
    paper = ndi.binary_opening(same_color & (bright > 0.55 * np.median(bright[middle])), iterations=4)
    regions, count = ndi.label(paper)
    inside = regions[middle][regions[middle] > 0]
    if len(inside) == 0:
        return np.ones(bright.shape, dtype=bool)
    return ndi.binary_fill_holes(regions == np.bincount(inside).argmax())


def find_digits(path):
    photo = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    shrink = min(1, 2000 / max(photo.size))
    photo = np.asarray(photo.resize((round(photo.width * shrink), round(photo.height * shrink)))).astype(float)
    paper = find_paper(photo)
    ys, xs = np.where(paper)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    scale = PAPER_WIDTH / (x1 - x0)
    size = (round((x1 - x0) * scale), round((y1 - y0) * scale))
    crop = np.asarray(Image.fromarray(photo[y0:y1, x0:x1].astype(np.uint8)).resize(size, Image.LANCZOS))
    paper = np.asarray(Image.fromarray((paper[y0:y1, x0:x1] * 255).astype(np.uint8)).resize(size, Image.NEAREST)) > 0
    paper = ndi.binary_erosion(paper, iterations=25)
    gray = crop.astype(float).mean(axis=2)
    background = ndi.median_filter(gray[::4, ::4], size=25)
    background = np.kron(background, np.ones((4, 4)))[:gray.shape[0], :gray.shape[1]]
    ink = ndi.binary_closing((background - gray > 38) & paper, iterations=2)
    regions, count = ndi.label(ndi.binary_dilation(ink, iterations=6))
    found = []
    for i, box in enumerate(ndi.find_objects(regions)):
        mask = ink[box] & (regions[box] == i + 1)
        if mask.sum() >= 60:
            found.append((box[0].start, box[1].start, box[0].stop, box[1].stop, mask))
    if not found:
        return [], crop, 0
    tall = np.percentile([item[2] - item[0] for item in found], 75)
    found.sort(key=lambda item: item[0] + item[2])
    rows = [[found[0]]]
    for item in found[1:]:
        first = rows[-1][0]
        if (item[0] + item[2]) / 2 - (first[0] + first[2]) / 2 > tall:
            rows.append([item])
        else:
            rows[-1].append(item)
    return [sorted(row, key=lambda item: item[1]) for row in rows], crop, tall


def read_row(row, bases, tall):
    images = np.array([to_28(item[4]) for item in row])
    total = np.sum(images ** 2, axis=1)
    residuals = np.stack([total - np.sum((images @ U) ** 2, axis=1) for U in bases], axis=1)
    guesses = np.argmin(residuals, axis=1)
    changed = 0
    for i, item in enumerate(row):
        mask = item[4]
        hole = (ndi.binary_fill_holes(mask).sum() - mask.sum()) / mask.size
        if guesses[i] == 0 and (hole > 0.1 or item[2] - item[0] > 0.6 * tall):
            guesses[i] = 1 + np.argmin(residuals[i, 1:])
            changed += 1
    return guesses, changed


def arabic(digit):
    return chr(0x0660 + digit)


path = sys.argv[1] if len(sys.argv) > 1 else "my_digits.png"
bases = np.load("bases.npy")
rows, paper, tall = find_digits(path)
if not rows:
    print("No digits found. Use a dark pen on light paper and keep the paper in the middle of the photo.")
    sys.exit()

fig = plt.figure(figsize=(8, 8 * paper.shape[0] / paper.shape[1]), dpi=120)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(paper)
ax.axis("off")
fixed = 0
for number, row in enumerate(rows):
    guesses, changed = read_row(row, bases, tall)
    fixed += changed
    print("Row", number + 1, "|", " ".join(str(g) for g in guesses), "|", " ".join(arabic(g) for g in guesses))
    for item, guess in zip(row, guesses):
        y0, x0, y1, x1 = item[0], item[1], item[2], item[3]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=ACCENT, linewidth=1.5))
        ax.text((x0 + x1) / 2, y0 - 8, str(guess), color="white", fontsize=13, fontweight="bold", ha="center", va="bottom", bbox=dict(facecolor=ACCENT, edgecolor="none", boxstyle="round,pad=0.25"))
print("Digits changed by the zero rule:", fixed)
fig.savefig("reading.png")
print("Saved reading.png")
