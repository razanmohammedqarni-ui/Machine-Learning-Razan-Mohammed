import sys
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

K_CHOICES = [5, 10, 20, 30, 40, 50]
MY_DIGITS = [[1, 2, 3], [0, 5, 6, 1, 8], [7, 0, 0, 5, 7, 8, 9]]
SPLIT = 50000
BACKGROUND = "#0E1117"
TEXT = "#F2F4F8"
MUTED = "#9AA4B2"
ACCENT = "#5EC4B6"
WARM = "#F2A65A"
ARABIC = "DejaVu Sans"


def load(folder):
    train_x = np.loadtxt(folder + "/csvTrainImages 60k x 784.csv", delimiter=",") / 255
    train_y = np.loadtxt(folder + "/csvTrainLabel 60k x 1.csv", dtype=int)
    test_x = np.loadtxt(folder + "/csvTestImages.csv", delimiter=",") / 255
    test_y = np.loadtxt(folder + "/csvTestLabel 10k x 1.csv", dtype=int)
    return train_x, train_y, test_x, test_y


def learn_bases(images, labels):
    bases = []
    for digit in range(10):
        A = images[labels == digit].T
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        bases.append(U)
    return bases


def residuals(images, bases, k):
    total = np.sum(images ** 2, axis=1)
    columns = []
    for U in bases:
        projection = images @ U[:, :k]
        columns.append(total - np.sum(projection ** 2, axis=1))
    return np.stack(columns, axis=1)


def predict(images, bases, k):
    return np.argmin(residuals(images, bases, k), axis=1)


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


def read_note(path):
    photo = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB")).astype(float)
    r, g, b = photo[:, :, 0], photo[:, :, 1], photo[:, :, 2]
    paper = (np.abs(r - b) < 22) & (np.abs(r - g) < 12) & (photo.min(axis=2) > 140)
    paper = ndi.binary_opening(paper, iterations=5)
    regions, count = ndi.label(paper)
    paper = regions == 1 + np.argmax(ndi.sum(paper, regions, range(1, count + 1)))
    paper = ndi.binary_erosion(ndi.binary_fill_holes(paper), iterations=25)
    gray = photo.mean(axis=2)
    background = ndi.median_filter(gray[::4, ::4], size=25)
    background = np.kron(background, np.ones((4, 4)))[:gray.shape[0], :gray.shape[1]]
    ink = ndi.binary_closing((background - gray > 38) & paper, iterations=2)
    regions, count = ndi.label(ndi.binary_dilation(ink, iterations=6))
    found = []
    for i, box in enumerate(ndi.find_objects(regions)):
        mask = ink[box] & (regions[box] == i + 1)
        if mask.sum() >= 60:
            found.append((box[0].start, box[1].start, box[0].stop, box[1].stop, mask))
    found.sort(key=lambda item: item[0])
    rows = [[found[0]]]
    for item in found[1:]:
        if item[0] - rows[-1][0][0] > 120:
            rows.append([item])
        else:
            rows[-1].append(item)
    rows = [sorted(row, key=lambda item: item[1]) for row in rows]
    y0 = max(0, min(item[0] for item in found) - 40)
    x0 = max(0, min(item[1] for item in found) - 40)
    y1 = max(item[2] for item in found) + 40
    x1 = max(item[3] for item in found) + 40
    crop = photo[y0:y1, x0:x1].astype(np.uint8)
    return [[to_28(item[4]) for item in row] for row in rows], crop


def picture(vector):
    return vector.reshape(28, 28).T


def arabic(digit):
    return chr(0x0660 + digit)


folder = sys.argv[1] if len(sys.argv) > 1 else "data"
note_path = sys.argv[2] if len(sys.argv) > 2 else "my_digits.png"
train_x, train_y, test_x, test_y = load(folder)
print("Training images:", train_x.shape[0], "| Test images:", test_x.shape[0])

bases = learn_bases(train_x[:SPLIT], train_y[:SPLIT])
best_k, best_score = 0, 0
for k in K_CHOICES:
    score = 100 * np.mean(predict(train_x[SPLIT:], bases, k) == train_y[SPLIT:])
    print("k =", k, "| validation accuracy =", round(score, 2), "%")
    if score > best_score:
        best_k, best_score = k, score
print("Best k:", best_k)

bases = learn_bases(train_x, train_y)
predictions = predict(test_x, bases, best_k)
accuracy = 100 * np.mean(predictions == test_y)
print("Test accuracy =", round(accuracy, 2), "%")

confusion = np.zeros((10, 10), dtype=int)
for truth, guess in zip(test_y, predictions):
    confusion[truth, guess] += 1
mistakes = confusion.copy()
np.fill_diagonal(mistakes, 0)
worst_truth, worst_guess = np.unravel_index(np.argmax(mistakes), mistakes.shape)
print("Most common mistake:", worst_truth, "read as", worst_guess, "|", mistakes[worst_truth, worst_guess], "times")

my_rows, note = read_note(note_path)
my_guesses = [predict(np.array(row), bases, best_k) for row in my_rows]
my_correct = sum(int(np.sum(guess == np.array(truth))) for guess, truth in zip(my_guesses, MY_DIGITS))
my_total = sum(len(row) for row in MY_DIGITS)
for guess, truth in zip(my_guesses, MY_DIGITS):
    print("My handwriting | wrote", truth, "| read", guess.tolist())
print("My handwriting:", my_correct, "of", my_total)

two_tone = LinearSegmentedColormap.from_list("two_tone", [WARM, BACKGROUND, ACCENT])
white = LinearSegmentedColormap.from_list("white", [BACKGROUND, TEXT])

fig = plt.figure(figsize=(8, 10), dpi=150, facecolor=BACKGROUND)
left, cell, gap = 0.06, 0.0808, 0.008
cell_h = cell * 8 / 10

fig.text(left, 0.945, "No neural network. Just linear algebra.", color=TEXT, fontsize=21, fontweight="bold")
fig.text(left, 0.912, "Reading Arabic handwritten digits with Singular Value Decomposition (SVD)", color=MUTED, fontsize=12)
fig.text(left, 0.828, str(np.floor(accuracy * 10) / 10) + "%", color=ACCENT, fontsize=40, fontweight="bold")
fig.text(left + 0.33, 0.851, "accuracy on " + format(test_x.shape[0], ",") + " handwritten digits it never saw", color=TEXT, fontsize=12)
fig.text(left + 0.33, 0.826, "trained on " + format(train_x.shape[0], ",") + " images  ·  k = " + str(best_k) + " vectors per digit", color=MUTED, fontsize=10.5)

fig.text(left, 0.772, "What it learned", color=TEXT, fontsize=13, fontweight="bold")
fig.text(left + 0.2, 0.772, "the top 3 singular vectors of each digit", color=MUTED, fontsize=10.5)
top = 0.712
for digit in range(10):
    x = left + digit * (cell + gap)
    fig.text(x + cell / 2 - 0.012, top + 0.008, arabic(digit), color=TEXT, fontsize=15, ha="center", fontfamily=ARABIC)
    fig.text(x + cell / 2 + 0.012, top + 0.009, str(digit), color=MUTED, fontsize=9, ha="center")
    for row in range(3):
        vector = bases[digit][:, row]
        if row == 0 and vector.sum() < 0:
            vector = -vector
        limit = np.abs(vector).max()
        ax = fig.add_axes([x, top - (row + 1) * cell_h - row * gap * 0.8, cell, cell_h])
        ax.imshow(picture(vector), cmap=two_tone, vmin=-limit, vmax=limit)
        ax.axis("off")

fig.text(left, 0.455, "My own handwriting", color=TEXT, fontsize=13, fontweight="bold")
fig.text(left + 0.27, 0.455, str(my_correct) + " of " + str(my_total) + " digits read correctly from a photo of a sticky note", color=MUTED, fontsize=10.5)
note_w = 0.36
note_h = note_w * 8 / 10 * note.shape[0] / note.shape[1]
top = 0.43
ax = fig.add_axes([left, top - note_h, note_w, note_h])
ax.imshow(note)
ax.axis("off")
small = 0.06
small_h = small * 8 / 10
step = note_h / len(my_rows)
for r, (row, guess, truth) in enumerate(zip(my_rows, my_guesses, MY_DIGITS)):
    for c in range(len(row)):
        x = left + note_w + 0.04 + c * (small + gap)
        y = top - r * step - small_h
        ax = fig.add_axes([x, y, small, small_h])
        ax.imshow(picture(row[c]), cmap=white, vmin=0, vmax=1)
        ax.axis("off")
        right = guess[c] == truth[c]
        fig.text(x + small / 2, y - 0.022, arabic(guess[c]) + (" ✓" if right else " ✗"), color=TEXT if right else WARM, fontsize=11.5, ha="center", fontfamily=ARABIC)

fig.text(left, 0.135, "How it reads", color=TEXT, fontsize=11, fontweight="bold")
fig.text(left + 0.2, 0.135, "project the image onto each digit's subspace, pick the closest", color=MUTED, fontsize=10.5)
fig.text(left, 0.105, "Where it fails", color=TEXT, fontsize=11, fontweight="bold")
fig.text(left + 0.2, 0.105, "most common mistake on the test set: " + str(worst_truth) + " read as " + str(worst_guess) + "  (" + str(mistakes[worst_truth, worst_guess]) + " of " + format(confusion[worst_truth].sum(), ",") + ")", color=MUTED, fontsize=10.5)

fig.text(left, 0.04, "Data: Arabic Handwritten Digits Dataset  ·  k chosen on a validation split, test set used once", color=MUTED, fontsize=9)
fig.savefig("arabic_digits_svd.png", facecolor=BACKGROUND)
print("Saved arabic_digits_svd.png")
