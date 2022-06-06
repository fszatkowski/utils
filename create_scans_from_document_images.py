import random
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from shutil import copy

import click
import cv2
import numpy as np
from tqdm import tqdm

random.seed(0)


@click.command()
@click.option("-i", "--input_dir", type=Path, default="ocr_benchmark/jats_dataset")
@click.option("-o", "--output_dir", type=Path, default="ocr_benchmark/pseudo_scans")
@click.option("-m", "--max_files", type=int, default=25)
@click.option(
    "-s",
    "--shots",
    type=int,
    default=2,
    help="Number of random modifications to generate for each image.",
)
def main(input_dir: Path, output_dir: Path, max_files: int, shots: int):
    docs = sorted(list(input_dir.glob("*")))
    image_paths = [img_path for doc in docs for img_path in doc.glob("*.jpg")]
    image_paths = sorted(image_paths)
    image_paths = random.sample(image_paths, max_files)

    image_paths = shots * image_paths
    _apply_random_transformations = partial(
        apply_random_transformations, output_dir=output_dir
    )
    with Pool(2) as p:
        results = list(
            tqdm(
                p.imap(_apply_random_transformations, image_paths),
                total=len(image_paths),
            )
        )

    assert all(results)


def apply_random_transformations(image_path: Path, output_dir: Path) -> bool:
    img = cv2.imread(str(image_path))

    file_name = f"{image_path.stem}"
    if random.choice([True, False]):
        brightness = random.randint(60, 100)
        file_name += f"_b{brightness}"
        img = change_brightness(img, brightness)

    if random.choice([True, False]):
        a, b, c = (4 * (random.random() - 0.5)), -1, (img.shape[0] * (random.random()))
        max_val = random.randint(30, 60)
        file_name += f"_a{a}_b{b}_c{c}_g{max_val}"
        img = change_brightness_perpendicular(img, max_val, a, b, c)

    if random.choice([True, False]):
        zero_perc = random.random() * 0.02
        file_name += f"_z{zero_perc}"
        img = rand_zero(img, zero_perc, channel_uniform=True)

    if random.choice([True, False]):
        compression_level = random.randint(1, 20)
        file_name += f"_l{compression_level}"
        img = compress(img, compression_level)

    file_name += image_path.suffix
    doc_dir = image_path.parent.name
    output_dir = output_dir / doc_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    annotation_file = list(image_path.parent.glob(f"*page{image_path.stem}.json"))[0]
    copy(annotation_file, output_dir / annotation_file.name)
    copy(image_path, output_dir / image_path.name)
    return cv2.imwrite(str(output_dir / file_name), img)


def change_brightness(img: np.ndarray, value: int) -> np.ndarray:
    img = deepcopy(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def change_brightness_perpendicular(
    img: np.ndarray, value: int, a: float, b: float, c: float
) -> np.ndarray:
    img = deepcopy(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Compute distance of each pixel to the line given by ax+by+c=0
    yy, xx = np.meshgrid(range(v.shape[0]), range(v.shape[1]), indexing="ij")
    line_distances = a * xx + b * yy + c
    line_distances /= (a ** 2 + b ** 2) ** 0.5

    # Compute pixel differences according to the distance to the line
    diff = line_distances / line_distances.max() * value

    # Add differences to original array
    v = v.astype(int) + diff.astype(int)
    v[v < 0] = 0
    v[v > 255] = 255
    v = v.astype(np.uint8)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def rand_zero(
    img: np.ndarray, perc: float = 0.01, channel_uniform: bool = False
) -> np.ndarray:
    img = deepcopy(img)
    if channel_uniform:
        rand_mask = np.random.random(img.shape[:2]) < perc
    else:
        rand_mask = np.random.random(img.shape) < perc
    img[rand_mask] = 0
    return img


def compress(img: np.ndarray, quality: int) -> np.ndarray:
    _, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(encoded, int(cv2.IMREAD_UNCHANGED))


if __name__ == "__main__":
    main()
