"""Extract red bounding boxes from annotated images and save them as JSON."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _find_line_segments(
    red_mask: np.ndarray, *, horizontal: bool, min_box_size: int
) -> list[tuple[int, int, int]]:
    """Return line centers and their start/end coordinates."""
    kernel_size = max(10, min_box_size // 2)
    if horizontal:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))

    line_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        line_mask, connectivity=8
    )
    segments = []

    for index in range(1, component_count):
        x, y, width, height, _ = stats[index]
        span = width if horizontal else height
        if span < min_box_size:
            continue

        if horizontal:
            segments.append((y + height // 2, x, x + width - 1))
        else:
            segments.append((x + width // 2, y, y + height - 1))

    return segments


def extract_red_boxes(
    image: np.ndarray, min_box_size: int = 15
) -> list[dict[str, int]]:
    """Detect axis-aligned red rectangles, including rectangles that overlap."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255])),
    )

    horizontal_lines = _find_line_segments(
        red_mask, horizontal=True, min_box_size=min_box_size
    )
    vertical_lines = _find_line_segments(
        red_mask, horizontal=False, min_box_size=min_box_size
    )
    tolerance = 3
    boxes = set()

    for (y1, hx1, hx2), (y2, hx3, hx4) in combinations(horizontal_lines, 2):
        top, bottom = sorted((y1, y2))
        if bottom - top < min_box_size:
            continue

        for (x1, vy1, vy2), (x2, vy3, vy4) in combinations(vertical_lines, 2):
            left, right = sorted((x1, x2))
            if right - left < min_box_size:
                continue

            top_line = (hx1, hx2) if y1 == top else (hx3, hx4)
            bottom_line = (hx3, hx4) if y1 == top else (hx1, hx2)
            left_line = (vy1, vy2) if x1 == left else (vy3, vy4)
            right_line = (vy3, vy4) if x1 == left else (vy1, vy2)

            corners_match = (
                top_line[0] <= left + tolerance
                and top_line[1] >= right - tolerance
                and bottom_line[0] <= left + tolerance
                and bottom_line[1] >= right - tolerance
                and left_line[0] <= top + tolerance
                and left_line[1] >= bottom - tolerance
                and right_line[0] <= top + tolerance
                and right_line[1] >= bottom - tolerance
            )
            if corners_match:
                boxes.add((left, top, right - left + 1, bottom - top + 1))

    return [
        {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        }
        for x, y, width, height in sorted(boxes, key=lambda box: (box[1], box[0]))
    ]


def extract_labels(input_dir: Path, output_path: Path, min_box_size: int) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {input_dir}")

    labels_dir = output_path.parent
    exclude_labels_dir = (
        labels_dir != input_dir and labels_dir.is_relative_to(input_dir)
    )
    image_paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not (
            exclude_labels_dir and path.is_relative_to(labels_dir)
        )
    )
    if not image_paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {input_dir}")

    image_records = []
    unreadable_images = []
    total_boxes = 0

    for image_path in image_paths:
        image_bytes = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            unreadable_images.append(str(image_path))
            continue

        height, width = image.shape[:2]
        boxes = extract_red_boxes(image, min_box_size)
        total_boxes += len(boxes)
        image_records.append(
            {
                "image": str(image_path.relative_to(input_dir)),
                "width": width,
                "height": height,
                "bounding_boxes": boxes,
            }
        )

    if unreadable_images:
        unreadable_list = "\n".join(unreadable_images)
        raise RuntimeError(f"Não foi possível abrir estas imagens:\n{unreadable_list}")

    result = {
        "total_images": len(image_records),
        "total_boxes": total_boxes,
        "images": image_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Imagens processadas: {len(image_records)}")
    print(f"Caixas detectadas: {total_boxes}")
    print(f"JSON salvo em: {output_path}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Extrai caixas delimitadoras vermelhas e salva as labels em JSON."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir / "preview",
        help="Pasta com imagens anotadas (padrão: ./preview).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "preview" / "labels" / "labels.json",
        help="Arquivo JSON de saída (padrão: ./preview/labels/labels.json).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=15,
        help="Largura e altura mínimas da caixa em pixels (padrão: 15).",
    )
    args = parser.parse_args()

    if args.min_size < 1:
        parser.error("--min-size precisa ser maior que zero.")

    extract_labels(args.input_dir.resolve(), args.output.resolve(), args.min_size)


if __name__ == "__main__":
    main()