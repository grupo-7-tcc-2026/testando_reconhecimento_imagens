"""Prepare the preview bounding boxes as a YOLO dataset and train a detector."""

import argparse
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SEED = 42
VALIDATION_RATIO = 0.2


def load_samples(labels_path: Path, images_dir: Path) -> list[dict[str, Any]]:
    if not labels_path.is_file():
        raise FileNotFoundError(f"Arquivo de labels não encontrado: {labels_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {images_dir}")

    data = json.loads(labels_path.read_text(encoding="utf-8"))
    image_records = data.get("images")
    if not isinstance(image_records, list) or not image_records:
        raise ValueError(f"O JSON não contém uma lista de imagens: {labels_path}")

    samples = []
    seen_paths = set()
    resolved_images_dir = images_dir.resolve()

    for record in image_records:
        if not isinstance(record, dict):
            raise ValueError("Cada item em 'images' precisa ser um objeto JSON.")

        image_name = record.get("image")
        if not isinstance(image_name, str) or not image_name.strip():
            raise ValueError("Cada imagem precisa ter um caminho válido no campo 'image'.")

        relative_path = Path(*PurePosixPath(image_name.replace("\\", "/")).parts)
        image_path = (images_dir / relative_path).resolve()
        if not image_path.is_relative_to(resolved_images_dir):
            raise ValueError(f"Caminho de imagem fora da pasta de origem: {image_name}")
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Formato de imagem não suportado: {image_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Imagem do JSON não encontrada: {image_path}")

        relative_key = relative_path.as_posix().casefold()
        if relative_key in seen_paths:
            raise ValueError(f"Imagem duplicada no JSON: {image_name}")
        seen_paths.add(relative_key)

        width = record.get("width")
        height = record.get("height")
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or not isinstance(height, int)
            or isinstance(height, bool)
            or width <= 0
            or height <= 0
        ):
            raise ValueError(f"Dimensões inválidas no JSON para {image_name}.")

        boxes = record.get("bounding_boxes")
        if not isinstance(boxes, list):
            raise ValueError(f"'bounding_boxes' precisa ser uma lista em {image_name}.")

        for box in boxes:
            if not isinstance(box, dict):
                raise ValueError(f"Caixa inválida em {image_name}: {box!r}")
            try:
                x, y, box_width, box_height = (
                    float(box[key]) for key in ("x", "y", "width", "height")
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Coordenadas inválidas em {image_name}: {box!r}") from error

            if not all(
                math.isfinite(value) for value in (x, y, box_width, box_height)
            ):
                raise ValueError(f"Coordenadas não finitas em {image_name}: {box!r}")
            if (
                x < 0
                or y < 0
                or box_width <= 0
                or box_height <= 0
                or x + box_width > width
                or y + box_height > height
            ):
                raise ValueError(f"Caixa fora dos limites da imagem {image_name}: {box!r}")

        samples.append(
            {
                "source": image_path,
                "relative_path": relative_path,
                "width": width,
                "height": height,
                "boxes": boxes,
            }
        )

    return samples


def split_samples(
    samples: list[dict[str, Any]], seed: int = SEED
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(samples) < 2:
        raise ValueError("São necessárias pelo menos duas imagens para treino e validação.")

    randomizer = random.Random(seed)
    train_samples = []
    validation_samples = []

    for is_positive in (True, False):
        group = [sample for sample in samples if bool(sample["boxes"]) == is_positive]
        randomizer.shuffle(group)
        if len(group) < 2:
            train_samples.extend(group)
            continue

        validation_count = max(1, round(len(group) * VALIDATION_RATIO))
        validation_count = min(validation_count, len(group) - 1)
        validation_samples.extend(group[:validation_count])
        train_samples.extend(group[validation_count:])

    if not train_samples or not validation_samples:
        raise ValueError("A divisão do dataset precisa ter imagens em treino e validação.")

    randomizer.shuffle(train_samples)
    randomizer.shuffle(validation_samples)
    return train_samples, validation_samples


def write_yolo_dataset(
    samples: list[dict[str, Any]],
    dataset_dir: Path,
    class_name: str,
    seed: int = SEED,
) -> tuple[Path, int, int]:
    if not class_name.strip():
        raise ValueError("O nome da classe não pode ficar vazio.")

    train_samples, validation_samples = split_samples(samples, seed)
    total_boxes = 0

    for split_name, split_samples_list in (
        ("train", train_samples),
        ("val", validation_samples),
    ):
        for sample in split_samples_list:
            relative_path = sample["relative_path"]
            image_destination = dataset_dir / "images" / split_name / relative_path
            label_destination = (
                dataset_dir
                / "labels"
                / split_name
                / relative_path.with_suffix(".txt")
            )
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.parent.mkdir(parents=True, exist_ok=True)

            image_destination.write_bytes(sample["source"].read_bytes())
            width = sample["width"]
            height = sample["height"]
            label_lines = []

            for box in sample["boxes"]:
                x = float(box["x"])
                y = float(box["y"])
                box_width = float(box["width"])
                box_height = float(box["height"])
                center_x = (x + box_width / 2) / width
                center_y = (y + box_height / 2) / height
                normalized_width = box_width / width
                normalized_height = box_height / height
                label_lines.append(
                    f"0 {center_x:.6f} {center_y:.6f} "
                    f"{normalized_width:.6f} {normalized_height:.6f}"
                )

            label_destination.write_text(
                "\n".join(label_lines), encoding="utf-8"
            )
            total_boxes += len(label_lines)

    yaml_path = dataset_dir / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        f"path: {json.dumps(dataset_dir.resolve().as_posix(), ensure_ascii=False)}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {json.dumps(class_name, ensure_ascii=False)}\n",
        encoding="utf-8",
    )

    print(f"Dataset YOLO criado em: {dataset_dir.resolve()}")
    print(
        f"Imagens de treino: {len(train_samples)} | "
        f"imagens de validação: {len(validation_samples)}"
    )
    print(f"Imagens sem caixas: {sum(not sample['boxes'] for sample in samples)}")
    print(f"Caixas no dataset: {total_boxes}")
    print(f"Configuração: {yaml_path.resolve()}")
    return yaml_path, len(train_samples), len(validation_samples)


def train_model(
    dataset_yaml: Path,
    model_weights: str,
    epochs: int,
    image_size: int,
    batch_size: int,
    workers: int | None,
) -> None:
    import torch
    from ultralytics import YOLO

    def print_epoch_summary(trainer: Any) -> None:
        epoch_number = trainer.epoch + 1
        print(f"\n=== Época {epoch_number}/{trainer.epochs} concluída ===")
        metrics = getattr(trainer, "metrics", None)
        if metrics is None:
            print("  métricas da validação ainda não disponíveis")
            return

        box_metrics = getattr(metrics, "box", None)
        if box_metrics is not None:
            precision = getattr(box_metrics, "p", None)
            recall = getattr(box_metrics, "r", None)
            map50 = getattr(box_metrics, "map50", None)
            map50_95 = getattr(box_metrics, "map", None)
            print(
                "  precision={:.4f} | recall={:.4f} | mAP50={:.4f} | mAP50-95={:.4f}".format(
                    float(precision) if precision is not None else 0.0,
                    float(recall) if recall is not None else 0.0,
                    float(map50) if map50 is not None else 0.0,
                    float(map50_95) if map50_95 is not None else 0.0,
                )
            )
        else:
            print("  métricas da validação indisponíveis neste momento")

    device = 0 if torch.cuda.is_available() else "cpu"
    worker_count = workers if workers is not None else (0 if os.name == "nt" else 2)
    model_name = Path(model_weights).stem
    run_name = f"{model_name}_com_lixo"

    model = YOLO(model_weights)
    model.add_callback("on_train_epoch_end", print_epoch_summary)
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        workers=worker_count,
        name=run_name,
        project=str(BASE_DIR / "runs" / "detect"),
        save=True,
        plots=True,
        device=device,
        seed=SEED,
        verbose=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Pesos finais não encontrados: {best_weights}")

    final_model_path = BASE_DIR / f"{model_name}.pt"
    final_model_path.write_bytes(best_weights.read_bytes())

    last_model_path = BASE_DIR / f"{model_name}_last.pt"
    last_weights = Path(results.save_dir) / "weights" / "last.pt"
    if last_weights.is_file():
        last_model_path.write_bytes(last_weights.read_bytes())

    print("Treinamento concluído.")
    print(f"Dispositivo: {device} | workers: {worker_count}")
    print(f"Melhores pesos: {best_weights}")
    print(f"Cópia final na raiz: {final_model_path}")
    print(f"Nome da execução: {run_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepara as caixas da pasta com_lixo e treina um detector YOLO."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=BASE_DIR / "com_lixo",
        help="Pasta das imagens (padrão: ./com_lixo).",
    )
    parser.add_argument(
        "--labels-json",
        type=Path,
        default=BASE_DIR / "com_lixo" / "labels" / "labels.json",
        help="JSON de caixas (padrão: ./com_lixo/labels/labels.json).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=BASE_DIR / "dataset_yolo",
        help="Pasta de saída do dataset YOLO (padrão: ./dataset_yolo).",
    )
    parser.add_argument(
        "--class-name",
        default="lixo",
        help="Nome da classe única das caixas (padrão: lixo).",
    )
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="Pesos iniciais do modelo YOLO (padrão: yolov8n.pt).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Processos de carregamento (padrão: 0 no Windows e 2 nos demais).",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Cria o dataset YOLO sem iniciar o treinamento.",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs precisa ser maior que zero.")
    if args.imgsz < 1:
        parser.error("--imgsz precisa ser maior que zero.")
    if args.batch < 1:
        parser.error("--batch precisa ser maior que zero.")
    if args.workers is not None and args.workers < 0:
        parser.error("--workers não pode ser negativo.")

    return args


def main() -> None:
    started_at = datetime.now().astimezone()
    print(
        f"Horário de início: {started_at.isoformat(sep=' ', timespec='seconds')}",
        flush=True,
    )
    try:
        args = parse_args()
        samples = load_samples(args.labels_json.resolve(), args.images_dir.resolve())
        dataset_yaml, _, _ = write_yolo_dataset(
            samples,
            args.dataset_dir.resolve(),
            args.class_name,
        )

        if not args.prepare_only:
            train_model(
                dataset_yaml,
                args.weights,
                args.epochs,
                args.imgsz,
                args.batch,
                args.workers,
            )
    finally:
        finished_at = datetime.now().astimezone()
        print(
            f"Horário final: {finished_at.isoformat(sep=' ', timespec='seconds')}",
            flush=True,
        )


if __name__ == "__main__":
    main()