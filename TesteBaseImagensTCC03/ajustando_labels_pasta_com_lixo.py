import json
from pathlib import Path

BASE_DIR = Path(r"C:\Users\alexa\OneDrive\Área de Trabalho\TCC\Teste Base de Imagens TCC 03")
COM_LIXO_DIR = BASE_DIR / "com_lixo"
LABELS_JSON = COM_LIXO_DIR / "labels" / "labels.json"


def listar_imagens_no_diretorio(diretorio: Path) -> set[str]:
    if not diretorio.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {diretorio}")

    imagens = set()
    for arquivo in diretorio.rglob("*"):
        if arquivo.is_file() and arquivo.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
            rel = arquivo.relative_to(diretorio).as_posix().casefold()
            imagens.add(rel)
    return imagens


def normalizar_path(label_path: str) -> str:
    if label_path is None:
        return ""
    return Path(label_path.replace("\\", "/")).as_posix().casefold()


def ajustar_json_labels() -> None:
    if not LABELS_JSON.exists():
        raise FileNotFoundError(f"Arquivo de labels não encontrado: {LABELS_JSON}")

    dados = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    imagens_lista = dados.get("images")
    if not isinstance(imagens_lista, list):
        raise ValueError(f"O JSON não contém uma lista 'images': {LABELS_JSON}")

    imagens_validas = listar_imagens_no_diretorio(COM_LIXO_DIR)

    nova_lista = []
    removidas = 0

    for registro in imagens_lista:
        if not isinstance(registro, dict):
            continue

        image_value = registro.get("image")
        if not isinstance(image_value, str):
            continue

        chave = normalizar_path(image_value)
        if chave in imagens_validas:
            nova_lista.append(registro)
        else:
            removidas += 1
            print(f"Removendo label sem imagem correspondente: {image_value}")

    dados["images"] = nova_lista
    dados["total_images"] = len(nova_lista)
    dados["total_boxes"] = sum(
        len(item.get("bounding_boxes", []))
        if isinstance(item.get("bounding_boxes"), list)
        else 0
        for item in nova_lista
    )

    LABELS_JSON.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Resumo final ===")
    print(f"Arquivo de labels: {LABELS_JSON}")
    print(f"Imagens válidas na pasta com_lixo: {len(imagens_validas)}")
    print(f"Labels restantes no JSON: {len(nova_lista)}")
    print(f"Labels removidas: {removidas}")
    print(f"Total de boxes finais: {dados['total_boxes']}")


if __name__ == "__main__":
    ajustar_json_labels()
