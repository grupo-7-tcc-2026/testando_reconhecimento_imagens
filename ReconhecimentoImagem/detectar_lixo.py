import json
import math
from pathlib import Path
from statistics import fmean

import cv2
from ultralytics import YOLO


PASTA = Path(__file__).resolve().parent
CAMINHO_MODELO = PASTA / "best.pt"
CAMINHO_IMAGEM = PASTA / "imagemTeste02.jpg"
LIMIAR_CONFIANCA = 0.30


def calcular_area_uniao_caixas(caixas: list[tuple[float, float, float, float]]) -> float:
    """Calcula a area total coberta pelas caixas, sem contar sobreposicoes duas vezes."""
    coordenadas_x = sorted({x for caixa in caixas for x in (caixa[0], caixa[2])})
    area = 0.0

    for inicio_x, fim_x in zip(coordenadas_x, coordenadas_x[1:]):
        if fim_x <= inicio_x:
            continue

        intervalos_y = sorted(
            (y1, y2)
            for x1, y1, x2, y2 in caixas
            if x1 < fim_x and x2 > inicio_x
        )
        altura_uniao = 0.0
        if intervalos_y:
            inicio_y, fim_y = intervalos_y[0]
            for proximo_inicio, proximo_fim in intervalos_y[1:]:
                if proximo_inicio > fim_y:
                    altura_uniao += fim_y - inicio_y
                    inicio_y, fim_y = proximo_inicio, proximo_fim
                else:
                    fim_y = max(fim_y, proximo_fim)
            altura_uniao += fim_y - inicio_y

        area += (fim_x - inicio_x) * altura_uniao

    return area


def main() -> None:
    if not CAMINHO_MODELO.is_file():
        raise FileNotFoundError(f"Modelo nao encontrado: {CAMINHO_MODELO}")
    if not CAMINHO_IMAGEM.is_file():
        raise FileNotFoundError(f"Imagem nao encontrada: {CAMINHO_IMAGEM}")

    modelo = YOLO(str(CAMINHO_MODELO))
    resultado = modelo.predict(
        source=str(CAMINHO_IMAGEM),
        conf=LIMIAR_CONFIANCA,
        verbose=False,
    )[0]

    altura, largura = resultado.orig_shape
    deteccoes = []
    caixas = []

    if resultado.boxes is not None:
        for caixa in resultado.boxes:
            x1, y1, x2, y2 = caixa.xyxy[0].cpu().tolist()
            x1 = max(0.0, min(float(largura), x1))
            y1 = max(0.0, min(float(altura), y1))
            x2 = max(0.0, min(float(largura), x2))
            y2 = max(0.0, min(float(altura), y2))
            coordenadas = (x1, y1, x2, y2)
            caixas.append(coordenadas)

            classe_id = int(caixa.cls[0].item())
            nomes = modelo.names
            nome_classe = nomes.get(classe_id, str(classe_id)) if isinstance(nomes, dict) else nomes[classe_id]
            deteccoes.append(
                {
                    "classe": nome_classe,
                    "confianca": round(float(caixa.conf[0].item()), 6),
                    "coordenadas": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                }
            )

    quantidade = len(deteccoes)
    area_total = float(largura * altura)
    area_uniao = calcular_area_uniao_caixas(caixas)
    confianca_media = fmean(item["confianca"] for item in deteccoes) * 100 if deteccoes else 0.0
    percentual_area = area_uniao / area_total * 100 if area_total else 0.0
    dados = {
        "imagem": CAMINHO_IMAGEM.name,
        "largura_imagem": largura,
        "altura_imagem": altura,
        "quantidade_lixo": quantidade,
        "confianca_media": math.trunc(confianca_media * 100) / 100,
        "percentual_area_ocupada_estimado": math.trunc(percentual_area * 100) / 100,
        "metrica_area": "Uniao das areas das caixas delimitadoras em relacao a area total da imagem.",
        "deteccoes": deteccoes,
    }

    caminho_json = CAMINHO_IMAGEM.with_suffix(".json")
    caminho_saida = CAMINHO_IMAGEM.with_name(
        f"{CAMINHO_IMAGEM.stem}_detectada{CAMINHO_IMAGEM.suffix}"
    )
    imagem_anotada = resultado.plot()
    if not cv2.imwrite(str(caminho_saida), imagem_anotada):
        raise OSError(f"Nao foi possivel salvar a imagem anotada: {caminho_saida}")
    caminho_json.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Imagem anotada salva em: {caminho_saida}")
    print(f"Dados salvos em: {caminho_json}")
    print(f"Lixos identificados: {quantidade}")
    print(f"Confianca media: {dados['confianca_media']:.2f}%")
    print(f"Area estimada ocupada: {dados['percentual_area_ocupada_estimado']:.2f}%")


if __name__ == "__main__":
    main()
