from pathlib import Path

BASE_DIR = Path(r"C:\Users\alexa\OneDrive\Área de Trabalho\TCC\Teste Base de Imagens TCC 03")
COM_LIXO_DIR = BASE_DIR / "com_lixo"
SEM_LIXO_DIR = BASE_DIR / "sem_lixo"


def listar_arquivos(diretorio: Path) -> set[str]:
    if not diretorio.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {diretorio}")

    arquivos = set()
    for arquivo in diretorio.rglob("*"):
        if arquivo.is_file():
            rel = arquivo.relative_to(diretorio).as_posix().casefold()
            arquivos.add(rel)
    return arquivos


def contar_arquivos(diretorio: Path) -> int:
    return sum(1 for _ in diretorio.rglob("*") if _.is_file())


def limpar_pasta(alvo_dir: Path, nome_pasta: str, arquivos_permitidos: set[str]) -> int:
    removidos = 0

    for arquivo in sorted(alvo_dir.rglob("*")):
        if not arquivo.is_file():
            continue

        rel = arquivo.relative_to(alvo_dir).as_posix().casefold()
        if rel not in arquivos_permitidos:
            arquivo.unlink()
            removidos += 1
            print(f"Removido em {nome_pasta}: {arquivo}")

    return removidos


def limpar_pastas() -> None:
    print(f"Base de comparação: {COM_LIXO_DIR}")
    print(f"Verificando: {SEM_LIXO_DIR}")

    arquivos_permitidos = listar_arquivos(COM_LIXO_DIR)

    removidos_com_lixo = limpar_pasta(COM_LIXO_DIR, "com_lixo", arquivos_permitidos)
    removidos_sem_lixo = limpar_pasta(SEM_LIXO_DIR, "sem_lixo", arquivos_permitidos)

    com_lixo_total = contar_arquivos(COM_LIXO_DIR)
    sem_lixo_total = contar_arquivos(SEM_LIXO_DIR)

    print("\n=== Resumo final ===")
    print(f"com_lixo: {com_lixo_total} arquivos")
    print(f"sem_lixo: {sem_lixo_total} arquivos")
    print(f"Arquivos removidos da pasta com_lixo: {removidos_com_lixo}")
    print(f"Arquivos removidos da pasta sem_lixo: {removidos_sem_lixo}")


if __name__ == "__main__":
    limpar_pastas()