import argparse
import json

from src.inferencia import predecir, guardar_imagen_etiquetada


def main():
    parser = argparse.ArgumentParser(description="Estima la probabilidad de melanoma de una imagen.")
    parser.add_argument("imagen", help="Ruta de la imagen dermatoscópica")
    parser.add_argument("--guardar", help="Ruta donde guardar la imagen etiquetada")
    args = parser.parse_args()

    try:
        resultado = predecir(args.imagen)
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        raise SystemExit(1)

    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    if args.guardar:
        guardar_imagen_etiquetada(args.imagen, resultado, args.guardar)


if __name__ == "__main__":
    main()