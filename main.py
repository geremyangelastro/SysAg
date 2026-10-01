import os
from PIL import Image
from transformers import pipeline


def main():
  print("Caricamento del modello in corso...")
  # Inizializza la pipeline di classificazione immagini
  classifier = pipeline("image-classification", model="trpakov/vit-face-expression")

  # Percorso dell'immagine di test (puoi metterla nella cartella principale)
  image_path = "volto_test.jpg"

  if not os.path.exists(image_path):
    print(
        f"Attenzione: Non ho trovato l'immagine '{image_path}'. Inseriscine"
        " una per continuare."
    )
    return

  # Carica l'immagine ed esegui la predizione
  image = Image.open(image_path)
  results = classifier(image)

  print("\n--- RISULTATI ANALISI ESPRESSIONE ---")
  for res in results:
    # Stampa l'emozione e la percentuale di confidenza
    print(f"Emoticon: {res['label']} -> Confidenza: {res['score']*100:.2f}%")


if __name__ == "__main__":
  main()