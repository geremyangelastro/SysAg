import os
from PIL import Image
from transformers import pipeline


def change_label(label):
  """Conversione delle etichette."""
  mapping = {
      # Rabbia
      "angry": "rabbia",
      "anger": "rabbia",
      # Disgusto
      "disgust": "disgusto",
      # Paura
      "fear": "paura",
      # Felicità/Gioia
      "happy": "gioia",
      "joy": "gioia",
      # Tristezza
      "sad": "tristezza",
      "sadness": "tristezza",
      # Sorpresa
      "surprise": "sorpresa",
      # Neutra
      "neutral": "neutro",
  }
  return mapping.get(label.lower(), label.lower())


def format_results(results, is_text_model=False):
  """Converte l'output della pipeline in un dizionario pulito {emozione: score} con label standard."""
  res_dict = {}
  items = results[0] if is_text_model else results

  for item in items:
    std_lbl = change_label(item["label"])
    res_dict[std_lbl] = item["score"]
  return res_dict


def valuta_incongruenza(dict_visivo, dict_testo, soglia_differenza=0.7):
  """Rileva incongruenza e/o masking."""
  top_visivo = max(dict_visivo, key=dict_visivo.get)
  score_visivo = dict_visivo[top_visivo]

  top_testo = max(dict_testo, key=dict_testo.get)
  score_testo = dict_testo[top_testo]

  print(f"\n[Analisi Modelli]")
  print(f" > Volto: {top_visivo} ({score_visivo*100:.2f}%)")
  print(f" > Testo: {top_testo} ({score_testo*100:.2f}%)")

  if top_visivo != top_testo:
    return "INCONGRUENTE", top_visivo, top_testo, 1

  diff = abs(score_visivo - score_testo)
  if diff > soglia_differenza:
    return "INCONGRUENTE", top_visivo, top_testo, diff
  else:
    return "CONGRUENTE", top_visivo, score_visivo, diff


def scegli_file(cartella, estensioni_valide):
  """Scansiona una cartella, mostra i file disponibili e permette all'utente di sceglierne uno."""
  if not os.path.exists(cartella):
    print(
        f"La cartella '{cartella}' non esiste. La creo automaticamente per te."
    )
    os.makedirs(cartella)
    return None

  # Filtra i file in base alle estensioni ammesse
  files = [
      f
      for f in os.listdir(cartella)
      if os.path.isfile(os.path.join(cartella, f))
      and f.lower().endswith(tuple(estensioni_valide))
  ]

  if not files:
    print(f"Nessun file compatibile trovato nella cartella '{cartella}'.")
    return None

  print(f"\n--- File disponibili in '{cartella}' ---")
  for i, nome_file in enumerate(files, 1):
    print(f" [{i}] {nome_file}")

  while True:
    try:
      scelta = int(input(f"Seleziona il numero del file (1-{len(files)}): "))
      if 1 <= scelta <= len(files):
        percorso_scelto = os.path.join(cartella, files[scelta - 1])
        return percorso_scelto
      else:
        print("Numero non valido. Riprova.")
    except ValueError:
      print("Inserisci un numero intero valido.")


def main():
  print("=" * 60)
  print("CONFRONTO MODELLO VISIVO vs MODELLO TESTUALE")
  print("=" * 60)

  # 1. Inizializzazione Pipeline
  print("Caricamento modelli in corso...")
  vision_classifier = pipeline(
      "image-classification", model="trpakov/vit-face-expression"
  )
  text_classifier = pipeline(
      "text-classification",
      model="j-hartmann/emotion-english-distilroberta-base",
      top_k=None,
  )

  # 2. Selezione dinamica Immagine
  print("\n[Selezione Immagine Volto]")
  image_path = scegli_file("test_images", [".jpg", ".jpeg", ".png"])
  if not image_path:
    print("Impossibile procedere senza un'immagine.")
    return

  print(f"Hai selezionato l'immagine: {image_path}")
  image = Image.open(image_path)
  vision_results = vision_classifier(image)

  print("\nRisultati:")
  for res in vision_results[:3]:
    std_label = change_label(res["label"])
    print(f" - {std_label.capitalize()} -> {res['score']*100:.2f}%")

  # 3. Selezione dinamica Testo
  print("\n[Selezione File di Testo]")
  text_path = scegli_file("test_texts", [".txt"])
  if not text_path:
    print("Impossibile procedere senza un file di testo.")
    return

  print(f"Hai selezionato il testo: {text_path}")
  with open(text_path, "r", encoding="utf-8") as file:
    frase_test = file.read().strip()

  print(f"Testo caricato: '{frase_test}'")
  text_results = text_classifier(frase_test)

  print("\nRisultati:")
  for res in text_results[0][:3]:
    std_label = change_label(res["label"])
    print(f" - {std_label.capitalize()} -> {res['score']*100:.2f}%")

  # Pulizia e standardizzazione nei dizionari
  dict_visivo = format_results(vision_results, is_text_model=False)
  dict_testo = format_results(text_results, is_text_model=True)

  # --- VALUTAZIONE INCONGRUENZA ---
  stato, emo1, emo2, diff = valuta_incongruenza(dict_visivo, dict_testo)

  if stato == "INCONGRUENTE":
    if diff == 1:
      print("\n⚠️ Rilevata INCONGRUENZA STRUTTURALE tra volto e testo")
    else:
      print(
          f"\n⚠️ Rilevata INCONGRUENZA tra volto e testo (differenza:"
          f" {diff*100:.2f}%)"
      )
    print("-> Fase di esplorazione...")
  else:
    print(
        f"\n✅ Modelli CONGRUENTI (differenza: {diff*100:.2f}%).\n-> Late"
        " Fusion..."
    )


if __name__ == "__main__":
  main()