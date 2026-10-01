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
    # Se è il modello di testo con top_k=None, i risultati sono dentro una lista [0]
    items = results[0] if is_text_model else results

    for item in items:
        std_lbl = change_label(item["label"])
        res_dict[std_lbl] = item["score"]
    return res_dict

def valuta_incongruenza(dict_visivo, dict_testo, soglia_differenza=0.7):
    """Rileva incongruenza e/o masking."""

    # 1. Selezione dell'emozione preponderante da ciascun modello
    top_visivo = max(dict_visivo, key=dict_visivo.get)
    score_visivo = dict_visivo[top_visivo]

    top_testo = max(dict_testo, key=dict_testo.get)
    score_testo = dict_testo[top_testo]

    print(f"\n[Analisi Modelli]")
    print(f" > Volto: {top_visivo} ({score_visivo*100:.2f}%)")
    print(f" > Testo: {top_testo} ({score_testo*100:.2f}%)")

    # 2. Se le emozioni principali sono DIVERSE -> Incongruenza strutturale
    if top_visivo != top_testo:
        return "INCONGRUENTE", top_visivo, top_testo, 1

    # 3. Se sono UGUALI -> Controlliamo la differenza di intensità
    diff = abs(score_visivo - score_testo)
    if diff > soglia_differenza:
        return "INCONGRUENTE", top_visivo, top_testo, diff
    else:
        return "CONGRUENTE", top_visivo, score_visivo, diff

def main():
    print("=" * 60)
    print("CONFRONTO MODELLO VISIVO vs MODELLO TESTUALE")
    print("=" * 60)

    # 1. Inizializzazione Pipeline
    vision_classifier = pipeline(
        "image-classification", model="trpakov/vit-face-expression"
    )
    text_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
    )

    # 2. Test Modello Visivo
    image_path = "volto_test.jpg"
    print(f"\n--- Analisi Immagine ({image_path}) ---")
    if os.path.exists(image_path):
        image = Image.open(image_path)
        vision_results = vision_classifier(image)

        for res in vision_results[:3]:
            std_label = change_label(res["label"])
            print(f" {std_label.capitalize()} (Orig: {res['label']}) ->"
                f" {res['score']*100:.2f}%"
            )
    else:
        print("Immagine di test non trovata.")

    # 3. Test Modello Testuale
    frase_test = "Il vostro Johnny è qui!"
    print(f"\n--- Analisi Testo ('{frase_test}') ---")
    text_results = text_classifier(frase_test)

    for res in text_results[0][:3]:
        std_label = change_label(res["label"])
        print(f" {std_label.capitalize()} (Orig: {res['label']}) ->"
              f" {res['score']*100:.2f}%"
        )

    # Pulizia e standardizzazione nei dizionari
    dict_visivo = format_results(vision_results, is_text_model=False)
    dict_testo = format_results(text_results, is_text_model=True)

    # --- VALUTAZIONE INCONGRUENZA ---
    stato, emo1, emo2, diff= valuta_incongruenza(dict_visivo, dict_testo)

    if stato == "INCONGRUENTE":
        if diff == 1:
            print(
                f"\n⚠️ Rilevata INCONGRUENZA STRUTTURALE tra volto e testo"
            )
        else:
            print(
                f"\n⚠️ Rilevata INCONGRUENZA tra volto e testo (differenza: {diff*100:.2f}%)"
            )
        print(
            "-> Fase di esplorazione..."
        )

    else:
        print(
            f"\n✅ Modelli CONGRUENTI (differenza: {diff*100:.2f}%). \n->Late Fusion..."
        )

if __name__ == "__main__":
  main()