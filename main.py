import threading
import time
import re
from collections import Counter
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, DetectorFactory
import webview
import gradio as gr

DetectorFactory.seed = 0

print("Caricamento dei modelli in corso...")
vision_classifier = pipeline("image-classification", model="trpakov/vit-face-expression")
text_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

print("Caricamento modello di traduzione...")
tokenizer_it_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-en")
model_it_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-it-en")
print("Modelli caricati con successo!")

def traduci_in_inglese(testo):
    inputs = tokenizer_it_en(testo, return_tensors="pt", padding=True)
    outputs = model_it_en.generate(**inputs)
    return tokenizer_it_en.decode(outputs[0], skip_special_tokens=True)

def change_label(label):
    mapping = {
        "angry": "rabbia", "anger": "rabbia",
        "disgust": "disgusto", "fear": "paura",
        "happy": "gioia", "joy": "gioia",
        "sad": "tristezza", "sadness": "tristezza",
        "surprise": "sorpresa", "neutral": "neutro",
    }
    return mapping.get(label.lower(), label.lower())

def format_results(results, is_text_model=False):
    res_dict = {}
    items = results[0] if is_text_model else results
    for item in items:
        std_lbl = change_label(item["label"])
        res_dict[std_lbl] = item["score"]
    return res_dict

def update_buffer(new_frame, current_buffer):
    if current_buffer is None:
        current_buffer = []
    if new_frame is not None:
        try:
            img = (new_frame.convert("RGB") if isinstance(new_frame, Image.Image)
                   else Image.fromarray(new_frame).convert("RGB"))
            current_buffer.append(img)
            current_buffer = current_buffer[-5:]
        except Exception:
            pass
    return current_buffer

def md_to_html(text):
    """Converte un testo markdown di base in HTML per la flip card."""
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = text.replace('\n', '<br>')
    return text

def genera_card_html(contenuto, titolo_front="📊 Analisi in attesa", testo_front="Avvia una chat per generare l'analisi.", colore_bordo="transparent", abilitato=False):
    """Genera il codice HTML leggendolo da un template esterno e applica il colore e lo stato del pulsante."""
    html_content = md_to_html(contenuto)
    
    try:
        with open("card_template.html", "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        return "Template HTML non trovato."
        
    if abilitato:
        pulsante_html = '<button class="flip-btn" onclick="this.closest(\'.flip-inner\').classList.toggle(\'is-flipped\')">🔄 Svela Risultati</button>'
    else:
        pulsante_html = '<button class="flip-btn" style="opacity: 0.5; cursor: not-allowed;" disabled>⏳ In attesa...</button>'
        
    return template.format(
        titolo_front=titolo_front, 
        testo_front=testo_front, 
        html_content=html_content,
        colore_bordo=colore_bordo,
        pulsante_front=pulsante_html
    )

def determina_colore_bordo(emozione_finale):
    """Mappa l'emozione finale nelle 4 Zone di Regolazione del modello a colori."""
    # Convertiamo l'emozione in minuscolo per sicurezza
    emo = emozione_finale.lower()
    
    # 1. ZONA BLU (Bassa energia / Tristezza / Stanchezza)
    if emo in ["tristezza", "sad", "sadness"]:
        return "#3498db" # Blu
        
    # 2. ZONA VERDE (Calma / Equilibrio / Neutro / Gioia moderata)
    elif emo in ["neutro", "neutral", "gioia", "happy", "joy"]:
        return "#2ecc71" # Verde
        
    # 3. ZONA GIALLA (Stress / Ansia / Paura / Sorpresa / Frustrazione)
    elif emo in ["paura", "fear", "sorpresa", "surprise", "disgusto", "disgust"]:
        return "#f1c40f" # Giallo
        
    # 4. ZONA ROSSA (Rabbia intensa / Crisi / Alta attivazione)
    elif emo in ["rabbia", "anger", "angry"]:
        return "#e74c3c" # Rosso
        
    # Colore di fallback predefinito se l'emozione non è mappata
    return "#95a5a6" # Grigio neutro

def risposta_chat(user_message, app_language, buffer_frames, history):
    if history is None:
        history = []

    if user_message is None or not str(user_message).strip():
        yield "", history, genera_card_html("Nessun testo da analizzare.", colore_bordo="transparent", abilitato=False)
        return

    user_message = str(user_message).strip()

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": "⏳ *Elaborazione in corso...*"})
    
    attesa_card = genera_card_html("⏳ *Calcolo in corso...*", "⚙️ Elaborazione...", "L'analisi multimodale sta lavorando sui dati.", colore_bordo="transparent", abilitato=False)
    yield "", history, attesa_card

    time.sleep(0.2)

    if not buffer_frames:
        history[-1]["content"] = "⚠️ Analisi fallita per mancanza di input visivo."
        yield "", history, genera_card_html("⚠️ Nessun fotogramma ricevuto dalla webcam.", "Errore", "Dati visivi mancanti.", colore_bordo="#e74c3c", abilitato=True)
        return

    try:
        avviso_lingua = ""
        try:
            detected_lang = detect(user_message)
        except Exception:
            detected_lang = "unk"

        mappa_lingue = {"it": "Italiano", "en": "Inglese", "es": "Spagnolo", "fr": "Francese", "de": "Tedesco", "pt": "Portoghese"}
        lingua_rilevata_nome = mappa_lingue.get(detected_lang, detected_lang.upper())

        if lingua_rilevata_nome != app_language and detected_lang != "unk":
            if lingua_rilevata_nome in ["Italiano", "Inglese"]:
                avviso_lingua = f"💡 Nota: Sembra che tu stia scrivendo in {lingua_rilevata_nome}! Ti suggeriamo di aprire le impostazioni ⚙️ e cambiare la lingua.\n\n"
            else:
                avviso_lingua = f"💡 Nota: Sembra che tu stia scrivendo in {lingua_rilevata_nome}, ma l'app è impostata su {app_language}.\n\n"
        
        if app_language == "Italiano":
            text_to_analyze = traduci_in_inglese(user_message)
        else:
            text_to_analyze = user_message

        # 1. Modello Testo
        raw_text = text_classifier(text_to_analyze)
        dict_testo = format_results(raw_text, is_text_model=True)
        top_testo = max(dict_testo, key=dict_testo.get)
        score_testo = dict_testo[top_testo]

        # 2. Frame Visivi
        risultati_visione = []
        for frame in buffer_frames:
            raw_vision = vision_classifier(frame)
            dict_visivo = format_results(raw_vision, is_text_model=False)
            top_emozione = max(dict_visivo, key=dict_visivo.get)
            score = dict_visivo[top_emozione]
            risultati_visione.append((top_emozione, score, dict_visivo))

        conteggio_emozioni = Counter([res[0] for res in risultati_visione])
        emozione_predominante = conteggio_emozioni.most_common(1)[0][0]

        miglior_risultato = None
        for res in risultati_visione:
            if res[0] == emozione_predominante:
                if miglior_risultato is None or res[1] > miglior_risultato[1]:
                    miglior_risultato = res

        top_visivo = miglior_risultato[0]
        score_visivo = miglior_risultato[1]
        dict_visivo = miglior_risultato[2]

        # 3. Costruzione output
        soglia_differenza = 0.7
        risposta_analisi = avviso_lingua
        risposta_analisi += f"🧠 **Analisi Multimodale:**\n"
        risposta_analisi += f"- **Volto (Predominante):** {top_visivo.capitalize()} ({score_visivo*100:.2f}%)\n"
        risposta_analisi += f"- **Testo (Analizzato):** \"{text_to_analyze}\"\n"
        risposta_analisi += f"- **Emozione Testo:** {top_testo.capitalize()} ({score_testo*100:.2f}%)\n\n"

        diff = abs(score_visivo - score_testo)

        if top_visivo != top_testo:
            risposta_analisi += "⚠️ **INCONGRUENZA STRUTTURALE** tra volto e testo. -> Fase di Esplorazione."
        else:
            if diff > soglia_differenza:
                risposta_analisi += f"⚠️ **Incongruenza di intensità** (differenza: {diff*100:.2f}%). -> Fase di Esplorazione."
            else:
                risposta_analisi += f"✅ **Modelli Congruenti** (differenza: {diff*100:.2f}%). -> Late Fusion."

        history[-1]["content"] = "✅ *Analisi completata. Clicca il pulsante sulla carta a sinistra per scoprire i dettagli tecnici.*\n\n*(Qui in futuro verrà inserita la risposta conversazionale dell'IA)*"
        
        # 4. Late fusion
        emozioni = set(dict_visivo.keys()).union(set(dict_testo.keys()))
        medie_emozioni = {}
        for emozione in emozioni:
            score_v = dict_visivo.get(emozione, 0.0)
            score_t = dict_testo.get(emozione, 0.0)
            medie_emozioni[emozione] = (score_v + score_t) / 2.0

        emozione_finale_media = max(medie_emozioni, key=medie_emozioni.get)
        score_finale_medio = medie_emozioni[emozione_finale_media]

        risposta_analisi += f"\n\n🔗 **Fusione Tardiva:**\n"
        risposta_analisi += f"- Emozione finale: **{emozione_finale_media.capitalize()}** (Media: {score_finale_medio*100:.2f}%)\n"
        
        risposta_analisi += "*(Medie per categoria:* "
        det_medie = [f"{k.capitalize()}: {v*100:.1f}%" for k, v in sorted(medie_emozioni.items(), key=lambda x: x[1], reverse=True)]
        risposta_analisi += " | ".join(det_medie) + ")*"

        # --- APPLICAZIONE ZONE DI REGOLAZIONE PER IL COLORE ---
        colore_dinamico = determina_colore_bordo(emozione_finale_media)

        card_finale = genera_card_html(
            risposta_analisi, 
            "📊 Analisi Completata", 
            "L'analisi è pronta!<br>Clicca il pulsante qui sotto per svelare i dettagli tecnici.",
            colore_bordo=colore_dinamico,
            abilitato=True
        )
        yield "", history, card_finale

    except Exception as e:
        history[-1]["content"] = "⚠️ Si è verificato un errore."
        yield "", history, genera_card_html(f"⚠️ Errore nell'elaborazione: {str(e)}", "Errore", "Si è verificato un problema.", colore_bordo="#e74c3c", abilitato=True)

try:
    with open("style.css", "r", encoding="utf-8") as f:
        custom_css = f.read()
except FileNotFoundError:
    print("Avviso: File style.css non trovato.")
    custom_css = ""

with gr.Blocks(css=custom_css) as demo:
    frame_buffer = gr.State([])

    # --- INTESTAZIONE ---
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("# 🧠 Sistema Multimodale Emotional Management")
        with gr.Column(scale=1, min_width=200):
            with gr.Accordion("⚙️ Impostazioni", open=False):
                lang_selector = gr.Dropdown(
                    choices=["Italiano", "Inglese"],
                    value="Italiano",
                    label="Lingua dell'applicazione"
                )

    # --- CORPO PRINCIPALE ---
    with gr.Row():
        # COLONNA SINISTRA (Webcam e Risultati)
        with gr.Column(scale=1):
            gr.Markdown("### 📷 Webcam Live")
            webcam_input = gr.Image(
                sources=["webcam"],
                streaming=True,
                type="pil",
                label=None,
                show_label=False,
                webcam_options=gr.WebcamOptions(
                    mirror=True,
                    constraints={"video": {"width": {"ideal": 320}, "height": {"ideal": 240}, "frameRate": {"ideal": 8, "max": 10}}},
                ),
            )
            
            # --- CARD 3D HTML ---
            card_analisi = gr.HTML(genera_card_html("<em>In attesa del primo messaggio...</em>", colore_bordo="transparent", abilitato=False))

        # COLONNA DESTRA (Chat)
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")
            chatbot = gr.Chatbot(height=500)

            with gr.Row():
                msg = gr.Textbox(placeholder="Scrivi qui come ti senti...", container=False, scale=8)
                btn_invia = gr.Button("Invia", variant="primary", scale=1)

    # Trigger eventi
    webcam_input.stream(fn=update_buffer, inputs=[webcam_input, frame_buffer], outputs=frame_buffer)
    msg.submit(fn=risposta_chat, inputs=[msg, lang_selector, frame_buffer, chatbot], outputs=[msg, chatbot, card_analisi])
    btn_invia.click(fn=risposta_chat, inputs=[msg, lang_selector, frame_buffer, chatbot], outputs=[msg, chatbot, card_analisi])

def avvia_server():
    demo.launch(server_port=7860, prevent_thread_lock=True, quiet=True, theme=gr.themes.Soft())

if __name__ == "__main__":
    t = threading.Thread(target=avvia_server, daemon=True)
    t.start()
    time.sleep(2)
    webview.create_window("Sistema Emozionale Multimodale", "http://127.0.0.1:7860", width=1300, height=800)
    webview.start()