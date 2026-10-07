import threading
import time
from collections import Counter
from PIL import Image
from transformers import pipeline
import webview
import gradio as gr

print("Caricamento dei modelli in corso...")
vision_classifier = pipeline("image-classification", model="trpakov/vit-face-expression")
text_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None,)
print("Modelli caricati con successo!")

def change_label(label):
    mapping = {
        "angry": "rabbia",
        "anger": "rabbia",
        "disgust": "disgusto",
        "fear": "paura",
        "happy": "gioia",
        "joy": "gioia",
        "sad": "tristezza",
        "sadness": "tristezza",
        "surprise": "sorpresa",
        "neutral": "neutro",
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
    """Mantiene un buffer aggiornato degli ultimi 5 frame provenienti dallo stream."""
    if current_buffer is None:
        current_buffer = []
    if new_frame is not None:
        try:
            img = (
                new_frame.convert("RGB")
                if isinstance(new_frame, Image.Image)
                else Image.fromarray(new_frame).convert("RGB")
            )
            current_buffer.append(img)
            current_buffer = current_buffer[-5:]
        except Exception:
            pass
    return current_buffer

def risposta_chat(user_message, buffer_frames, history):
    """Riceve testo, buffer degli ultimi 5 frame e storico, restituendo un feedback in streaming."""
    if history is None:
        history = []

    if user_message is None or not str(user_message).strip():
        yield "", history
        return

    user_message = str(user_message).strip()

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": "⏳ *Analisi multimodale in corso... attendi.*"})
    
    yield "", history

    time.sleep(0.2)

    if not buffer_frames:
        history[-1]["content"] = "⚠️ **Attenzione:** Nessun fotogramma ricevuto dalla webcam. Assicurati di aver concesso i permessi della fotocamera e attendi qualche istante."
        yield "", history
        return

    try:
        # 1. Esecuzione modello Testo
        raw_text = text_classifier(user_message)
        dict_testo = format_results(raw_text, is_text_model=True)
        top_testo = max(dict_testo, key=dict_testo.get)
        score_testo = dict_testo[top_testo]

        # 2. Analisi dei 5 Frame Visivi
        risultati_visione = []
        for frame in buffer_frames:
            raw_vision = vision_classifier(frame)
            dict_visivo = format_results(raw_vision, is_text_model=False)
            top_emozione = max(dict_visivo, key=dict_visivo.get)
            score = dict_visivo[top_emozione]
            risultati_visione.append((top_emozione, score, dict_visivo))

        # Trova l'emozione predominante tra i 5 frame
        conteggio_emozioni = Counter([res[0] for res in risultati_visione])
        emozione_predominante = conteggio_emozioni.most_common(1)[0][0]

        # Seleziona l'analisi con l'accuratezza maggiore per l'emozione predominante
        miglior_risultato = None
        for res in risultati_visione:
            if res[0] == emozione_predominante:
                if miglior_risultato is None or res[1] > miglior_risultato[1]:
                    miglior_risultato = res

        top_visivo = miglior_risultato[0]
        score_visivo = miglior_risultato[1]
        dict_visivo = miglior_risultato[2]

        # 3. Valutazione incongruenza
        soglia_differenza = 0.7

        risposta_bot = f"🧠 **Analisi Multimodale (su {len(buffer_frames)} frame):**\n"
        risposta_bot += f"- **Volto (Predominante):** {top_visivo.capitalize()} ({score_visivo*100:.2f}%)\n"
        risposta_bot += f"- **Testo:** {top_testo.capitalize()} ({score_testo*100:.2f}%)\n\n"

        if top_visivo != top_testo:
            risposta_bot += "⚠️ **Rilevata INCONGRUENZA STRUTTURALE** tra volto e testo. -> *Fase di Esplorazione*."
        else:
            diff = abs(score_visivo - score_testo)
            if diff > soglia_differenza:
                risposta_bot += f"⚠️ **Incongruenza di intensità** (differenza: {diff*100:.2f}%). -> *Fase di Esplorazione*."
            else:
                risposta_bot += f"✅ **Modelli Congruenti** (differenza: {diff*100:.2f}%). -> *Late Fusion*."

        history[-1]["content"] = risposta_bot
        yield "", history

    except Exception as e:
        history[-1]["content"] = f"⚠️ Errore nell'elaborazione: {str(e)}"
        yield "", history

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Sistema Multimodale Emotional Management")

    frame_buffer = gr.State([])

    with gr.Row():
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
                    constraints={
                        "video": {
                            "width":     {"ideal": 320},
                            "height":    {"ideal": 240},
                            "frameRate": {"ideal": 8, "max": 10},
                        }
                    },
                ),
            )

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")
            chatbot = gr.Chatbot(height=500)

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Scrivi qui come ti senti...", container=False, scale=8
                )
                btn_invia = gr.Button("Invia", variant="primary", scale=1)

    webcam_input.stream(
        fn=update_buffer,
        inputs=[webcam_input, frame_buffer],
        outputs=frame_buffer
    )

    msg.submit(
        fn=risposta_chat,
        inputs=[msg, frame_buffer, chatbot], 
        outputs=[msg, chatbot],
    )

    btn_invia.click(
        fn=risposta_chat,
        inputs=[msg, frame_buffer, chatbot], 
        outputs=[msg, chatbot],
    )

def avvia_server():
    demo.launch(
        server_port=7860,
        prevent_thread_lock=True,
        quiet=True,
        theme=gr.themes.Soft(),
    )

if __name__ == "__main__":
    t = threading.Thread(target=avvia_server, daemon=True)
    t.start()

    time.sleep(2)

    webview.create_window(
        "Sistema Emozionale Multimodale",
        "http://127.0.0.1:7860",
        width=1300,
        height=800,
    )
    webview.start()