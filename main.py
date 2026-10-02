import threading
import time
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

def risposta_chat(user_message, webcam_frame, history):
    """Riceve testo, frame PIL dalla webcam e storico."""
    if history is None:
        history = []

    if user_message is None or not str(user_message).strip():
        return "", history

    user_message = str(user_message).strip()

    if webcam_frame is None:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "⚠️ **Attenzione:** Nessun fotogramma ricevuto dalla webcam. Assicurati di aver concesso i permessi della fotocamera."})
        return "", history

    try:
        frame_corrente = (
            webcam_frame.convert("RGB")
            if isinstance(webcam_frame, Image.Image)
            else Image.fromarray(webcam_frame).convert("RGB")
        )
    except Exception as e:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": f"⚠️ Errore nell'elaborazione del fotogramma: {str(e)}"})
        return "", history

    # 1. Esecuzione dei modelli
    raw_vision = vision_classifier(frame_corrente)
    raw_text = text_classifier(user_message)

    dict_visivo = format_results(raw_vision, is_text_model=False)
    dict_testo = format_results(raw_text, is_text_model=True)

    top_visivo = max(dict_visivo, key=dict_visivo.get)
    score_visivo = dict_visivo[top_visivo]

    top_testo = max(dict_testo, key=dict_testo.get)
    score_testo = dict_testo[top_testo]

    # 2. Valutazione incongruenza
    soglia_differenza = 0.7

    risposta_bot = "🧠 **Analisi Multimodale:**\n"
    risposta_bot += f"- **Volto:** {top_visivo.capitalize()} ({score_visivo*100:.2f}%)\n"
    risposta_bot += f"- **Testo:** {top_testo.capitalize()} ({score_testo*100:.2f}%)\n\n"

    if top_visivo != top_testo:
        risposta_bot += "⚠️ **Rilevata INCONGRUENZA STRUTTURALE** tra volto e testo. -> *Fase di Esplorazione*."
    else:
        diff = abs(score_visivo - score_testo)
        if diff > soglia_differenza:
            risposta_bot += f"⚠️ **Incongruenza di intensità** (differenza: {diff*100:.2f}%). -> *Fase di Esplorazione*."
        else:
            risposta_bot += f"✅ **Modelli Congruenti** (differenza: {diff*100:.2f}%). -> *Late Fusion*."

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": risposta_bot})

    return "", history

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Sistema Multimodale Emotional Management")

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

    msg.submit(
        fn=risposta_chat,
        inputs=[msg, webcam_input, chatbot],
        outputs=[msg, chatbot],
    )

    btn_invia.click(
        fn=risposta_chat,
        inputs=[msg, webcam_input, chatbot],
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

    # Attendi che il server Gradio sia pronto prima di aprire la WebView
    time.sleep(2)

    webview.create_window(
        "Sistema Emozionale Multimodale",
        "http://127.0.0.1:7860",
        width=1300,
        height=800,
    )
    webview.start()