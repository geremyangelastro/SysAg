import threading
import time
import re
from collections import Counter
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, DetectorFactory
from langchain_ollama import ChatOllama
import webview
import gradio as gr

DetectorFactory.seed = 0

print("Caricamento dei modelli HuggingFace in corso...")
vision_classifier = pipeline("image-classification", model="trpakov/vit-face-expression")
text_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

print("Caricamento modello di traduzione...")
tokenizer_it_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-it-en")
model_it_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-it-en")

print("Inizializzazione del modello Ollama...")
NOME_MODELLO_OLLAMA = "qwen2.5:7b"
try:
    llm = ChatOllama(model=NOME_MODELLO_OLLAMA, temperature=0.3)
    print("Ollama collegato con successo!")
except Exception as e:
    print(f"Attenzione: Impossibile collegare Ollama ({e}). Verificare che 'ollama serve' sia attivo.")
    llm = None

# --- MACCHINA A STATI PER IL DIALOGO ---
STATI = {
    "saluto": {
        "dire": "Sei un assistente empatico ed umano. Accogli l'utente in modo caloroso e chiedigli come si sente o come sta andando la sua giornata.",
        "obiettivo": "L'utente ha risposto al saluto o ha espresso come si sente o come sta andando la sua giornata.",
        "prossimo": "esplorazione"
    },
    "esplorazione": {
        "dire": "Sei un assistente empatico ed umano. L'utente ha condiviso il suo stato d'animo. Esplora delicatamente le sue sensazioni fisiche o l'ambiente circostante senza fare diagnosi o usare terminologia tecnica.",
        "obiettivo": "L'utente ha fornito un ulteriore dettaglio o approfondimento sulle sue sensazioni o sulla situazione che sta vivendo.",
        "prossimo": "regolazione"
    },
    "regolazione": {
        "dire": "Sei un assistente empatico ed umano. Basandoti sulla situazione e sullo stato emotivo rilevato, spiega la sensazione in modo semplice e consiglia una piccola strategia pratica di supporto (es. pausa, respirazione, musica, supporto sensoriale). Chiedi se desidera provare.",
        "obiettivo": "L'utente ha risposto al consiglio (ha ringraziato, detto che ci proverà, o ha commentato il suggerimento).",
        "prossimo": "mappatura_somatica"
    },
    "mappatura_somatica": {
        "dire": "Sei un assistente empatico ed umano. Guida l'utente in una breve esplorazione interocettiva/mappatura somatica, chiedendogli di concentrarsi su dove sente fisicamente l'emozione nel corpo (es. petto, pancia, spalle, gola, nodo al collo) o se nota particolari viscerali o tensioni muscolari.",
        "obiettivo": "L'utente ha fornito un riscontro sulla sensazione corporea o sul punto del corpo in cui percepisce l'emozione.",
        "prossimo": "de_escalation"
    },
    "de_escalation": {
        "dire": "Sei un assistente empatico ed umano. Offri una breve tecnica di de-escalation o radicamento (grounding) mirata a rilasciare la tensione corporea individuata. Rassicura l'utente e dagli un ultimo consiglio pratico prima della chiusura.",
        "obiettivo": "L'utente ha risposto o commentato la tecnica di de-escalation.",
        "prossimo": "fine"
    },
    "fine": {
        "dire": "Ringrazia caldamente l'utente per la conversazione, auguragli una buona giornata e ricordagli che se lo desidera può fare una nuova sessione cliccando su Ricomincia.",
        "obiettivo": None,
        "prossimo": None
    }
}

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
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = text.replace('\n', '<br>')
    return text

def genera_card_html(contenuto, titolo_front="📊 Analisi in attesa", testo_front="Avvia una chat per generare l'analisi.", colore_bordo="transparent", abilitato=False):
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
    emo = emozione_finale.lower()
    if emo in ["tristezza", "sad", "sadness"]:
        return "#3498db", "Zona Blu (Bassa energia / Stanchezza)"
    elif emo in ["neutro", "neutral", "gioia", "happy", "joy"]:
        return "#2ecc71", "Zona Verde (Calma / Equilibrio)"
    elif emo in ["paura", "fear", "sorpresa", "surprise", "disgusto", "disgust"]:
        return "#f1c40f", "Zona Gialla (Attivazione / Stress / Ansia)"
    elif emo in ["rabbia", "anger", "angry"]:
        return "#e74c3c", "Zona Rossa (Alta attivazione / Crisi)"
    return "#95a5a6", "Zona Grigia (Neutro)"

def _estrai_testo_sicuro(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parti = []
        for elem in content:
            if isinstance(elem, str):
                parti.append(elem)
            elif isinstance(elem, dict):
                parti.append(str(elem.get("text", elem.get("content", elem.get("value", "")))))
        return " ".join(parti)
    if isinstance(content, dict):
        return str(content.get("text", content.get("content", "")))
    return str(content) if content is not None else ""

def is_conferma_webcam(user_message, history):
    bot_chiedeva_webcam = False
    for msg in reversed(history):
        txt_bot = ""
        is_assistant = False
        
        if isinstance(msg, dict):
            if msg.get("role") == "assistant":
                is_assistant = True
                txt_bot = _estrai_testo_sicuro(msg.get("content", "")).lower()
        elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
            is_assistant = True
            txt_bot = _estrai_testo_sicuro(msg[1]).lower()

        if is_assistant:
            if "webcam" in txt_bot or "fotocamera" in txt_bot or "immagine" in txt_bot:
                bot_chiedeva_webcam = True
            break

    if not bot_chiedeva_webcam:
        return False

    txt_user = user_message.strip().lower()
    frasi_tipiche = ["e ora", "ora", "fatto", "attivata", "accesa", "funziona", "ora va", "ora funziona", "ok", "ci sei", "abilitata", "l'ho attivata", "ho attivato"]
    if any(f in txt_user for f in frasi_tipiche) and len(txt_user) < 35:
        return True

    if llm:
        try:
            prompt = (
                f"L'utente ha scritto: \"{user_message}\" dopo che l'assistente gli aveva segnalato che la webcam non era attiva.\n"
                "L'utente sta esprimendo un'emozione/stato d'animo oppure sta solo chiedendo/confermando che la fotocamera è attiva (es. 'e ora?', 'fatto')?\n"
                "Rispondi ESCLUSIVAMENTE 'TECNICO' o 'EMOZIONE'."
            )
            res = llm.invoke(prompt).content.strip().upper()
            return "TECNICO" in res
        except Exception:
            pass
    return False

def obiettivo_raggiunto(stato, messaggio_utente):
    if not llm or not STATI[stato]["obiettivo"]:
        return True

    domanda = (
        f"Obiettivo dello stato attuale: {STATI[stato]['obiettivo']}\n"
        f"Messaggio dell'utente: \"{messaggio_utente}\"\n"
        "L'obiettivo è stato raggiunto da questo messaggio dell'utente? Rispondi ESCLUSIVAMENTE con la parola 'SI' oppure 'NO'."
    )
    try:
        risposta = llm.invoke(domanda).content
        return risposta.strip().upper().startswith("S")
    except Exception as e:
        print(f"Errore nella verifica dell'obiettivo con Ollama: {e}")
        return True

def parla_ollama(stato, messaggio_utente, contesto_emotivo=""):
    if not llm:
        return "Mi spiace, il modello Ollama non è attualmente connesso."

    istruzioni_sistema = (
        f"{STATI[stato]['dire']}\n"
        f"Contesto emotivo rilevato dall'analisi: {contesto_emotivo}\n"
        "REGOLA FONDAMENTALE: Rispondi SEMPRE in italiano, in modo breve, naturale e umano (massimo 2-3 frasi).\n"
        "DIVIETO ASSOLUTO: Non menzionare MAI termini tecnici come 'Late Fusion', 'Zona Blu', 'Zona Verde', 'Zona Gialla', 'Zona Rossa', 'incongruenza strutturale', né percentuali o nomi di modelli."
    )

    try:
        risposta = llm.invoke([
            ("system", istruzioni_sistema),
            ("human", messaggio_utente)
        ])
        return risposta.content.strip()
    except Exception as e:
        print(f"Errore nella generazione del messaggio con Ollama: {e}")
        return "Sono qui per ascoltarti. Come posso esserti d'aiuto in questo momento?"

def gestisci_azione(user_message, app_language, buffer_frames, current_state, history):
    if history is None:
        history = []

    # 1. RESET (Pulsante "Ricomincia" cliccato nello stato "fine")
    if current_state == "fine":
        initial_history = [{"role": "assistant", "content": "(saluto) Ciao! Benvenuto. Come stai? Raccontami pure come ti senti in questo momento."}]
        card_reset = genera_card_html("<em>In attesa del primo messaggio...</em>", colore_bordo="transparent", abilitato=False)
        yield gr.update(value="", interactive=True, placeholder="Scrivi qui come ti senti..."), initial_history, card_reset, "saluto", gr.update(value="Invia", variant="primary")
        return

    # 2. CONTROLLO MESSAGGIO VUOTO
    if user_message is None or not str(user_message).strip():
        yield gr.update(), history, genera_card_html("Nessun testo da analizzare.", colore_bordo="transparent", abilitato=False), current_state, gr.update()
        return

    # 3. CONTROLLO SE È UNA CONFERMA TECNICA DI RIPRISTINO WEBCAM (es. "e ora?")
    if is_conferma_webcam(user_message, history):
        user_message = str(user_message).strip()
        history.append({"role": "user", "content": user_message})
        
        if buffer_frames:
            history.append({"role": "assistant", "content": f"({current_state}) Perfetto, ora la fotocamera è attiva e riesco a vederti! Raccontami pure come ti senti o come sta andando la tua giornata."})
            card_attesa = genera_card_html("Pronto per l'analisi. Condividi come ti senti.", "In attesa", "Scrivi come ti senti per avviare l'analisi.", colore_bordo="#3498db", abilitato=False)
        else:
            history.append({"role": "assistant", "content": f"({current_state}) Ancora non riesco a cogliere l'immagine dalla webcam. Assicurati di aver fornito i permessi al browser e che sia accesa."})
            card_attesa = genera_card_html("⚠️ Nessun fotogramma ricevuto dalla webcam.", "Errore", "Dati visivi mancanti.", colore_bordo="#e74c3c", abilitato=False)
            
        yield gr.update(value=""), history, card_attesa, current_state, gr.update()
        return

    # 4. AVVIO ELABORAZIONE NORMALE
    user_message = str(user_message).strip()
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": "⏳ *Sto elaborando...*"})
    
    attesa_card = genera_card_html("⏳ *Elaborazione in corso...*", "⚙️ Analisi...", "Il sistema sta valutando i dati.", colore_bordo="transparent", abilitato=False)
    yield gr.update(value=""), history, attesa_card, current_state, gr.update()

    time.sleep(0.2)

    # 5. CONTROLLO WEBCAM PRIMA DELL'ANALISI
    if not buffer_frames:
        history[-1]["content"] = f"({current_state}) Hmm, non riesco a cogliere bene l'immagine dalla webcam. Assicurati che sia attiva e di aver fornito i permessi!"
        card_err = genera_card_html("⚠️ Nessun fotogramma ricevuto dalla webcam.", "Errore", "Dati visivi mancanti.", colore_bordo="#e74c3c", abilitato=False)
        yield gr.update(), history, card_err, current_state, gr.update()
        return

    # 6. VALUTAZIONE TRANSIZIONE DI STATO TRAMITE OLLAMA
    if obiettivo_raggiunto(current_state, user_message):
        prossimo_stato = STATI[current_state]["prossimo"]
        if prossimo_stato:
            current_state = prossimo_stato

    # 7. ANALISI MULTIMODALE COMPLETA
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
                avviso_lingua = f"💡 Nota: Sembra che tu stia scrivendo in {lingua_rilevata_nome}! Ti suggeriamo di cambiare la lingua nelle impostazioni ⚙️.\n\n"
        
        if app_language == "Italiano":
            text_to_analyze = traduci_in_inglese(user_message)
        else:
            text_to_analyze = user_message

        raw_text = text_classifier(text_to_analyze)
        dict_testo = format_results(raw_text, is_text_model=True)
        top_testo = max(dict_testo, key=dict_testo.get)
        score_testo = dict_testo[top_testo]

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

        ha_incongruenza_strutturale = (top_visivo != top_testo)

        if ha_incongruenza_strutturale:
            emozione_finale_media = top_visivo
            colore_dinamico, nome_zona = determina_colore_bordo(emozione_finale_media)
            
            risposta_analisi = avviso_lingua
            risposta_analisi += f"🎯 **Zona di Regolazione Rilevata:** {nome_zona}\n\n"
            risposta_analisi += f"⚠️ **Nota:** È emersa una discrepanza tra la tua espressione visiva ({top_visivo}) e le parole utilizzate ({top_testo}).\n\n"
            risposta_analisi += f"🧠 **Dettagli Tecnici:**\n- Volto: {top_visivo.capitalize()} ({score_visivo*100:.1f}%)\n- Testo: {top_testo.capitalize()} ({score_testo*100:.1f}%)\n"

            contesto_per_ollama = f"È stata rilevata una discrepanza tra l'espressione visiva ({top_visivo}) e le parole dell'utente ({top_testo}). L'utente potrebbe nascondere stanchezza o sforzo."
        else:
            emozioni = set(dict_visivo.keys()).union(set(dict_testo.keys()))
            medie_emozioni = {}
            for emozione in emozioni:
                score_v = dict_visivo.get(emozione, 0.0)
                score_t = dict_testo.get(emozione, 0.0)
                medie_emozioni[emozione] = (score_v + score_t) / 2.0

            emozione_finale_media = max(medie_emozioni, key=medie_emozioni.get)
            colore_dinamico, nome_zona = determina_colore_bordo(emozione_finale_media)

            risposta_analisi = avviso_lingua
            risposta_analisi += f"🎯 **Zona di Regolazione Rilevata:** {nome_zona}\n\n"
            risposta_analisi += f"🔗 **Fusione Tardiva:** Emozione prevalente stimata: **{emozione_finale_media.capitalize()}**\n\n"
            det_medie = [f"{k.capitalize()}: {v*100:.1f}%" for k, v in sorted(medie_emozioni.items(), key=lambda x: x[1], reverse=True)]
            risposta_analisi += f"*(Medie analitiche: {' | '.join(det_medie)})*"

            contesto_per_ollama = f"L'utente si trova in uno stato emotivo di tipo '{emozione_finale_media}' ed è associato alla {nome_zona}."

        # 8. GENERAZIONE RISPOSTA CHAT TRAMITE OLLAMA CON PREFISSO DELLO STATO
        risposta_bot = parla_ollama(current_state, user_message, contesto_per_ollama)
        history[-1]["content"] = f"({current_state}) {risposta_bot}"

        card_finale = genera_card_html(
            risposta_analisi, 
            "📊 Analisi Completata", 
            "L'analisi è pronta!<br>Clicca il pulsante qui sotto per svelare i dettagli tecnici.",
            colore_bordo=colore_dinamico,
            abilitato=True
        )

        # 9. GESTIONE FINE SESSIONE
        if current_state == "fine":
            yield gr.update(interactive=False, value="", placeholder="Sessione conclusa. Clicca Ricomincia..."), history, card_finale, current_state, gr.update(value="🔄 Ricomincia", variant="secondary")
        else:
            yield gr.update(interactive=True, value="", placeholder="Scrivi qui..."), history, card_finale, current_state, gr.update()

    except Exception as e:
        history[-1]["content"] = f"({current_state}) Mi dispiace, c'è stato un piccolo intoppo tecnico nell'elaborazione."
        yield gr.update(), history, genera_card_html(f"⚠️ Errore: {str(e)}", "Errore", "Problema rilevato.", colore_bordo="#e74c3c", abilitato=True), current_state, gr.update()

try:
    with open("style.css", "r", encoding="utf-8") as f:
        custom_css = f.read()
except FileNotFoundError:
    custom_css = ""

with gr.Blocks() as demo:
    frame_buffer = gr.State([])
    dialogue_state = gr.State("saluto")

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
        # COLONNA SINISTRA (Webcam e Card)
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

        # COLONNA DESTRA (Chat con saluto iniziale automatico e stato visibile)
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "(saluto) Ciao! Benvenuto. Come stai? Raccontami pure come ti senti in questo momento."}],
                height=500
            )

            with gr.Row():
                msg = gr.Textbox(placeholder="Scrivi qui come ti senti...", container=False, scale=8)
                btn_invia = gr.Button("Invia", variant="primary", scale=1)

    outputs_list = [msg, chatbot, card_analisi, dialogue_state, btn_invia]
    
    webcam_input.stream(fn=update_buffer, inputs=[webcam_input, frame_buffer], outputs=frame_buffer)
    
    msg.submit(
        fn=gestisci_azione, 
        inputs=[msg, lang_selector, frame_buffer, dialogue_state, chatbot], 
        outputs=outputs_list
    )
    btn_invia.click(
        fn=gestisci_azione, 
        inputs=[msg, lang_selector, frame_buffer, dialogue_state, chatbot], 
        outputs=outputs_list
    )
    
def avvia_server():
    demo.launch(server_port=7860, css=custom_css, prevent_thread_lock=True, quiet=True, theme=gr.themes.Soft())

if __name__ == "__main__":
    t = threading.Thread(target=avvia_server, daemon=True)
    t.start()
    time.sleep(2)
    webview.create_window("Sistema Emozionale Multimodale", "http://127.0.0.1:7860", width=1300, height=800)
    webview.start()