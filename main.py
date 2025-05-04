# Voraussetzungen: pip install openai qrcode[pil] PyMuPDF tiktoken python-dotenv

import fitz               # PyMuPDF
import os
import json
import sqlite3
import tiktoken
import qrcode
import re
import string
from openai import OpenAI
from dotenv import load_dotenv

# === Konfiguration ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# === Hilfsfunktionen ===
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def trim_to_token_limit(text, max_tokens=14000, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:max_tokens])

def clean_iban(iban_str):
    return re.sub(r"[^A-Z0-9]", "", iban_str.upper())

def clean_filename(name):
    allowed = "-_() " + string.ascii_letters + string.digits
    return "".join(c for c in name if c in allowed)

def create_sepa_qr(iban, empfaenger, betrag, verwendungszweck, save_path):
    empfaenger = empfaenger.strip()[:70]
    verwendungszweck = verwendungszweck.strip()[:140]
    payload = "\n".join([
        "BCD", "001", "1", "SCT", "",
        empfaenger, iban.replace(" ", ""),
        f"EUR{betrag}", "", "", verwendungszweck, "", ""
    ])
    qr = qrcode.QRCode(version=5, box_size=10, border=4)
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill="black", back_color="white")
    img.save(save_path)
    print(f"SEPA-QR-Code gespeichert unter: {save_path}")

def init_db():
    conn = sqlite3.connect("dokumente.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dokumente (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datum TEXT,
            beschreibung TEXT,
            dateiname TEXT,
            massnahme TEXT,
            massnahmen_text TEXT,
            ist_rechnung BOOLEAN,
            betrag_eur TEXT,
            empfaenger TEXT,
            iban TEXT,
            verwendungszweck TEXT,
            zahlungsziel TEXT,
            klassifikation TEXT,
            zusammenfassung TEXT,
            dokumententyp TEXT,
            qr_code_path TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_document(data, qr_path=None):
    conn = sqlite3.connect("dokumente.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO dokumente (
            datum, beschreibung, dateiname, massnahme, massnahmen_text, ist_rechnung,
            betrag_eur, empfaenger, iban, verwendungszweck, zahlungsziel,
            klassifikation, zusammenfassung, dokumententyp, qr_code_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("datum"),                               #  1
        data.get("beschreibung"),                        #  2
        data.get("dateiname"),                           #  3
        data.get("massnahme"),                           #  4
        data.get("massnahmen_text"),                     #  5
        data.get("ist_rechnung"),                        #  6
        data.get("betrag_eur"),                          #  7
        data.get("empfaenger"),                          #  8
        data.get("iban"),                                #  9
        data.get("verwendungszweck"),                    # 10
        data.get("zahlungsziel"),                        # 11
        ", ".join(data.get("klassifikation", [])),       # 12
        data.get("zusammenfassung"),                     # 13
        data.get("dokumententyp"),                       # 14
        qr_path                                          # 15
    ))
    conn.commit()
    conn.close()

def analyze_with_gpt(text):
    def run_model(text, model_name):
        prompt = f"""
Analysiere folgendes Dokument und gib die Informationen im JSON-Format zurück.

Wenn es sich offensichtlich um eine Rechnung handelt, ergänze auch die Felder 
'betrag_eur', 'iban', 'empfaenger', 'verwendungszweck' und 'zahlungsziel'.

Wenn eine Rechnungsnummer erkannt wird, trage sie in 'verwendungszweck' ein.

Wenn ein Datum im Dokument steht, gib es immer im Format "YYYY-MM-DD" im Feld "datum" zurück.
Ist kein klares Datum vorhanden, nimm das Ausstellungsdatum des Schreibens oder ein plausibles Datum aus dem Kontext.

Erzeuge außerdem einen passenden, beschreibenden Dateinamen im Feld "dateiname".
Der Name soll den Inhalt eindeutig beschreiben, idealerweise mit Absender, Art des Dokuments und Datum,
z. B. "Rechnung_PVS_2024-05" oder "Bescheid_BARMER_Juli_2025".

Wenn im Dokument steht, dass der Betrag per Lastschrift eingezogen wird
(z. B. durch Begriffe wie „Lastschrifteinzug“, „Einzug vom Konto“, „Mandatsreferenznummer“ oder „wird automatisch abgebucht“),
dann ist es **keine Rechnung**, auch wenn ein Betrag genannt wird.

In diesem Fall:
- Setze "ist_rechnung": false
- Lasse "betrag_eur", "iban", "empfaenger", "verwendungszweck" und "zahlungsziel" leer
- Gib als "massnahme" einfach "ablegen" an
- Gib in "massnahmen_text" an, ob auf das Schreiben geantwortet oder anderweitig gehandelt werden muss

Ermittle zusätzlich den passenden Dokumententyp im Feld "dokumententyp".
Wähle aus: "Rechnung", "Gutschrift", "Bescheid", "Beitragsrechnung", "Kontoauszug", "Vertrag", "Versicherungsschein", "Mahnung", "Schreiben", "Information", "Zertifikat", "Sonstiges"

Antwortformat:
{{
  "datum": "...",
  "beschreibung": "...",
  "dateiname": "...",
  "massnahme": "ablegen" | "massnahme_erforderlich",
  "massnahmen_text": "...",
  "ist_rechnung": true | false,
  "betrag_eur": "...",
  "iban": "...",
  "empfaenger": "...",
  "verwendungszweck": "...",
  "zahlungsziel": "...",
  "klassifikation": ["...", "..."],
  "zusammenfassung": "...",
  "dokumententyp": "..."
}}

Dokument:
{text}
"""
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    try:
        message = run_model(text, "gpt-3.5-turbo")
        return json.loads(message)
    except json.JSONDecodeError:
        try:
            print("🔁 Versuche es mit GPT-4 ...")
            message = run_model(text, "gpt-4")
            return json.loads(message)
        except:
            return {
                "datum": "0000-00-00",
                "beschreibung": "Fehlerhafte GPT-Antwort",
                "dateiname": "Fehler",
                "massnahme": "ablegen",
                "massnahmen_text": "JSON konnte nicht gelesen werden",
                "ist_rechnung": False,
                "betrag_eur": "",
                "iban": "",
                "empfaenger": "",
                "verwendungszweck": "",
                "zahlungsziel": "",
                "klassifikation": [],
                "zusammenfassung": "",
                "dokumententyp": "Sonstiges"
            }

def main():
    init_db()
    dokumenten_ordner = os.path.join(os.path.dirname(__file__), "dokumente")
    if not os.path.exists(dokumenten_ordner):
        print("Ordner './dokumente' nicht gefunden.")
        return

    for filename in os.listdir(dokumenten_ordner):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(dokumenten_ordner, filename)
        print(f"📄 Verarbeite: {filename}")

        # Text extrahieren
        text = extract_text_from_pdf(pdf_path)

        # Prüfen, ob das PDF durchsuchbaren Text enthält
        if len(text.strip()) < 50:
            print("🔍 Dokument nicht lesbar – markiere zur manuellen Prüfung.")
            summary = {
                "datum": "0000-00-00",
                "beschreibung": "PDF nicht durchsuchbar – manuelle Prüfung erforderlich",
                "dateiname": os.path.splitext(filename)[0],
                "massnahme": "massnahme_erforderlich",
                "massnahmen_text": "Bitte manuell bearbeiten (Text fehlt).",
                "ist_rechnung": False,
                "betrag_eur": "",
                "iban": "",
                "empfaenger": "",
                "verwendungszweck": "",
                "zahlungsziel": "",
                "klassifikation": ["nicht_lesbar"],
                "zusammenfassung": "",
                "dokumententyp": "Sonstiges"
            }
        else:
            # Normaler GPT-Durchlauf
            token_count = count_tokens(text)
            if token_count > 14000:
                print(f"⚠️ {token_count} Tokens – Text wird gekürzt.")
                text = trim_to_token_limit(text)
            else:
                print(f"✅ Tokenanzahl: {token_count} – alles im grünen Bereich.")
            summary = analyze_with_gpt(text)
            if summary.get("ist_rechnung") and "iban" in summary:
                summary["iban"] = clean_iban(summary["iban"])

        # Dateiumbenennung
        datum = summary.get("datum", "0000-00-00")
        dateiname_raw = summary.get("dateiname", os.path.splitext(filename)[0])
        dateiname_clean = clean_filename(f"{datum} {dateiname_raw}")
        neuer_pdf_name = f"{dateiname_clean}.pdf"
        neuer_pdf_pfad = os.path.join(dokumenten_ordner, neuer_pdf_name)
        if not os.path.exists(neuer_pdf_pfad):
            os.rename(pdf_path, neuer_pdf_pfad)
            print(f"🔁 Umbenannt in: {neuer_pdf_name}")
        else:
            print(f"⚠️ Datei {neuer_pdf_name} existiert bereits – nicht umbenannt.")

        # QR-Code falls Rechnung
        qr_filename = None
        if summary.get("ist_rechnung"):
            qr_filename = f"{dateiname_clean}.png"
            qr_path = os.path.join(dokumenten_ordner, qr_filename)
            create_sepa_qr(
                summary["iban"],
                summary["empfaenger"],
                summary["betrag_eur"],
                summary["verwendungszweck"],
                qr_path
            )

        # Eintrag in die Datenbank
        insert_document(summary, qr_filename)

if __name__ == "__main__":
    main()
