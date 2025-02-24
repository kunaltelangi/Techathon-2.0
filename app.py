import re
import threading
from datetime import datetime
import json
from flask import Flask, render_template, make_response
from flask_socketio import SocketIO, emit
import assemblyai as aai
from constant import assemblyai_api_key  # Your AssemblyAI API key
from pyppeteer import launch

app = Flask(__name__)
socketio = SocketIO(app)
aai.settings.api_key = assemblyai_api_key

transcriber = None
session_id = None
transcriber_lock = threading.Lock()
# On re-analysis, previous analysis is replaced .
report_transcript = ""
current_language = "english"  # default language for realtime transcription

# Global variables for chart data
symptom_counts = {}      
severity_trends = []     
symptom_timeline = []    

user_location = None  # Store user's location

base_prompt = '''You are a medical transcript analyzer. Your task is to return the exact transcript with the following modifications:
1. Wrap any Protected Health Information (PHI) (such as names, ages, nationalities, gender identities, organizations) in <span style="color: red;"> ... </span>.
2. Highlight any Medical History (illnesses, symptoms, conditions) using <span style="background-color: lightgreen;"> ... </span>.
3. Italicize mentions of Anatomy (body parts) using <em> ... </em>.
4. Wrap any Medication names in <span style="background-color: yellow;"> ... </span>.
5. Wrap any Tests, Treatments, & Procedures in <span style="color: darkblue;"> ... </span>.
6. Based solely on the transcript, predict the most likely diagnosis and output it on its own separate line, enclosed in <span style="color: blue;"> and </span> with no extra text.
If there is insufficient information to predict a diagnosis or apply modifications, simply return the transcript verbatim.
Return only the formatted transcript without any additional commentary.'''

precautions_prompt = '''You are a medical advisor. Based on the predicted diagnosis provided below, suggest practical precautions (including recommended food items and home remedies).
Format your answer in HTML as follows:
<div class="precautions">
  <ul>
    <li>Precaution 1</li>
    <li>Precaution 2</li>
  </ul>
</div>
Do not include extra text.'''

severity_prompt = '''You are a medical advisor. Based on the predicted diagnosis provided below, evaluate its severity and provide a short recommendation.
Format your answer in HTML as follows:
<span class="severity">Severity: HIGH - Please consult a doctor immediately.</span>
or
<span class="severity">Severity: MODERATE - Monitor your symptoms and consider consulting a doctor.</span>
or
<span class="severity">Severity: LOW - Maintain healthy habits.</span>
Return only the HTML formatted text.'''

def on_open(session_opened: aai.RealtimeSessionOpened):
    global session_id
    session_id = session_opened.session_id
    print("Session ID:", session_id)

def on_data(transcript: aai.RealtimeTranscript):
    if not transcript.text:
        return
    # For final transcripts, trigger analysis.
    if isinstance(transcript, aai.RealtimeFinalTranscript):
        socketio.emit('transcript', {'text': transcript.text})
        analyze_transcript(transcript.text, reanalysis=False)
    else:
        socketio.emit('partial_transcript', {'text': transcript.text})

def on_error(error: aai.RealtimeError):
    print("An error occurred:", error)

def on_close():
    global session_id
    session_id = None
    print("Closing Session")

def transcribe_real_time(language):
    global transcriber, current_language
    current_language = language.lower() if language else "english"
    transcriber = aai.RealtimeTranscriber(
        sample_rate=16_000,
        on_data=on_data,
        on_error=on_error,
        on_open=on_open,
        on_close=on_close
    )
    transcriber.connect()
    microphone_stream = aai.extras.MicrophoneStream(sample_rate=16_000)
    transcriber.stream(microphone_stream)

def analyze_transcript(transcript, reanalysis=False):
    global report_transcript, symptom_counts, severity_trends, symptom_timeline, current_language

    # --- Extract structured graph data ---
    graph_data_prompt = f'''You are an assistant that extracts structured data from a medical transcript.
Given the following transcript:
{transcript}
Produce a JSON object with the following keys:
"symptom_counts": a dictionary mapping each symptom (string) to its frequency (integer),
"severity_trends": an array of objects, each with keys "time" (formatted as HH:MM:SS) and "severity" (one of HIGH, MODERATE, LOW),
"symptom_timeline": an array of objects, each with keys "time" (formatted as HH:MM:SS) and "symptom" (string).
Return only valid JSON with no additional commentary.'''
    try:
        graph_result = aai.Lemur().task(
            graph_data_prompt,
            input_text=graph_data_prompt,
            final_model=aai.LemurModel.claude3_5_sonnet
        )
        data = json.loads(graph_result.response)
        symptom_counts = data.get("symptom_counts", symptom_counts)
        severity_trends = data.get("severity_trends", severity_trends)
        symptom_timeline = data.get("symptom_timeline", symptom_timeline)
        print("Graph data extracted:", data)
    except Exception as e:
        print("Error parsing graph data from AI:", e)
        keywords = ["fever", "cough", "pain", "nausea", "dizziness", "headache"]
        transcript_lower = transcript.lower()
        for keyword in keywords:
            count = transcript_lower.count(keyword)
            if count > 0:
                symptom_counts[keyword] = symptom_counts.get(keyword, 0) + count

    # --- Generate formatted transcript ---
    result = aai.Lemur().task(
        base_prompt,
        input_text=transcript,
        final_model=aai.LemurModel.claude3_5_sonnet
    )
    formatted = result.response.strip()
    print("Formatted transcript:", formatted)
    if reanalysis:
        report_transcript = formatted + "<br>"
    else:
        report_transcript += formatted + "<br>"
    socketio.emit('formatted_transcript', {'text': formatted})
    
    # --- Extract diagnosis from formatted transcript ---
    diagnosis_matches = re.findall(r'<span\s+style="color:\s*blue;">(.*?)<\/span>', formatted, re.IGNORECASE)
    if diagnosis_matches:
        predicted_diagnosis = diagnosis_matches[0]
        print("Predicted diagnosis:", predicted_diagnosis)
        
        precautions_result = aai.Lemur().task(
            precautions_prompt,
            input_text=predicted_diagnosis,
            final_model=aai.LemurModel.claude3_5_sonnet
        )
        precautions_html = precautions_result.response.strip()
        print("Precautions:", precautions_html)
        socketio.emit('precautions', {'text': precautions_html})
        report_transcript += precautions_html + "<br>"
        
        severity_result = aai.Lemur().task(
            severity_prompt,
            input_text=predicted_diagnosis,
            final_model=aai.LemurModel.claude3_5_sonnet
        )
        severity_html = severity_result.response.strip()
        print("Severity:", severity_html)
        socketio.emit('severity', {'text': severity_html})
        report_transcript += severity_html + "<br>"
        
        severity_match = re.search(r'Severity:\s*(HIGH|MODERATE|LOW)', severity_html, re.IGNORECASE)
        if severity_match:
            severity_level = severity_match.group(1).upper()
            severity_trends.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "severity": severity_level
            })
        
        # --- Generate clinic/hospital suggestions ---
        if user_location:
            loc_str = f"latitude {user_location['latitude']}, longitude {user_location['longitude']}"
        else:
            loc_str = "a generic urban area"
        
        clinic_prompt = f'''You are a healthcare advisor. The patient is located at {loc_str}. Based on the predicted diagnosis provided below, suggest a list of nearby clinics and hospitals that specialize in this condition.
Format your answer in HTML as an unordered list (<ul>...</ul>).
For each suggestion, include:
- The Clinic/Hospital Name
- The Address
- The Contact Information
- A "Get Directions" button that is an anchor tag (<a>) opening Google Maps in a new tab.
Format the "Get Directions" link so that the href is: "https://www.google.com/maps/search/?api=1&query=CLINIC_ADDRESS"
Do not include any extra commentary.'''
        
        clinic_result = aai.Lemur().task(
            clinic_prompt,
            input_text=predicted_diagnosis,
            final_model=aai.LemurModel.claude3_5_sonnet
        )
        clinic_html = clinic_result.response.strip()
        print("Clinic suggestions:", clinic_html)
        socketio.emit('clinic_suggestions', {'text': clinic_html})
        report_transcript += clinic_html + "<br>"

@socketio.on('suggest_correction')
def handle_suggest_correction(data):
    unclear_word = data.get('word')
    context = data.get('context', '')
    suggestion_prompt = f"""
You are an AI language assistant specialized in medical transcription.
An unclear word "{unclear_word}" appears in the following transcript context:
"{context}"
Please provide three correction suggestions that are phonetically similar and medically appropriate.
Return your answer as a JSON array of strings with no extra commentary.
"""
    try:
        suggestion_result = aai.Lemur().task(
            suggestion_prompt,
            input_text=suggestion_prompt,
            final_model=aai.LemurModel.claude3_5_sonnet
        )
        suggestions = json.loads(suggestion_result.response)
        emit('correction_suggestions', {'word': unclear_word, 'suggestions': suggestions})
    except Exception as e:
        print("Error generating suggestions:", e)
        emit('correction_suggestions', {'word': unclear_word, 'suggestions': []})

@socketio.on('re_analyze_transcript')
def handle_re_analyze_transcript(data):
    updated_transcript = data.get('updated_transcript', '')
    if updated_transcript:
        analyze_transcript(updated_transcript, reanalysis=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trans.html')
def trans():
    return render_template('trans.html')

@app.route('/login.html')
def login():
    return render_template('login.html')

@app.route('/chart_data')
def chart_data():
    global symptom_counts, severity_trends, symptom_timeline
    if not symptom_counts:
        symptom_counts = {"fever": 0, "cough": 0, "pain": 0}
    if not severity_trends:
        severity_trends = [{"time": datetime.now().strftime("%H:%M:%S"), "severity": "LOW"}]
    data = {
        "symptom_counts": symptom_counts,
        "severity_trends": severity_trends,
        "symptom_timeline": symptom_timeline
    }
    return json.dumps(data)

@app.route('/report')
def report():
    global report_transcript
    phi_items = re.findall(r'<span\s+style="color:\s*red;">(.*?)<\/span>', report_transcript, re.IGNORECASE)
    medical_history_items = re.findall(r'<span\s+style="background-color:\s*lightgreen;">(.*?)<\/span>', report_transcript, re.IGNORECASE)
    anatomy_items = re.findall(r'<em>(.*?)<\/em>', report_transcript, re.IGNORECASE)
    medication_items = re.findall(r'<span\s+style="background-color:\s*yellow;">(.*?)<\/span>', report_transcript, re.IGNORECASE)
    tests_items = re.findall(r'<span\s+style="color:\s*darkblue;">(.*?)<\/span>', report_transcript, re.IGNORECASE)
    diagnosis_items = re.findall(r'<span\s+style="color:\s*blue;">(.*?)<\/span>', report_transcript, re.IGNORECASE)
    severity_items = re.findall(r'<span\s+class="severity">(.*?)<\/span>', report_transcript, re.IGNORECASE)
    return render_template(
        'report.html',
        phi=phi_items,
        medical_history=medical_history_items,
        anatomy=anatomy_items,
        medication=medication_items,
        tests=tests_items,
        diagnosis=diagnosis_items,
        severity=severity_items,
        report_transcript=report_transcript
    )

def generate_pdf():
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    async def _generate():
        browser = await launch(executablePath=chrome_path, args=['--no-sandbox'])
        page = await browser.newPage()
        await page.goto('http://localhost:5000/report', {'waitUntil': 'networkidle0'})
        await page.waitFor(3000)
        pdf_bytes = await page.pdf({'format': 'A4', 'printBackground': True})
        await browser.close()
        return pdf_bytes
    import asyncio
    return asyncio.get_event_loop().run_until_complete(_generate())

@app.route('/download_pdf')
def download_pdf():
    pdf_bytes = generate_pdf()
    response = make_response(pdf_bytes)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=report.pdf'
    return response

@socketio.on('toggle_transcription')
def handle_toggle_transcription(data):
    language = data.get('language', 'english')
    location = data.get('location')
    global user_location, transcriber, session_id
    if location:
        user_location = location
    with transcriber_lock:
        if session_id:
            if transcriber:
                print("Closing transcriber session")
                transcriber.close()
                transcriber = None
                session_id = None
        else:
            print("Starting transcriber session with language:", language)
            threading.Thread(target=transcribe_real_time, args=(language,)).start()

if __name__ == '__main__':
    socketio.run(app, debug=True)
