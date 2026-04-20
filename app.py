import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 🔥 INCREASED LIMIT TO 2GB (2000 MB)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024 
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODELS = {
    'fast': 'base',
    'balanced': 'medium',
    'high': 'large-v3'
}

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_srt_and_txt(segments, target_lang):
    srt_content = ""
    txt_content = []
    
    # Setup translator if the user wants a language other than English/Original
    # (Because Whisper handles English translation natively)
    translator = None
    if target_lang not in ['original', 'en']:
        try:
            translator = GoogleTranslator(source='auto', target=target_lang)
        except Exception as e:
            print(f"Translator init error: {e}")

    for i, segment in enumerate(segments, start=1):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip()
        
        # Translate the text segment if a specific language was requested
        if translator:
            try:
                text = translator.translate(text)
            except Exception as e:
                print(f"Translation failed for segment, keeping original: {e}")
                
        srt_content += f"{i}\n{start} --> {end}\n{text}\n\n"
        txt_content.append(text)
        
    return srt_content, "\n".join(txt_content)

@app.route('/')
def index():
    return jsonify({"status": "AI Backend is running with 2GB limit and Multi-Lang Translation!"})

@app.route('/process', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    accuracy = request.form.get('accuracy', 'balanced')
    target_lang = request.form.get('target_lang', 'original')
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        model_size = MODELS.get(accuracy, 'medium')
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        # If target is English, let Whisper do it natively (it's faster and more accurate)
        task = "translate" if target_lang == 'en' else "transcribe"
        
        segments, info = model.transcribe(filepath, beam_size=5, task=task)
        segments_list = list(segments)
        
        # Generate SRT and translate if necessary
        srt_data, txt_data = generate_srt_and_txt(segments_list, target_lang)
        
        if os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify({
            'success': True,
            'srt': srt_data,
            'txt': txt_data,
            'detected_language': info.language,
        })

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
