import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS # NEW: Import CORS
from werkzeug.utils import secure_filename
from faster_whisper import WhisperModel

app = Flask(__name__)
# NEW: Allow Vercel/Netlify to talk to this backend
CORS(app, resources={r"/*": {"origins": "*"}}) 

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
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

def generate_srt(segments):
    srt_content = ""
    for i, segment in enumerate(segments, start=1):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        srt_content += f"{i}\n{start} --> {end}\n{segment.text.strip()}\n\n"
    return srt_content

@app.route('/')
def index():
    return jsonify({"status": "AI Backend is running!"})

@app.route('/process', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    accuracy = request.form.get('accuracy', 'balanced')
    translate = request.form.get('translate') == 'true'
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        model_size = MODELS.get(accuracy, 'medium')
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        task = "translate" if translate else "transcribe"
        
        segments, info = model.transcribe(filepath, beam_size=5, task=task)
        segments_list = list(segments)
        
        srt_data = generate_srt(segments_list)
        txt_data = "\n".join([s.text.strip() for s in segments_list])
        
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
    # NEW: Hugging Face Spaces requires port 7860
    app.run(host='0.0.0.0', port=7860)