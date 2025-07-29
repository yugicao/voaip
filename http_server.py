
#!/usr/bin/env python3
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import uuid
import hashlib
import os
from datetime import datetime
import sip_user_manager
import soundfile as sf
import numpy as np
import librosa
from infer import AntiSpoofing
import time
import threading
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
import numpy as np
from scipy.spatial.distance import cosine

app = Flask(__name__)
CORS(app)

DB_FILE = "voice_system.db"
AUDIO_DIR = "user_voices"
os.makedirs(AUDIO_DIR, exist_ok=True)


detector = AntiSpoofing(
    config_path="./config/AASIST.conf",
    weights_path="./models/weights/AASIST.pth",
    device="cpu"
)


speaker_classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa"
)


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        phone TEXT UNIQUE,
        password TEXT,
        fullname TEXT,
        voice_filename TEXT,
        created_at TEXT,
        last_check_result TEXT,
        last_check_time TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS call_status (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        caller TEXT NOT NULL,
        callee TEXT NOT NULL,
        status TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS voice_verification_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        call_id INTEGER NOT NULL,
        user_id TEXT NOT NULL,
        opponent_id TEXT NOT NULL,
        result TEXT NOT NULL,
        speaker_id TEXT,
        speaker_name TEXT,
        speaker_phone TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS speaker_embeddings (
        user_id TEXT PRIMARY KEY,
        embedding BLOB,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)
    
    conn.commit()
    conn.close()

init_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def deepfake_detect(path):
    try:
        res = detector.predict(path)
        print(f"score={res['score']:.4f}, label={res['label']}")
        return res['label']
    except Exception as e:
        print("Cannot deepfake detect:", e)
        return "error"

def get_embedding(audio_path):
    """Trích xuất embedding từ file âm thanh"""
    try:
        signal, fs = torchaudio.load(audio_path)
        
        # Mono
        if signal.shape[0] > 1:
            signal = signal[:1, :]
            
        embeddings = speaker_classifier.encode_batch(signal)
        emb = embeddings.squeeze().detach().cpu().numpy().astype(np.float32)
        
        # Check vector 1D
        if emb.ndim > 1:
            emb = emb.flatten()
            
        return emb
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None

def save_embedding(user_id, embedding):
    """Lưu embedding vào database"""
    if embedding is None:
        return False
        
    try:
        emb_blob = embedding.tobytes()
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO speaker_embeddings (user_id, embedding)
            VALUES (?, ?)
        """, (user_id, emb_blob))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving embedding: {e}")
        return False

def identify_speaker(embedding, threshold=0.7):
    """Xác định người nói từ embedding"""
    if embedding is None:
        return None
        
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.id, u.fullname, u.phone, se.embedding 
            FROM speaker_embeddings se
            JOIN users u ON se.user_id = u.id
        """)
        rows = cursor.fetchall()
        conn.close()
        
        best_match = None
        best_score = float('inf')
        
        for row in rows:
            user_id, name, phone, emb_blob = row
            stored_emb = np.frombuffer(emb_blob, dtype=np.float32)
            

            probe_emb = embedding.flatten() if embedding.ndim > 1 else embedding
            stored_emb = stored_emb.flatten() if stored_emb.ndim > 1 else stored_emb
            
            dist = cosine(probe_emb, stored_emb)
            
            if dist < best_score:
                best_score = dist
                best_match = (user_id, name, phone)
        

        if best_score < threshold:
            return best_match
        return None
    except Exception as e:
        print(f"Error identifying speaker: {e}")
        return None

# Hàm xử lý giọng nói trong luồng riêng
def process_voice_in_thread(temp_path, user_id, call_id, opponent_id):
    try:
        # 1. Kiểm tra deepfake
        label = deepfake_detect(temp_path)
        
        # 2. Nhận diện người nói
        current_emb = get_embedding(temp_path)
        speaker_info = identify_speaker(current_emb) if current_emb is not None else None
        
        # 3. Lưu kết quả vào database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        if speaker_info:
            speaker_id, speaker_name, speaker_phone = speaker_info
            cursor.execute("""
                INSERT INTO voice_verification_results 
                (call_id, user_id, opponent_id, result, speaker_id, speaker_name, speaker_phone) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (call_id, user_id, opponent_id, label, speaker_id, speaker_name, speaker_phone))
        else:
            cursor.execute("""
                INSERT INTO voice_verification_results 
                (call_id, user_id, opponent_id, result) 
                VALUES (?, ?, ?, ?)
            """, (call_id, user_id, opponent_id, label))
            
        conn.commit()
        conn.close()
        print(f"Đã lưu kết quả cho user {user_id}: {label}")
    except Exception as e:
        print(f"Lỗi khi xử lý giọng nói: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route("/register", methods=["POST"])
def register():
    phone = request.form.get("phone")
    password = request.form.get("password")
    fullname = request.form.get("fullname")
    voice = request.files.get("voice")

    if not all([phone, password, fullname, voice]):
        return jsonify({"success": False, "error": "Thiếu thông tin"}), 400

    user_id = str(uuid.uuid4())
    filename = f"{user_id}.wav"
    voice_path = os.path.join(AUDIO_DIR, filename)
    voice.save(voice_path)

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute("INSERT INTO users (id, phone, password, fullname, voice_filename, created_at) VALUES (?, ?, ?, ?, ?, ?)", 
            (user_id, phone, hash_password(password), fullname, filename, datetime.now().isoformat()))

        conn.commit()
        conn.close()
        
        # Trích xuất và lưu embedding
        embedding = get_embedding(voice_path)
        save_embedding(user_id, embedding)
        
        sip_user_manager.add_user(phone, password)
        return jsonify({"success": True, "user_id": user_id}), 200
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "error": "Số điện thoại đã tồn tại"}), 409
    except Exception as e:
        print(f"Lỗi khi đăng ký: {str(e)}")
        return jsonify({"success": False, "error": "Lỗi hệ thống"}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    phone = data.get("phone")
    password = data.get("password")
    
    if not phone or not password:
        return jsonify({"success": False, "error": "Thiếu thông tin"}), 400

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, password, fullname FROM users WHERE phone = ?", (phone,))
    user = cursor.fetchone()
    conn.close()

    if user and hash_password(password) == user[1]:
        return jsonify({
            "success": True,
            "user_id": user[0],
            "fullname": user[2]
        }), 200
    else:
        return jsonify({"success": False, "error": "Sai số điện thoại hoặc mật khẩu"}), 401

@app.route("/save-call-status", methods=["GET", "POST"])
def save_call_status():
    if request.method == "GET":
        caller = request.args.get("caller")
        callee = request.args.get("callee")
        status = request.args.get("status")
    else:
        data = request.get_json(silent=True) or request.form
        caller = data.get("caller")
        callee = data.get("callee")
        status = data.get("status")
    
    if not all([caller, callee, status]):
        return "Thiếu tham số", 400
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        if status == "calling":
            cursor.execute("""
                INSERT INTO call_status (caller, callee, status)
                VALUES (?, ?, ?)
            """, (caller, callee, status))
        else:
            cursor.execute("""
                UPDATE call_status 
                SET status = ?
                WHERE id = (
                    SELECT id 
                    FROM call_status 
                    WHERE (caller = ? OR callee = ?)
                    AND status = 'calling'
                    ORDER BY timestamp DESC 
                    LIMIT 1
                )
            """, (status, caller, caller))
        
        conn.commit()
        conn.close()
        return "OK", 200
    except Exception as e:
        return f"Lỗi: {str(e)}", 500

@app.route("/verify-voice", methods=["POST"])
def verify_voice():
    user_id = request.form.get("user_id")
    voice = request.files.get("voice")

    if not all([user_id, voice]):
        return jsonify({"success": False, "error": "Thiếu thông tin"}), 400


    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT cs.id, cs.caller, cs.callee 
        FROM call_status cs
        JOIN users u ON (u.phone = cs.caller OR u.phone = cs.callee)
        WHERE u.id = ? AND cs.status = 'calling'
        ORDER BY cs.timestamp DESC 
        LIMIT 1
    """, (user_id,))
    call_info = cursor.fetchone()
    
    if not call_info:
        conn.close()
        return jsonify({"success": False, "error": "Không tìm thấy cuộc gọi đang hoạt động"}), 404
    
    call_id, caller_phone, callee_phone = call_info
    

    cursor.execute("SELECT phone FROM users WHERE id = ?", (user_id,))
    user_info = cursor.fetchone()
    
    if not user_info:
        conn.close()
        return jsonify({"success": False, "error": "Người dùng không tồn tại"}), 404
    
    user_phone = user_info[0]
    

    opponent_phone = caller_phone if user_phone == callee_phone else callee_phone
    
    cursor.execute("SELECT id FROM users WHERE phone = ?", (opponent_phone,))
    opponent_info = cursor.fetchone()
    
    if not opponent_info:
        conn.close()
        return jsonify({"success": False, "error": "Không tìm thấy đối phương"}), 404
    
    opponent_id = opponent_info[0]
    conn.close()

    # Save temp file
    temp_path = os.path.join(AUDIO_DIR, f"temp_{user_id}_{int(time.time())}.wav")
    voice.save(temp_path)
    
    # Create a thread for handle
    threading.Thread(
        target=process_voice_in_thread,
        args=(temp_path, user_id, call_id, opponent_id)
    ).start()
    
    # Wait for opponent result
    start_time = time.time()
    timeout = 15
    
    while time.time() - start_time < timeout:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check opponent result
        cursor.execute("""
            SELECT result, speaker_id, speaker_name, speaker_phone 
            FROM voice_verification_results 
            WHERE call_id = ? AND user_id = ?
        """, (call_id, opponent_id))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            label, speaker_id, speaker_name, speaker_phone = result
            
            response = {
                "success": True,
                "label": label
            }
            
            if speaker_id:
                response["speaker"] = {
                    "id": speaker_id,
                    "name": speaker_name,
                    "phone": speaker_phone
                }
            
            return jsonify(response), 200
        
        time.sleep(0.5)
    
    return jsonify({
        "success": False,
        "error": "Timeout: Không nhận được kết quả từ đối phương"
    }), 408

@app.route("/status", methods=["POST"])
def get_user_status():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "Invalid JSON"}), 400
            
        user_id = data.get("user_id")
        if not user_id:
            return jsonify({"success": False, "error": "Missing user_id"}), 400

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute("SELECT phone FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return jsonify({"success": False, "error": "User not found"}), 404
        
        phone = user[0]
        
        cursor.execute("""
            SELECT status FROM call_status 
            WHERE (caller = ? OR callee = ?) 
            AND status = 'calling'
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (phone, phone))
        
        active_call = cursor.fetchone()
        conn.close()
        
        status = "calling" if active_call else "idle"
        
        return jsonify({
            "success": True,
            "status": status
        }), 200
        
    except sqlite3.OperationalError as e:
        return jsonify({
            "success": False,
            "error": f"Database error: {str(e)}",
            "solution": "Please ensure the call_status table exists"
        }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


