import sqlite3
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio
from scipy.spatial.distance import cosine

import sys
sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = "speakers.db"

# Load model một lần
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa"
)

def delete_user(name: str):
    """Xóa user khỏi database theo tên"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM speakers WHERE name = ?", (name,))
    changes = conn.total_changes
    conn.commit()
    conn.close()
    return changes > 0


def list_users():
    """Liệt kê tên các user đã enroll"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM speakers ORDER BY name")
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]

def authenticate_speaker(threshold: float = 0.7):
    # 1) Ghi âm
    wav_file = "probe.wav"
    record_audio(wav_file, duration=10.0)
    # 2) Lấy embedding
    probe_emb = get_embedding(wav_file)
    # 3) So sánh với tất cả embedding đã enroll
    candidates = load_all_embeddings()
    best_match, best_score = None, 1.0  # vì cosine distance trong [0,2]
    for name, emb in candidates:
        dist = cosine(probe_emb, emb)
        if dist < best_score:
            best_score, best_match = dist, name

    # 4) Kiểm tra với ngưỡng
    if best_score < threshold:
        similarity = 1 - best_score
        print(f"Xác thực thành công! Đây có thể là `{best_match}` (similarity={similarity:.2f})")
    else:
        print("Xác thực thất bại: không khớp với bất kỳ người dùng nào.")


def enroll_speaker(duration=15.0):
    name = input("Nhập tên người dùng để enroll: ").strip()
    wav_file = f"{name}.wav"
    record_audio(wav_file, duration)
    emb = get_embedding(wav_file)
    save_embedding(name, emb)
    print(f"Đã enroll `{name}` thành công.")


def get_embedding(wav_path: str):
    signal, fs = torchaudio.load(wav_path)
    embeddings = classifier.encode_batch(signal)  # [1,1,192]
    emb = embeddings.squeeze().detach().cpu().numpy().astype(np.float32)
    return emb


def record_audio(filename: str, duration: float = 10.0, fs: int = 16000):
    """
    Ghi âm từ micro trong `duration` giây, lưu thành WAV mono với sampling rate = fs.
    """
    print(f"Đang ghi âm trong {duration} giây...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    # Chuyển về int16 để lưu WAV
    audio_int16 = (audio * 32767).astype('int16')
    write(filename, fs, audio_int16)
    print(f"Đã lưu file: {filename}")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS speakers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

def save_embedding(name: str, embedding: np.ndarray):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Chuyển numpy array thành BLOB
    emb_blob = embedding.tobytes()
    c.execute("INSERT OR REPLACE INTO speakers (name, embedding) VALUES (?, ?)",
              (name, emb_blob))
    conn.commit()
    conn.close()

def load_all_embeddings():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM speakers")
    rows = c.fetchall()
    conn.close()
    data = []
    for name, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        data.append((name, emb))
    return data

def remove_speaker():
    name = input("Nhập tên người dùng cần xóa: ").strip()
    if delete_user(name):
        print(f"Đã xóa user `{name}` khỏi database.")
    else:
        print(f"Không tìm thấy user `{name}` để xóa.")


def show_users():
    users = list_users()
    if users:
        print("Danh sách user đã enroll:")
        for u in users:
            print(f" - {u}")
    else:
        print("Hiện chưa có user nào trong database.")



#=====================Test API======================
def main():
    init_db()
    while True:
        print("\n--- Menu ---")
        print("1) Enroll (ghi âm & lưu)")
        print("2) Authenticate (ghi âm & so sánh)")
        print("3) Delete user")
        print("4) List users")
        print("0) Thoát")
        choice = input("Chọn [0-4]: ").strip()
        if choice == "1":
            enroll_speaker()
        elif choice == "2":
            authenticate_speaker()
        elif choice == "3":
            remove_speaker()
        elif choice == "4":
            show_users()
        elif choice == "0":
            break
        else:
            print("Lựa chọn không hợp lệ.")


if __name__ == "__main__":
    main()
