






#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import sqlite3
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
DB_FILE = "voice_system.db"  # Cùng DB với http_server.py

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Parse query string
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)
            caller = query_params.get('caller', [''])[0]
            callee = query_params.get('callee', [''])[0]
            status = query_params.get('status', [''])[0]

            logging.info(f"Caller: {caller}, Callee: {callee}, Status: {status}")

            # Lưu vào database
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO call_status (caller, callee, status)
                VALUES (?, ?, ?)
            """, (caller, callee, status))
            conn.commit()
            conn.close()

            # Send HTTP 200 response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"result": "ok"}')
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            self.send_response(500)
            self.end_headers()

    def log_message(self, format, *args):
        # Sử dụng logging thay vì in ra console
        logging.info("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format%args))

def run(server_class=HTTPServer, handler_class=RequestHandler, port=4573):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info(f"Server running on port {port}...")
    httpd.serve_forever()

if __name__ == '__main__':
    run()


# #!/usr/bin/env python3

# from http.server import BaseHTTPRequestHandler, HTTPServer
# from urllib.parse import urlparse, parse_qs
# import os
# import time
# from infer import AntiSpoofing
# from datetime import datetime
# import sqlite3

# import soundfile as sf
# import numpy as np
# import librosa

# DB_FILE = "voice_system.db"

# detector = AntiSpoofing(
#     config_path="./config/AASIST.conf",
#     weights_path="./models/weights/AASIST.pth",
#     device="cpu"     # hoặc "cuda"
# )

# def update_result_to_db(phone, label):
#     conn = sqlite3.connect(DB_FILE)
#     cursor = conn.cursor()
#     cursor.execute("""
#         UPDATE users 
#         SET last_check_result = ?, last_check_time = ?
#         WHERE phone = ?
#     """, (label, datetime.now().isoformat(), phone))
#     conn.commit()
#     conn.close()

# class RequestHandler(BaseHTTPRequestHandler):
#     def do_GET(self):
#         parsed = urlparse(self.path)
#         if parsed.path == "/process_audio":
#             # query = parse_qs(parsed.query)
#             # filename = query.get('filename', [''])[0]
#             # full_path = f"/var/spool/asterisk/monitor/{filename}"
            
#             # print(f"\n[{time.ctime()}] NHẬN YÊU CẦU XỬ LÝ:")
#             # print(f"File: {full_path}")
#             # print(f"Kích thước: {os.path.getsize(full_path) if os.path.exists(full_path) else 0} bytes")
#             query = parse_qs(parsed.query)
#             filename = query.get('filename', [''])[0]
#             caller = query.get('caller', [''])[0]
#             callee = query.get('callee', [''])[0]
#             full_path = f"/var/spool/asterisk/monitor/{filename}"

#             print(f"\n[{time.ctime()}] NHẬN YÊU CẦU XỬ LÝ:")
#             print(f"File: {full_path}")
#             print(f"Kích thước: {os.path.getsize(full_path) if os.path.exists(full_path) else 0} bytes")
#             print(f"Caller: {caller}, Callee: {callee}")
            
#             if os.path.exists(full_path):
#                 label = self.deepfake_detect(full_path)
#                 if caller:
#                     update_result_to_db(caller, label)
#                 if callee:
#                     update_result_to_db(callee, label)
#                 os.remove(full_path)
#                 print("XỬ LÝ THÀNH CÔNG")
#                 self.respond(200, "OK")
#             else:
#                 print("FILE KHÔNG TỒN TẠI")
#                 self.respond(404, "File not found")
#         else:
#             self.respond(404, "Not Found")

#     def respond(self, code, message):
#         self.send_response(code)
#         self.send_header('Content-type', 'text/plain')
#         self.end_headers()
#         self.wfile.write(message.encode())
    
#     def deepfake_detect(self, path):
#         try:
#             res = detector.predict(path)
#             print(f"score={res['score']:.4f}, label={res['label']}")
#             # return True if (res['label'] == 'genuine') else False
#             return res['label']
#         except Exception as e:
#             print("Cannot deepfake detect:", e)


#     # def deepfake_detect(self, path):
#     #     try:
#     #         wav, sr = librosa.load(path, sr=16000, mono=True)
#     #         # sliding window: 4s window, 2s hop
#     #         win_len = sr * 4
#     #         hop_len = sr * 2
#     #         scores = []
#     #         for start in range(0, len(wav) - win_len + 1, hop_len):
#     #             segment = wav[start:start+win_len]
#     #             # save tmp segment
#     #             temp = "/tmp/seg.wav"
#     #             sf.write(temp, segment, sr)
#     #             res = detector.predict(temp)
#     #             scores.append(res['score'])
#     #         if not scores:
#     #             # nếu file <4s, pad về 4s rồi test
#     #             padded = np.pad(wav, (0, max(0, win_len-len(wav))), mode='constant')
#     #             sf.write("/tmp/seg.wav", padded, sr)
#     #             res = detector.predict("/tmp/seg.wav")
#     #             scores = [res['score']]
#     #         avg = float(np.mean(scores))
#     #         label = "genuine" if avg >= detector.threshold else "spoof"
#     #         print(f"Segments: {len(scores)}, Scores: {scores}, Avg: {avg:.4f}, Label: {label}")
#     #         return label
#     #     except Exception as e:
#     #         print("Cannot deepfake detect:", e)
#     #         return "error"


# if __name__ == '__main__':
#     port = 4573
#     server = HTTPServer(('0.0.0.0', port), RequestHandler)
#     print(f"[{time.ctime()}]HTTP SERVER ĐÃ SẴN SÀNG TRÊN PORT {port}")
#     try:
#         server.serve_forever()
#     except KeyboardInterrupt:
#         print("\nDỪNG SERVER")
#         server.server_close()


# #!/usr/bin/env python3
# from http.server import BaseHTTPRequestHandler, HTTPServer
# from urllib.parse import urlparse, parse_qs

# class RequestHandler(BaseHTTPRequestHandler):
#     def do_GET(self):
#         # Parse query string
#         parsed_url = urlparse(self.path)
#         query_params = parse_qs(parsed_url.query)
#         caller = query_params.get('caller', [''])[0]
#         callee = query_params.get('callee', [''])[0]
#         status = query_params.get('status', [''])[0]

#         print(f"Caller: {caller}, Callee: {callee}, Status: {status}")

#         # Send HTTP 200 response
#         self.send_response(200)
#         self.send_header('Content-type', 'application/json')
#         self.end_headers()
#         self.wfile.write(b'{"result": "ok"}')

#     def log_message(self, format, *args):
#         # Tắt log mặc định cho gọn terminal
#         return

# def run(server_class=HTTPServer, handler_class=RequestHandler, port=4573):
#     server_address = ('', port)
#     httpd = server_class(server_address, handler_class)
#     print(f"Server running on port {port}...")
#     httpd.serve_forever()

# if __name__ == '__main__':
#     run()



