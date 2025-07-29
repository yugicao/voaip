#!/usr/bin/env python3
import subprocess
import time
import os

# Change to your folder
working_dir = "/home/cak/nckh/yugicao_VoAI"

try:
    subprocess.run(["sudo", "systemctl", "restart", "asterisk"], check=True)
    print("Đã restart Asterisk thành công.")
except subprocess.CalledProcessError as e:
    print("Lỗi khi restart Asterisk:", e)
    exit(1)

time.sleep(1)



try:
    subprocess.Popen([
        "gnome-terminal",
        "--", "bash", "-c",
        "sudo asterisk -rvvv; exec bash"
    ])
    print("Đã mở cửa sổ terminal mới để chạy 'asterisk -rvvv'.")
except FileNotFoundError:
    print("Không tìm thấy gnome-terminal. Bạn có thể thay bằng xterm hoặc terminal khác.")


time.sleep(1)

http_script_path = os.path.join(working_dir, "http_server.py")
try:
    subprocess.Popen([
        "gnome-terminal",
        "--", "bash", "-c",
        f"source {working_dir}/venv/bin/activate && python {http_script_path}; exec bash"
    ])
    print("Đã mở terminal mới để chạy 'http_server.py'.")
except FileNotFoundError:
    print("Không tìm thấy gnome-terminal. Hãy thay bằng terminal khác nếu cần.")


# python_venv = os.path.join(working_dir, "venv/bin/python")
# script_path = os.path.join(working_dir, "fastagi_server.py")


# time.sleep(1)

# try:
#     subprocess.run(
#         ["sudo", python_venv, script_path],
#         cwd=working_dir
#     )
# except Exception as e:
#     print("Lỗi khi chạy fastagi_server.py:", e)
#     exit(1)



