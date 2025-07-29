#!/usr/bin/env python3
import sys
import os
import subprocess
import time

SIP_CONF_PATH = "/etc/asterisk/sip.conf"  # Thay đổi nếu file bạn ở vị trí khác

def read_sip_conf():
    if not os.path.exists(SIP_CONF_PATH):
        print(f"Không tìm thấy file: {SIP_CONF_PATH}")
        sys.exit(1)
    with open(SIP_CONF_PATH, "r") as f:
        return f.read()

def write_sip_conf(content):
    with open(SIP_CONF_PATH, "w") as f:
        f.write(content)

def add_user(username, secret):
    conf = read_sip_conf()
    if f"[{username}]" in conf:
        print(f"User [{username}] đã tồn tại.")
        return

    user_block = f"""
[{username}]
type=friend
host=dynamic
secret={secret}
context=internal
""".strip()

    conf += "\n\n" + user_block
    write_sip_conf(conf)
    print(f"Đã thêm user [{username}].")
    reload_asterisk()

def delete_user(username):
    conf = read_sip_conf()
    lines = conf.splitlines()
    new_lines = []
    in_user_block = False
    removed = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"[{username}]"):
            in_user_block = True
            removed = True
            continue
        if in_user_block and stripped.startswith("[") and stripped.endswith("]"):
            in_user_block = False
        if not in_user_block:
            new_lines.append(line)

    if not removed:
        print(f"⚠️ Không tìm thấy user [{username}].")
    else:
        new_conf = "\n".join(new_lines)
        write_sip_conf(new_conf)
        print(f"Đã xóa user [{username}].")
        reload_asterisk()

def reload_asterisk():
    try:
        subprocess.run(["sudo", "asterisk", "-rx", "sip reload"], check=True)
        print("Đã reload Asterisk (sip.conf).")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi reload Asterisk: {e}")

def print_usage():
    print("Cách dùng:")
    print("  python sip_user_manager.py add <username> <secret>")
    print("  python sip_user_manager.py delete <username>")

if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print_usage()
    #     sys.exit(1)

    # action = sys.argv[1]
    # if action == "add" and len(sys.argv) == 4:
    #     add_user(sys.argv[2], sys.argv[3])
    # elif action == "delete" and len(sys.argv) == 3:
    #     delete_user(sys.argv[2])
    # else:
    #     print_usage()
    add_user("7003", "7003")
    add_user("7004", "7004")

    time.sleep(10)
    delete_user("7003")
