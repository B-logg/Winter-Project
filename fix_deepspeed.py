import os
import deepspeed
import sys

# 1. DeepSpeed가 설치된 실제 경로를 자동으로 찾습니다.
ds_root = os.path.dirname(deepspeed.__file__)
target_file = os.path.join(ds_root, "elasticity", "elastic_agent.py")

print(f"🔍 DeepSpeed 경로 감지됨: {target_file}")

if not os.path.exists(target_file):
    print("❌ 파일을 찾을 수 없습니다. DeepSpeed가 제대로 설치되었는지 확인해주세요.")
    sys.exit(1)

# 2. 교체할 코드 (소켓 함수 수동 구현)
new_code_block = """
import logging
import socket

# [Manual Patch] _get_socket_with_port 구현
def _get_socket_with_port():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        return sock
    except Exception as e:
        print(f"Error binding socket: {e}")
        raise e

log = logging.getLogger(__name__)
"""

# 3. 파일 수정 시작
try:
    with open(target_file, "r") as f:
        lines = f.readlines()

    modified_lines = []
    patched = False

    for line in lines:
        # 에러를 유발하는 import 문을 찾아서 교체
        if "from torch.distributed.elastic.agent.server.api" in line and "import" in line:
            # 이미 패치된 적이 있는지 확인 (중복 방지)
            if "_get_socket_with_port" in line and "def" not in line:
                modified_lines.append(new_code_block)
                patched = True
                print("   ✅ 문제의 Import 구문을 찾아 패치 코드로 교체했습니다.")
            elif "log" in line:
                 modified_lines.append(new_code_block)
                 patched = True
                 print("   ✅ (구버전) 문제의 Import 구문을 찾아 패치 코드로 교체했습니다.")
            else:
                # 혹시 모를 다른 import 구문은 유지하되, 우리가 패치하려는 대상이면 교체
                modified_lines.append(new_code_block)
                patched = True
        else:
            modified_lines.append(line)

    # 저장
    with open(target_file, "w") as f:
        f.writelines(modified_lines)

    if patched:
        print("🎉 수정 완료! 이제 다시 run_a100_forest.sh를 실행하세요.")
    else:
        print("⚠️ 수정할 부분을 찾지 못했습니다. 이미 수정되었거나 파일 내용이 예상과 다릅니다.")
        print("파일 내용을 직접 확인해보세요.")

except Exception as e:
    print(f"❌ 오류 발생: {e}")