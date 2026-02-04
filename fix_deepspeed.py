import os

# 에러 로그에서 가져온 정확한 절대 경로
target_file = "/home/sbosung1789/miniconda3/envs/glamm/lib/python3.10/site-packages/deepspeed/elasticity/elastic_agent.py"

print(f"🔧 타겟 파일 경로: {target_file}")

if not os.path.exists(target_file):
    print("❌ 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
    exit(1)

# 교체할 코드 (소켓 함수 수동 구현 + 로깅 설정)
new_code_block = """
import logging
import socket

# [Manual Patch] _get_socket_with_port 구현 (PyTorch 2.x 호환용)
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

# 파일 읽기
with open(target_file, "r") as f:
    lines = f.readlines()

new_lines = []
patched = False

for line in lines:
    # 1. 문제가 되는 import 구문들 제거
    if "from torch.distributed.elastic.agent.server.api" in line and ("log" in line or "_get_socket_with_port" in line):
        # 중복 패치 방지: 이미 패치된 코드가 있다면 건너뜀
        if patched: 
            continue
            
        print("   ✅ 문제의 Import 구문을 발견하여 교체합니다.")
        new_lines.append(new_code_block)
        patched = True
    
    # 2. 이미 패치된 코드(우리가 넣은 함수 정의)가 보이면 중복해서 넣지 않음
    elif "[Manual Patch]" in line:
        print("   ⚠️ 이미 패치가 적용된 파일 같습니다. 내용을 덮어씁니다.")
        patched = True
        new_lines.append(line)
        
    # 3. 그 외 정상 코드는 유지
    else:
        new_lines.append(line)

# 파일 쓰기
with open(target_file, "w") as f:
    f.writelines(new_lines)

print("🎉 DeepSpeed 수정 완료! 이제 다시 학습 스크립트(run_a100_forest.sh)를 실행하세요.")