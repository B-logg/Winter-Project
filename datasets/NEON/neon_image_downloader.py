import pandas as pd
import neonutilities as nu
import os

# 1. 전처리한 CSV 파일 로드
csv_path = "/Users/bosung/Desktop/GLaMM/datasets/NEON/NEON_dataset"
df = pd.read_csv(csv_path)

# 2. 이미지가 저장될 루트 경로
save_root = "/Users/bosung/Desktop/NEON/images"
os.makedirs(save_root, exist_ok=True)

# 3. 사이트 및 연도별로 그룹화하여 효율적으로 다운로드
# RGB 이미지 상품 ID: DP3.30010.001
for (site, year), group in df.groupby(['site', 'year']):
    print(f"📥 {site} 지역 ({year}년) 이미지 다운로드 시도 중...")
    
    # 중복 제거된 동거/북거 좌표 리스트 추출
    eastings = group['utm_e'].astype(int).unique().tolist()
    northings = group['utm_n'].astype(int).unique().tolist()
    
    try:
        # 지정된 좌표와 교차하는 AOP 타일만 다운로드
        nu.by_tile_aop(
            dpid="DP3.30010.001",
            site=site,
            year=str(year),
            easting=eastings,
            northing=northings,
            savepath=save_root,
            check_size=False  # 다운로드 전 용량 확인 절차 생략
        )
    except Exception as e:
        print(f"{site} 다운로드 중 오류 발생: {e}")

print("\n모든 요청된 타일의 다운로드 프로세스가 완료되었습니다.")