# 한 이미지에 대해서 흩어져 있는 모든 정보를 CSV 형태로 모으는 코드
# BART, GRSM, HARV 모두 한 CSV 파일로 병햡

import geopandas as gpd
import pandas as pd
import os
import glob
import numpy as np

def estimate_dbh(height, area):
    """ 
        나무의 형태학적 비율(Height/Crown diameter)을 분석하여 수종(활엽수 vs 침엽수)을 판별하고
        Jucker et al. 공식으로 나무의 직경 추론
    """

    # 1. 수관 직경(Crown diameter) 계산
    cd = 2 * np.sqrt(area / np.pi)
    
    # 2. 수종 판별(활엽수 vs 침엽수)
    h_cd_ratio = height / cd
    is_conifer = h_cd_ratio > 2.5

    # 3. Jucker et al. 계수 설정(활엽수 vs 침엽수)
    alpha, beta = (-1.164, 0.706) if is_conifer else (-1.130, 0.755) 
    
    # 4. DBH(나무 직경) 추론 - Jucker 공식 적용 (DBH 단위: cm)
    log_d = alpha + beta * np.log(height * cd)
    return np.exp(log_d), "Conifer" if is_conifer else "Broadleaf"

base_path = "/Users/bosung/Desktop/NEON"
output_path = "/Users/bosung/Desktop/GLaMM/datasets/NEON"
sites = ['BART', 'GRSM', 'HARV']
output_csv = os.path.join(output_path, "NEON_dataset")

all_data = []

for site in sites:
    site_path = os.path.join(base_path, site)
    # .shp 파일들만 리스트업
    shp_files = glob.glob(os.path.join(site_path, "*.shp"))

    for shp_file in shp_files:
        # 파일 이름에서 정보 추출하기(예: 2022_GRSM_6_270000_3937000_image.shp)
        filename = os.path.basename(shp_file)
        parts = filename.split('_')

        # geopandas 도구가 .shp 뿐만 아니라 이름이 같은 다른 확장자를 묶어줌
        # 여러 확장자에 흩어져있는 한 이미지에 대한 여러 정보들을 하나의 데이터프레임으로 합쳐줌
        gdf = gpd.read_file(shp_file)

        # 수관 면적 계산
        calculated_areas = gdf.geometry.area

        # 나무별로 수종 판별 및 DBH(나무 직경) 추론
        dbhs_and_types = [estimate_dbh(h, a) for h, a in zip(gdf['height'], calculated_areas)]
        dbhs = [item[0] for item in dbhs_and_types]
        tree_types = [item[1] for item in dbhs_and_types]

        # 한 장의 이미지에 대해 흩어져있는 모든 정보 병합
        summary = {
            'site': site,
            'tile_id': filename.replace(".shp", ""),
            'year': parts[0],
            'utm_e': parts[3], # 중앙 자오선으로부터 동쪽으로 몇 미터 떨어져있는지
            'utm_n': parts[4], # 적도로부터 북쪽으로 몇 미터 떨어져있는지

            # 이미지 전체 통계 정보
            'tree_count': len(gdf), # 이미지 내 나무 수
            'avg_height': gdf['height'].mean(), # 나무 평균 높이
            'max_height': gdf['height'].max(), # 나무 최대 높이
            'total_crown_area': calculated_areas.sum(), # 총 수관 면적
            'avg_crown_area': calculated_areas.mean(), # 평균 수관 면적

            # 개별 나무 정보
            'bboxes': gdf.geometry.apply(lambda g: list(g.bounds)).tolist(), # 개별 나무 영역 좌표
            'individual_heights': gdf['height'].tolist(), # 개별 나무 높이
            'individual_crown_area': calculated_areas.tolist(), # 개별 나무 수관 면적
            'individual_dbhs': dbhs, # 개별 나무 직경
            'individual_tree_types': tree_types # 개별 나무 판별(활엽수 vs 침엽수)
        }

        all_data.append(summary)

df = pd.DataFrame(all_data)
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"통합 완료, 저장 위치: {output_csv}")
print(df.head(10))