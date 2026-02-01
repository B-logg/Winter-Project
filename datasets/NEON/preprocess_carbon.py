import pandas as pd
import numpy as np
import ast

INPUT_PATH = "NEON_dataset.csv"  # 로컬 경로
OUTPUT_PATH = "NEON_dataset_with_carbon.csv" # 새로 생성될 파일

TEST_MODE = False

def calculate_carbon_data(row):

    # 한 이미지(Row) 내의 모든 나무에 대해 탄소량을 계산하여 리스트로 반환

    try:
        # 문자열 리스트 파싱 ("[1, 2]" -> [1, 2])
        heights = np.array(ast.literal_eval(row['individual_heights']) if isinstance(row['individual_heights'], str) else row['individual_heights'])
        areas = np.array(ast.literal_eval(row['individual_crown_areas']) if isinstance(row['individual_crown_areas'], str) else row['individual_crown_areas'])
        dbhs = np.array(ast.literal_eval(row['individual_dbhs']) if isinstance(row['individual_dbhs'], str) else row['individual_dbhs'])
        
        # 1. 연간 탄소 흡수량 (수관 면적 기반)
        # 공식: 면적 * 0.28
        carbon_annual = areas * 0.28
        
        # 2. 총 탄소 저장량 (Biomass 기반)
        # AGB = 0.25 * D^2 * H
        # TB = AGB * 1.2 -> Dry = TB * 0.725 -> C = Dry * 0.5 -> CO2 = C * 3.67
        agb = 0.25 * (dbhs**2) * heights
        total_biomass = agb * 1.2
        dry_weight = total_biomass * 0.725
        carbon_content = dry_weight * 0.5
        carbon_stored_co2 = carbon_content * 3.67

        sum_annual = np.sum(carbon_annual)
        sum_stored = np.sum(carbon_stored_co2)
        
        # 소수점 4째자리까지 반올림 후 리스트로 변환
        return pd.Series([
            list(np.round(carbon_annual, 4)), 
            list(np.round(carbon_stored_co2, 4)),
            np.round(sum_annual, 4),
            np.round(sum_stored, 4)
        ])
        
    except Exception as e:
        # 데이터가 비어있거나 에러날 경우 빈 리스트 반환
        return pd.Series([[], [], 0.0, 0.0])
    
if __name__ == "__main__":

    df = pd.read_csv(INPUT_PATH)

    if TEST_MODE:
        df = df.head(5)
    
    print(f"Calculating Carbon Data for {len(df)} images...")

    carbon_cols = [
            'individual_carbon_annual', # 이미지 내 개별 나무 탄소 흡수량 (수관 면적 기반)
            'individual_carbon_stored', # 이미지 내 개별 나무 탄소 흡수량 (Biomass 기반)
            'sum_carbon_annual',        # 이미지 내 총 탄소 흡수량(수관 면적 기반)
            'sum_carbon_stored'         # 이미지 내 총 탄소 흡수량(Biomass 기반)
        ]
    
    df[carbon_cols] = df.apply(calculate_carbon_data, axis=1)

    output_df = df[['tile_id'] + carbon_cols]

    output_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print("Calculation Done!")


