import json
import os
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv



CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(CURRENT_DIR, ".env"))
GEMINI_API_KEY = os.getenv("Gemini_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-pro')

CSV_PATH = os.path.join(CURRENT_DIR, "NEON_dataset.csv")
CARBON_CSV_PATH = os.path.join(CURRENT_DIR, "NEON_dataset_with_carbon.csv")
L3_JSON_PATH = os.path.join(CURRENT_DIR, "l3_dataset/l3_dataset.json")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "l4_dataset/l4_dataset.json")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def safe_eval(val):
    try:
        if isinstance(val, str):
            return eval(val)
        return val
    except:
        return []

def analyze_tree_geometry(height, area, dbh):
    # 나무의 기하학적 형태 분석 (침엽수 vs 활엽수)
    if area <= 0: return "Unknown", 0
    radius = math.sqrt(area / math.pi)
    crown_diameter = radius * 2
    
    # 형태 지수 (Height / Crown Diameter)
    # 2.0 이상이면 좁고 긴 형태 -> 침엽수 추정
    shape_index = height / crown_diameter if crown_diameter > 0 else 0
    tree_type = "Conifer" if shape_index > 2.0 else "Broadleaf"
    
    # 형상비 (H/D Ratio)
    # 60~70: 건강한 상태, 80이상: 풍해나 설해에 매우 취약해서 간벌이 필요(간벌을 통해 흉고직경을 증대시켜 형상비를 낮추는 것이 필요)
    hd_ratio = (height * 100) / dbh if dbh > 0 else 0
    
    return tree_type, hd_ratio

def analyze_stand_details(row):
    # 한 타일의 모든 통계 데이터 추출
    heights = safe_eval(row['individual_heights'])
    areas = safe_eval(row['individual_crown_areas'])
    dbhs = safe_eval(row['individual_dbhs'])
    carbons_a = safe_eval(row['individual_carbon_annual'])
    carbons_s = safe_eval(row['individual_carbon_stored'])
    
    if not heights: return None
    
    # 그룹별 데이터 저장소
    groups = {
        "Conifer": {'h': [], 'a': [], 'd': [], 'c_s': [], 'c_a': [], 'hd': []},
        "Broadleaf": {'h': [], 'a': [], 'd': [], 'c_s': [], 'c_a': [], 'hd': []}
    }
    
    for i in range(len(heights)):
        h, a, d = heights[i], areas[i], dbhs[i]
        c_a, c_s = carbons_a[i], carbons_s[i]
        
        t_type, hd_ratio = analyze_tree_geometry(h, a, d)
        
        g = groups[t_type]
        g['h'].append(h)
        g['a'].append(a)
        g['d'].append(d)
        g['c_s'].append(c_s)
        g['c_a'].append(c_a)
        g['hd'].append(hd_ratio)
        
    # 통계 요약 함수 (최대/최소/평균/합계 모두 포함)
    def summarize(data):
        count = len(data['h'])
        if count == 0: return None
        return {
            "count": count,
            "height": {
                "mean": round(np.mean(data['h']), 2),
                "max": round(np.max(data['h']), 2),
                "min": round(np.min(data['h']), 2)
            },
            "crown_area": {
                "mean": round(np.mean(data['a']), 2),
                "total": round(np.sum(data['a']), 2)
            },
            "dbh": {
                "mean": round(np.mean(data['d']), 2),
                "max": round(np.max(data['d']), 2)
            },
            "carbon": {
                "total_stored": round(np.sum(data['c_s']), 2),
                "total_annual": round(np.sum(data['c_a']), 2),
                "per_tree_efficiency": round(np.mean(data['c_a']), 2) # 그루당 효율
            },
            "hd_ratio": round(np.mean(data['hd']), 1)
        }

    return {
        "conifer": summarize(groups['Conifer']),
        "broadleaf": summarize(groups['Broadleaf']),
        "total_trees": len(heights),
        "total_carbon_stored": round(sum(carbons_s), 2),
        "total_carbon_annual": round(sum(carbons_a), 2)
    }


def create_expert_report_prompt(tile_id, year, analysis):
    
    # 데이터 텍스트화 함수
    def format_stats(name, stats):
        if not stats: return f"- {name}: 식별되지 않음."
        return f"""
        [{name} 그룹 분석]
        - 개체수: {stats['count']}본
        - 수고(Height): 평균 {stats['height']['mean']}m (범위: {stats['height']['min']}~{stats['height']['max']}m)
        - 흉고직경(DBH): 평균 {stats['dbh']['mean']}cm (최대: {stats['dbh']['max']}cm)
        - 수관면적(Crown Area): 평균 {stats['crown_area']['mean']}m², 총 피복면적 {stats['crown_area']['total']}m²
        - 탄소 지표: 총 저장량 {stats['carbon']['total_stored']}kg, 연간 흡수량 {stats['carbon']['total_annual']}kg
        - 흡수 효율: 그루당 연간 {stats['carbon']['per_tree_efficiency']}kg 흡수
        - 평균 형상비(H/D): {stats['hd_ratio']}
        """

    conifer_txt = format_stats("침엽수(Conifer)", analysis['conifer'])
    broadleaf_txt = format_stats("활엽수(Broadleaf)", analysis['broadleaf'])
    
    # 전체 비중 계산
    c_ratio = 0
    if analysis['conifer']:
        c_ratio = round(analysis['conifer']['count'] / analysis['total_trees'] * 100, 1)
    b_ratio = 100 - c_ratio

    prompt = f"""
    당신은 산림 자원 통계 분석가(Forest Statistics Analyst)입니다.
    제공된 정밀 측정 데이터를 바탕으로, 주관적 의견을 배제하고 철저히 데이터에 입각한 '산림 자원 현황 상세 보고서'를 작성해 주세요.

    [분석 대상지 개요]
    - ID: {tile_id} (조사연도: {year})
    - 전체 임목 본수: {analysis['total_trees']}본
    - 임상 구성: 침엽수 {c_ratio}% vs 활엽수 {b_ratio}%
    - 총 탄소 저장량: {analysis['total_carbon_stored']} kg CO2eq
    - 연간 탄소 흡수량: {analysis['total_carbon_annual']} kg/yr

    [상세 통계 데이터]
    - 침엽수: {conifer_txt}
    - 활엽수: {broadleaf_txt}

    [보고서 작성 요청사항]
    다음 3가지 섹션으로 구성된 데이터 중심의 보고서를 작성하세요.

    1. 임분 구조 및 생육 현황 (Stand Structure Analysis)
    - 침엽수와 활엽수의 수고(Height) 분포 범위(Min-Max)와 평균 직경(DBH)을 비교하여, 숲의 층위 구조(상층/하층)를 분석하세요.
    - 수관 면적(Crown Area) 데이터를 인용하여, 숲의 울폐도(Canopy Closure)나 생육 공간의 여유 정도를 수치적으로 설명하세요.
    - 수종 분류 근거: 나무의 높이 대비 수관폭 비율(Slenderness)을 기준으로 기하학적 분류를 수행했음을 명시하세요.

    2. 활력도 및 형상비 정밀 진단 (Vitality & Stability)
    - 형상비(H/D Ratio) 수치를 인용하여 나무의 물리적 안정성을 진단하세요. (일반적으로 80 이상이면 세장(Slender)한 상태임)
    - 형상비 60~70: 건강한 상태, 80이상: 풍해나 설해에 매우 취약해서 간벌이 필요(간벌을 통해 흉고직경을 증대시켜 형상비를 낮추는 것이 필요)을 기반으로 서술하세요.
    - 형상비를 통해 숲의 현재 발달 단계(성숙림/유령림 등)를 통계적으로 추론하세요.

    3. 탄소 흡수 효율성 비교 (Carbon Sequestration Efficiency)
    - 두 수종 그룹(침엽수 vs 활엽수)의 '개체당 연간 평균 흡수량(Efficiency)'을 비교하여, 이 구역에서 어떤 수종이 탄소 흡수에 더 기여하고 있는지 분석하세요.
    - 전체 탄소 저장량과 연간 흡수량을 승용차 배출량 등 이해하기 쉬운 지표와 비교하거나, 생태학적 가치를 수치 중심으로 서술하세요.

    4. 종합 경영 제언 (Management Prescription)
    - 데이터에 기반하여 [보존 / 솎아베기(간벌) / 수종 갱신] 중 하나의 처방을 내리고 그 이유를 설명하세요.

    5. 출력 요구
    - 제공된 분석 대상지 개요와 상세 통계 데이터를 모두 출력하세요.
    - 나무를 지칭할때는 항상 뒤에 '[SEG]' 를 붙이세요. (예: 침엽수보다 활엽수의 비율이 높고, 이 나무들은 [SEG] 형상비가 ... (생략))
    """
    return prompt


def generate_l4_dataset():
    if not os.path.exists(L3_JSON_PATH):
        print(f"Error: {L3_JSON_PATH} not found")
        return
    with open(L3_JSON_PATH, 'r') as f:
        l3_data = json.load(f)

    l3_map = {
        item['id']: {
            'image': item['image'], 
            'mask_path': item.get('mask_path')
        }
        for item in l3_data   
    }
    print(f"Loaded {len(l3_map)} entries from Level-3.")
    
    if not os.path.exists(CSV_PATH) or not os.path.exists(CARBON_CSV_PATH):
        print("CSV Files Missing.")
        return

    df_org = pd.read_csv(CSV_PATH)
    df_carbon = pd.read_csv(CARBON_CSV_PATH)
    
    df_merged = pd.merge(df_org, df_carbon, on='tile_id', how='inner')
    
    # [테스트 설정] 딱 10개만 슬라이싱 / [실제 설정] 딱 1000개 슬라이싱
    df_test = df_merged.iloc[:10]
    
    l4_results = []
    print(f"Generating Expert Reports for {len(df_test)} tiles...")

    save_interval = 50
    
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
        tile_id = row['tile_id']

        if tile_id not in l3_map:
            continue

        l3_info = l3_map[tile_id]
        
        try:
            # 1. 정밀 분석
            analysis = analyze_stand_details(row)
            if not analysis: continue

            # 2. 프롬프트 생성
            prompt = create_expert_report_prompt(tile_id, row['year'], analysis)
            
            # 3. Gemini 생성
            response = model.generate_content(prompt)
            report_text = response.text
            
            # 4. 저장
            l4_entry = {
                "id": tile_id + "_L4",
                "image": l3_info['image'],
                "year": int(row['year']),
                "metadata": analysis,
                "conversations": [
                    {
                        "from": "human", 
                        "value": "제공된 산림 통계 데이터를 바탕으로 수종별(침엽수 vs 활엽수) 생육 특성과 탄소 흡수 효율을 상세 비교하는 보고서를 작성해줘."
                    },
                    {
                        "from": "gpt", 
                        "value": report_text
                    }
                ]
            }
            l4_results.append(l4_entry)

            if (len(l4_results)) % save_interval == 0:
                with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                    json.dump(l4_results, f, indent=4, ensure_ascii=False)

            time.sleep(1.2) # API 쿨타임
            
        except Exception as e:
            print(f"Error processing {tile_id}: {e}")
            continue

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(l4_results, f, indent=4, ensure_ascii=False)
        
    print(f"Expert L4 Dataset ({len(l4_results)} samples) Created! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_l4_dataset()