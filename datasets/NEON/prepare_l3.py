import numpy as np

# generate_l3에 넣기 위한 전처리
def neon_l2_bridge(row, tree_idx=0, custom_bbox=None):
    
    site = row['site']
    tile_id = row['tile_id']

    # generate_l3에서 UTM 좌표를 정규화(0~1000)해서 넘겨줌 = custom_bbox
    if custom_bbox is not None:
        normalized_bbox = custom_bbox
    else: # 정규화된 좌표가 없는 경우 .. 
        raise ValueError(f"정규화된 좌표가 입력되지 않았습니다. Tile: {tile_id}, Tree: {tree_idx} ")
    
    h = row['individual_heights'][tree_idx] # 나무 높이
    a = row['individual_crown_areas'][tree_idx] # 나무 수관면적
    d = row['individual_dbhs'][tree_idx] # 나무 직경(추론값)
    tree_type = row['individual_tree_types'][tree_idx] # 나무 종류(활엽수 vs 침엽수, 추론값)

    human_query = "이 구역 나무[SEG]의 탄소 흡수 효율과 생태학적 가치를 계산 근거와 함께 분석해 주세요."

    prompt = f"""
    당신은 산림 생태학자이자 탄소 흡수율 측정 전문가입니다. 
    제공된 이미지와 아래의 실측 수치를 바탕으로 특정 나무[SEG]의 탄소 흡수 효율을 분석하세요.

    [실측 데이터 정보]
    - 측정할 나무 위치(Bounding Box): {normalized_bbox} (상대 좌표 0~1000 기준)
    - 측정할 나무 높이(Height): {h:.2f} m
    - 측정할 나무 수관 투영 면적(Crown Area): {a:.2f} m2
    - 측정할 나무 직경: {d:.2f} cm    (Jucker et al. 모델 적용)

    [분석 가이드 및 필수 적용 공식]

    1. 수관 면적 기반 계산: 
       - 연간 탄소(C) 흡수량 = 수관 면적(m2) * 0.28 kg C / m2 / 년
       - 예: 수관 면적이 10m2인 나무는 연간 약 2.8kg의 탄소를 흡수함.
    
    2. 개체별 생장 기반 정밀 공식:
       - 단계별 계산 과정
       - 지상부 Biomass(AGB) 추정: 0.25 * (D^2) * H (D: 나무의 직경, H: 나무의 높이)
       - Total Biomass(TB) = 지상부 바이오매스(AGB) * 1.2
       - 건조 중량 변환: TB * 72.5% (수분 제외 건조 무게)
       - 탄소 함량 산출: 건조 중량 * 50%
       - 최종 CO2 환산: 산출된 탄소량(C) * 3.67

    공식1: 연간 탄소 흡수량 - 나무가 1년 동안 새롭게 빨아들인 탄소의 양
    공식2: 평생 흡수한 탄소양(kg) - 나무가 자라면서 대기 중에서 빨아들여 자기 몸(줄기, 뿌리 등)으로 만든 탄소의 전체 무게

    [수행 과제]
    1. 제시된 Bounding Box 영역 내의 내무를 식별하고 [SEG] 토큰을 사용하여 지칭하세요.
    2. 위의 두 공식을 단계별(Step-by-step)로 적용하여 계산하세요.
    3. 계산 과정의 각 단계를 논리적으로 설명하고, 공식1, 공식2의 설명에 따라 연간 탄소 흡수량과 총 탄소 흡수량을 설명하세요.
    4. 최종적으로 이 나무 [SEG]가 탄소 중립에 기여하는 가치를 캡션으로 작성하세요

    [출력 형식(JSON)]
    {{
        "dense_caption": "이 나무[SEG]는 높이 {h:.1f}m, 수관 면적 {a:.1f}m2의 개체로... (중략) ...따라서 연간 약 X kg의 CO2를 흡수할 것으로 분석됩니다.",
        "objects": {{"target_tree": [중심_x, 중심_y]}}
    }}
    """
    return {
        "id": f"{tile_id}_{tree_idx}",
        "image": f"{tile_id}.jpg",
        "human_query": human_query,
        "prompt": prompt,
        "landmark": f"Tree at {normalized_bbox}"
        }