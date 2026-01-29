import numpy as np

# generate_l3에 넣기 위한 전처리
def neon_l2_bridge(stats, tile_id_suffix=""):
    
    heights = np.array(stats['heights']) # 타일(1024x1024) 내 존재하는 나무들의 높이 정보 배열
    areas = np.array(stats['areas']) # 타일(1024x1024) 내 존재하는 나무들의 수관 면적 정보 배열
    dbhs = np.array(stats['dbhs']) # 타일(1024x1024) 내 존재하는 나무들의 직경 정보 배열

    tree_count = len(heights) # # 타일(1024x1024) 내 존재하는 나무의 수

    if tree_count == 0:
        return None
    
    # 탄소흡수율 측정 오차범위를 최소화하기 위한 평균값 산출(QMD)
    # 평균 높이 (산술 평균)
    avg_h = np.mean(heights)

    # 평균 수관 면적 
    avg_a = np.mean(areas)

    # 평균 직경(QMD: Quadratic Mean Diameter)
    # 바이오매스 공식(D^2)의 오차를 보정하기 위해 제곱평균제곱근 사용
    avg_d_qmd = np.sqrt(np.sum(dbhs**2) / tree_count)

    human_query = "이 구역(Tile) 나무[SEG]의 탄소 흡수 효율과 생태학적 가치를 계산 근거와 함께 분석해 주세요."

    prompt = f"""
    당신은 산림 생태학자이자 탄소 흡수율 측정 전문가입니다. 
    제공된 이미지는 1024x1024 해상도의 드론 촬영 숲 구역(Tile)입니다.
    제공된 이미지와 아래의 실측 수치를 바탕으로 특정 나무[SEG]의 탄소 흡수 효율을 분석하세요.

    아래 제공된 '구역 통계 데이터'는 이 이미지 안에 있는 모든 개별 나무들을 정밀 측정한 결과의 요약값입니다.
    이를 바탕으로 이 구역 전체의 탄소 흡수 효율을 분석하세요.

    [실측 기반 구역(Tile) 통계 데이터]
    - 식별된 나무 수: {tree_count} 그루
    - 평균 나무 높이(Height): {avg_h:.3f} m
    - 평균 수관 투영 면적(Crown Area): {avg_a:.3f} m2
    - 평균 나무 직경: {avg_d_qmd:.23f} cm    (Jucker et al. 모델 적용) 

    [분석 가이드 및 필수 적용 공식]

    1. 수관 면적 기반 계산: 
       - 1그루의 연간 탄소(C) 흡수량 = 수관 면적(m2) * 0.28 kg C / m2 / 년
       - 예: 수관 면적이 10m2인 나무는 연간 약 2.8kg의 탄소를 흡수함.
       - 구역(Tile) 전체 탄소 흡수량 = 1구루의 연간 탄소(C) 흡수량 * {tree_count} (kg/년)
    
    2. 개체별 생장 기반 정밀 공식:
       - 단계별 계산 과정
       - 지상부 Biomass(AGB) 추정 =  0.25 * (D^2) * H (D: 나무의 직경, H: 나무의 높이)
       - Total Biomass(TB) = 지상부 바이오매스(AGB) * 1.2
       - 건조 중량 변환 = TB * 72.5% (수분 제외 건조 무게)
       - 탄소 함량 산출 = 건조 중량 * 50%
       - 최종 CO2 환산 = 산출된 탄소량(C) * 3.67
       - 구역 전체 저장량 = 최종 CO2 환산 * {tree_count} (kg CO2eq)

    공식1: 연간 탄소 흡수량 - 나무가 1년 동안 새롭게 빨아들인 탄소의 양
    공식2: 평생 흡수한 탄소양(kg) - 나무가 자라면서 대기 중에서 빨아들여 자기 몸(줄기, 뿌리 등)으로 만든 탄소의 전체 무게

    [수행 과제]
    1. 제시된 Bounding Box 영역 내의 내무를 식별하고 [SEG] 토큰을 사용하여 지칭하세요.
    2. 위의 두 공식을 단계별(Step-by-step)로 적용하여 계산하세요.
    3. 계산 과정의 각 단계를 논리적으로 설명하고, 공식1, 공식2의 설명에 따라 연간 탄소 흡수량과 총 탄소 흡수량을 설명하세요.
    4. 최종적으로 이 나무 [SEG]가 탄소 중립에 기여하는 가치를 캡션으로 작성하세요

    [출력 형식(JSON)]
    {{
        "dense_caption": "이 구역은 약 {tree_count}그루의 나무가 밀집해있으며, 나무[SEG]들은 평균적으로 높이 {avg_h:.3f}m, 평균적으로 수관 면적 {avg_a:.3f}m2의 개체로... (중략) ...따라서 해당 구역(Tile)은 연간 약 X kg의 CO2를 흡수할 것이며, 총 탄소 흡수량은 Y kg으로 분석됩니다.",
        "objects": {{"forest_tile": [1024, 1024]}}
    }}
    """
    return {
        "id": f"{tile_id_suffix}", 
        "image": f"{tile_id_suffix}.jpg",
        "human_query": human_query,
        "prompt": prompt,
        "landmark": "Forest Tile"
        }