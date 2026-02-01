import numpy as np

# generate_l3에 넣기 위한 전처리
def neon_l2_bridge(stats, tile_id_suffix=""):
    
    heights = np.array(stats['heights']) # 타일(1024x1024) 내 존재하는 나무들의 높이 정보 배열
    areas = np.array(stats['areas']) # 타일(1024x1024) 내 존재하는 나무들의 수관 면적 정보 배열
    dbhs = np.array(stats['dbhs']) # 타일(1024x1024) 내 존재하는 나무들의 직경 정보 배열

    
    tree_count = stats['tree_count'] # 타일(1024x1024) 내 존재하는 나무의 수
    if tree_count == 0: return None

    avg_h = stats['avg_height']
    avg_a = stats['avg_area']
    avg_d = stats['avg_diameter']
    sum_annual = stats['sum_carbon_annual']
    sum_stored = stats['sum_carbon_stored']

    human_query = "이 구역(Tile)에 붉은 박스로 표시된 나무 [SEG]들의 탄소 흡수 효율과 생태학적 가치를 분석해 주세요."

    prompt = f"""
    당신은 산림 생태학자이자 탄소 흡수율 측정 전문가입니다. 
    제공된 이미지는 1024x1024 해상도의 드론 촬영 숲 구역(Tile)입니다.

    [시각적 가이드]
    - 이미지에 빨간색 박스(Bounding Box)로 표시된 객체들이 분석 대상인 개별 나무들입니다.


    [정밀 분석 데이터 (Ground Truth)]
    - 이미 현장 정밀 조사를 통해 확보된 데이터입니다.
    - 아래 수치를 '사실'로 받아들이고, 직접 계산하지 말고 그대로 인용하여 설명하세요.

    1. 식생 밀도: 총 {tree_count} 그루 식별됨
    2. 평균 제원: 평균 높이 {avg_h:.1f}m, 평균 수관 면적 {avg_a:.1f}m2, 평균 나무 직경: {avg_d:.1f}cm (Jucker et al. 모델 적용)
    3. 탄소 분석 결과 (Total)
        - 연간 탄소 흡수량: 약 {sum_annual:.2f} kg/yr (수관 면적 모델 기반)
        - 총 탄소 저장량: 약 {sum_stored:.2f} kg CO2eq (Biomass 모델 기반)

    [용어 정의]
    연간 탄소 흡수량 - 나무가 1년 동안 새롭게 빨아들인 탄소의 양
    평생 흡수한 탄소양(kg) - 나무가 자라면서 대기 중에서 빨아들여 자기 몸(줄기, 뿌리 등)으로 만든 탄소의 전체 무게

    [작성 요청]
    1. Dense Caption: 위 데이터를 바탕으로 이 숲 구역의 탄소 흡수량을 설명하는 전문적인 캡션을 작성하세요.
    2. 논리적 서술: 붉은 박스로 식별된 {tree_count}그루의 나무들은..." 처럼 시각적 요소를 언급하며 시작하세요.
    3. [SEG] 토큰 사용: 붉은 박스로 식별된 나무들을 지칭할때는 뒤에 [SEG] 토큰을 반드시붙이세요.
    4. 수치 인용: 제공된 탄소 흡수량과 저장량 수치를 반드시 포함하여 서술하세요.
    5. 가치 서술: 최종적으로 이미지에 포함된 나무들이 탄소 중립에 기여하는 가치를 캡션으로 작성하세요

    [출력 형식(JSON)]
    {{
        "dense_caption": "이미지에는 붉은 박스로 표시된 {tree_count}그루의 나무 [SEG]가 밀집해있으며, 나무[SEG]들은 평균적으로 높이 {avg_h:.3f}m, 평균적으로 수관 면적 {avg_a:.3f}m2, 평균적으로 직경 {avg_d:.3f}의 개체로... (중략) ...따라서 해당 구역(Tile)은 연간 약 X kg의 CO2를 흡수할 것이며, 총 탄소 흡수량은 Y kg으로 분석됩니다.",
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