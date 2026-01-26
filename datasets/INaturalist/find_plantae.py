import pandas as pd
import os
import sys

# 파일 경로 설정
base_path = os.path.expanduser("~/inaturalist-open-data-20251227")
taxa_file = os.path.join(base_path, "taxa.csv")


taxa = pd.read_csv(
    taxa_file,
    sep='\t',
    usecols=['taxon_id', 'ancestry', 'name', 'rank', 'active'],
    dtype={'ancestry': str, 'name': str, 'active': bool}
)

# 식물계(47126)의 후손인 모든 데이터 필터링
"""
    47124: 소나무, 잣나무 등 대부분의 침엽수
    47150: 참나무, 단풍나무 등 대부분의 활엽수
    48370: 소철류(원시적인 나무 형태)
    48369: 은행나무류
    47163: 야자나무, 대나무 등(외떡잎 식물 중 나무 형태)
    ...
"""
plants = taxa[taxa['ancestry'].str.contains(r'(^|/)47126(/|$)', regex=True, na=False)]

# 주요 나무 그룹 추출
# 47128(소나무목), 47853(참나무목), 48374(단풍/무환자), 48375(버드나무), 48460(벚나무/장미목 중 목본)
tree_roots = ['47128', '47853', '48374', '48375', '48460', '48469', '48369']
root_pattern = '|'.join([f'(^|/){r}(/|$)' for r in tree_roots])


trees = plants[
    (plants['rank'].str.lower() == 'species') & 
    (plants['active'] == True) &
    (plants['ancestry'].str.contains(root_pattern, regex=True, na=False))
]


print(f"분석 결과 요약")
print(f"- 전체 식물 데이터: {len(plants):,}개\n")
print(f"- 필터링된 주요 나무 종: {len(trees):,}개\n")

tree_ids = trees['taxon_id'].unique().tolist()
with open("all_tree_ids.txt", "w") as f:
    f.write("\n".join(map(str, tree_ids)))

print("나무 종 ID 리스트가 'tree_taxon_ids.txt'에 저장되었습니다.")