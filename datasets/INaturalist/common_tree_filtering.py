import json
import time
import pandas as pd
from pyinaturalist import get_observation_species_counts

def fetch_woody_species(root_ids):
    all_tree_data = {}
    total_species_count = 0

    # 초본(Grass/Herb)이 많은 계통에서 제외할 대표적인 속(Genus) 목록
    excluded_genera = [
        'Trifolium', 'Medicago', 'Vicia', 'Fragaria', 'Rubus', 'Rosa', 
        'Potentilla', 'Viola', 'Mentha', 'Lamium', 'Veronica', 'Plantago'
    ]


    for r_id in root_ids:
        try:
            # 1. 해당 Root 하위의 종별 관측 수 가져오기 (상위 100종)
            results = get_observation_species_counts(
                taxon_id=r_id,
                quality_grade='research', # 신뢰도 높은 데이터만
                rank='species',
                per_page=100
            )

            taxon_name = ""
            species_list = []

            for obs in results['results']:
                taxon = obs['taxon']
                s_name = taxon['name']
                c_name = taxon.get('preferred_common_name', 'N/A')
                
                # 풀/꽃 필터링: 제외 리스트에 속명이 포함되면 스킵
                genus_name = s_name.split()[0]
                if genus_name in excluded_genera:
                    continue

                # 종 정보 저장
                species_list.append({
                    'taxon_id': taxon['id'],
                    'scientific_name': s_name,
                    'common_name': c_name,
                    'observations_count': obs['count'],
                    'rank': taxon['rank']
                })

            # 첫 번째 결과에서 상위 계통 이름 확인
            if species_list:
                # API 호출을 줄이기 위해 첫 번째 종의 조상 정보 활용 (간략화)
                root_name = f"Root_{r_id}" 
                all_tree_data[r_id] = {
                    'root_id': r_id,
                    'species_count': len(species_list),
                    'species': species_list
                }
                total_species_count += len(species_list)
                print(f"ID {r_id}: {len(species_list)}개 종 수집 완료")

            time.sleep(0.5) # API 매너

        except Exception as e:
            print(f"ID {r_id} 수집 오류: {e}")

    print(f"\n총 수집된 나무 종 수: {total_species_count}개")
    return all_tree_data

target_roots = [47562, 47374, 70279, 47853, 47729]

# 실행
master_tree_data = fetch_woody_species(target_roots)

# JSON 파일로 저장
output_path = "common_tree.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(master_tree_data, f, indent=2, ensure_ascii=False)

print(f"최종 데이터가 '{output_path}'에 저장되었습니다.")