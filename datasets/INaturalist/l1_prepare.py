import json
import time
from pyinaturalist import get_observations, get_taxa_by_id
from tqdm import tqdm

input_path = "common_tree.json"
output_path = "l1_prepare_data.json"

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

all_species_to_process = []
for root_id, info in data.items():
    if 'species' in info:
        all_species_to_process.extend(info['species'])

print(f"처리할 종의 수: {len(all_species_to_process)}개")

final_data = []

API_SLEEP = 1.0

for idx, species_info in enumerate(tqdm(all_species_to_process)):
    taxon_id = species_info['taxon_id']
    scientific_name = species_info['scientific_name']
    common_name = species_info.get('common_name', '')
    species_name = f"{scientific_name}"
    if common_name and common_name != "N/A":
        species_name += f" ({common_name})"

    entry = {
        "taxonomy": {
            "Kingdom" : None,
            "Phylum": None,
            "Class": None,
            "Order": None,
            "Family": None,
            "Genus": None,
            "Species": species_name
        },
        "image_urls" : [],
        "tree_type": "None" # 기본 값
    }

    try:
        taxon_details = get_taxa_by_id(taxon_id)
        tree_type = ""

        if taxon_details and 'results' in taxon_details and taxon_details['results']:
            curr_taxon = taxon_details['results'][0]
            ancestors = curr_taxon.get('ancestors', [])

            for anc in ancestors:
                rank = anc['rank'].capitalize()
                name = anc['name']

                if rank in entry['taxonomy']:
                    entry['taxonomy'][rank] = name

                if rank == 'Class':
                    if name == 'Pinopsida':
                        tree_type = "Conifer (침엽수)"
                    elif name == 'Magnoliopsida':
                        tree_type = "Broad-leaf (활엽수)"
                    elif name == 'Ginkgoopsida':
                        tree_type = "Ginkgo (은행나무)"
                    else:
                        tree_type = "None"

        entry['tree_type'] = tree_type

               
        observations = get_observations(
            taxon_id=taxon_id,
            quality_grade='research',
            photos=True,
            per_page=20
        )

        image_urls = []
        if 'results' in observations:
            for obs in observations['results']:
                if obs['photos']:
                    # medium_url이 일반적, 원본 - original_url 사용 가능(용량 큼)
                    img_url = obs['photos'][0]['url'].replace('square', 'medium')
                    image_urls.append(img_url)

        entry['image_urls'] = image_urls

        if image_urls:
            final_data.append(entry)

        time.sleep(API_SLEEP)

    except Exception as e:
        print(f"{taxon_id} ({scientific_name}) 처리 중 오류 발생: {e}")
        time.sleep(5)

    if (idx + 1) % 50 == 0: # 중간 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
    
with open(output_path, 'w', encoding='utf-8') as f: # 최종 저장
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print(f"총 {len(final_data)}개의 데이터 수집 완료")


