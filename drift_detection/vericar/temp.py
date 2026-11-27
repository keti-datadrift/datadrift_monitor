import os
import shutil
import re
import json
import glob

def parse_label_file(file_path):
    labels = []
    pattern = re.compile(r"\('([^']*)',\s*'([^']*)',\s*'([^']*)',\s*'([^']*)'\)")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    labels.append(match.groups())
    except FileNotFoundError:
        print(f"경고: {file_path} 파일을 찾을 수 없습니다. 이 단계는 건너뜁니다.")
    return labels

def copy_files_for_step(step_name, labels, base_src_path, base_dst_path):
    """
    주어진 라벨에 해당하는 파일들을 원본 경로에서 대상 경로로 복사합니다.
    """
    print(f"--- {step_name} 처리 시작 ---")
    
    dst_images_path = os.path.join(base_dst_path, step_name, 'images')
    dst_labels_path = os.path.join(base_dst_path, step_name, 'labels')
    os.makedirs(dst_images_path, exist_ok=True)
    os.makedirs(dst_labels_path, exist_ok=True)

    for label_parts in labels:
        relative_path = os.path.join(*label_parts)
        src_image_dir = os.path.join(base_src_path, 'all', 'images', relative_path)
        src_label_dir = os.path.join(base_src_path, 'all', 'labels', relative_path)
        dst_image_dir = os.path.join(dst_images_path, relative_path)
        dst_label_dir = os.path.join(dst_labels_path, relative_path)

        if os.path.exists(src_image_dir):
            if not os.path.exists(dst_image_dir):
                shutil.copytree(src_image_dir, dst_image_dir)
                print(f"복사 완료 (images): {src_image_dir} -> {dst_image_dir}")
            else:
                print(f"이미 존재하여 건너뜁니다 (images): {dst_image_dir}")
        if os.path.exists(src_label_dir):
            if not os.path.exists(dst_label_dir):
                shutil.copytree(src_label_dir, dst_label_dir)
                print(f"복사 완료 (labels): {src_label_dir} -> {dst_label_dir}")
            else:
                print(f"이미 존재하여 건너뜁니다 (labels): {dst_label_dir}")

def update_json_labels_in_directory(target_path):
    print(f"--- JSON 파일 업데이트 시작: {target_path} ---")
    json_files = glob.glob(os.path.join(target_path, '**', '*.json'), recursive=True)

    if not json_files:
        print("업데이트할 JSON 파일을 찾지 못했습니다.")
        return

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            attributes = data.get('car', {}).get('attributes', {})
            model = attributes.get('model')
            
            modified = False
            if model == '기아_봉고':
                attributes['brand'] = '기아자동차'
                attributes['model'] = '봉고'
                modified = True
            elif model == '현대_포터':
                attributes['brand'] = '현대자동차'
                attributes['model'] = '포터'
                modified = True

            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                print(f"업데이트 완료: {file_path}")

        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"파일 처리 중 오류 발생: {file_path} | 오류: {e}")
    print("--- JSON 파일 업데이트 완료 ---")

if __name__ == "__main__":
    BASE_PATH = r"F:\dataset_final"
    LABEL_FILE_PATH = r"c:\Users\DMLab\GITHUB\vericar2\txt" 

    for i in range(4):
        step = f"step{i}"
        label_file = os.path.join(LABEL_FILE_PATH, f"{step}.txt")
        
        labels_to_copy = parse_label_file(label_file)
        if labels_to_copy:
            copy_files_for_step(step, labels_to_copy, BASE_PATH, BASE_PATH)

    json_update_target_path = r"F:\dataset_final\step1\labels\화물"
    update_json_labels_in_directory(json_update_target_path)