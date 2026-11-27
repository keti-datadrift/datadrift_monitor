import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    # 경로 설정
    source_dir = Path(r"D:\Datasets\Vehicle\자동차 차종-연식-번호판 인식용 영상\Train\images")
    ref_dir = Path(r"F:\IPIU2026\dataset_final\all\images")
    dest_dir = Path(r"E:\viewpoint\used")

    # 1. 참조할 pt 파일들의 이름(stem) 수집
    print(f"Scanning reference files in: {ref_dir}")
    ref_stems = set()
    if not ref_dir.exists():
        print(f"Error: Reference directory {ref_dir} does not exist.")
        return

    # .pt 파일 검색 (재귀적)
    for p in ref_dir.rglob("*.pt"):
        ref_stems.add(p.stem)
    
    print(f"Found {len(ref_stems)} reference .pt files.")

    if not ref_stems:
        print("No reference files found. Exiting.")
        return

    # 2. 저장할 디렉토리 생성
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 3. 소스 디렉토리에서 jpg 파일 검색 및 복사
    print(f"Scanning source files in: {source_dir}")
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist.")
        return
    
    # jpg 파일 검색 (대소문자 확장자 모두 고려)
    source_files_set = set(source_dir.rglob("*.jpg"))
    source_files_set.update(source_dir.rglob("*.JPG"))
    
    print(f"Found {len(source_files_set)} .jpg files. Filtering and copying...")

    copied_count = 0
    for src_path in tqdm(source_files_set):
        # 파일 이름(확장자 제외)이 참조 리스트에 있는지 확인
        if src_path.stem in ref_stems:
            dest_path = dest_dir / src_path.name
            
            try:
                shutil.copy2(src_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"Failed to copy {src_path}: {e}")

    print(f"Operation complete. Copied {copied_count} files to {dest_dir}")

if __name__ == "__main__":
    main()