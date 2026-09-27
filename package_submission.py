import zipfile
import os
import shutil
from pathlib import Path

def create_submission():
    base_dir = Path("c:/Projects/coding/ML_SCHOOL/Noob-coders-main/Noob-coders-main")
    zip_path = base_dir / "NoobCoders_submission.zip"
    
    print(f"Creating submission package at {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Add Output files (moving them from output/test/ to output/ in the zip)
        output_dir = base_dir / "output"
        matching_file = output_dir / "matching_results.tsv"
        candidate_file = output_dir / "candidate_pairs.tsv"
        
        if matching_file.exists():
            print("Adding matching_results.tsv...")
            zipf.write(matching_file, arcname="output/matching_results.tsv")
        else:
            print(f"Warning: {matching_file} not found!")
            
        if candidate_file.exists():
            print("Adding candidate_pairs.tsv...")
            zipf.write(candidate_file, arcname="output/candidate_pairs.tsv")
        else:
            print(f"Warning: {candidate_file} not found!")
            
        # 2. Add Code files to code/business_entity_resolution/
        code_prefix = "code/business_entity_resolution/"
        
        # Add src directory
        src_dir = base_dir / "src"
        if src_dir.exists():
            print("Adding src/ directory...")
            for root, dirs, files in os.walk(src_dir):
                for file in files:
                    if file.endswith('.pyc') or '__pycache__' in root:
                        continue
                    file_path = Path(root) / file
                    arcname = code_prefix + str(file_path.relative_to(base_dir)).replace('\\', '/')
                    zipf.write(file_path, arcname=arcname)
                    
        # Add README.md and requirements.txt
        for file_name in ["README.md", "requirements.txt", "model.xgb"]: # I am adding model.xgb as well just in case they need it to run the pipeline
            file_path = base_dir / file_name
            if file_path.exists():
                print(f"Adding {file_name} to code folder...")
                zipf.write(file_path, arcname=code_prefix + file_name)
                
        # 3. Add Documentation_template.md to root
        doc_file = base_dir / "Documentation_template.md"
        if doc_file.exists():
            print("Adding Documentation_template.md...")
            zipf.write(doc_file, arcname="Documentation_template.md")
            
    print("Submission package created successfully!")

if __name__ == "__main__":
    create_submission()
