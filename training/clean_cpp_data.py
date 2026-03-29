import os
import hashlib

RAW_GENERAL_PATH = "data/raw/general_cpp"
RAW_CP_PATH = "data/raw/cp_problems"

CLEAN_GENERAL_OUTPUT = "data/cleaned/general_cpp/general_cpp.txt"
CLEAN_CP_OUTPUT = "data/cleaned/cp_pairs/cp_solutions.txt"

VALID_EXTENSIONS = (".cpp", ".cc", ".cxx", ".hpp", ".h")

MIN_LINES = 10
MAX_LINES = 800  # safety for long files

def normalize_code(code):
    code = code.replace("\r\n", "\n")
    code = code.replace("\t", "    ")
    return code.strip()

def is_valid_file(content):
    lines = content.split("\n")
    if len(lines) < MIN_LINES:
        return False
    if len(lines) > MAX_LINES:
        return False
    return True

def collect_cpp_files(root_dir):
    cpp_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(VALID_EXTENSIONS):
                cpp_files.append(os.path.join(root, file))
    return cpp_files

def process_files(file_paths, output_path):
    seen_hashes = set()
    kept = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for path in file_paths:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                content = normalize_code(content)

                if not is_valid_file(content):
                    continue

                file_hash = hashlib.md5(content.encode()).hexdigest()
                if file_hash in seen_hashes:
                    continue

                seen_hashes.add(file_hash)

                out.write("<bos>\n")
                out.write(content)
                out.write("\n<eos>\n\n")

                kept += 1

            except Exception as e:
                continue

    print(f"Saved {kept} cleaned files to {output_path}")

if __name__ == "__main__":
    os.makedirs("data/cleaned/general_cpp", exist_ok=True)
    os.makedirs("data/cleaned/cp_pairs", exist_ok=True)

    print("Processing general C++ repos...")
    general_files = collect_cpp_files(RAW_GENERAL_PATH)
    process_files(general_files, CLEAN_GENERAL_OUTPUT)

    print("Processing CP solution repos...")
    cp_files = collect_cpp_files(RAW_CP_PATH)
    process_files(cp_files, CLEAN_CP_OUTPUT)

    print("Cleaning completed.")