import os

RAW_PATH = "data/raw/general_cpp"
OUTPUT_PATH = "data/cleaned/cp_finetune.txt"

VALID_EXT = (".cpp", ".cc", ".hpp", ".h")

MIN_FILE_SIZE = 200   # skip tiny files
MAX_FILE_SIZE = 20000 # skip huge libraries


def is_valid_file(path):
    if not path.endswith(VALID_EXT):
        return False

    size = os.path.getsize(path)
    if size < MIN_FILE_SIZE or size > MAX_FILE_SIZE:
        return False

    return True


def clean_code(text):
    # Remove excessive empty lines
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if line.strip() == "":
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_title(filepath):
    name = os.path.basename(filepath)
    name = name.replace(".cpp", "")
    name = name.replace(".cc", "")
    name = name.replace("_", " ")
    name = name.replace("-", " ")
    return name.strip()


def build_dataset():
    print("Building CP fine-tune dataset...")

    count = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:

        for root, _, files in os.walk(RAW_PATH):
            for file in files:

                full_path = os.path.join(root, file)

                if not is_valid_file(full_path):
                    continue

                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()

                    code = clean_code(code)
                    title = extract_title(full_path)

                    formatted = (
                        "<bos>\n"
                        "<problem>\n"
                        f"Title: {title}\n"
                        "<code>\n"
                        f"{code}\n"
                        "<eos>\n\n"
                    )

                    out.write(formatted)
                    count += 1

                except:
                    continue

    print(f"Total CP pairs created: {count}")


if __name__ == "__main__":
    build_dataset()