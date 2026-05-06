import os
import time
import hashlib
from icrawler.builtin import BingImageCrawler

# ================= CONFIG =================
BASE_DIR = "dataset"
NUM_IMAGES = 100   # 🔥 Increase for better accuracy (100–300 recommended)

# ================= DATASET =================
dataset = {
    "tomato": ["healthy", "early blight", "late blight"],
    "potato": ["healthy", "early blight", "late blight"],
    "apple": ["healthy", "scab", "black rot"],
    "grape": ["healthy", "black rot"],
    "corn": ["healthy", "leaf blight"],
    "rice": ["healthy", "brown spot", "leaf blast"],
    "wheat": ["healthy", "rust", "powdery mildew"],
    "cotton": ["healthy", "leaf curl"],
    "mango": ["healthy", "anthracnose"],
    "banana": ["healthy", "leaf spot"]
}

# ================= HELPER =================
def clean_name(text):
    return text.replace(" ", "_").lower()

def get_file_hash(filepath):
    """Generate hash to remove duplicates"""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def remove_duplicates(folder):
    """Remove duplicate images"""
    hashes = set()
    removed = 0

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        try:
            file_hash = get_file_hash(path)

            if file_hash in hashes:
                os.remove(path)
                removed += 1
            else:
                hashes.add(file_hash)

        except:
            os.remove(path)

    print(f"🧹 Removed {removed} duplicates from {folder}")


# ================= DOWNLOAD =================
def download_dataset():
    os.makedirs(BASE_DIR, exist_ok=True)

    for plant, diseases in dataset.items():
        for disease in diseases:

            class_name = f"{clean_name(plant)}___{clean_name(disease)}"
            folder_path = os.path.join(BASE_DIR, class_name)

            os.makedirs(folder_path, exist_ok=True)

            query = f"{plant} leaf {disease}"

            print(f"\n⬇️ Downloading: {query}")

            crawler = BingImageCrawler(
                storage={'root_dir': folder_path},
                downloader_threads=4
            )

            try:
                crawler.crawl(
                    keyword=query,
                    max_num=NUM_IMAGES,
                    file_idx_offset=0
                )

                time.sleep(1)

                # 🧹 Clean duplicates
                remove_duplicates(folder_path)

            except Exception as e:
                print(f"❌ Error: {e}")

    print("\n✅ Dataset Download Completed!")


# ================= LABEL GENERATOR =================
def create_labels():
    classes = sorted(os.listdir(BASE_DIR))

    import json
    with open("labels.json", "w") as f:
        json.dump(classes, f)

    print("✅ labels.json created!")


# ================= MAIN =================
if __name__ == "__main__":
    download_dataset()
    create_labels()





