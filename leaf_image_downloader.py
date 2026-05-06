import os
from icrawler.builtin import BingImageCrawler

# All categories (fruits, vegetables, crops, diseases)
leaf_classes = [
    # 🌾 Crops
    "rice leaf", "wheat leaf", "maize leaf", "corn leaf",
    
    # 🍅 Vegetables
    "tomato leaf", "potato leaf", "brinjal leaf", "chilli leaf",
    "cabbage leaf", "spinach leaf", "okra leaf",
    
    # 🍎 Fruits
    "apple leaf", "banana leaf", "mango leaf", "grape leaf",
    "orange leaf", "papaya leaf", "guava leaf",
    
    # 🌿 Healthy
    "healthy plant leaf",
    
    # 🦠 Diseases
    "leaf early blight", "leaf late blight",
    "leaf bacterial spot", "leaf powdery mildew",
    "leaf rust disease"
]

# Number of images per class
NUM_IMAGES = 5


for leaf in leaf_classes:
    folder_name = leaf.replace(" ", "_")
    os.makedirs(folder_name, exist_ok=True)

    print(f"Downloading: {leaf}")

    crawler = BingImageCrawler(storage={'root_dir': folder_name})
    crawler.crawl(keyword=leaf, max_num=NUM_IMAGES)

print("✅ All images downloaded successfully!")
