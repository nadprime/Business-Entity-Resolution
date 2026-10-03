import argparse
import os
import shutil
import urllib.request
import zipfile

DATA_URL = "https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    a = ap.parse_args()
    for sub in ("train", "test"):
        os.makedirs(os.path.join(a.data_dir, sub), exist_ok=True)
    if os.path.exists(os.path.join(a.data_dir, "train", "train_source1.tsv")):
        print("Data already present - nothing to do.")
        return
    zip_path, tmp = os.path.join(a.data_dir, "dataset.zip"), os.path.join(a.data_dir, "_tmp")
    print("Downloading ...")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    for root, _, files in os.walk(tmp):
        for f in files:
            if f.endswith(".tsv"):
                rel = os.path.relpath(root, tmp).lower()   # judge only the path inside the archive
                folder = "train" if "train" in rel or "train" in f.lower() else "test"
                shutil.copy(os.path.join(root, f), os.path.join(a.data_dir, folder, f))
    shutil.rmtree(tmp)
    os.remove(zip_path)
    print("Done.")


if __name__ == "__main__":
    main()
