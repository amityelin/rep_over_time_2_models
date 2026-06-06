# upload_pyr_npz_files_to_drive.py
# Upload pyrImg*.npz files to Google Drive.
# Assumes the new StimulusPyramidProcessor already created .npz files.

try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    USING_PYDRIVE2 = True
except ImportError:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    USING_PYDRIVE2 = False

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import numpy as np
import os
import time


# ============================================================
# Settings
# ============================================================

PYRAMID_FOLDER = r"D:/NSD/pyramid_expand/all_subjects_henderson256"
DRIVE_FOLDER_ID = "1BhUUnIt28R9_xowOGe7ncl9qlKa7FBEA"
CLIENT_SECRETS_JSON = r"D:/NSD/client_secrets.json"

RETRIES = 3
SLEEP_BETWEEN_RETRIES = 3.0
VERBOSE_UPLOAD_LINKS = True
TEST_SMALL_UPLOAD = True

LOG_PATH = os.path.join(PYRAMID_FOLDER, "drive_upload.log")


# ============================================================
# Helpers
# ============================================================

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(kind, fname, link=""):
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"{ts()} {kind} {fname} {link}\n")


def verify_npz(npz_path, retries=3, sleep=0.2):
    """
    Verify that the .npz file opens and contains the expected pyramid keys.
    """
    required_keys = ["bigImg", "sumOri", "modelOri", "numLevels", "numOrientations"]

    last_err = None

    for _ in range(retries):
        try:
            with np.load(npz_path, allow_pickle=True) as z:
                for key in required_keys:
                    if key not in z:
                        raise KeyError(f"Missing key: {key}")

                _ = z["bigImg"]
                _ = z["sumOri"]
                _ = z["modelOri"]

            return True

        except Exception as e:
            last_err = e
            time.sleep(sleep)

    print(f"verify_npz failed for {npz_path}: {last_err}")
    return False


def list_in_folder(drive, folder_id):
    """
    List files already present in the target Google Drive folder.
    Shared-Drive safe.
    """
    files = []
    page_token = None

    params = {
        "q": f"'{folder_id}' in parents and trashed=false",
        "includeItemsFromAllDrives": True,
        "supportsAllDrives": True,
        "corpora": "allDrives",
    }

    while True:
        if page_token:
            params["pageToken"] = page_token

        fl = drive.ListFile(params)
        batch = fl.GetList()
        files.extend(batch)

        page_token = getattr(fl, "metadata", {}).get("nextPageToken", None)

        if not page_token:
            break

    return files


def upload_file_to_folder(drive, local_path, dest_name, folder_id):
    """
    Upload one local file to Google Drive folder.
    """
    try:
        f = drive.CreateFile({
            "title": dest_name,
            "parents": [{"id": folder_id}],
        })

        f.SetContentFile(local_path)
        f.Upload(param={"supportsAllDrives": True})

        link = f.get("webViewLink") or f.get("alternateLink") or ""
        return True, link

    except Exception as e:
        print(f"Upload error [{dest_name}]: {e}")
        return False, None


def fetch_folder_meta(drive, folder_id):
    """
    Print folder metadata to confirm folder ID and access.
    """
    try:
        folder = drive.CreateFile({"id": folder_id})

        try:
            if USING_PYDRIVE2:
                folder.FetchMetadata(
                    fields="id,name,title,mimeType,driveId,parents,owners,webViewLink"
                )
            else:
                folder.FetchMetadata()
        except Exception:
            folder.FetchMetadata()

        print(
            "Folder check:",
            "id=", folder.get("id"),
            "name=", folder.get("name") or folder.get("title"),
            "mimeType=", folder.get("mimeType"),
            "driveId=", folder.get("driveId"),
        )

    except Exception as e:
        print(f"Could not fetch folder metadata for {folder_id}: {e}")


# ============================================================
# Step 1: Scan local .npz files
# ============================================================

print("Scanning local pyramid folder...")

pyramid_files = sorted([
    f for f in os.listdir(PYRAMID_FOLDER)
    if f.startswith("pyrImg") and f.endswith(".npz")
])

print(f"Found local .npz files: {len(pyramid_files)}")

if len(pyramid_files) == 0:
    raise SystemExit(f"No pyrImg*.npz files found in:\n{PYRAMID_FOLDER}")


# ============================================================
# Step 2: Verify local files
# ============================================================

print("Verifying local .npz files...")

valid_items = []
invalid_files = []

for fname in tqdm(pyramid_files, desc="Verify", unit="file"):
    npz_path = os.path.join(PYRAMID_FOLDER, fname)

    if verify_npz(npz_path):
        valid_items.append({
            "fname": fname,
            "path": npz_path,
        })
    else:
        invalid_files.append(fname)
        log_line("FAILED_VERIFY", fname)

print(f"Valid files: {len(valid_items)}")
print(f"Invalid files: {len(invalid_files)}")

if invalid_files:
    print("Invalid files were skipped.")
    print(invalid_files[:20])


# ============================================================
# Step 3: Authenticate with Google Drive
# ============================================================

print("Authenticating with Google Drive...")

gauth = GoogleAuth()
gauth.LoadClientConfigFile(CLIENT_SECRETS_JSON)

try:
    scopes = gauth.settings.get("oauth_scope")
    if not scopes:
        gauth.settings["oauth_scope"] = ["https://www.googleapis.com/auth/drive"]
except Exception:
    pass

try:
    gauth.LocalWebserverAuth()
except Exception:
    gauth.CommandLineAuth()

drive = GoogleDrive(gauth)

fetch_folder_meta(drive, DRIVE_FOLDER_ID)


# ============================================================
# Step 4: Fetch already uploaded files
# ============================================================

print("Fetching already-uploaded files in Drive folder...")

existing_drive_files = list_in_folder(drive, DRIVE_FOLDER_ID)

existing_drive_names = set(
    (f.get("title") or f.get("name"))
    for f in existing_drive_files
)

existing_drive_map = {
    (f.get("title") or f.get("name")): (
        f.get("webViewLink") or f.get("alternateLink") or ""
    )
    for f in existing_drive_files
}

print(f"Already in Drive folder: {len(existing_drive_names)} files")


# ============================================================
# Step 5: Optional test upload
# ============================================================

if TEST_SMALL_UPLOAD:
    probe = Path(PYRAMID_FOLDER) / "____drive_probe.txt"
    probe.write_text("drive probe ok", encoding="utf-8")

    ok, link = upload_file_to_folder(
        drive,
        str(probe),
        "____drive_probe.txt",
        DRIVE_FOLDER_ID,
    )

    print("Test upload:", "OK" if ok else "FAILED", "|", link or "")

    if not ok:
        raise SystemExit("Drive test upload failed. Check folder ID and permissions.")


# ============================================================
# Step 6: Upload .npz files
# ============================================================

print("Uploading .npz files...")

uploaded = 0
skipped = 0
failed = 0
t0 = time.time()

for item in tqdm(valid_items, desc="Upload", unit="file"):
    fname = item["fname"]
    local_path = item["path"]

    # Skip if already uploaded
    if fname in existing_drive_names:
        skipped += 1
        link = existing_drive_map.get(fname, "")

        if VERBOSE_UPLOAD_LINKS and link:
            tqdm.write(f"Skipped already in Drive: {fname}  {link}")
        else:
            tqdm.write(f"Skipped already in Drive: {fname}")

        log_line("SKIPPED", fname, link)
        continue

    # Upload with retries
    success = False
    link = ""

    for attempt in range(1, RETRIES + 1):
        ok, link = upload_file_to_folder(
            drive,
            local_path,
            fname,
            DRIVE_FOLDER_ID,
        )

        if ok:
            success = True
            uploaded += 1

            existing_drive_names.add(fname)
            existing_drive_map[fname] = link or ""

            if VERBOSE_UPLOAD_LINKS and link:
                tqdm.write(f"Uploaded: {fname}  {link}")
            else:
                tqdm.write(f"Uploaded: {fname}")

            log_line("UPLOADED", fname, link or "")
            break

        tqdm.write(
            f"Retrying {fname}: attempt {attempt}/{RETRIES} "
            f"in {SLEEP_BETWEEN_RETRIES}s..."
        )
        time.sleep(SLEEP_BETWEEN_RETRIES)

    if not success:
        failed += 1
        tqdm.write(f"FAILED: {fname}")
        log_line("FAILED_UPLOAD", fname)


# ============================================================
# Summary
# ============================================================

print("\nDone.")
print(f"Time: {time.time() - t0:.1f}s")
print(f"Uploaded: {uploaded}")
print(f"Skipped: {skipped}")
print(f"Failed: {failed}")
print(f"Invalid local files skipped: {len(invalid_files)}")
print(f"Log written to: {LOG_PATH}")