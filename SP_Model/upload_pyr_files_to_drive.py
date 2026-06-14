# upload_pyr_files_to_drive.py
# Convert pyrImg*.mat -> .npz (per file), verify, delete .mat (low disk), upload .npz to Google Drive.
# Logs every result to drive_upload.log

# --- Prefer PyDrive2; fallback to PyDrive ---
try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    USING_PYDRIVE2 = True
except ImportError:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    USING_PYDRIVE2 = False

from tqdm import tqdm
import scipy.io as sio
import numpy as np
import os
import time
import tempfile
from pathlib import Path
from datetime import datetime

# ======================
# TOGGLES / SETTINGS
# ======================
# Disk-space friendly:
PRECLEAN_DUP_MATS = True                     # delete .mat if a valid .npz already exists locally
DELETE_MAT_IMMEDIATELY_AFTER_CONVERT = True  # delete .mat right after creating & verifying .npz
DELETE_MAT_IF_ALREADY_IN_DRIVE = False       # if .npz already on Drive, also delete local .mat?
RETRIES = 3                                   # upload retries per file
SLEEP_BETWEEN_RETRIES = 3.0                   # seconds between retries
VERBOSE_UPLOAD_LINKS = True                   # print link for each uploaded/skipped file
TEST_SMALL_UPLOAD = True                     # quick probe upload to verify folder access

# Paths / IDs
isub = 1  # Subject number (1–8)
nsd_design_path = r"D:/NSD/experiments/nsd_expdesign.mat"
pyramid_folder = r"D:/NSD/pyramid_expand/configurable_256_no_background"   # where pyrImg*.mat / .npz live
drive_folder_id  = "1G69j71-WJ6dyMymJ9vEY4dsluiPHfIrn"  # target Drive folder ID
CLIENT_SECRETS_JSON = r"D:/NSD/client_secrets.json"      # OAuth client secrets

# Log file
LOG_PATH = os.path.join(pyramid_folder, "drive_upload.log")

# ======================
# Helpers
# ======================
def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_line(kind, fname, link=""):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as log:
            log.write(f"{ts()} {kind} {fname} {link}\n")
    except Exception as e:
        tqdm.write(f"Couldn't write log for {fname}: {e}")

def safe_delete(path):
    try:
        os.remove(path)
    except Exception as e:
        print(f" Couldn't delete {path}: {e}")

def save_npz_atomic(npz_path, **arrays):
    """Atomic write of .npz to avoid partial/corrupt files."""
    folder = os.path.dirname(npz_path) or "."
    with tempfile.NamedTemporaryFile(dir=folder, delete=False) as tmp:
        np.savez_compressed(tmp, **arrays)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = tmp.name
    os.replace(temp_name, npz_path)

def verify_npz(npz_path, retries=3, sleep=0.2):
    """Try opening .npz; fallback to allow_pickle for object arrays."""
    last_err = None
    for _ in range(retries):
        try:
            with np.load(npz_path) as z:
                _ = z["sumOri"]; _ = z["modelOri"]
            return True
        except Exception as e1:
            last_err = e1
            try:
                with np.load(npz_path, allow_pickle=True) as z:
                    _ = z["sumOri"]; _ = z["modelOri"]
                return True
            except Exception as e2:
                last_err = e2
                time.sleep(sleep)
    print(f"verify_npz failed for {npz_path}: {last_err}")
    return False

def list_in_folder(drive, folder_id):
    """List files in a folder; Shared-Drive safe."""
    files, page_token = [], None
    params = {
        'q': f"'{folder_id}' in parents and trashed=false",
        'includeItemsFromAllDrives': True,
        'supportsAllDrives': True,
        'corpora': 'allDrives',
    }
    while True:
        if page_token:
            params['pageToken'] = page_token
        fl = drive.ListFile(params)
        batch = fl.GetList()
        files.extend(batch)
        page_token = getattr(fl, "metadata", {}).get('nextPageToken', None)
        if not page_token:
            break
    return files

def upload_file_to_folder(drive, local_path, dest_name, folder_id):
    """Upload one file to folder; returns (success, link)."""
    try:
        f = drive.CreateFile({'title': dest_name, 'parents': [{'id': folder_id}]})
        f.SetContentFile(local_path)
        f.Upload(param={'supportsAllDrives': True})
        link = f.get('webViewLink') or f.get('alternateLink') or ""
        return True, link
    except Exception as e:
        print(f" Upload error [{dest_name}]: {e}")
        return False, None

def fetch_folder_meta(drive, folder_id):
    """Print folder metadata to confirm ID and access."""
    try:
        folder = drive.CreateFile({'id': folder_id})
        try:
            if USING_PYDRIVE2:
                folder.FetchMetadata(fields='id,name,title,mimeType,driveId,parents,owners,webViewLink')
            else:
                folder.FetchMetadata()
        except Exception:
            folder.FetchMetadata()
        print("Folder check:",
              "id=", folder.get('id'),
              "name=", folder.get('name') or folder.get('title'),
              "mimeType=", folder.get('mimeType'),
              "driveId=", folder.get('driveId'))
    except Exception as e:
        print(f" Could not fetch folder metadata for {folder_id}: {e}")

# ======================
# Step 1: Load NSD design
# ======================
print(f" Loading NSD design for subject {isub}...")
nsdDesign = sio.loadmat(nsd_design_path)
subjectim = nsdDesign["subjectim"]  # [8, N_images_per_subject]
masterordering = nsdDesign["masterordering"].flatten() - 1  # 0-based
valid_masterordering = masterordering[masterordering < subjectim.shape[1]]
allImgs = np.unique(subjectim[isub - 1, valid_masterordering])  # 1-based image IDs

# ======================
# Step 2: Scan local files
# ======================
existing_files = []  # dicts: {'fname': <npz>, 'npz': path, 'base': base}
to_convert = []      # list of bases to convert
missing = []         # bases with no mat/npz

print(" Checking pyramid file existence and mapping...")
for img_id in allImgs:
    file_index = int(img_id) - 1
    base = f"pyrImg{file_index}"
    mat_path = os.path.join(pyramid_folder, base + ".mat")
    npz_path = os.path.join(pyramid_folder, base + ".npz")
    if os.path.exists(npz_path):
        existing_files.append({'fname': base + ".npz", 'npz': npz_path, 'base': base})
    elif os.path.exists(mat_path):
        to_convert.append(base)
    else:
        missing.append(base)

print(f" Total image IDs: {len(allImgs)}")
print(f" Files ready locally (.npz): {len(existing_files)}")
print(f" Files to convert from .mat: {len(to_convert)}")
if missing:
    print(f"  Missing locally (.mat not found): {len(missing)}")

# ======================
# Step 3: Auth & folder sanity
# ======================
print(" Authenticating with Google Drive...")
gauth = GoogleAuth()
gauth.LoadClientConfigFile(CLIENT_SECRETS_JSON)
try:
    scopes = gauth.settings.get('oauth_scope')
    if not scopes:
        gauth.settings['oauth_scope'] = ['https://www.googleapis.com/auth/drive']
except Exception:
    pass

try:
    gauth.LocalWebserverAuth()
except Exception:
    gauth.CommandLineAuth()
drive = GoogleDrive(gauth)

fetch_folder_meta(drive, drive_folder_id)

print(" Fetching already-uploaded files in Drive folder...")
existing_drive_files = list_in_folder(drive, drive_folder_id)
existing_drive_names = set((f.get('title') or f.get('name')) for f in existing_drive_files)
existing_drive_map = {
    (f.get('title') or f.get('name')): (f.get('webViewLink') or f.get('alternateLink') or "")
    for f in existing_drive_files
}

if TEST_SMALL_UPLOAD:
    probe = Path(pyramid_folder) / "____drive_probe.txt"
    probe.write_text("drive probe ok", encoding="utf-8")
    ok, link = upload_file_to_folder(drive, str(probe), "____drive_probe.txt", drive_folder_id)
    print(" Test upload:", "OK" if ok else "FAILED", "|", link or "")
    if not ok:
        raise SystemExit("Drive test upload failed; check folder ID and permissions.")

# ======================
# Step 4 + 5: Convert → Verify → Delete .mat → Upload (low disk)
# ======================
# Pre-clean: if a valid .npz already exists, delete the duplicate .mat
if PRECLEAN_DUP_MATS:
    for rec in existing_files:
        base = rec['base']
        mat_path = os.path.join(pyramid_folder, base + ".mat")
        npz_path = rec['npz']
        if os.path.exists(mat_path) and os.path.exists(npz_path) and verify_npz(npz_path):
            safe_delete(mat_path)

# Build work list: convert-needing first (frees disk sooner), then already-converted
items = (
    [{'base': b,
      'npz': os.path.join(pyramid_folder, b + ".npz"),
      'fname': b + ".npz",
      'needs_convert': True} for b in to_convert]
    +
    [{'base': d['base'], 'npz': d['npz'], 'fname': d['fname'], 'needs_convert': False}
     for d in existing_files]
)

print(" Convert → Verify → Delete .mat → Upload (low-disk mode)...")
uploaded = skipped = converted = mats_deleted = 0
t0 = time.time()

for item in tqdm(items, desc="Convert→Upload", unit="file"):
    base, npz_path, fname = item['base'], item['npz'], item['fname']
    mat_path = os.path.join(pyramid_folder, base + ".mat")

    # Convert if needed
    if item['needs_convert']:
        if not os.path.exists(mat_path):
            tqdm.write(f" Missing .mat: {base}")
            continue
        try:
            m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            sumOri = m["sumOri"]; modelOri = m["modelOri"]
            sumOri = np.asarray(sumOri); modelOri = np.asarray(modelOri)
            save_npz_atomic(npz_path, sumOri=sumOri, modelOri=modelOri)
            converted += 1
        except KeyError as e:
            tqdm.write(f" Keys missing in {base}.mat: {e}")
            continue
        except Exception as e:
            tqdm.write(f" Convert failed {base}: {e}")
            continue

    # Verify .npz (retry + pickle fallback)
    if not verify_npz(npz_path):
        if os.path.exists(mat_path):
            try:
                m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
                sumOri = np.asarray(m["sumOri"]); modelOri = np.asarray(m["modelOri"])
                save_npz_atomic(npz_path, sumOri=sumOri, modelOri=modelOri)
            except Exception as e:
                tqdm.write(f" Rebuild failed {base}: {e}")
                log_line("FAILED(VERIFY)", fname)
                continue
            if not verify_npz(npz_path):
                tqdm.write(f" NPZ still invalid after rebuild: {npz_path}")
                log_line("FAILED(VERIFY)", fname)
                continue
        else:
            tqdm.write(f" NPZ invalid and .mat missing: {base}")
            log_line("FAILED(VERIFY)", fname)
            continue

    # Delete .mat immediately after successful conversion/verification
    if DELETE_MAT_IMMEDIATELY_AFTER_CONVERT and os.path.exists(mat_path):
        safe_delete(mat_path)
        mats_deleted += 1

    # Already in Drive?
    if fname in existing_drive_names:
        skipped += 1
        link = existing_drive_map.get(fname, "")
        if VERBOSE_UPLOAD_LINKS and link:
            tqdm.write(f"↩ Skipped (already in Drive): {fname}  {link}")
        else:
            tqdm.write(f"↩ Skipped (already in Drive): {fname}")
        log_line("SKIPPED", fname, link)
        if DELETE_MAT_IF_ALREADY_IN_DRIVE and os.path.exists(mat_path):
            safe_delete(mat_path)
            mats_deleted += 1
        continue

    # Upload with retries
    success, link = False, None
    for attempt in range(1, RETRIES + 1):
        ok, link = upload_file_to_folder(drive, npz_path, fname, drive_folder_id)
        if ok:
            success = True
            uploaded += 1
            existing_drive_names.add(fname)           # so later files can detect it
            existing_drive_map[fname] = link or ""    # so future runs can print the link on skip
            if VERBOSE_UPLOAD_LINKS and link:
                tqdm.write(f"✔ Uploaded: {fname}  {link}")
            else:
                tqdm.write(f"✔ Uploaded: {fname}")
            log_line("UPLOADED", fname, link or "")
            break
        else:
            tqdm.write(f"   retrying {attempt}/{RETRIES} in {SLEEP_BETWEEN_RETRIES}s...")
            time.sleep(SLEEP_BETWEEN_RETRIES)

    if not success:
        tqdm.write(f"✖ FAILED: {fname}")
        log_line("FAILED", fname)

print("\n Done.")
print(f" Time: {time.time()-t0:.1f}s | Converted: {converted} | Uploaded: {uploaded} | Skipped: {skipped} | .mat deleted: {mats_deleted}")
print(f" Log written to: {LOG_PATH}")