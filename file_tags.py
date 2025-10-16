import json
import os
from datetime import datetime, timezone

def set_tags(file_path, tags: dict):
    """Assign tags to a file using NTFS Alternate Data Streams."""
    ads_path = f"{file_path}:tags"
    mtime = os.path.getmtime(file_path)
    atime = os.path.getatime(file_path)
    os.utime(file_path, (atime, mtime))  # Update access time to current time
    with open(ads_path, "w", encoding="utf-8") as ads:
        json.dump(tags, ads)
    os.utime(file_path, (atime, mtime))  # Restore original access and modification times


def get_tags(file_path):
    """Retrieve tags from a file."""
    ads_path = f"{file_path}:tags"
    if not os.path.exists(file_path):
        raise FileNotFoundError("Base file does not exist.")
    try:
        with open(ads_path, "r", encoding="utf-8") as ads:
            return json.load(ads)
    except FileNotFoundError:
        return {}

# Test
file = r"C:\Users\jaff_\Desktop\sql.txt"
tags = {"last_acces_time": datetime.now(tz = timezone.utc).isoformat(), "status": "Calibrated", "user": os.getlogin()}

set_tags(file, tags)
print("Stored tags:", get_tags(file))
