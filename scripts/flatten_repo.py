import os
import subprocess
from pathlib import Path
from datetime import datetime

# --- Configuration ---
OUTPUT_FILE = "repo_context.txt"
MAX_FILE_SIZE_KB = 500  # Skip files larger than this (avoids data dumps)

# Even with git ls-files, we want to explicitly skip certain extensions 
# (like images/binaries that might be tracked)
BINARY_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.tif', '.tiff', 
    '.pyc', '.iso', '.bin', '.exe', '.dll', '.so', '.dylib', '.pdf',
    '.parquet', '.zip', '.tar', '.gz'
}

def get_git_files():
    """Retrieves all version-controlled files using git ls-files."""
    try:
        # standard git command to list tracked files
        result = subprocess.check_output(
            ["git", "ls-files"], 
            stderr=subprocess.DEVNULL
        ).decode("utf-8")
        files = result.splitlines()
        return sorted(files) # Deterministic ordering
    except subprocess.CalledProcessError:
        print("❌ Error: Not a git repository or git is not installed.")
        return []

def is_text_file(file_path):
    """
    Checks if a file is text-based.
    1. Checks extension against binary blocklist.
    2. Checks for null bytes in the first 1024 bytes (heuristic).
    """
    if Path(file_path).suffix.lower() in BINARY_EXTENSIONS:
        return False
        
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:
                return False
    except Exception:
        return False
        
    return True

def generate_tree(files):
    """Generates a visual directory tree string from a list of files."""
    tree = ["Project Directory Structure:"]
    for f in files:
        tree.append(f"├── {f}")
    return "\n".join(tree)

def flatten_repo():
    repo_root = Path(".").resolve()
    all_git_files = get_git_files()
    
    if not all_git_files:
        print("No files found. Make sure you run this from the repo root.")
        return

    processed_files = []
    skipped_files = []
    total_chars = 0

    print(f"🔍 Found {len(all_git_files)} tracked files. Processing...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        # --- 1. Header & Manifest ---
        out.write(f"REPO CONTEXT SNAPSHOT\n")
        out.write(f"Generated: {datetime.now().isoformat()}\n")
        out.write(f"Source: {repo_root.name}\n")
        out.write("=" * 40 + "\n\n")

        # --- 2. Directory Tree ---
        out.write(generate_tree(all_git_files))
        out.write("\n\n" + "=" * 40 + "\n\n")

        # --- 3. File Contents ---
        for rel_path in all_git_files:
            if rel_path == OUTPUT_FILE:
                continue
                
            full_path = repo_root / rel_path
            
            # Size Check
            if full_path.stat().st_size > (MAX_FILE_SIZE_KB * 1024):
                skipped_files.append(f"{rel_path} (Too Large: {full_path.stat().st_size/1024:.1f}KB)")
                continue

            # Binary/Text Check
            if not is_text_file(full_path):
                skipped_files.append(f"{rel_path} (Binary detected)")
                continue

            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
                
                # Visual separator + XML wrapper
                out.write(f"----- START FILE: {rel_path} -----\n")
                out.write(f"<file path=\"{rel_path}\">\n")
                out.write(content)
                out.write(f"\n</file>\n")
                out.write(f"----- END FILE: {rel_path} -----\n\n")
                
                processed_files.append(rel_path)
                total_chars += len(content)
                
            except Exception as e:
                skipped_files.append(f"{rel_path} (Read Error: {str(e)})")

        # --- 4. Append Skipped Files Log at the end ---
        if skipped_files:
            out.write("\n" + "=" * 40 + "\n")
            out.write("SKIPPED FILES LOG:\n")
            for item in skipped_files:
                out.write(f"- {item}\n")

    # Console Summary
    print(f"✅ Success! Flattened {len(processed_files)} files to {OUTPUT_FILE}")
    print(f"🚫 Skipped {len(skipped_files)} files (see end of output file for details).")
    print(f"📊 Total Content Size: {total_chars/1024:.1f} KB")

if __name__ == "__main__":
    flatten_repo()