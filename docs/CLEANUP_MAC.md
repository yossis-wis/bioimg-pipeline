# bioimg-pipeline — macOS Cleanup & Reset Guide

Follow these steps to cleanly remove old versions of the pipeline, environments, and configuration before a fresh install.

**Use this guide when:**
- Setting up on a new machine that has old configurations
- Resetting after a broken environment
- Switching from Anaconda to Miniforge

---

## 1) Backup important data first

Before cleaning up, ensure you've: 

- [ ] Pushed all important code changes to GitHub
- [ ] Saved any local-only files or data you want to keep
- [ ] Noted any custom configurations you want to preserve

---

## 2) Remove standalone Spyder (if installed)

The standalone Spyder app uses its own Python and won't work with your conda environment.

### Uninstall the app:

1. Open **Finder → Applications**
2. Find **Spyder** and drag to Trash
3. Empty Trash

### Remove Spyder settings:

```zsh
rm -rf ~/.spyder-py3
rm -rf ~/Library/Application\ Support/spyder-py3
```

**Note:** After setup, you'll use Spyder from within the conda environment instead. 

---

## 3) Remove the conda environment

This ensures no conflicting library versions remain.

1. Deactivate the current environment (if active):
   ```zsh
   conda deactivate
   ```

2. Remove the environment:
   ```zsh
   conda remove --na bioimg-slice0 --all
   ```

3. Verify it's gone:
   ```zsh
   conda env list
   ```

   Ensure `bioimg-slice0` is not in the list.

---

## 4) Archive or delete old repository

### Option A: Delete (if you're sure)

```zsh
rm -rf ~/Code/bioimg-pipeline
```

### Option B:  Rename/Archive (Recommended)

Move the old repo to a backup folder so the name is free for a fresh clone:

```zsh
mv ~/Code/bioimg-pipeline ~/Code/bioimg-pipeline_OLD_$(date +%Y%m%d)
```

---

## 5) Remove from GitHub Desktop

1. Open **GitHub Desktop**
2. Find the old repository in the left sidebar
3. Right-click → **Remove**
4. Optionally check **"Also move this repository to Trash"**

---

## 6) Clean environment variable (if changing data location)

If you previously set `BIOIMG_DATA_ROOT` to a different location:

1. Open your config: 
   ```zsh
   nano ~/.zshrc
   ```

2. Find the line starting with `export BIOIMG_DATA_ROOT=... `

3. Delete it or update the path. 

4. Save and apply: 
   ```zsh
   source ~/.zshrc
   ```

---

## 7) Reset data bench (optional)

If you want to clear old run data but keep raw inputs:

```zsh
# Clear all previous run outputs
rm -rf ~/bioimg-data/runs/*

# Clear cache if needed
rm -rf ~/bioimg-data/cache/*
```

---

## 8) Clean conda installation (optional - full reset)

Only do this if you want to completely reinstall conda. 

### Check your current installation:

```zsh
which conda
```

- `/Users/yourname/anaconda3/... ` → Anaconda
- `/Users/yourname/miniconda3/...` → Miniconda  
- `/Users/yourname/miniforge3/...` → Miniforge

### Remove conda entirely:

```zsh
# Remove the conda directory (adjust based on your installation)
rm -rf ~/anaconda3      # if Anaconda
rm -rf ~/miniconda3     # if Miniconda
rm -rf ~/miniforge3     # if Miniforge

# Remove conda config files
rm -rf ~/.conda
rm -f ~/.condarc
```

### Remove conda from shell config:

```zsh
nano ~/.zshrc
```

Find and delete the block that looks like:
```
# >>> conda initialize >>>
... 
# <<< conda initialize <<<
```

Save and reload: 
```zsh
source ~/.zshrc
```

---

## 9) Clean caches (optional - frees disk space)

```zsh
# Conda cache (can be several GB)
conda clean --all -y

# Pip cache
rm -rf ~/Library/Caches/pip

# VS Code Python cache
rm -rf ~/Library/Caches/com.microsoft.VSCode

# Python bytecode files
find ~/Code -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
```

---

## 10) Verify clean state

Before starting the setup guide: 

```zsh
# Should show no bioimg-slice0
conda env list

# Should say "No such file or directory"
ls ~/Code/bioimg-pipeline

# Should be empty or show your desired value
echo $BIOIMG_DATA_ROOT
```

---

## Quick cleanup script

Save this as `cleanup.sh` and run with `bash cleanup.sh`:

```bash
#!/bin/bash
# cleanup.sh - Bioimg pipeline cleanup script
# Review before running! 

echo "🧹 Bioimg Pipeline Cleanup"
echo "=========================="
echo ""
read -p "This will remove the conda environment and clean caches. Continue? (y/n) " -n 1 -r
echo ""

if [[ !  $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Step 1: Deactivating conda..."
conda deactivate 2>/dev/null

echo ""
echo "Step 2: Removing conda environment 'bioimg-slice0'..."
conda remove --name bioimg-slice0 --all -y 2>/dev/null && echo "  ✅ Removed" || echo "  ℹ️  Not found"

echo ""
echo "Step 3: Cleaning conda cache..."
conda clean --all -y 2>/dev/null && echo "  ✅ Cleaned"

echo ""
echo "Step 4: Cleaning pip cache..."
rm -rf ~/Library/Caches/pip && echo "  ✅ Cleaned"

echo ""
echo "Step 5: Removing Spyder settings..."
rm -rf ~/.spyder-py3 && echo "  ✅ Cleaned"

echo ""
echo "🎉 Cleanup complete!"
echo ""
echo "Next steps:"
echo "  1.  Optionally remove old repo:  rm -rf ~/Code/bioimg-pipeline"
echo "  2. Optionally uninstall standalone Spyder from Applications"
echo "  3. Follow SETUP_MAC.md for fresh installation"
```

---

## Next steps

After completing cleanup, proceed to **[SETUP_MAC.md](SETUP_MAC.md)** for fresh installation.
