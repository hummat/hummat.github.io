# Large File Storage (Cloudflare R2)

Large binary assets (images, figures, data files) are stored in **Cloudflare R2** and served via a custom domain.

## Storage Details

- **Bucket name:** `hummat-assets`
- **Public URL:** `https://assets.hummat.com`
- **Directories in R2:**
  - `images/` – post thumbnails, banners, inline images
  - `figures/` – interactive HTML figures (Plotly, etc.)
  - `data/` – numpy arrays, PLY point clouds, other data files
  - `blender/` – Blender project files

## Referencing Assets in Posts

Use absolute URLs pointing to R2:

```markdown
image: https://assets.hummat.com/images/my-thumbnail.jpg
banner: https://assets.hummat.com/images/my-banner.jpg
```

```html
<img src="https://assets.hummat.com/images/example.png" />
<div data-include="https://assets.hummat.com/figures/my-plot.html"></div>
```

## Adding New Files

### Automated (via Git Hook)

A pre-commit hook automatically uploads assets to R2 when you commit.

1. **Enable the hook** (one-time setup):
   ```bash
   git config core.hooksPath .githooks
   ```

2. **Add files to the upload folder** (gitignored, won't be committed):
   ```bash
   mkdir -p _assets/images
   cp /path/to/photo.jpg _assets/images/
   ```

3. **Commit your post** (the hook runs automatically):
   ```bash
   git add my-post.md
   git commit -m "Add new post"
   ```
   The hook uploads files from `_assets/` to R2, then moves them to `_assets/.uploaded/` to prevent re-uploading.

4. **Reference in your post** using `https://assets.hummat.com/images/photo.jpg`

### Manual (via rclone)

1. **Prerequisites:** Install and configure `rclone` with the R2 remote named `r2`. Configuration is stored at `~/.config/rclone/rclone.conf`.

2. **Upload files:**
   ```bash
   # Single file
   rclone copy /path/to/file.jpg r2:hummat-assets/images/

   # Entire directory
   rclone copy ./local-figures/ r2:hummat-assets/figures/ --progress

   # Sync (mirror local to remote, deletes removed files)
   rclone sync ./images/ r2:hummat-assets/images/ --progress
   ```

3. **Verify upload:**
   ```bash
   rclone ls r2:hummat-assets/images/ | grep "file.jpg"
   ```

4. **Reference in post:** Use the public URL `https://assets.hummat.com/<path>` in your Markdown or HTML.

## Useful rclone Commands

```bash
# List all directories in bucket
rclone lsd r2:hummat-assets

# List files in a directory
rclone ls r2:hummat-assets/images/

# Check bucket size
rclone size r2:hummat-assets

# Delete a file
rclone delete r2:hummat-assets/images/old-file.jpg
```

## Migration Note

This site previously used Git LFS via Netlify Large Media. That has been deprecated and all assets migrated to R2. The `.gitattributes` file is now empty (LFS tracking removed), and `.lfsconfig` has been deleted. Do not commit large binary files directly to the repository.
