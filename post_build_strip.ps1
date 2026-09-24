# post_build_strip.ps1 — Strip unnecessary bloat from PyInstaller build
# Run this AFTER pyinstaller BGsub.spec

$dist = Join-Path $PSScriptRoot "dist\BGsub\_internal"
if (!(Test-Path $dist)) {
    Write-Host "Build directory not found. Run pyinstaller first."
    exit 1
}

$removed = 0
$dirs = @(
    "imageio_ffmpeg",   # -84 MB, video codec, not needed
    "imageio",           # imageio wrapper, not needed
    "pyFAI",             # -8 MB, not used by BGsub
    "OpenGL",            # -2 MB, not used
    "jedi",              # -3 MB, IDE autocomplete
    "parso",             # jedi dep
    "tornado"            # web server, not needed
)

foreach ($d in $dirs) {
    $p = Join-Path $dist $d
    if (Test-Path $p) {
        $sz = (Get-ChildItem -Recurse $p -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        Remove-Item -Recurse -Force $p
        Write-Host ("Removed {0,-20} {1,7:F1} MB" -f $d, ($sz/1MB))
        $removed += $sz
    }
}

# Remove PySide6 translations (not needed for English-only app)
$trans = Join-Path $dist "PySide6\translations"
if (Test-Path $trans) {
    $sz = (Get-ChildItem -Recurse $trans -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    Remove-Item -Recurse -Force $trans
    Write-Host ("Removed {0,-20} {1,7:F1} MB" -f "PySide6/translations", ($sz/1MB))
    $removed += $sz
}

$total = (Get-ChildItem -Recurse (Split-Path $dist -Parent) | Measure-Object -Property Length -Sum).Sum
Write-Host ("`nRemoved total: {0:F1} MB" -f ($removed/1MB))
Write-Host ("Final dist size: {0:F1} MB" -f ($total/1MB))
