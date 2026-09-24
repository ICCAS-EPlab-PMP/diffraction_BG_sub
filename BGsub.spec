# -*- mode: python ; coding: utf-8 -*-
"""
BGsub.spec �?PyInstaller build spec
Packages BGsub 2D GUI as standalone executable.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

SPEC_DIR = Path(SPECPATH).resolve()
WORKSPACE_DIR = SPEC_DIR
OUTPUT_DIR = SPEC_DIR.parent / 'RELEASE'


def collect_no_tests(pkg):
    """Collect data/bins from a package but exclude test files."""
    datas, bins, imports = collect_all(pkg)
    clean_datas = []
    for src, dst in datas:
        p = Path(src)
        parts = p.parts
        if any(part.lower() in ("tests", "test") for part in parts):
            continue
        if p.suffix in (".py", ".pyc") and p.parent.name == "tests":
            continue
        clean_datas.append((src, dst))
    return clean_datas, bins, imports


datas = []
binaries = []
hiddenimports = [
    "dateutil",
    "silx.gui.widgets.CollapsibleWidget",
    "silx.gui.widgets.FlowLayout",
    "silx.gui.qt",
    "silx.gui.utils",
    "PySide6.QtWidgets",
    "PySide6.QtCore",
    "PySide6.QtGui",
]

for pkg in ("silx", "fabio", "h5py", "scipy"):
    d, b, i = collect_no_tests(pkg)
    datas += d
    binaries += b
    hiddenimports += i

a = Analysis(
    [str(WORKSPACE_DIR / "run_app.py")],
    pathex=[str(WORKSPACE_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "cv2", "opencv", "opencv_contrib_python",
        "pyarrow",
        "torch", "torchaudio", "torchvision",
        "onnxruntime", "tensorflow", "keras",
        "numba", "llvmlite",
        "pymatgen",
        "sklearn", "skimage",
        "huggingface_hub",
        "streamlit", "plotly", "altair", "pydeck",
        "flask", "django", "fastapi", "uvicorn",
        "selenium", "playwright",
        "jupyter", "jupyterlab", "notebook",
        "nbconvert", "nbformat",
        "ipywidgets", "ipython", "ipykernel",
        "PyQt5", "PyQt6", "pyqtgraph",
        "PyOpenGL", "pyopencl",
        "PySide6.QtQml", "PySide6.QtQuick",
        "PySide6.QtQuickWidgets",
        "PySide6.QtBluetooth", "PySide6.QtNfc",
        "PySide6.QtPositioning", "PySide6.QtLocation",
        "PySide6.QtSensors", "PySide6.QtSerialPort",
        "PySide6.QtSerialBus", "PySide6.QtWebSockets",
        "PySide6.QtWebChannel",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineQuick",
        "PySide6.QtWebView",
        "PySide6.Qt3DCore", "PySide6.Qt3DRender",
        "PySide6.Qt3DInput", "PySide6.Qt3DAnimation",
        "PySide6.Qt3DLogic", "PySide6.Qt3DExtras",
        "PySide6.QtCharts", "PySide6.QtDataVisualization",
        "PySide6.QtGraphs", "PySide6.QtGraphsWidgets",
        "PySide6.QtScxml", "PySide6.QtStateMachine",
        "PySide6.QtRemoteObjects", "PySide6.QtPdf",
        "PySide6.QtPdfWidgets", "PySide6.QtHelp",
        "PySide6.QtTextToSpeech",
        "PySide6.QtSpatialAudio",
        "PySide6.QtNetworkAuth", "PySide6.QtHttpServer",
        "PySide6.QtExampleIcons",
        "PySide6.QtQuick3D", "PySide6.QtQuickTest",
        "PySide6.QtQuickControls2",
        "PySide6.QtConcurrent",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtXml", "PySide6.QtDBus",
        "PySide6.QtSql",
        "scipy._lib.tests", "scipy.optimize.tests",
        "scipy.ndimage.tests", "scipy.sparse.tests",
        "scipy.linalg.tests", "scipy.signal.tests",
        "scipy.stats.tests", "scipy.interpolate.tests",
        "scipy.io.tests", "scipy.spatial.tests",
        "scipy.special.tests", "scipy.integrate.tests",
        "scipy.fft.tests", "scipy.cluster.tests",
        "scipy.odr.tests",
        "numpy.tests", "numpy.f2py.tests",
        "matplotlib.tests",
        "silx.math.test", "silx.gui.plot.test",
        "pytest", "ruff", "mypy",
        "imageio_ffmpeg", "imageio",
        "pyFAI", "pyfai",
        "OpenGL", "PyOpenGL",
        "jedi", "parso",
        "tornado",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BGsub",
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(SPEC_DIR / 'logo.ico')],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name="BGsub",
    pathex=[str(OUTPUT_DIR)],
)

