"""On-the-fly C++ node compilation service."""
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from app.config import settings

# Path to the SDK headers (shipped with the repo)
_SDK_INCLUDE = Path(__file__).parent.parent.parent.parent / "shared" / "cpp" / "include"
_CMAKE_TEMPLATE = Path(__file__).parent.parent.parent.parent / "shared" / "cpp" / "CMakeLists.txt.template"


def _source_hash(source_code: str, compile_flags: list[str]) -> str:
    content = source_code + "||" + json.dumps(sorted(compile_flags))
    return hashlib.sha256(content.encode()).hexdigest()


def compile_node(
    source_code: str,
    compile_flags: list[str] | None = None,
    extra_libs: list[str] | None = None,
) -> dict:
    """
    Compile C++ source to .so. Returns dict with keys:
      status: 'ok' | 'error'
      so_hash: str (only on ok)
      so_path: str (only on ok)
      stderr_output: str
      compiled_at: str ISO
    """
    flags = compile_flags or ["-O2", "-march=native"]
    src_hash = _source_hash(source_code, flags)

    # Check cache
    cached_so = settings.compiled_dir / src_hash / "node.so"
    if cached_so.exists():
        return {
            "status": "ok",
            "so_hash": src_hash,
            "so_path": str(cached_so),
            "stderr_output": "",
            "compiled_at": datetime.now(timezone.utc).isoformat(),
        }

    # Write source to temp dir
    build_root = Path(tempfile.mkdtemp(prefix=f"cvflow_build_{src_hash[:8]}_"))
    try:
        (build_root / "node.cpp").write_text(source_code)

        # Copy CMakeLists.txt with user flags substituted
        cmake_template = _CMAKE_TEMPLATE.read_text()
        user_flags = "\n".join(f"    {f}" for f in flags)
        cmake_content = cmake_template.replace("    # USER_FLAGS_PLACEHOLDER", user_flags)

        # Add extra_libs if any
        extra_cmake = ""
        for lib in (extra_libs or []):
            extra_cmake += f"\nfind_package({lib} REQUIRED)\ntarget_link_libraries(node PRIVATE {lib}::{lib})\n"
        cmake_content = cmake_content.replace("# EXTRA_LIBS_PLACEHOLDER", extra_cmake)
        (build_root / "CMakeLists.txt").write_text(cmake_content)

        # Copy SDK headers
        include_dst = build_root / "include"
        shutil.copytree(_SDK_INCLUDE, include_dst)

        # Run cmake configure
        configure_result = subprocess.run(
            ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
            cwd=build_root, capture_output=True, text=True, timeout=60,
        )
        if configure_result.returncode != 0:
            return {
                "status": "error",
                "so_hash": src_hash,
                "stderr_output": configure_result.stderr + configure_result.stdout,
                "compiled_at": datetime.now(timezone.utc).isoformat(),
            }

        # Run cmake build
        build_result = subprocess.run(
            ["cmake", "--build", "build", "--parallel"],
            cwd=build_root, capture_output=True, text=True, timeout=120,
        )
        stderr_combined = build_result.stderr + build_result.stdout

        if build_result.returncode != 0:
            return {
                "status": "error",
                "so_hash": src_hash,
                "stderr_output": stderr_combined,
                "compiled_at": datetime.now(timezone.utc).isoformat(),
            }

        # Copy .so to cache
        built_so = build_root / "build" / "node.so"
        if not built_so.exists():
            return {
                "status": "error",
                "so_hash": src_hash,
                "stderr_output": "Build succeeded but node.so not found",
                "compiled_at": datetime.now(timezone.utc).isoformat(),
            }

        dest_dir = settings.compiled_dir / src_hash
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_so, cached_so)

        return {
            "status": "ok",
            "so_hash": src_hash,
            "so_path": str(cached_so),
            "stderr_output": stderr_combined,
            "compiled_at": datetime.now(timezone.utc).isoformat(),
        }

    finally:
        shutil.rmtree(build_root, ignore_errors=True)
