# Source this before running anything: `source env.sh`.
# Headless MuJoCo rendering through EGL.
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# WSL2 only: Mesa's D3D12 driver on the integrated GPU was the fastest renderer measured here
# (15 steps/s with two 128x128 cameras, against 4 for the default llvmpipe).
if grep -qi microsoft /proc/version 2>/dev/null; then
  export GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD
fi
# MediaPipe needs libGLESv2 (Ubuntu package libgles2). Without root, extract it into syslibs/.
if [ -d "$(dirname "${BASH_SOURCE[0]}")/syslibs/root" ]; then
  export LD_LIBRARY_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/syslibs/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi
