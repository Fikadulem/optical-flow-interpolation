# optical-flow-interpolation

Frame interpolation using symmetric optical flow (Farneback), with spatial regularization and occlusion handling. The program reads pairs of frames, synthesizes the intermediate frame, computes quality metrics (MAIE, PSNR, SSIM), and writes outputs per dataset.

**Key Features**
- **Two Interpolation Modes:** Raw symmetric flow and Spatial regularized + Occlusion-aware.
- **Metrics:** Mean Absolute Interpolation Error (MAIE), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM).
- **Per-dataset outputs:** Saves `mid_raw.png` and `mid_reg.png` under `interpolated/<dataset>/`.

**Requirements**
- **C++:** `C++17` or newer.
- **Build tools:** `cmake` 3.16+ and `make` (or Ninja).
- **OpenCV:** 4.x built with contrib modules (for `ximgproc`).
	- macOS (Homebrew): `brew install cmake opencv`
	- If `ximgproc` is missing, build OpenCV from source with `opencv_contrib`.

**Data Layout**
Place datasets under the workspace in either `data/` (preferred) or `inputframes/`. The app auto-detects `data/`, falling back to `inputframes/` if `data/` is not present.

**Run**
Execute the binary from the `build/` directory:

```zsh
./optical_flow_interpolation
```
