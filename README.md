# Ndev Vision Lab

A self-contained, from-scratch visual-matching experiment. It has no pretrained weights, downloaded vision model, hosted vision API, OpenCV, TensorFlow, or PyTorch.

Its first capability is multi-scale template matching: find where a reference image appears inside a larger image, while testing different coordinates and sizes.

## Project structure

    Ndev_vision/
    |-- app.py                  Streamlit interface and progress display
    |-- requirements.txt        Python dependencies
    |-- README.md               This guide
    |-- core/
    |   |-- __init__.py         Core package marker
    |   '-- vision_matcher.py   Matching algorithm and result object
    '-- tests/
        '-- test_matcher.py     Synthetic verification test

## What it does

1. Upload a larger search image.
2. Upload a smaller reference image or object crop to find.
3. Convert both images to grayscale.
4. Resize the reference through every selected scale.
5. Compare it against each candidate x/y coordinate.
6. Measure mean absolute pixel difference.
7. Keep the lowest-error result and draw its bounding box.

The UI reports live progress, best similarity, final scale, top-left coordinate, comparison count, and shows the best match with a red rectangle.

## Algorithm flow

    search image + reference image
                |
                v
    grayscale conversion
                |
                v
    resize reference for each scale
                |
                v
    compare every candidate x/y patch
                |
                v
    retain lowest error
                |
                v
    display score and bounding box

Calculation:

    error      = average(abs(search_patch_pixel - reference_pixel))
    similarity = 1 - error

Values are normalized to the 0-1 range. Exact same pixels produce 1.0 or 100% similarity.

## Source files

### app.py

The Streamlit user-interface layer. It reads uploads, shows controls, calls the core algorithm, updates live progress, and draws the result. The algorithm is intentionally kept separate, so it can later be used from a webcam, batch benchmark, or command-line utility.

### core/vision_matcher.py

The independent, from-scratch algorithm layer.

- prepare_image converts to grayscale and limits working size for responsive searching.
- _features returns either pixel brightness or a hand-built edge-magnitude map.
- match_template performs the coordinate and scale search.
- MatchResult carries score, error, location, dimensions, scale, and comparison count.

### tests/test_matcher.py

Creates a synthetic patch at coordinate (47, 31), then asserts the matcher recovers it exactly at scale 1.0.

## Interface controls

| Control | Purpose | Tradeoff |
|---|---|---|
| Comparison signal | Choose pixels or edges | Edges cope better with brightness changes |
| Reference scale range | Smallest and largest reference size | Wider range creates more work |
| Scale samples | Number of sizes tested | Finer scale estimate but slower |
| Coordinate stride | Space between candidate positions | Lower is more exact but slower |
| Working image limit | Maximum processed image side | More detail but more comparisons |

Pixels mode is direct grayscale comparison: the closest expression of the original coordinate-by-coordinate idea. It works best for similar crop, pose, illumination, and image quality.

Edges mode calculates simple horizontal and vertical brightness gradients with NumPy and compares edge strength. It is still our own hand-written feature and can be less sensitive to overall brightness.

## Setup and run

### Prerequisite

Install a standard 64-bit Python 3.11 or later distribution. When this project was initialized, the computer exposed only Microsoft Store launcher stubs for python and py; therefore no virtual environment could be created and the interface/test could not yet be run.

After Python is installed, run in PowerShell:

    cd E:\Ndev\Ndev_vision
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    streamlit run app.py

Streamlit prints a local address, normally http://localhost:8501.

If activation is blocked for this terminal session:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .\.venv\Scripts\Activate.ps1

## Verification

With the virtual environment active:

    python -m pytest

Expected result: one passing synthetic test.

## Known limits

This is deliberately a baseline, not semantic vision:

- Rotation, perspective, and major viewpoint changes break pixel correspondence.
- Occlusion can hide parts of the reference.
- Major lighting changes lower direct-pixel similarity.
- Dense searches on large images can be CPU-intensive.
- It does not yet recognize concepts or classes such as cats, cars, or faces.

## Roadmap

1. Test rotated references.
2. Add RGB/HSV comparison and brightness normalization.
3. Implement hand-designed features: corners, local gradients, orientation histograms, and image pyramids.
4. Add coarse-to-fine search: low-resolution scan, then detailed local refinement.
5. Create a private evaluation dataset with expected boxes; track score, position error, scale error, and runtime.
6. Later, define a small neural network with random initialization and train it only on collected or licensed data. The current matcher remains the baseline to beat.
7. Add webcam or frame-by-frame input and graph scores over time.

## Development rules

- Keep UI code in app.py and algorithms in core/.
- Add a synthetic test for every new core feature.
- Change one experimental variable at a time.
- Treat score as relative until calibrated on an evaluation dataset.
- Record failures: each failure describes the next improvement needed.

## Dependencies

- Streamlit: local interface.
- NumPy: array and numerical computation.
- Pillow: image loading and resizing.
- pytest: automated testing.

None of these supplies pretrained vision weights.
