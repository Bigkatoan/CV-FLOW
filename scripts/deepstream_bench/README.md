# DeepStream benchmark (for the comparison in the repo's top-level README)

Custom NVIDIA DeepStream pipeline running the SAME `tests/fixtures/yolov8n.onnx`
model used everywhere else in this repo's benchmarks, for a fair comparison
against cv-flow / a naive Python pipeline / plain GStreamer. Not part of the
`cv_flow` Python package — standalone, one-off benchmark tooling.

DeepStream ships bbox parsers for YOLOv3/v4-style outputs only. YOLOv8's ONNX
export has a different raw tensor layout, so a custom parser is required —
`nvdsinfer_yolov8_parser.cpp` decodes it using the exact same math as
`cv_flow/nodes/postprocess.py::run_nms()` (features-first `(84, 8400)` layout,
cx/cy/w/h → xyxy, per-box argmax over 80 COCO classes), so this is a fair
same-algorithm comparison across runtimes, not a different detector.

## Reproduce

```bash
# 1. Install DeepStream (one-time, ~632MB):
sudo apt install -y deepstream-7.1 libgstrtspserver-1.0-0 libjsoncpp25

# 2. Build the custom parser:
cd scripts/deepstream_bench
make

# 3. Generate a baseline-profile H.264 test file (Tegra's HW decoder rejects
#    x264enc's default profile — must be explicit, see Makefile/this command):
gst-launch-1.0 videotestsrc num-buffers=150 pattern=ball ! \
  video/x-raw,width=1280,height=720,framerate=30/1 ! videoconvert ! \
  x264enc tune=zerolatency speed-preset=ultrafast key-int-max=15 ! \
  video/x-h264,profile=baseline,stream-format=byte-stream ! h264parse ! \
  mp4mux ! filesink location=test_input.mp4

# 4. Uncapped throughput (uses the engine cache after the first run, which
#    builds a TensorRT engine next to the model — that first run is slow):
deepstream-app -c deepstream_app_config_file.txt

# 5. Live USB camera (replace camera-v4l2-dev-node in the config with your
#    actual /dev/videoN — see README.md's camera notes for finding it):
deepstream-app -c deepstream_app_config_camera.txt
```

Both configs disable `[tiled-display]` and `[osd]` (`enable=0`) — on this
headless Jetson (no `$DISPLAY`), leaving either enabled made `deepstream-app`
segfault during pipeline startup. FPS is read from the periodic
`**PERF:  <fps> (<avg>)` lines (`enable-perf-measurement=1`).

## Verifying the parser decodes real detections (not just "runs fast")

```bash
gst-launch-1.0 -e v4l2src device=/dev/videoN num-buffers=45 ! \
  video/x-raw,width=1280,height=720,framerate=30/1 ! \
  nvvideoconvert ! "video/x-raw(memory:NVMM),format=NV12" ! \
  m.sink_0 nvstreammux name=m batch-size=1 width=1280 height=720 live-source=1 ! \
  nvinfer config-file-path=config_infer_primary_yolov8.txt ! \
  nvdsosd ! nvvideoconvert ! video/x-raw,format=I420 ! jpegenc ! \
  multifilesink location="frame_%02d.jpg"
```

Inspect the last frame — should show real `person`/etc. boxes correctly
placed on actual objects in view.
