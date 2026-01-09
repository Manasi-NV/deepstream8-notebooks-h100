#!/usr/bin/env python
# coding: utf-8

# DeepStream 8.0 - Queue Sizing for Crowded Retail Store
# V-Shaped Boundary with Cashier Exclusion Zone

import sys
import time
import os

sys.path.append('/opt/nvidia/deepstream/deepstream-8.0/sources/deepstream_python_apps/apps')

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
from common.bus_call import bus_call
import pyds

Gst.init(None)

print(f"GStreamer version: {Gst.version_string()}")

PGIE_CLASS_ID_PERSON = 0

# Input/Output paths
INPUT_VIDEO_FILE = '/app/notebooks/videos/queue_detection_crowded.mp4'
OUTPUT_VIDEO_NAME = '/app/notebooks/queue_sizing_crowded_out.mp4'

# Config files
PGIE_CONFIG_FILE = '/app/notebooks/models/peoplenet/pgie_peoplenet_config.txt'

# Video dimensions (640x480 for this video)
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

BOUNDARY_P1 = (170, 0)      # Top start
BOUNDARY_P2 = (340, 200)    # First bend
BOUNDARY_P3 = (420, 300)    # Bottom corner (lowest point)
BOUNDARY_P4 = (550, 200)    # Going back up
BOUNDARY_P5 = (600, 160)    # End point (top right)

DEBUG_MODE = True

print("="*60)
print("QUEUE SIZING - CROWDED RETAIL STORE")
print("="*60)
print(f"Input video: {INPUT_VIDEO_FILE}")
print(f"Output video: {OUTPUT_VIDEO_NAME}")
print(f"Video dimensions: {VIDEO_WIDTH}x{VIDEO_HEIGHT}")
print(f"Boundary: {BOUNDARY_P1} → {BOUNDARY_P2} → {BOUNDARY_P3} → {BOUNDARY_P4} → {BOUNDARY_P5}")
print(f"Rule: Between boundary lines = QUEUE (count), Outside = CASHIER (exclude)")
print("="*60)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_elm_or_print_err(factoryname, name, printedname, detail=""):
    """Create a GStreamer element or print error message"""
    print(f"Creating {printedname}...")
    elm = Gst.ElementFactory.make(factoryname, name)
    if not elm:
        sys.stderr.write(f"Unable to create {printedname}\n")
        if detail:
            sys.stderr.write(detail)
    return elm


def get_left_boundary_x(y):
    """Get X on the LEFT boundary (P1->P2->P3)"""
    # For 640x480: P1(170,0), P2(340,200), P3(420,300)
    if y <= 200:
        # P1(170,0) to P2(340,200)
        return 170 + (y / 200) * 170  # 340-170=170
    elif y <= 300:
        # P2(340,200) to P3(420,300)
        return 340 + ((y - 200) / 100) * 80  # 420-340=80
    else:
        # Below P3, return P3's x
        return 420


def get_right_boundary_x(y):
    """Get X on the RIGHT boundary (P5->P4->P3)"""
    # For 640x480: P5(600,160), P4(550,200), P3(420,300)
    if y <= 160:
        # Above P5, extend to frame edge (queue extends to right)
        return 640
    elif y <= 200:
        # P5(600,160) to P4(550,200)
        return 600 - ((y - 160) / 40) * 50  # 600-550=50
    elif y <= 300:
        # P4(550,200) to P3(420,300)
        return 550 - ((y - 200) / 100) * 130  # 550-420=130
    else:
        # Below P3, return P3's x
        return 420


def is_in_cashier_zone(box_left, box_top, box_width, box_height):
    """
    Check if a person's bounding box is in the CASHIER ZONE.
    Returns True if person should be EXCLUDED from queue count.
    
    Zone logic:
    - Y > 300: ALL CASHIER (bottom area behind counter)
    - Y <= 300: CASHIER if LEFT of P1-P2-P3 OR RIGHT of P3-P4-P5
                QUEUE if BETWEEN the two boundary lines
    """
    box_center_x = box_left + box_width / 2
    box_center_y = box_top + box_height / 2
    
    # Below Y=300 is ALL cashier zone
    if box_center_y > 300:
        return True
    
    # Above Y=300, check if between the two boundaries (queue) or outside (cashier)
    left_boundary = get_left_boundary_x(box_center_y)
    right_boundary = get_right_boundary_x(box_center_y)
    
    # If between boundaries = QUEUE (not cashier)
    if left_boundary <= box_center_x <= right_boundary:
        return False
    
    # Otherwise = CASHIER (left of left boundary or right of right boundary)
    return True


# ============================================================
# CREATE PIPELINE
# ============================================================

print("\n" + "="*60)
print("CREATING PIPELINE")
print("="*60)

# Create Pipeline
pipeline = Gst.Pipeline()
if not pipeline:
    sys.stderr.write("Unable to create Pipeline\n")

# Source elements
source = make_elm_or_print_err("filesrc", "file-source", "File Source")
demux = make_elm_or_print_err("qtdemux", "qt-demux", "QT Demuxer")

# Decode elements - Video is MPEG4 (FMP4), not H264
mpeg4parser = make_elm_or_print_err("mpeg4videoparse", "mpeg4-parser", "MPEG4 Parser")
decoder = make_elm_or_print_err("nvv4l2decoder", "nvv4l2-decoder", "NV Decoder")
streammux = make_elm_or_print_err("nvstreammux", "stream-muxer", "Stream Muxer")

# Inference element
pgie = make_elm_or_print_err("nvinfer", "primary-inference", "Primary Inference")

# Tracker element
tracker = make_elm_or_print_err("nvtracker", "tracker", "Object Tracker")

# Display and output elements
nvvidconv = make_elm_or_print_err("nvvideoconvert", "convertor", "NV Video Converter 1")
nvosd = make_elm_or_print_err("nvdsosd", "onscreendisplay", "On-Screen Display")
nvvidconv2 = make_elm_or_print_err("nvvideoconvert", "convertor2", "NV Video Converter 2")
capsfilter = make_elm_or_print_err("capsfilter", "caps", "Caps Filter")
sw_videoconvert = make_elm_or_print_err("videoconvert", "sw-videoconvert", "Software Video Converter")

encoder = make_elm_or_print_err("x264enc", "encoder", "H264 Software Encoder")
h264parser2 = make_elm_or_print_err("h264parse", "h264-parser2", "H264 Parser 2")
mp4mux = make_elm_or_print_err("mp4mux", "mp4mux", "MP4 Muxer")
sink = make_elm_or_print_err("filesink", "filesink", "File Sink")


# ============================================================
# CONFIGURE ELEMENTS
# ============================================================

print("\n" + "="*60)
print("CONFIGURING ELEMENTS")
print("="*60)

# File source
source.set_property('location', INPUT_VIDEO_FILE)
print(f"Input: {INPUT_VIDEO_FILE}")

# Streammux - 640x480 for this video
streammux.set_property('width', VIDEO_WIDTH)
streammux.set_property('height', VIDEO_HEIGHT)
streammux.set_property('batch-size', 1)
streammux.set_property('batched-push-timeout', 4000000)
print(f"Stream muxer: {VIDEO_WIDTH}x{VIDEO_HEIGHT}, batch-size=1")

# Primary inference
pgie.set_property('config-file-path', PGIE_CONFIG_FILE)
print(f"PGIE config: {PGIE_CONFIG_FILE}")

# Tracker
tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream-8.0/lib/libnvds_nvmultiobjecttracker.so')
tracker.set_property('ll-config-file', '/opt/nvidia/deepstream/deepstream-8.0/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml')
tracker.set_property('tracker-width', 640)
tracker.set_property('tracker-height', 480)
tracker.set_property('display-tracking-id', 1)
print("Tracker: NvDCF configured")

# Caps filter
caps = Gst.Caps.from_string("video/x-raw, format=I420")
capsfilter.set_property("caps", caps)

# Encoder
encoder.set_property("bitrate", 8000)
encoder.set_property("speed-preset", "medium")
encoder.set_property("tune", "zerolatency")
print("Encoder: 8 Mbps, medium preset")

# Sink
sink.set_property('location', OUTPUT_VIDEO_NAME)
sink.set_property('sync', False)
print(f"Output: {OUTPUT_VIDEO_NAME}")


# ============================================================
# BUILD PIPELINE
# ============================================================

print("\n" + "="*60)
print("BUILDING PIPELINE")
print("="*60)

def on_demux_pad_added(demux, pad, mpeg4parser):
    """Callback for qtdemux dynamic pad linking"""
    pad_name = pad.get_name()
    print(f"Demux pad added: {pad_name}")
    if pad_name.startswith("video"):
        sink_pad = mpeg4parser.get_static_pad("sink")
        if not sink_pad.is_linked():
            ret = pad.link(sink_pad)
            print(f"Demux linked to mpeg4parser: {ret}")

# Add elements to pipeline
print("Adding elements to pipeline...")
for elem in [source, demux, mpeg4parser, decoder, streammux, pgie, tracker,
             nvvidconv, nvosd, nvvidconv2, capsfilter, sw_videoconvert,
             encoder, h264parser2, mp4mux, sink]:
    pipeline.add(elem)
print("All elements added")

# Link elements
print("Linking elements...")
source.link(demux)
demux.connect("pad-added", on_demux_pad_added, mpeg4parser)
mpeg4parser.link(decoder)

# Streammux pad
sinkpad = streammux.request_pad_simple("sink_0")
srcpad = decoder.get_static_pad("src")
srcpad.link(sinkpad)

# Link the rest
streammux.link(pgie)
pgie.link(tracker)
tracker.link(nvvidconv)
nvvidconv.link(nvosd)
nvosd.link(nvvidconv2)
nvvidconv2.link(capsfilter)
capsfilter.link(sw_videoconvert)
sw_videoconvert.link(encoder)
encoder.link(h264parser2)
h264parser2.link(mp4mux)
mp4mux.link(sink)
print("Pipeline linked: source → decode → pgie → tracker → osd → encode → output")


# ============================================================
# METADATA PROBE FOR QUEUE COUNTING
# ============================================================

# Statistics tracking
video_stats = {
    'total_frames': 0,
    'total_person_detections': 0,
    'total_queue_detections': 0,
    'total_cashier_detections': 0,
    'max_queue_count': 0,
    'max_total_count': 0,
    'unique_tracker_ids': set(),
    'confidence_scores': [],
    'box_widths': [],
    'box_heights': [],
    'box_areas': [],
    'detections_per_frame': [],
}

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """
    Probe callback to count people in queue and draw debug overlays.
    """
    global video_stats
    
    frame_number = 0
    queue_count = 0
    cashier_count = 0
    total_persons = 0
    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK
    
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        frame_number = frame_meta.frame_num
        
        # Process each detected object
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                
                if obj_meta.class_id == PGIE_CLASS_ID_PERSON:
                    total_persons += 1
                    rect = obj_meta.rect_params
                    box_left = rect.left
                    box_top = rect.top
                    box_width = rect.width
                    box_height = rect.height
                    box_bottom = box_top + box_height
                    box_center_x = box_left + box_width / 2
                    box_center_y = box_top + box_height / 2
                    
                    # Collect stats for comparison
                    video_stats['unique_tracker_ids'].add(obj_meta.object_id)
                    video_stats['confidence_scores'].append(obj_meta.confidence)
                    video_stats['box_widths'].append(box_width)
                    video_stats['box_heights'].append(box_height)
                    video_stats['box_areas'].append(box_width * box_height)
                    
                    # Classification logic
                    # Get text params for this object to change label
                    txt_params = obj_meta.text_params
                    
                    # Simple logic: Cashier zone = Staff, everything else = Queue
                    if is_in_cashier_zone(box_left, box_top, box_width, box_height):
                        cashier_count += 1
                        video_stats['total_cashier_detections'] += 1
                        # Color: Red for cashier (excluded)
                        rect.border_color.set(1.0, 0.0, 0.0, 1.0)
                        rect.border_width = 2
                        txt_params.display_text = "Cashier"
                        
                    else:
                        # Everyone else is in the queue!
                        queue_count += 1
                        video_stats['total_queue_detections'] += 1
                        # Color: Bright GREEN for queue (counted!)
                        rect.border_color.set(0.0, 1.0, 0.0, 1.0)
                        rect.border_width = 4
                        txt_params.display_text = "Queue"
                    
                    # Debug output for first few frames
                    if DEBUG_MODE and frame_number < 5:
                        print(f"  Person: center=({box_center_x:.0f},{box_center_y:.0f}) "
                              f"size={box_width:.0f}x{box_height:.0f} bottom={box_bottom:.0f}")
                
            except StopIteration:
                break
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        
        # Create display overlay
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        
        # ========== TEXT OVERLAYS ==========
        display_meta.num_labels = 3
        
        # Main queue count (large, top-left)
        py_nvosd_text_params = display_meta.text_params[0]
        py_nvosd_text_params.display_text = f"QUEUE: {queue_count}"
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 30
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 14
        py_nvosd_text_params.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)  # Green
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.8)
        
        # Frame info
        text1 = display_meta.text_params[1]
        text1.display_text = f"Frame: {frame_number} | Total: {total_persons}"
        text1.x_offset = 10
        text1.y_offset = 55
        text1.font_params.font_name = "Serif"
        text1.font_params.font_size = 10
        text1.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        text1.set_bg_clr = 1
        text1.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)
        
        # Cashier count (excluded)
        text2 = display_meta.text_params[2]
        text2.display_text = f"Cashiers (excl): {cashier_count}"
        text2.x_offset = 400
        text2.y_offset = 30
        text2.font_params.font_name = "Serif"
        text2.font_params.font_size = 10
        text2.font_params.font_color.set(1.0, 0.3, 0.3, 1.0)  # Red
        text2.set_bg_clr = 1
        text2.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)
        
        # ========== DRAW 5-POINT BOUNDARY LINE ==========
        display_meta.num_lines = 4
        
        # Segment 1: P1 to P2
        line0 = display_meta.line_params[0]
        line0.x1 = BOUNDARY_P1[0]
        line0.y1 = BOUNDARY_P1[1]
        line0.x2 = BOUNDARY_P2[0]
        line0.y2 = BOUNDARY_P2[1]
        line0.line_width = 3
        line0.line_color.set(1.0, 0.5, 0.0, 1.0)  # Orange
        
        # Segment 2: P2 to P3
        line1 = display_meta.line_params[1]
        line1.x1 = BOUNDARY_P2[0]
        line1.y1 = BOUNDARY_P2[1]
        line1.x2 = BOUNDARY_P3[0]
        line1.y2 = BOUNDARY_P3[1]
        line1.line_width = 3
        line1.line_color.set(1.0, 0.5, 0.0, 1.0)  # Orange
        
        # Segment 3: P3 to P4
        line2 = display_meta.line_params[2]
        line2.x1 = BOUNDARY_P3[0]
        line2.y1 = BOUNDARY_P3[1]
        line2.x2 = BOUNDARY_P4[0]
        line2.y2 = BOUNDARY_P4[1]
        line2.line_width = 3
        line2.line_color.set(1.0, 0.5, 0.0, 1.0)  # Orange
        
        # Segment 4: P4 to P5
        line3 = display_meta.line_params[3]
        line3.x1 = BOUNDARY_P4[0]
        line3.y1 = BOUNDARY_P4[1]
        line3.x2 = BOUNDARY_P5[0]
        line3.y2 = BOUNDARY_P5[1]
        line3.line_width = 3
        line3.line_color.set(1.0, 0.5, 0.0, 1.0)  # Orange
        
        # No rectangle needed
        display_meta.num_rects = 0
        
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        
        # Update video stats
        video_stats['total_frames'] += 1
        video_stats['total_person_detections'] += total_persons
        video_stats['detections_per_frame'].append(total_persons)
        video_stats['max_queue_count'] = max(video_stats['max_queue_count'], queue_count)
        video_stats['max_total_count'] = max(video_stats['max_total_count'], total_persons)
        
        # Console output every 30 frames
        if frame_number % 30 == 0:
            print(f"Frame {frame_number:4d}: Queue={queue_count} | Cashier={cashier_count} | Total={total_persons}")
        
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    
    return Gst.PadProbeReturn.OK


# Attach probe
osdsinkpad = nvosd.get_static_pad("sink")
if osdsinkpad:
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    print("Probe attached to OSD sink pad")
else:
    print("ERROR: Could not get OSD sink pad")


# ============================================================
# RUN PIPELINE
# ============================================================

# Bus handler
loop = GLib.MainLoop()
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", bus_call, loop)

print("\n" + "="*60)
print("STARTING QUEUE DETECTION PIPELINE")
print("="*60)
print(f"Input: {INPUT_VIDEO_FILE}")
print(f"Output: {OUTPUT_VIDEO_NAME}")
print()
print("Legend:")
print("  GREEN box = Customer in QUEUE (counted)")
print("  RED box = Cashier (excluded)")
print("  ORANGE line = V-shaped boundary")
print("="*60 + "\n")

start_time = time.time()

# Start pipeline
ret = pipeline.set_state(Gst.State.PLAYING)
if ret == Gst.StateChangeReturn.FAILURE:
    print("ERROR: Unable to set pipeline to PLAYING state")
    sys.exit(1)

try:
    loop.run()
except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"\nError: {e}")
finally:
    pipeline.set_state(Gst.State.NULL)
    
    elapsed_time = time.time() - start_time
    print(f"\n" + "="*60)
    print(f"PIPELINE COMPLETED")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Output saved to: {OUTPUT_VIDEO_NAME}")
    print("="*60)


# ============================================================
# VERIFY OUTPUT
# ============================================================

if os.path.exists(OUTPUT_VIDEO_NAME):
    file_size = os.path.getsize(OUTPUT_VIDEO_NAME)
    print(f"\nOutput file: {OUTPUT_VIDEO_NAME}")
    print(f"Size: {file_size / (1024*1024):.2f} MB")
else:
    print(f"\nWARNING: Output file not found: {OUTPUT_VIDEO_NAME}")


# ============================================================
# PRINT COMPARISON STATISTICS
# ============================================================

print("\n" + "="*60)
print("DETECTION STATISTICS (for video comparison)")
print("="*60)

print(f"\n{'='*40}")
print("SUMMARY METRICS")
print(f"{'='*40}")
print(f"  Total frames processed:      {video_stats['total_frames']}")
print(f"  Total person detections:     {video_stats['total_person_detections']}")
print(f"  Unique people tracked:       {len(video_stats['unique_tracker_ids'])}")
print(f"  Max simultaneous queue:      {video_stats['max_queue_count']}")
print(f"  Max simultaneous total:      {video_stats['max_total_count']}")

if video_stats['total_frames'] > 0:
    avg_per_frame = video_stats['total_person_detections'] / video_stats['total_frames']
    print(f"  Avg detections per frame:    {avg_per_frame:.2f}")

print(f"\n{'='*40}")
print("QUEUE vs CASHIER BREAKDOWN")
print(f"{'='*40}")
print(f"  Total queue detections:      {video_stats['total_queue_detections']}")
print(f"  Total cashier detections:    {video_stats['total_cashier_detections']}")

if video_stats['confidence_scores']:
    import statistics
    avg_conf = statistics.mean(video_stats['confidence_scores'])
    min_conf = min(video_stats['confidence_scores'])
    max_conf = max(video_stats['confidence_scores'])
    print(f"\n{'='*40}")
    print("DETECTION CONFIDENCE")
    print(f"{'='*40}")
    print(f"  Average confidence:          {avg_conf:.3f}")
    print(f"  Min confidence:              {min_conf:.3f}")
    print(f"  Max confidence:              {max_conf:.3f}")

if video_stats['box_areas']:
    avg_area = statistics.mean(video_stats['box_areas'])
    min_area = min(video_stats['box_areas'])
    max_area = max(video_stats['box_areas'])
    avg_width = statistics.mean(video_stats['box_widths'])
    avg_height = statistics.mean(video_stats['box_heights'])
    min_width = min(video_stats['box_widths'])
    min_height = min(video_stats['box_heights'])
    print(f"\n{'='*40}")
    print("BOUNDING BOX SIZES")
    print(f"{'='*40}")
    print(f"  Average box size:            {avg_width:.0f} x {avg_height:.0f} px")
    print(f"  Smallest box:                {min_width:.0f} x {min_height:.0f} px")
    print(f"  Average box area:            {avg_area:.0f} px²")
    print(f"  Min box area:                {min_area:.0f} px²")
    print(f"  Max box area:                {max_area:.0f} px²")

print("\n" + "="*60)
print("USE THESE METRICS TO COMPARE VIDEOS")
print("Higher quality video should show:")
print("  - More unique tracker IDs")
print("  - Higher avg detections per frame")
print("  - Smaller min bounding box (detecting farther people)")
print("  - Higher average confidence scores")
print("="*60)

