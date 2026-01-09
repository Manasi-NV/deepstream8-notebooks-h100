#!/usr/bin/env python
# coding: utf-8

# # DeepStream 8.0 - Queue Sizing
# 
# 
# 

# In[1]:


# Import Required Libraries
import sys
import time

sys.path.append('/opt/nvidia/deepstream/deepstream-8.0/sources/deepstream_python_apps/apps')

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
from common.bus_call import bus_call
import pyds

Gst.init(None)

print(f"GStreamer version: {Gst.version_string()}")


# In[2]:


# Configuration - Define class labels and file paths

# Object class IDs (PeopleNet model)
PGIE_CLASS_ID_PERSON = 0
PGIE_CLASS_ID_BAG = 1
PGIE_CLASS_ID_FACE = 2

# ============================================================
# INPUT/OUTPUT CONFIGURATION
# ============================================================
INPUT_VIDEO_FILE = '/app/notebooks/videos/queue_retail.mp4'  # Queue monitoring video
OUTPUT_VIDEO_NAME = '/app/notebooks/queue_sizing_out.mp4'

# Config files - Using PeopleNet for better person detection
PGIE_CONFIG_FILE = '/app/notebooks/models/peoplenet/pgie_peoplenet_config.txt'
TRACKER_CONFIG_FILE = '/app/notebooks/tracker_config.txt'
ANALYTICS_CONFIG_FILE = '/app/notebooks/nvdsanalytics_config.txt'

# Display configuration
print("="*60)
print("QUEUE SIZING DETECTION WITH PEOPLENET")
print("="*60)
print(f"Model: PeopleNet (person, bag, face)")
print(f"Input video: {INPUT_VIDEO_FILE}")
print(f"Output video: {OUTPUT_VIDEO_NAME}")
print(f"PGIE config: {PGIE_CONFIG_FILE}")
print(f"Tracker config: {TRACKER_CONFIG_FILE}")
print(f"Analytics config: {ANALYTICS_CONFIG_FILE}")


# In[3]:


def make_elm_or_print_err(factoryname, name, printedname, detail=""):
    """Create a GStreamer element or print error message"""
    print(f"Creating {printedname}...")
    elm = Gst.ElementFactory.make(factoryname, name)
    if not elm:
        sys.stderr.write(f"Unable to create {printedname}\n")
    if detail:
            sys.stderr.write(detail)
    return elm



# In[4]:


print("\n" + "="*60)
print("CREATING PIPELINE")
print("="*60)

# Create Pipeline
pipeline = Gst.Pipeline()
if not pipeline:
    sys.stderr.write("Unable to create Pipeline\n")

# Create source elements for MP4 file
source = make_elm_or_print_err("filesrc", "file-source", "File Source")
demux = make_elm_or_print_err("qtdemux", "qt-demux", "QT Demuxer")

# Create decode elements
h264parser = make_elm_or_print_err("h264parse", "h264-parser", "H264 Parser")
decoder = make_elm_or_print_err("nvv4l2decoder", "nvv4l2-decoder", "NV Decoder")
streammux = make_elm_or_print_err("nvstreammux", "stream-muxer", "Stream Muxer")

# Create inference element
pgie = make_elm_or_print_err("nvinfer", "primary-inference", "Primary Inference (Person Detection)")

# Create tracker element (for consistent object IDs across frames)
tracker = make_elm_or_print_err("nvtracker", "tracker", "Object Tracker")

# Create analytics element (for ROI-based counting)
analytics = make_elm_or_print_err("nvdsanalytics", "analytics", "NV DS Analytics")

# Create display and output elements
nvvidconv = make_elm_or_print_err("nvvideoconvert", "convertor", "NV Video Converter 1")
nvosd = make_elm_or_print_err("nvdsosd", "onscreendisplay", "On-Screen Display")
nvvidconv2 = make_elm_or_print_err("nvvideoconvert", "convertor2", "NV Video Converter 2")
capsfilter = make_elm_or_print_err("capsfilter", "caps", "Caps Filter")
sw_videoconvert = make_elm_or_print_err("videoconvert", "sw-videoconvert", "Software Video Converter")

encoder = make_elm_or_print_err("x264enc", "encoder", "H264 Software Encoder")
h264parser2 = make_elm_or_print_err("h264parse", "h264-parser2", "H264 Parser 2")
mp4mux = make_elm_or_print_err("mp4mux", "mp4mux", "MP4 Muxer")
sink = make_elm_or_print_err("filesink", "filesink", "File Sink")



# In[5]:


# Configure element properties

print("\n" + "="*60)
print("CONFIGURING ELEMENTS")
print("="*60)

# File source configuration
source.set_property('location', INPUT_VIDEO_FILE)
print(f"Input file: {INPUT_VIDEO_FILE}")

# Streammux: Set batch properties (640x360 for retail video)
streammux.set_property('width', 640)
streammux.set_property('height', 360)
streammux.set_property('batch-size', 1)
streammux.set_property('batched-push-timeout', 4000000)
print("Stream muxer: 640x360, batch-size=1")

# Primary inference: Set config file
pgie.set_property('config-file-path', PGIE_CONFIG_FILE)
print(f"PGIE config: {PGIE_CONFIG_FILE}")

# Tracker: Set config file
tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream-8.0/lib/libnvds_nvmultiobjecttracker.so')
tracker.set_property('ll-config-file', '/opt/nvidia/deepstream/deepstream-8.0/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml')
tracker.set_property('tracker-width', 640)
tracker.set_property('tracker-height', 480)
tracker.set_property('display-tracking-id', 1)
print("Tracker: NvDCF tracker configured")

# Analytics: Set config file for ROI-based counting
analytics.set_property('config-file', ANALYTICS_CONFIG_FILE)
print(f"Analytics config: {ANALYTICS_CONFIG_FILE}")

# Capsfilter: Set format for encoder
caps = Gst.Caps.from_string("video/x-raw, format=I420")
capsfilter.set_property("caps", caps)
print("Caps filter: I420 format")

# Encoder: Set bitrate
encoder.set_property("bitrate", 4000)
encoder.set_property("speed-preset", "ultrafast")
encoder.set_property("tune", "zerolatency")  # 4 Mbps
print("Encoder bitrate: 4 Mbps")

# Sink: Set output file
sink.set_property('location', OUTPUT_VIDEO_NAME)
sink.set_property('sync', False)
print(f"Output file: {OUTPUT_VIDEO_NAME}")

print("\nAll elements configured!")


# In[6]:


# Add elements to pipeline and link them

print("\n" + "="*60)
print("BUILDING PIPELINE")
print("="*60)

# Callback for qtdemux dynamic pad linking (MP4 files have dynamic pads)
def on_demux_pad_added(demux, pad, h264parser):
    """Called when qtdemux creates a new pad (when video stream is found)"""
    pad_name = pad.get_name()
    print(f"Demux pad added: {pad_name}")

    # Only link video pads (ignore audio)
    if pad_name.startswith("video"):
        sink_pad = h264parser.get_static_pad("sink")
        if not sink_pad.is_linked():
            pad.link(sink_pad)
            print("Demux linked to h264parser")

# Add all elements to pipeline
print("Adding elements to pipeline...")
pipeline.add(source)
pipeline.add(demux)
pipeline.add(h264parser)
pipeline.add(decoder)
pipeline.add(streammux)
pipeline.add(pgie)
pipeline.add(tracker)      # NEW: Add tracker
pipeline.add(analytics)    # NEW: Add analytics
pipeline.add(nvvidconv)
pipeline.add(nvosd)
pipeline.add(nvvidconv2)
pipeline.add(capsfilter)
pipeline.add(sw_videoconvert)
pipeline.add(encoder)
pipeline.add(h264parser2)
pipeline.add(mp4mux)
pipeline.add(sink)
print("All elements added (including tracker and analytics)")

# Link elements
print("\nLinking elements...")

# Link source → demux (static)
source.link(demux)
print("Linked: source → demux")

# Connect demux pad-added callback for dynamic linking
demux.connect("pad-added", on_demux_pad_added, h264parser)
print("Demux pad-added callback connected")

# Link h264parser → decoder (static)
h264parser.link(decoder)
print("Linked: h264parser → decoder")

# Create pads for streammux
sinkpad = streammux.request_pad_simple("sink_0")
if not sinkpad:
    sys.stderr.write("Unable to get sink pad of streammux\n")
srcpad = decoder.get_static_pad("src")
if not srcpad:
    sys.stderr.write("Unable to get source pad of decoder\n")
srcpad.link(sinkpad)
print("Linked: decoder → streammux")

# Link remaining elements (NEW: added tracker and analytics)
streammux.link(pgie)
pgie.link(tracker)
tracker.link(analytics)
analytics.link(nvvidconv)
nvvidconv.link(nvosd)
nvosd.link(nvvidconv2)
nvvidconv2.link(capsfilter)
capsfilter.link(sw_videoconvert)
sw_videoconvert.link(encoder)
encoder.link(h264parser2)
h264parser2.link(mp4mux)
mp4mux.link(sink)
print("Linked: streammux → pgie → tracker → analytics → nvvidconv → nvosd → encoder → sink")

print("\nPipeline built successfully!")


# In[7]:


# Define metadata probe function for queue counting with OVERLAP-BASED logic

# ROI coordinates for queue area
ROI_LEFT = 20
ROI_TOP = 72
ROI_RIGHT = 640
ROI_BOTTOM = 300

# Filters
MIN_OVERLAP_RATIO = 0.3
MIN_BOX_WIDTH = 50
MIN_BOX_HEIGHT = 80
MIN_BOX_BOTTOM_Y = 200
CASHIER_ZONE_X = 490
CASHIER_ZONE_BOTTOM = 350

def calculate_overlap_ratio(box_left, box_top, box_width, box_height):
    box_right = box_left + box_width
    box_bottom = box_top + box_height
    inter_left = max(box_left, ROI_LEFT)
    inter_top = max(box_top, ROI_TOP)
    inter_right = min(box_right, ROI_RIGHT)
    inter_bottom = min(box_bottom, ROI_BOTTOM)
    if inter_left >= inter_right or inter_top >= inter_bottom:
        return 0.0
    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    box_area = box_width * box_height
    if box_area <= 0:
        return 0.0
    return inter_area / box_area

def is_foreground_person(box_left, box_top, box_width, box_height, box_bottom):
    if box_width < MIN_BOX_WIDTH or box_height < MIN_BOX_HEIGHT:
        return False
    if box_bottom < MIN_BOX_BOTTOM_Y:
        return False
    if box_left > CASHIER_ZONE_X and box_bottom > CASHIER_ZONE_BOTTOM:
        return False
    return True

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """Callback function to count people in queue using OVERLAP-BASED counting"""

    frame_number = 0
    people_in_queue = 0
    total_people = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    # Retrieve batch metadata from buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num

        # Count persons using OVERLAP-BASED logic with filters
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                if obj_meta.class_id == PGIE_CLASS_ID_PERSON:
                    total_people += 1
                    rect = obj_meta.rect_params
                    box_bottom = rect.top + rect.height

                    # Skip background/small people and cashier zone
                    if not is_foreground_person(rect.left, rect.top, rect.width, rect.height, box_bottom):
                        try:
                            l_obj = l_obj.next
                        except StopIteration:
                            break
                        continue

                    # Calculate overlap with ROI
                    overlap = calculate_overlap_ratio(rect.left, rect.top, rect.width, rect.height)
                    if overlap >= MIN_OVERLAP_RATIO:
                        people_in_queue += 1
            except StopIteration:
                break
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Add display metadata
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]

        # Set display text - show queue count prominently
        py_nvosd_text_params.display_text = "QUEUE COUNT: {} | Frame: {} | Total Detected: {}".format(
            people_in_queue, frame_number, total_people
        )

        # Position and style
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 40
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 12
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)  # Yellow
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)

        # Draw ROI rectangle (green box showing the queue area)
        display_meta.num_rects = 1
        rect_params = display_meta.rect_params[0]
        rect_params.left = ROI_LEFT
        rect_params.top = ROI_TOP
        rect_params.width = ROI_RIGHT - ROI_LEFT
        rect_params.height = ROI_BOTTOM - ROI_TOP
        rect_params.border_width = 3
        rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)  # Green border
        rect_params.has_bg_color = 0

        # Print to console every 50 frames
        if frame_number % 50 == 0:
            print(f"Frame {frame_number}: Queue={people_in_queue}, Total={total_people}")

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

print("Queue counting: OVERLAP-BASED with filters")
print(f"ROI: ({ROI_LEFT},{ROI_TOP}) to ({ROI_RIGHT},{ROI_BOTTOM})")
print(f"Filters: min_box={MIN_BOX_WIDTH}x{MIN_BOX_HEIGHT}, cashier_zone=X>{CASHIER_ZONE_X}")


# In[8]:


# Attach probe to OSD element

osdsinkpad = nvosd.get_static_pad("sink")
if not osdsinkpad:
    sys.stderr.write("Unable to get sink pad of nvosd\n")
else:
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    print("Metadata probe attached to OSD element")


# In[9]:


# Setup bus message handler

# Create event loop
loop = GLib.MainLoop()
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", bus_call, loop)

print("Bus message handler configured")


# In[10]:


# Run the pipeline

print("\n" + "="*60)
print("STARTING QUEUE SIZING PIPELINE")
print("="*60)
print(f"Input: {INPUT_VIDEO_FILE}")
print(f"Output: {OUTPUT_VIDEO_NAME}")
print("ROI: 60% of frame (center)")
print("Counting: People inside green ROI box")
print("="*60 + "\n")

start_time = time.time()

# Start pipeline
ret = pipeline.set_state(Gst.State.PLAYING)
if ret == Gst.StateChangeReturn.FAILURE:
    print(" ERROR: Unable to set pipeline to PLAYING state")
else:
    try:
        # Run event loop (blocks until EOS or error)
        loop.run()
    except KeyboardInterrupt:
        print("\n Interrupted by user")
    except Exception as e:
        print(f"\n Error: {e}")
    finally:
        # Cleanup
        print("\nCleaning up...")
        pipeline.set_state(Gst.State.NULL)

        elapsed_time = time.time() - start_time
        print(f"\n" + "="*60)
        print(f"PIPELINE COMPLETED")
        print(f" Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Output saved to: {OUTPUT_VIDEO_NAME}")
        print("="*60)


# In[11]:


import os

if os.path.exists(OUTPUT_VIDEO_NAME):
    file_size = os.path.getsize(OUTPUT_VIDEO_NAME)
    print(f" Output file exists")
    print(f"Location: {OUTPUT_VIDEO_NAME}")
    print(f"Size: {file_size / (1024*1024):.2f} MB")
    print(f"\nOn your host machine: ~/deepstream8/notebooks/ds_out.mp4")
else:
    print(f"Output file not found: {OUTPUT_VIDEO_NAME}")


# In[12]:


# Display output video with HTML5 player

from IPython.display import HTML
import os

if os.path.exists(OUTPUT_VIDEO_NAME):
    # Create HTML5 video player
    html = f"""
    <div style="text-align: center; margin: 20px;">
        <h3>🎬 Queue Sizing Detection Output</h3>
        <video width="800" controls>
            <source src="queue_sizing_out.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <p style="margin-top: 10px;">
            <strong>File:</strong> queue_sizing_out.mp4 | 
            <strong>Size:</strong> {os.path.getsize(OUTPUT_VIDEO_NAME) / (1024*1024):.2f} MB
        </p>
    </div>
    """
    display(HTML(html))
else:
    print(f"Video not found: {OUTPUT_VIDEO_NAME}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




