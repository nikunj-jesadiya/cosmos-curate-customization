# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RF-DETR + BoxMOT tracking stage for pseudo-labeling.

This stage processes video clips using RF-DETR for object detection
and BoxMOT trackers for multi-object tracking. It generates structured
pseudo-labels with instance and per-frame annotations.
"""

from __future__ import annotations

import colorsys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
import nvtx  # type: ignore[import-untyped]
import torch
from loguru import logger
from PIL import Image

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.infra.gpu_start_helper import (
    gpu_stage_cleanup,
    gpu_stage_startup,
)
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.pipelines.video.utils.data_model import (
    BoundingBox2D,
    Clip,
    FrameAnnotation,
    FrameInstance,
    SplitPipeTask,
    TrackedInstance,
    TrackingConfig,
    TrackingResult,
)

if TYPE_CHECKING:
    import numpy.typing as npt

TrackerType = Literal["deepocsort", "botsort", "bytetrack", "boosttrack", "hybridsort"]

# Constants for track array indexing
TRACK_CONFIDENCE_IDX = 5
TRACK_CLASS_ID_IDX = 6


def _generate_distinct_color(index: int) -> list[int]:
    """Generate a distinct RGB color for a given index using golden ratio distribution.

    Args:
        index: Index of the color to generate.

    Returns:
        List of [R, G, B] values (0-255).

    """
    hue = (index * 0.618033988749895) % 1.0  # Golden ratio for distribution
    saturation = 0.7 + (index % 3) * 0.1
    value = 0.9 - (index % 2) * 0.1
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return [int(r * 255), int(g * 255), int(b * 255)]


def _draw_detections(
    image: np.ndarray,
    bboxes: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    """Draw detection bounding boxes and labels on an image.

    Args:
        image: Input image (BGR format).
        bboxes: Bounding boxes array of shape (N, 4) in [x1, y1, x2, y2] format.
        class_ids: Class ID array of shape (N,).
        class_names: List of class names.

    Returns:
        Annotated image.

    """
    img = image.copy()
    line_thickness = 1
    font_scale = 0.4

    for bbox, class_id_raw in zip(bboxes, class_ids, strict=False):
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(class_id_raw)

        # Generate color based on class_id
        color = _generate_distinct_color(class_id)
        # Convert RGB to BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, line_thickness)

        # Prepare label (class name only, no confidence)
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        label = class_name

        # Draw label text (white with black outline for visibility, no background box)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_pos = (x1 + 2, y1 - 4)
        # Draw black outline for contrast
        cv2.putText(img, label, text_pos, font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        # Draw white text on top
        cv2.putText(img, label, text_pos, font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def _draw_tracks(
    image: np.ndarray,
    instances: list[dict[str, Any]],
    instances_info: dict[str, dict[str, Any]],
) -> np.ndarray:
    """Draw tracking bounding boxes and labels on an image.

    Args:
        image: Input image (BGR format).
        instances: List of instance dicts with object_id, bounding_box_2d_tight, etc.
        instances_info: Dict mapping object_id to instance info (for color).

    Returns:
        Annotated image.

    """
    img = image.copy()
    line_thickness = 1
    font_scale = 0.4

    for inst in instances:
        object_id = inst["object_id"]
        bbox = inst["bounding_box_2d_tight"]
        class_id = inst.get("semantic_id", 0)

        x1, y1, x2, y2 = map(int, bbox)

        # Get color from instances_info (based on track_id for consistency)
        if object_id in instances_info:
            color = instances_info[object_id]["color"]
        else:
            color = _generate_distinct_color(class_id)

        # Convert RGB to BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, line_thickness)

        # Prepare label with track ID (no confidence score)
        label = object_id

        # Draw label text (white with black outline for visibility, no background box)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_pos = (x1 + 2, y1 - 4)
        # Draw black outline for contrast
        cv2.putText(img, label, text_pos, font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        # Draw white text on top
        cv2.putText(img, label, text_pos, font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def _encode_frames_to_video(
    frames: list[np.ndarray],
    fps: float = 30.0,
) -> bytes:
    """Encode a list of frames to MP4 video bytes.

    Args:
        frames: List of frames (BGR format, numpy arrays).
        fps: Frames per second for the output video.

    Returns:
        MP4 video as bytes.

    """
    if not frames:
        return b""

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
        for frame in frames:
            writer.write(frame)
        writer.release()

        video_bytes = tmp_path.read_bytes()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return video_bytes


def _get_tracker(
    tracker_name: TrackerType,
    device: str,
    config: TrackingConfig,
) -> Any:  # noqa: ANN401
    """Create and return a BoxMOT tracker instance.

    Args:
        tracker_name: Name of the tracker to use.
        device: Device to run the tracker on.
        config: Tracking configuration with per_class, thresholds, etc.

    Returns:
        Tracker instance.

    Raises:
        ImportError: If boxmot is not installed.
        ValueError: If tracker_name is not valid.

    """
    try:
        from boxmot import BoostTrack, BotSort, ByteTrack, DeepOcSort, HybridSort  # noqa: PLC0415
    except ImportError as e:
        msg = "Please install boxmot: pip install boxmot"
        raise ImportError(msg) from e

    trackers = {
        "deepocsort": DeepOcSort,
        "botsort": BotSort,
        "bytetrack": ByteTrack,
        "boosttrack": BoostTrack,
        "hybridsort": HybridSort,
    }

    if tracker_name not in trackers:
        msg = f"Unknown tracker: {tracker_name}. Available: {list(trackers.keys())}"
        raise ValueError(msg)

    # ReID-based trackers need additional parameters
    reid_trackers = {"deepocsort", "botsort", "boosttrack", "hybridsort"}

    if tracker_name in reid_trackers:
        return trackers[tracker_name](
            device=device,
            per_class=config.per_class,
            iou_threshold=config.iou_threshold,
            det_thresh=config.detection_threshold,
            max_age=config.max_age,
            min_hits=config.min_hits,
        )
    # ByteTrack and other non-ReID trackers
    return trackers[tracker_name](
        per_class=config.per_class,
        iou_threshold=config.iou_threshold,
        max_age=config.max_age,
        min_hits=config.min_hits,
    )


class RFDETRTrackingStage(CuratorStage):
    """Stage for object detection and tracking using RF-DETR + BoxMOT.

    This stage processes video clips to detect and track objects, generating
    structured pseudo-labels including:
    - Instance definitions (object_id, class, track_id, etc.)
    - Per-frame annotations (bounding boxes, confidence, etc.)

    The output is stored in the clip's `tracking_result` field.
    """

    def __init__(
        self,
        tracking_config: TrackingConfig | None = None,
        num_gpus_per_worker: float = 1.0,
        caption_source_video: Literal["raw", "detection", "tracking"] = "raw",
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the RF-DETR tracking stage.

        Args:
            tracking_config: Configuration for tracking. If None, uses defaults.
            num_gpus_per_worker: Number of GPUs per worker.
            caption_source_video: Which video to use for captioning:
                - "raw": Use original raw video (default, no swap)
                - "detection": Use detection visualization video
                - "tracking": Use tracking visualization video
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._config = tracking_config or TrackingConfig()
        self._num_gpus_per_worker = num_gpus_per_worker
        self._caption_source_video = caption_source_video
        self._verbose = verbose
        self._log_stats = log_stats
        self._model: Any = None
        self._process_count = 0
        self._swap_count = 0
        self._skip_count = 0

    def stage_setup(self) -> None:
        """Initialize stage resources and load RF-DETR model."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)

        try:
            from rfdetr import RFDETRBase  # noqa: PLC0415
            from rfdetr.util.coco_classes import COCO_CLASSES  # noqa: PLC0415
        except ImportError as e:
            msg = "Please install rfdetr: pip install rfdetr"
            raise ImportError(msg) from e

        # Store COCO_CLASSES for use in processing
        self._coco_classes: list[str] = COCO_CLASSES

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = RFDETRBase(device=device)

        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)
        logger.info(f"RF-DETR model loaded on {device}")

        if self._caption_source_video != "raw":
            logger.info(f"Captioning will use {self._caption_source_video} video instead of raw video")

    def destroy(self) -> None:
        """Clean up resources."""
        gpu_stage_cleanup(self.__class__.__name__)
        self._model = None

        if self._caption_source_video != "raw":
            logger.info(f"Video swap for captioning completed: {self._swap_count} swaps, {self._skip_count} skipped")

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage."""
        return CuratorStageResource(gpus=self._num_gpus_per_worker)

    def _create_frame_instances(  # noqa: PLR0913
        self,
        tracks: np.ndarray,
        result: TrackingResult,
        frame_idx: int,
        width: int,
        height: int,
        class_instance_counters: dict[int, int],
    ) -> list[FrameInstance]:
        """Create frame instances from tracking results.

        Args:
            tracks: Tracking results from BoxMOT.
            result: Tracking result object to update.
            frame_idx: Current frame index.
            width: Frame width.
            height: Frame height.
            class_instance_counters: Counter dict for class instances.

        Returns:
            List of frame instances.

        """
        frame_instances: list[FrameInstance] = []

        for track in tracks:
            # track format: [x1, y1, x2, y2, track_id, confidence, class_id, ...]
            track_id = int(track[4])
            bbox_tight = BoundingBox2D(
                xmin=float(track[0]),
                ymin=float(track[1]),
                xmax=float(track[2]),
                ymax=float(track[3]),
            )
            confidence = float(track[TRACK_CONFIDENCE_IDX]) if len(track) > TRACK_CONFIDENCE_IDX else 1.0
            class_id = int(track[TRACK_CLASS_ID_IDX]) if len(track) > TRACK_CLASS_ID_IDX else 0

            # Filter by target classes if specified
            class_name = self._coco_classes[class_id] if class_id < len(self._coco_classes) else "unknown"
            if self._config.target_classes is not None and class_name not in self._config.target_classes:
                continue

            # Create unique object_id
            object_id = f"{class_name}_{track_id}"

            # Calculate loose bbox (expanded)
            bbox_loose = bbox_tight.expand(
                self._config.bbox_expansion_ratio,
                width,
                height,
            )

            # Add to instances if not exists
            if object_id not in result.instances:
                if class_id not in class_instance_counters:
                    class_instance_counters[class_id] = 0
                class_instance_counters[class_id] += 1
                instance_id = class_instance_counters[class_id]

                result.instances[object_id] = TrackedInstance(
                    object_id=object_id,
                    object_type=class_name,
                    instance_id=instance_id,
                    semantic_id=class_id,
                    color=_generate_distinct_color(track_id),
                    caption=f"{class_name} (track {track_id})",
                    track_id=track_id,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    confidence_avg=confidence,
                    frame_count=1,
                )
            else:
                # Update existing instance
                inst = result.instances[object_id]
                inst.last_frame = frame_idx
                inst.frame_count += 1
                # Running average of confidence
                n = inst.frame_count
                inst.confidence_avg = inst.confidence_avg + (confidence - inst.confidence_avg) / n

            # Add frame instance
            frame_instances.append(
                FrameInstance(
                    object_id=object_id,
                    instance_id=result.instances[object_id].instance_id,
                    semantic_id=class_id,
                    bounding_box_2d_tight=bbox_tight,
                    bounding_box_2d_loose=bbox_loose,
                    confidence=confidence,
                )
            )

        return frame_instances

    def _generate_visualizations(
        self,
        frame: np.ndarray,
        detections: Any,  # noqa: ANN401
        frame_instances: list[FrameInstance],
        result: TrackingResult,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate detection and tracking visualization frames.

        Args:
            frame: Input frame (BGR).
            detections: Detection results from RF-DETR.
            frame_instances: Frame instances for this frame.
            result: Tracking result with instance info.

        Returns:
            Tuple of (detection_vis, tracking_vis) frames.

        """
        # Convert instances to dict format for draw_tracks
        instances_dict = [
            {
                "object_id": inst.object_id,
                "bounding_box_2d_tight": inst.bounding_box_2d_tight.to_list(),
                "confidence": inst.confidence,
                "semantic_id": inst.semantic_id,
            }
            for inst in frame_instances
        ]
        instances_info = {obj_id: {"color": inst.color} for obj_id, inst in result.instances.items()}

        # Draw detection visualization
        det_img = _draw_detections(
            image=frame,
            bboxes=detections.xyxy if len(detections.xyxy) > 0 else np.empty((0, 4)),
            class_ids=detections.class_id if len(detections.xyxy) > 0 else np.empty((0,)),
            class_names=self._coco_classes,
        )

        # Draw tracking visualization
        track_img = _draw_tracks(
            image=frame,
            instances=instances_dict,
            instances_info=instances_info,
        )

        return det_img, track_img

    def _process_clip(self, clip: Clip) -> None:  # noqa: C901, PLR0912, PLR0915
        """Process a single clip with detection and tracking.

        Args:
            clip: Clip to process.

        """
        if clip.encoded_data is None:
            clip.errors["tracking"] = "no_encoded_data"
            return

        # Decode video frames
        frames = self._decode_clip_frames(clip.encoded_data)
        if frames is None or len(frames) == 0:
            clip.errors["tracking"] = "frame_decode_failed"
            return

        height, width = frames[0].shape[:2]
        device = "0" if torch.cuda.is_available() else "cpu"

        # Extract FPS from clip metadata
        clip_fps = 30.0  # Default fallback
        try:
            metadata = clip.extract_metadata()
            if metadata and metadata.get("framerate"):
                clip_fps = float(metadata["framerate"])
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Could not extract FPS from clip {clip.uuid}, using default {clip_fps}: {e}")

        # Create tracker for this clip
        tracker = _get_tracker(
            tracker_name=self._config.tracker_name,  # type: ignore[arg-type]
            device=device,
            config=self._config,
        )

        # Initialize tracking result
        result = TrackingResult(
            source_path=clip.source_video,
            fps=clip_fps,
            width=width,
            height=height,
            total_frames=len(frames),
            tracker_name=self._config.tracker_name,
            detection_threshold=self._config.detection_threshold,
        )

        # Track instance counters per class
        class_instance_counters: dict[int, int] = {}

        # Collect frames for video generation if configured
        det_frames_for_video: list[np.ndarray] = []
        track_frames_for_video: list[np.ndarray] = []

        # Process each frame
        for frame_idx, frame in enumerate(frames):
            # Convert BGR to RGB for RF-DETR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Run detection
            detections = self._model.predict(pil_img, threshold=self._config.detection_threshold)

            # Convert to BoxMOT format: (x1, y1, x2, y2, conf, cls)
            if len(detections.xyxy) > 0:
                dets = np.column_stack(
                    (
                        detections.xyxy,
                        detections.confidence,
                        detections.class_id.astype(int),
                    )
                )
            else:
                dets = np.empty((0, 6))

            # Run tracking
            tracks = tracker.update(dets, frame)

            # Build frame annotation
            frame_key = f"clip_{clip.uuid}_frame_{frame_idx:06d}"
            frame_instances = self._create_frame_instances(
                tracks=tracks,
                result=result,
                frame_idx=frame_idx,
                width=width,
                height=height,
                class_instance_counters=class_instance_counters,
            )

            # Store frame annotation
            result.frames[frame_key] = FrameAnnotation(
                frame_number=frame_idx,
                width=width,
                height=height,
                instances=frame_instances,
                detection_count=max(0, len(detections.xyxy)),
            )

            # Save RGB frame if configured
            if self._config.save_rgb_frames:
                _, rgb_encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                result.rgb_frames.append((frame_idx, rgb_encoded.tobytes()))

            # Generate visualization frames/video if configured
            if self._config.save_vis_frames or self._config.save_video:
                det_img, track_img = self._generate_visualizations(
                    frame=frame,
                    detections=detections,
                    frame_instances=frame_instances,
                    result=result,
                )

                # Encode as JPEG and store (only if save_vis_frames is enabled)
                if self._config.save_vis_frames:
                    _, det_encoded = cv2.imencode(".jpg", det_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    _, track_encoded = cv2.imencode(".jpg", track_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    result.vis_frames.append((frame_idx, det_encoded.tobytes(), track_encoded.tobytes()))

                # Collect raw frames for video generation
                if self._config.save_video:
                    det_frames_for_video.append(det_img)
                    track_frames_for_video.append(track_img)

        # Generate videos if configured
        if self._config.save_video and det_frames_for_video:
            # Use FPS extracted from clip metadata (already stored in result.fps)
            result.detection_video_bytes = _encode_frames_to_video(det_frames_for_video, fps=result.fps)
            result.tracking_video_bytes = _encode_frames_to_video(track_frames_for_video, fps=result.fps)

            if self._verbose:
                logger.info(
                    f"Clip {clip.uuid}: Generated detection video "
                    f"({len(result.detection_video_bytes) / 1024:.1f} KB) and tracking video "
                    f"({len(result.tracking_video_bytes) / 1024:.1f} KB)"
                )

        # Filter tracks by minimum frame count
        if self._config.min_track_frames > 1:
            filtered_object_ids = {
                obj_id for obj_id, info in result.instances.items() if info.frame_count < self._config.min_track_frames
            }

            # Remove filtered instances
            result.instances = {
                obj_id: info for obj_id, info in result.instances.items() if obj_id not in filtered_object_ids
            }

            # Remove filtered instances from frame annotations
            for frame_key in result.frames:
                result.frames[frame_key].instances = [
                    inst for inst in result.frames[frame_key].instances if inst.object_id not in filtered_object_ids
                ]

            if self._verbose and len(filtered_object_ids) > 0:
                logger.info(
                    f"Clip {clip.uuid}: Filtered {len(filtered_object_ids)} tracks "
                    f"with < {self._config.min_track_frames} frames"
                )

        # Store result in clip
        clip.tracking_result = result

        if self._verbose:
            logger.info(
                f"Clip {clip.uuid}: Detected {len(result.instances)} unique objects across {result.total_frames} frames"
            )

    def _decode_clip_frames(self, encoded_data: bytes) -> list[npt.NDArray[np.uint8]] | None:
        """Decode video frames from encoded clip data.

        Args:
            encoded_data: Encoded video bytes.

        Returns:
            List of frames as numpy arrays (BGR format), or None if decoding fails.

        """
        frames: list[npt.NDArray[np.uint8]] = []

        # Write to temp file for OpenCV
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            tmp.write(encoded_data)
            tmp.flush()

            vid = cv2.VideoCapture(tmp.name)
            if not vid.isOpened():
                return None

            while True:
                ret, frame = vid.read()
                if not ret:
                    break
                frames.append(frame)

            vid.release()

        return frames if frames else None

    def _swap_video_for_captioning(self, clip: Clip, video_type: str) -> bool:
        """Swap clip's encoded_data with annotated video for captioning.

        Args:
            clip: Clip to process.
            video_type: Type of video to use ("detection" or "tracking").

        Returns:
            True if swap was successful, False otherwise.

        """
        # Skip if no tracking result
        if clip.tracking_result is None:
            logger.warning(
                f"Clip {clip.uuid}: No tracking result available, cannot use {video_type} video for captioning"
            )
            return False

        # Get the appropriate video bytes based on source type
        if video_type == "detection":
            video_bytes = clip.tracking_result.detection_video_bytes
        elif video_type == "tracking":
            video_bytes = clip.tracking_result.tracking_video_bytes
        else:
            return False

        # Check if the video is available
        if video_bytes is None or len(video_bytes) == 0:
            logger.warning(
                f"Clip {clip.uuid}: {video_type} video not available, using original raw video for captioning"
            )
            return False

        # Swap the encoded_data with the annotated video
        original_size = len(clip.encoded_data) if clip.encoded_data else 0
        clip.encoded_data = video_bytes
        new_size = len(clip.encoded_data)

        # Always log video swap for visibility
        logger.info(
            f"Clip {clip.uuid}: Swapped to {video_type} video for captioning "
            f"(original: {original_size / 1024:.1f} KB, new: {new_size / 1024:.1f} KB)"
        )

        return True

    @nvtx.annotate("RFDETRTrackingStage")  # type: ignore[misc]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # noqa: C901
        """Process video clips with detection and tracking.

        Args:
            tasks: Tasks containing videos to process.

        Returns:
            Processed tasks with tracking results.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video

            with self._timer.time_process(len(video.clips)):
                for clip in video.clips:
                    if clip.encoded_data is None:
                        continue
                    try:
                        self._process_clip(clip)
                    except Exception:  # noqa: BLE001
                        logger.exception(f"Error tracking clip {clip.uuid}")
                        clip.errors["tracking"] = "tracking_failed"

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        # Swap video source for captioning if configured
        if self._caption_source_video != "raw":
            for task in tasks:
                video = task.video
                for clip in video.clips:
                    success = self._swap_video_for_captioning(clip, self._caption_source_video)
                    if success:
                        self._swap_count += 1
                    else:
                        self._skip_count += 1

        # Free memory periodically
        self._process_count += 1
        if self._process_count % 10 == 0:
            torch.cuda.empty_cache()

        return tasks
