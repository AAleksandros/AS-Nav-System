"""Main entry point for the autonomous navigation system.

This module provides the CLI interface and orchestrates the full processing pipeline.
"""

import argparse
import sys
import time
from pathlib import Path

from src.config import load_config, merge_config_overrides, print_config
from src.utils.logger import setup_logger
from src.utils.video_utils import VideoReader, VideoWriter
from src.perception import ObjectDetector
from src.planning import NavigationPlanner
from src.control import AgentController
from src.visualization import VisualizationRenderer


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description=(
            "Autonomous Navigation Agent - Process drone footage "
            "with YOLO-based obstacle detection"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m src.main -i drone.mp4 -o result.mp4

  # Custom config
  python -m src.main -i input.mp4 -o output.mp4 -c config/high_accuracy.yaml

  # Fast processing (every 3rd frame)
  python -m src.main -i input.mp4 -o output.mp4 --frame-skip 3

  # Debug mode
  python -m src.main -i input.mp4 -o output.mp4 -v

  # Process first 100 frames only
  python -m src.main -i input.mp4 -o output.mp4 --max-frames 100
        """
    )

    # Required arguments
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input video file path'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output video file path'
    )

    # Optional arguments
    parser.add_argument(
        '-c', '--config',
        type=str,
        default=None,
        help='Configuration file path (default: config/default_config.yaml)'
    )
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=None,
        help='Process every Nth frame (overrides config)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to process (overrides config, -1 = all)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable DEBUG logging'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Force CPU processing (disable GPU)'
    )

    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    """Validate input/output paths.

    Args:
        args: Parsed command-line arguments

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If output path is invalid

    """
    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video file not found: {args.input}")

    # Check output directory exists or can be created
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    """Main entry point for the navigation system.

    Returns:
        Exit code (0 = success, 1 = error)
    """
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1

    # Apply CLI overrides
    overrides = {}
    if args.frame_skip is not None:
        overrides['frame_skip'] = args.frame_skip
    if args.max_frames is not None:
        overrides['max_frames'] = args.max_frames
    if args.no_gpu:
        overrides['model.device'] = 'cpu'

    if overrides:
        config = merge_config_overrides(config, overrides)

    # Determine log level
    log_level = 'DEBUG' if args.verbose else config.logging.level

    # Setup logging
    logger = setup_logger(
        name='navigation',
        level=log_level,
        log_file=config.logging.log_file,
        log_to_console=config.logging.log_to_console
    )

    # Validate paths
    try:
        validate_paths(args)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Path validation failed: {e}")
        return 1

    # Log configuration summary
    logger.info("=" * 60)
    logger.info("Autonomous Navigation System - Starting")
    logger.info("=" * 60)
    logger.info(f"Input video: {args.input}")
    logger.info(f"Output video: {args.output}")
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Device: {config.model.device}")
    logger.info(f"Confidence threshold: {config.model.confidence_threshold}")
    logger.info(f"Frame skip: {config.processing.frame_skip}")
    logger.info(f"Max frames: {config.processing.max_frames}")
    logger.info("=" * 60)

    if args.verbose:
        logger.debug("Full configuration:")
        print_config(config)

    # Initialize pipeline components
    try:
        logger.info("Initializing pipeline components...")

        # Initialize detector
        detector = ObjectDetector(config)  # type: ignore
        logger.info("✓ Object detector initialized")

        # Initialize other components
        planner = NavigationPlanner(config)  # type: ignore
        controller = AgentController(config)  # type: ignore
        renderer = VisualizationRenderer(config)  # type: ignore

        logger.info("✓ All pipeline components initialized")

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return 1

    # Process video
    try:
        with VideoReader(args.input) as reader:
            with VideoWriter(
                args.output,
                fps=reader.fps,
                width=reader.width,
                height=reader.height
            ) as writer:
                logger.info("=" * 60)
                logger.info("Starting video processing...")
                logger.info("=" * 60)

                start_time = time.time()
                frames_processed = 0
                frames_written = 0
                total_detections = 0
                state_counts = {"navigate": 0, "avoid": 0, "stop": 0}

                # Process each frame
                for frame_idx, frame in enumerate(reader):
                    try:
                        # Apply frame skip
                        frame_skip = config.processing.frame_skip  # type: ignore
                        if frame_idx % frame_skip != 0:
                            # Write original frame without processing
                            writer.write_frame(frame)
                            frames_written += 1
                            continue

                        # Check max frames limit (after frame skip)
                        max_frames = config.processing.max_frames  # type: ignore
                        if max_frames > 0:
                            if frames_processed >= max_frames:
                                logger.info(
                                    f"Reached max frames limit ({max_frames})"
                                )
                                break

                        # 1. Detect obstacles
                        detections = detector.detect_frame(frame)
                        total_detections += len(detections)

                        # 2. Get agent state
                        agent_state = controller.get_agent_state()

                        # 3. Determine state and action
                        state, action = planner.update(detections, agent_state)
                        state_counts[state.value] += 1

                        # 4. Execute action
                        controller.execute_action(action)

                        # 5. Render overlays
                        annotated_frame = renderer.render(
                            frame,
                            detections,
                            controller.get_agent_state(),
                            state
                        )

                        # 6. Write output frame
                        writer.write_frame(annotated_frame)
                        frames_written += 1
                        frames_processed += 1

                        # Progress logging (every 30 frames or 5%)
                        if frame_idx % 30 == 0 and frame_idx > 0:
                            elapsed = time.time() - start_time
                            fps = frames_processed / elapsed
                            logger.info(
                                f"Frame {frame_idx}: "
                                f"State={state.value}, "
                                f"Detections={len(detections)}, "
                                f"FPS={fps:.1f}"
                            )

                    except Exception as e:
                        logger.warning(
                            f"Error processing frame {frame_idx}: {e}"
                        )
                        # Write original frame on error
                        writer.write_frame(frame)
                        frames_written += 1
                        continue

                # Processing complete
                elapsed = time.time() - start_time
                logger.info("=" * 60)
                logger.info("Processing complete!")
                logger.info(f"Frames processed: {frames_processed}")
                logger.info(f"Frames written: {frames_written}")
                logger.info(f"Total detections: {total_detections}")
                logger.info(f"Processing time: {elapsed:.1f}s")
                avg_fps = frames_processed / elapsed if elapsed > 0 else 0
                logger.info(f"Average FPS: {avg_fps:.1f}")
                logger.info("State distribution:")
                for state_name, count in state_counts.items():
                    pct = (
                        (count / frames_processed * 100)
                        if frames_processed > 0
                        else 0
                    )
                    logger.info(f"  {state_name}: {count} ({pct:.1f}%)")
                logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
