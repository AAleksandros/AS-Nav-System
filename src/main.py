"""Main entry point for the autonomous navigation system.

This module provides the CLI interface and orchestrates the full processing pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.config import load_config, merge_config_overrides, print_config
from src.utils.logger import setup_logger


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Autonomous Navigation Agent - Process drone footage with YOLO-based obstacle detection",
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

    # TODO: Pipeline processing will be implemented in later phases
    logger.info("Pipeline processing not yet implemented")
    logger.info("Placeholder: This is where video processing will occur")

    logger.info("=" * 60)
    logger.info("Processing complete")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
