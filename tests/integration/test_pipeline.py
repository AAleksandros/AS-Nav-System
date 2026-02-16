"""Integration tests for the APF simulation pipeline."""

import cv2  # type: ignore
import sys

from src.main import main


class TestPipelineIntegration:
    """Integration tests for the full APF simulation pipeline."""

    def test_corridor_scenario_runs(self, tmp_path, monkeypatch):
        """Test corridor scenario runs and produces output video."""
        output_path = tmp_path / "corridor.mp4"

        test_args = [
            'src.main',
            '--scenario', 'corridor',
            '--duration', '1',
            '-o', str(output_path),
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()

        assert exit_code == 0
        assert output_path.exists()

        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count > 0
        cap.release()

    def test_gauntlet_scenario_runs(self, tmp_path, monkeypatch):
        """Test gauntlet scenario runs and produces output video."""
        output_path = tmp_path / "gauntlet.mp4"

        test_args = [
            'src.main',
            '--scenario', 'gauntlet',
            '--duration', '1',
            '-o', str(output_path),
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()

        assert exit_code == 0
        assert output_path.exists()

    def test_dynamic_scenario_runs(self, tmp_path, monkeypatch):
        """Test dynamic scenario runs and produces output video."""
        output_path = tmp_path / "dynamic.mp4"

        test_args = [
            'src.main',
            '--scenario', 'dynamic',
            '--duration', '1',
            '-o', str(output_path),
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()

        assert exit_code == 0
        assert output_path.exists()

    def test_random_scenario_with_seed(self, tmp_path, monkeypatch):
        """Test random scenario with seed produces output."""
        output_path = tmp_path / "random.mp4"

        test_args = [
            'src.main',
            '--scenario', 'random',
            '--seed', '42',
            '--duration', '1',
            '-o', str(output_path),
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()

        assert exit_code == 0
        assert output_path.exists()

    def test_invalid_scenario_fails(self, tmp_path, monkeypatch):
        """Test that an invalid scenario returns exit code 1."""
        output_path = tmp_path / "invalid.mp4"

        test_args = [
            'src.main',
            '--scenario', 'nonexistent_scenario',
            '--duration', '1',
            '-o', str(output_path),
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()

        assert exit_code == 1

    def test_duration_override(self, tmp_path, monkeypatch):
        """Test --duration 1 limits output to approximately 30 frames."""
        output_path = tmp_path / "short.mp4"

        test_args = [
            'src.main',
            '--scenario', 'corridor',
            '--duration', '1',
            '-o', str(output_path),
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()

        assert exit_code == 0
        assert output_path.exists()

        cap = cv2.VideoCapture(str(output_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 1 second at 30fps ~= 30 frames (dt=0.033 -> 30 steps)
        assert 25 <= frame_count <= 35

    def test_output_video_dimensions(self, tmp_path, monkeypatch):
        """Test output video has correct 800x600 dimensions."""
        output_path = tmp_path / "dims.mp4"

        test_args = [
            'src.main',
            '--scenario', 'corridor',
            '--duration', '1',
            '-o', str(output_path),
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()
        assert exit_code == 0

        cap = cv2.VideoCapture(str(output_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        assert width == 800
        assert height == 600

    def test_verbose_mode(self, tmp_path, monkeypatch):
        """Test verbose mode doesn't crash."""
        output_path = tmp_path / "verbose.mp4"

        test_args = [
            'src.main',
            '--scenario', 'corridor',
            '--duration', '1',
            '-o', str(output_path),
            '-v',
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()

        assert exit_code == 0
        assert output_path.exists()

    def test_slalom_scenario_runs(self, tmp_path, monkeypatch):
        """Test slalom scenario runs and produces output video."""
        output_path = tmp_path / "slalom.mp4"

        test_args = [
            'src.main',
            '--scenario', 'slalom',
            '--duration', '1',
            '-o', str(output_path),
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()

        assert exit_code == 0
        assert output_path.exists()

    def test_narrow_gap_scenario_runs(self, tmp_path, monkeypatch):
        """Test narrow_gap scenario runs and produces output video."""
        output_path = tmp_path / "narrow_gap.mp4"

        test_args = [
            'src.main',
            '--scenario', 'narrow_gap',
            '--duration', '1',
            '-o', str(output_path),
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()

        assert exit_code == 0
        assert output_path.exists()

    def test_u_turn_scenario_runs(self, tmp_path, monkeypatch):
        """Test u_turn scenario runs and produces output video."""
        output_path = tmp_path / "u_turn.mp4"

        test_args = [
            'src.main',
            '--scenario', 'u_turn',
            '--duration', '1',
            '-o', str(output_path),
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()

        assert exit_code == 0
        assert output_path.exists()

    def test_crossing_scenario_runs(self, tmp_path, monkeypatch):
        """Test crossing scenario runs and produces output video."""
        output_path = tmp_path / "crossing.mp4"

        test_args = [
            'src.main',
            '--scenario', 'crossing',
            '--duration', '1',
            '-o', str(output_path),
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()

        assert exit_code == 0
        assert output_path.exists()

    def test_dense_scenario_runs(self, tmp_path, monkeypatch):
        """Test dense scenario runs and produces output video."""
        output_path = tmp_path / "dense.mp4"

        test_args = [
            'src.main',
            '--scenario', 'dense',
            '--duration', '1',
            '-o', str(output_path),
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()

        assert exit_code == 0
        assert output_path.exists()

    def test_corridor_5s_agent_moves(self, tmp_path, monkeypatch):
        """Test that agent moves significantly in a 5s corridor scenario."""
        output_path = tmp_path / "corridor_5s.mp4"

        test_args = [
            'src.main',
            '--scenario', 'corridor',
            '--duration', '5',
            '-o', str(output_path),
        ]
        monkeypatch.setattr(sys, 'argv', test_args)

        exit_code = main()

        assert exit_code == 0
        assert output_path.exists()

        cap = cv2.VideoCapture(str(output_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 5 seconds at 30fps ~= 150 frames
        assert frame_count >= 100
