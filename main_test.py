from _pytest.capture import CaptureFixture

from main import main


def test_main(capsys: CaptureFixture[str]) -> None:
    """Test that main() runs without error and prints expected output."""
    main()
    captured = capsys.readouterr()
    assert captured.out == "Hello World!\n"
