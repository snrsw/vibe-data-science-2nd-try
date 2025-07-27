from unittest.mock import patch


from vibe_data_science_2nd_try.utils.logging import configure_logging, get_logger


def test_configure_logging_default():
    with patch("structlog.configure") as mock_configure:
        configure_logging()
        assert mock_configure.called


def test_configure_logging_custom_level():
    with patch("structlog.configure") as mock_configure:
        configure_logging(level="DEBUG")
        assert mock_configure.called


def test_get_logger():
    with patch("structlog.get_logger") as mock_get_logger:
        get_logger("test_logger")
        mock_get_logger.assert_called_with("test_logger")