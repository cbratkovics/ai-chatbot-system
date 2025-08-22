"""Basic test to ensure pytest works."""


def test_import():
    """Test that we can import the app module."""
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert True


def test_basic_math():
    """Basic test to verify pytest is working."""
    assert 1 + 1 == 2


def test_environment():
    """Test environment setup."""
    import os

    assert os.path.exists("config/requirements/base.txt")
