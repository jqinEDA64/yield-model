import pytest

def pytest_addoption(parser):
    # This adds the --plot flag to the pytest command
    parser.addoption(
        "--plot", action="store_true", default=False, help="Display plots during tests"
    )
    # This adds the --print flag to the pytest command
    parser.addoption(
        "--print", action="store_true", default=False, help="Print output during tests"
    )

@pytest.fixture
def do_plot(request):
    # This fixture allows your tests to "ask" if the flag was used
    return request.config.getoption("--plot")

@pytest.fixture
def do_print(request):
    # This fixture allows your tests to "ask" if the flag was used
    return request.config.getoption("--print")