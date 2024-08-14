from pytest import fixture
from tufseg.scripts import configuration


@fixture(scope="session")
def example(request):
    return request.param if hasattr(request, "param") else None
