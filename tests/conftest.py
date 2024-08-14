from pytest import fixture
from tufseg import configuration


@fixture(scope="session")
def example(request):
    return request.param if hasattr(request, "param") else None
