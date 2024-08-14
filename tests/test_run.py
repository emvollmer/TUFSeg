from pytest import mark


@mark.parametrize("example", ["example"], indirect=True)
def test_example(example):
    assert example == "example"
