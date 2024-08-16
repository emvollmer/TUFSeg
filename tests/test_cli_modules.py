import os
import subprocess
import pytest

MODULES_TO_TEST = [
    "tufseg.scripts.registration.generate_dirtree",
    "tufseg.scripts.registration.register_RGB_TIR",
    "tufseg.scripts.setup.generate_segm_masks",
    "tufseg.scripts.setup.train_test_split",
    "tufseg.scripts.segm_models.train_UNet",
    "tufseg.scripts.segm_models.evaluate_UNet",
    "tufseg.scripts.segm_models.infer_UNet",
]


@pytest.mark.parametrize("module", MODULES_TO_TEST)
def test_module_help(module):
    """
    Test that running 'python -m <module> --help' works without errors.
    """
    # skip the test if the module requires CUDA and CUDA is not available
    if (
            module == "tufseg.scripts.segm_models.train_UNet"
            and not os.getenv("CUDA_HOME")
    ):
        pytest.skip(
            "Skipping test for train_UNet because CUDA is not available"
        )

    result = subprocess.run(
        ["python", "-m", module, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, \
        f"Error running --help on {module}: {result.stderr}"
    assert "--help" in result.stdout or result.stderr, \
        f"Help output not found for {module}"
