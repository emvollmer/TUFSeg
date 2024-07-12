from setuptools import find_packages, setup
from pathlib import Path

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = Path(PARENT, 'README.md').read_text(encoding='utf-8')

with open(Path(PARENT, "requirements.txt"), "r") as fh:
    REQUIREMENTS = [line.strip() for line in fh]

setup(
    name='TUFSeg',
    version='0.0.1',
    python_requires='>=3.8',
    license='BSD-3-Clause',
    description='Thermal Urban Feature Segmentation',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/emvollmer/TUFSeg',
    author='Elena Vollmer, Leon Klug, James Kahn',
    author_email='elena.vollmer@kit.edu',
    packages=find_packages(),  # required
    install_requires=REQUIREMENTS,
    classifiers=[
        'Intended Audience :: Information Technology',
        'Intended Audience :: System Administrators',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD-3-Clause License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition'],
)
