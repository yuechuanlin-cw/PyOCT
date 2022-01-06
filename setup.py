import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyOCT", # Replace with your own username
    version="3.0.0",
    author="Yuechuan Lin",
    author_email="linyuechuan1989@gmail.com",
    description="Optical imaging reconstruction for both spectral-domain OCT and off-axis digital holography microscopy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeversayEverLin/PyOCT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

