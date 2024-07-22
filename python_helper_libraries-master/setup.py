import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="python_helper_library",  # Replace with your own username
    version="0.0.1",
    author="Shriya Dikshith",
    author_email="shriya.dikshith@hp.com",
    description="Python helper package aiming to ease development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.azc.ext.hp.com/shriya-dikshith/python_helper_libraries",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
