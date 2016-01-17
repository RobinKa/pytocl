from distutils.core import setup

setup(
    name="pytocl",
    version= "0.1.1",
    description="Converts Python functions to OpenCL functions",
    author="Robin Kahlow (Toraxxx)",
    author_email="xtremegosugaming@gmail.com",
    maintainer="Robin Kahlow (Toraxxx)",
    maintainer_email="xtremegosugaming@gmail.com",
    url="https://github.com/ToraxXx/pytocl",
    requires=["numpy", "pyopencl"],
    license= "MIT",
    package_dir={"": "src"},
    packages=["pytocl"],
)
