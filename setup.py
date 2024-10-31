from setuptools import find_packages, setup

setup(
    name="threedsim",
    version="0.1",
    description="3D SiM",
    author="Julian Buechel, Athanasios Vasilopoulos",
    author_email="jub@zurich.ibm.com, atv@zurich.ibm.com",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_required=">=3.10",
)
