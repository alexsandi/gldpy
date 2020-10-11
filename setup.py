import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gldpy", 
    version="0.1",
    author="Aleksandra Alekseeva",
    author_email="alex_sandi@mail.ru",
    description="Generalized Lambda Distribution for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    url="https://github.com/alexsandi/gldpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
       ]
)