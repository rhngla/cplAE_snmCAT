import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

print("====Listing packages found:====")
print(setuptools.find_packages())
print("===============================")
setuptools.setup(
    name="cplAE_snmCAT",
    version="1.0",
    author="Rohan Gala",
    author_email="rhngla@gmail.com",
    description="Coupled autoencoders for snmCAT data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
)
