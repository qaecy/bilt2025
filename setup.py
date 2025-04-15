from setuptools import setup, find_packages

setup(
    name="built2025",
    version="0.1.0",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    author="Christian Frausing",
    author_email="christian.frausing@qaecy.com",
    description="Graph RAG in AEC Workshop Materials",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/qaecy/built2025",
)
