from setuptools import setup, find_packages

setup(
    name="llm-cerebras",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "llm",
        "httpx",
    ],
    entry_points={
        "llm": [
            "cerebras=llm_cerebras.cerebras",
        ],
    },
    author="Thomas (Thomasthomas) Hughes",
    author_email="irthomasthomas@gmail.com",
    description="llm plugin to prompt Cerebras hosted models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
