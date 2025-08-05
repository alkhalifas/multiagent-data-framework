from setuptools import setup, find_packages

setup(
    name="multiagent-data-framework",
    version="0.1.0",
    description="A multi‑agent framework for exploring supply chain, manufacturing and sales data via LLMs",
    author="Your Name",
    packages=find_packages(include=["multiagent", "multiagent.*"]),
    include_package_data=True,
    install_requires=[
        "pandas>=1.0",
        "langchain",
        "openai",
        "streamlit",
    ],
)