from setuptools import setup, find_packages

setup(
    name="fractal-block-sparse-attention",
    version="0.1.0",
    description="Sub-Quadratic Block-Sparse Attention via Odd Binomial Fractal Tree Topologies",
    author="Göksenin Akkavim",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
    ],
    extras_require={
        "triton": ["triton>=2.1.0"],
    },
    python_requires=">=3.9",
)
