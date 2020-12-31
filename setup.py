from setuptools import setup

setup(
    name="PhotosynthesisAI",
    version="1.0",
    description="AI for the board game Photosynthesis",
    author="Tom Clements",
    author_email="thomasalbertclements@gmail.com",
    packages=["PhotosynthesisAI"],
    install_requires=["numpy", "hexy", "matplotlib", "dataclasses", "scikit-learn", "tensorflow"],
)
