from setuptools import setup


def get_requirements():
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    return requirements


setup(
    name='colab-notebook-utils',
    version='1.0.0',
    packages=['colab_notebook_utils'],
    url='',
    license='',
    author='',
    author_email='',
    description="Supporting utilities for Reboot Motion's interactive biomechanics notebook.",
    classifiers=[],
    install_requires=get_requirements(),
    entry_points={}
)