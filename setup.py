#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = []
with open("requirements_dev.txt", "r") as f:
    requirements = f.read().splitlines()

test_requirements = [ ]

setup(
    author="Miaoqi Chu",
    author_email='mqichu@anl.gov',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A high-performance correlation (multi-tau/two-time) package running on GPU and CPU",
    entry_points={
        'console_scripts': [
            'boost_corr=boost_corr.cli:main',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='boost_corr',
    name='boost_corr',
    packages=find_packages(include=['boost_corr', 'boost_corr.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/AZjk/boost_corr',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    zip_safe=False,
)
