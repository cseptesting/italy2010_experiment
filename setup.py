from setuptools import setup, find_packages

setup(
    name='fe_italy_ti',
    version='0.1.0',
    author='Pablo Iturrieta',
    author_email='pciturri@gfz-potsdam.de',
    license='LICENSE',
    description='fe_italy_ti',
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    entry_points={
                  'console_scripts': ['fecsep = fecsep.main:fecsep']
                  },
    url='git@git.gfz-potsdam.de:csep-group/fecsep-quadtree.git'
)
