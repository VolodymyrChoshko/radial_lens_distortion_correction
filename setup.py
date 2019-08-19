import setuptools

setuptools.setup(
    name='tetra3',
    version='0.1',
    author='Gustav Pettersson @ ESA',
    author_email='gustavpettersson@live.com',
    description='A fast Python lost-in-space plate solver for star trackers.',
    url='https://github.com/esa/tetra3',
    packages=setuptools.find_packages(),
    install_requires=['numpy >= 1.17.0', 'scipy >= 1.3.1', 'Pillow >= 6.1.0'],
)