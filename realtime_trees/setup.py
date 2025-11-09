from distutils.core import setup

package_name = 'realtime_trees'

setup(
    name='realtime_trees',
    version='0.1',
    author='Leonard Freissmuth',
    author_email='l.freissmuth@gmail.com',
    packages=[package_name, f"{package_name}/utils"],
    package_dir={'': 'src'},
    python_requires='>=3.6',
    description='Toolset for online tree reconstruction',
)