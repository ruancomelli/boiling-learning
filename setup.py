# For instance,
# pip install -r git+https://github.com/ruancomelli/boiling-learning#egg=boiling-learning[dev,scripts]

from pathlib import Path
from typing import List, Union

from setuptools import find_packages, setup

project_path = Path(__file__).parent


def read_lines(path: Union[str, Path]) -> List[str]:
    return Path(path).read_text().splitlines()


author = 'ruancomelli'
project_name = 'boiling-learning'

REQUIRES = read_lines(project_path / 'requirements.txt')

EXTRAS_REQUIRE = {
    extra: read_lines(project_path / f'requirements-{extra}.txt')
    for extra in ('dev', 'scripts')
}
EXTRAS_REQUIRE['all'] = [
    extra_req
    for extra_reqs in EXTRAS_REQUIRE.values()
    for extra_req in extra_reqs
]

README = (project_path / 'README.md').read_text()
VERSION = (project_path / 'VERSION').read_text().strip()

setup(
    name=project_name,
    version=VERSION,
    author=author,
    author_email='ruancomelli@gmail.com',
    maintainer='Ruan Cardoso Comelli',
    maintainer_email='ruancomelli@gmail.com',

    description="A project for learning heat flux estimation from boiling images.",
    long_description=README,
    long_description_content_type="text/markdown",

    url=f'https://github.com/{author}/{project_name}',
    download_url=f'https://github.com/{author}/{project_name}/dist/{project_name}-{VERSION}.tar.gz',

    license='proprietary',

    python_requires='==3.6',
    packages=find_packages(),

    install_requires=REQUIRES,
    extras_require=EXTRAS_REQUIRE,

    keywords=[
        'boiling',
        'cnn',
        'convolutional-neural-network'
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Manufacturing',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Physics'
    ],

    project_urls={
        'Documentation': f'https://github.com/{author}/{project_name}/blob/main/README.md',
        'Source': f'https://github.com/{author}/{project_name}',
        'Tracker': f'https://github.com/{author}/{project_name}/issues',
    },
)
