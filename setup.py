# For instance,
# pip install -r git+https://github.com/ruancomelli/boiling-learning#egg=boiling-learning[colab,dev]

from pathlib import Path
from pkg_resources import parse_requirements
import re
from setuptools import setup, find_packages
from typing import List, Union


project_path = Path(__file__).parent


def read_requirements(path: Union[str, Path]) -> List[str]:
    with Path(path).open('r') as file:
        requirements = list(map(str.strip, file))

    pattern = re.compile(r'git\+https://github\.com/(?P<author>.+)/(?P<repo>.+)\.git.*')

    requirements = [
        (
            # see https://stackoverflow.com/a/53706140/5811400
            ' @ '.join((
                match.group('repo'),
                requirement
            ))
            if match
            else requirement
        )
        for requirement, match in zip(requirements, map(pattern.match, requirements))
    ]
    return requirements


author = 'ruancomelli'
project_name = 'boiling-learning'


REQUIRES = read_requirements(project_path / 'requirements.txt')
print(REQUIRES)

EXTRAS_REQUIRE = {
    extra: read_requirements(
        project_path / f'requirements-{extra}.txt'
    )
    for extra in (
        'colab',
        'dev',
        'scripts'
    )
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

    description="A package for learning heat flux estimation from boiling images.",
    long_description=README,
    long_description_content_type="text/markdown",
    url=f'https://github.com/{author}/{project_name}',
    download_url = f'https://github.com/{author}/{project_name}/dist/{project_name}-{VERSION}.tar.gz',

    license='proprietary',

    python_requires='>=3.7',
    packages=find_packages(),

    install_requires=list(map(str, parse_requirements(REQUIRES))),
    extras_require=EXTRAS_REQUIRE,

    keywords=[
        'boiling',
        'cnn',
        'convolutional-neural-network'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
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
