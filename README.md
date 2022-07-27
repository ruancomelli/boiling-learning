<h1 align="center">
Boiling Learning
</h1>

[![Github Actions](https://github.com/commitizen-tools/commitizen/workflows/Python%20package/badge.svg?style=flat-square)](https://github.com/ruancomelli/boiling-learning/actions)
[![Sourcery](https://img.shields.io/badge/Sourcery-enabled-brightgreen)](https://sourcery.ai)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![SemVer](https://img.shields.io/badge/semver-2.0.0-green)](https://semver.org/spec/v2.0.0.html)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![Author: ruancomelli](https://img.shields.io/badge/ruancomelli-blue?style=flat&label=author)](https://github.com/ruancomelli)

<h1 align="center">
<a href="https://github.com/ruancomelli/boiling-learning">
<img
  src=https://user-images.githubusercontent.com/22752929/179121666-171ab746-a072-4e62-80fa-448c4bd3f646.png
  alt="Robot underwater surrounded by bubbles"
  width="300"
  align="center"
>
</a>
</h1>

_Image generated by [DALL-E mini](https://www.craiyon.com/) using the prompt "robot watching bubbles in water"_

---

About
---

Project developed by me, [Ruan Comelli](https://github.com/ruancomelli), at [UFSC (Federal University of Santa Catarina)](https://ufsc.br/) in order to obtain a Master's degree in Mechanical Engineering.

This project has evolved a lot since its beginning, so please be kind when judging it. In particular, much of the CI scripts were developed only after a good portion of the core functionality was already implemented.

If you are curious about the evolution of this project, take a look at the [changelog](CHANGELOG.md).

<h1 align="center">
<a href="https://github.com/ruancomelli/boiling-learning">
<img
  src=https://user-images.githubusercontent.com/22752929/181357665-f1fb9c59-ec02-46f8-b723-f38c15576bfd.gif
  alt="Nemo - Bubbles! Bubbles! Bubbles! Bubbles!"
  width="200"
  align="center"
>
</a>
</h1>

Versioning
---

This project uses [ZeroVer](https://0ver.org/), a versioning scheme in which software never leaves major version **0**. This means that breaking changes are expected frequently.

Given a version number `0.x.y`, the minor version `x` will be incremented whenever intentional breaking changes are introduced. If no new breaking changes are added, new releases will only increment the patch version `y` for both bug fixes and new features. This is similar to how [SemVer](https://semver.org/) treats the major and the minor version numbers, respectively.

Installation
---

First of all, thank you for your interest in this project!

To install Boiling Learning in your local machine, first clone it from this repository:
```sh
git clone https://github.com/ruancomelli/boiling-learning.git
```
and then move to your new local repository:
```sh
cd boiling-learning
```

When you're there, make sure that you start a new [virtual environment](https://docs.python.org/3/tutorial/venv.html) to encapsulate the packages you are about to install:
```sh
python3 -m venv .venv
source .venv/bin/activate
```

If everything runs smoothly, you can just install the Boiling Learning's requirements:
```sh
pip install -r requirements.txt
```

If you also wish to execute the provided [Python scripts](boiling_learning/scripts), install the scripts requirements:
```sh
pip install -r requirements-scripts.txt
```

To install Boiling Learning, run:
```
python setup.py install
```

Now we're all set to start learning phase change using neural nets!

Contributing
---

### Setting up your local environment

To set up your local environment for development, first follow the steps outlined in [Installation](#installation).

After everything is installed correctly, install the development requirements:
```sh
pip install -r requirements-dev.txt
```

Then install the [pre-commit](https://pre-commit.com/) git hooks:
```sh
pre-commit install
pre-commit install --hook-type commit-msg
```

### Making changes

For now, there isn't really a standard for making changes. Try to stick to the pattern you see in the code you are writing.

When you're done with your changes, and before committing anything, make sure that you didn't break already implemented functionality by running the tests. Tests can be executed with:
```sh
python -m unittest tests/test_*
```

Code coverage can be easily inspected with the script [`coverage.sh`](scripts/coverage.sh):
```sh
. ./scripts/coverage.sh
```

### Committing

After you make and stage your changes, instead of running `git commit`, use the standardized committing script [`commit.sh`](scripts/commit.sh), which uses [Commitizen](https://github.com/commitizen-tools/commitizen):
```sh
. ./scripts/commit.sh
```

### Releasing new versions

If you are a maintainer and you wish to bump the project's version, please run our [release script](scripts/release.sh):
```sh
. ./scripts/release.sh
```

This uses [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) and [Semantic Versioning](https://semver.org/) to decide which version to bump the project to. If you are unhappy with this, you can manually decide which version number to increment:
```sh
. ./scripts/release.sh --increment [MAJOR|MINOR|PATCH]
```

The release scripts forwards all arguments to [Commitizen's bump command](https://github.com/commitizen-tools/commitizen/blob/master/docs/bump.md), so anything they accept works here as well. In particular, another thing you may wish to do is to bump a prerelease version:
```sh
. ./scripts/release.sh --prerelease [alpha|beta|rc]
```

Upcoming
---

- Improved README.
