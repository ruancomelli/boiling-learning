import atexit
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify
import more_itertools as mit
import portion as P
import requests


def _get_command():
    system = platform.system()
    if system == "Darwin":
        command = "ngrok"
    elif system == "Windows":
        command = "ngrok.exe"
    elif system == "Linux":
        command = "ngrok"
    else:
        raise Exception("{system} is not supported".format(system=system))
    return command


def _run_ngrok(port):
    command = _get_command()
    ngrok_path = str(Path(tempfile.gettempdir(), "ngrok"))
    _download_ngrok(ngrok_path)
    executable = str(Path(ngrok_path, command))
    os.chmod(executable, 0o777)
    ngrok = subprocess.Popen([executable, 'http', str(port)])
    atexit.register(ngrok.terminate)
    localhost_url = "http://localhost:4040/api/tunnels"  # Url with tunnel details
    time.sleep(1)
    tunnel_url = requests.get(localhost_url).text  # Get the tunnel information
    j = json.loads(tunnel_url)

    tunnel_url = j['tunnels'][0]['public_url']  # Do the parsing of the get
    tunnel_url = tunnel_url.replace("https", "http")
    return tunnel_url


def _download_ngrok(ngrok_path):
    if Path(ngrok_path).exists():
        return
    system = platform.system()
    if system == "Darwin":
        url = "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-darwin-amd64.zip"
    elif system == "Windows":
        url = "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-windows-amd64.zip"
    elif system == "Linux":
        url = "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip"
    else:
        raise Exception('{system} is not supported'.format(system=system))
    download_path = _download_file(url)
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(ngrok_path)


def _download_file(url):
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    download_path = str(Path(tempfile.gettempdir(), local_filename))
    with open(download_path, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    return download_path


def start_ngrok(port):
    ngrok_address = _run_ngrok(port)
    print(' * Running on {ngrok_address}'.format(ngrok_address=ngrok_address))
    print(' * Traffic stats available on http://127.0.0.1:4040')
    return ngrok_address


def run_with_ngrok(app):
    """
    The provided Flask app will be securely exposed to the public internet via ngrok when run,
    and the its ngrok address will be printed to stdout
    :param app: a Flask application object
    :return: None
    """
    old_run = app.run

    def new_run(*args, **kwargs):
        port = kwargs.get('port', 5000)
        ngrok_address = start_ngrok(port)

        address_path = kwargs.pop('address_path', 'server_data/url.txt')
        address_path = Path(address_path)
        address_path.parent.mkdir(exist_ok=True, parents=True)
        address_path.write_text(ngrok_address)

        old_run(*args, **kwargs)
    app.run = new_run


def discretize_portion(p: P.Interval) -> P.Interval:
    def first_step(s: P.Interval):
        return (
            P.OPEN,
            (s.lower - 1 if s.left is P.CLOSED else s.lower),
            (s.upper + 1 if s.right is P.CLOSED else s.upper),
            P.OPEN
        )

    def second_step(s: P.Interval):
        return (
            P.CLOSED,
            (s.lower + 1 if s.left is P.OPEN else s.lower),
            (s.upper - 1 if s.right is P.OPEN else s.upper),
            P.CLOSED
        )
    if p.empty:
        return p
    else:
        return p.apply(first_step).apply(second_step)


class SequenceDistributorCase:
    def __init__(self, name: str, length: int):
        self._name: str = name
        self._universe: P.Interval = discretize_portion(P.closedopen(0, length))
        self._assigned: P.Interval = P.empty()
        self._completed: P.Interval = P.empty()

    @property
    def universe(self) -> P.Interval:
        return self._universe

    @property
    def assigned(self) -> P.Interval:
        return self._assigned

    @property
    def completed(self) -> P.Interval:
        return self._completed

    @property
    def remainder(self) -> P.Interval:
        return discretize_portion(self._universe - self._completed)

    def mark_completed(self, index: int) -> None:
        if index not in self._universe:
            raise IndexError(f'index {index} is not in the universe {self._universe}')
        self._completed = discretize_portion(self._completed | P.singleton(index))

    def interrupt(self, index: int) -> None:
        if index not in self._universe:
            raise IndexError(f'index {index} is not in the universe {self._universe}')
        self._assigned = discretize_portion(self._assigned - P.singleton(index))

    def complete(self, index: int) -> None:
        self.interrupt(index)
        self.mark_completed(index)

    def assign(self) -> Optional[int]:
        index = mit.first(
            P.iterate(self._universe - (self._assigned | self._completed), step=1),
            None
        )

        if index is not None:
            self._assigned = discretize_portion(self._assigned | P.singleton(index))

        return index


s = {}

app = Flask(__name__)
run_with_ngrok(app)


@app.route("/hello")
def hello_world():
    name = request.args.get('name', 'World')
    return '<h1>Hello, {name}!</h1>'.format(name=name)


@app.route('/')
def home():
    return '<h1>Welcome home</h1>'


@app.route('/assign', methods=['GET'])
def assign():
    case_name = request.args['case_name']
    seq_len = int(request.args['seq'])

    if case_name not in s:
        s[case_name] = SequenceDistributorCase(case_name, seq_len)

    return jsonify(s[case_name].assign())


@app.route('/restart', methods=['PUT'])
def restart():
    s.clear()


@app.route('/erase', methods=['PUT'])
def erase():
    case_name = request.args['case_name']
    del s[case_name]


@app.route('/complete', methods=['PUT'])
def complete():
    case_name = request.args['case_name']
    index = int(request.args['seq'])
    s[case_name].complete(index)


@app.route('/interrupt', methods=['PUT'])
def interrupt():
    case_name = request.args['case_name']
    index = int(request.args['index'])
    s[case_name].interrupt(index)


if __name__ == '__main__':
    kwargs = {}

    if len(sys.argv) >= 2:
        port = int(sys.argv[1])
        kwargs.update(port=port)

    if len(sys.argv) >= 3:
        address_path = Path(sys.argv[2])
        kwargs.update(address_path=address_path)

    app.run(**kwargs)
