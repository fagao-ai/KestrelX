# Publishing to PyPI

The distribution name is `kestrel-agent-core`; the import package is
`kestrel_agent_core`.

## GitHub Actions release

The `Python Release` workflow uses PyPI Trusted Publishing, so no API token or
repository secret is needed.

Before the first run, configure a pending trusted publisher:

- PyPI project name: `kestrel-agent-core`
- GitHub owner: `fagao-ai`
- Repository: `kestrel`
- Workflow: `python-release.yml`
- Environment: `pypi`

For a TestPyPI rehearsal, add the same pending publisher on TestPyPI with
environment `testpypi`. Then create matching GitHub environments under
`Settings -> Environments` (`pypi` and optionally `testpypi`).

The workflow builds the sdist plus macOS arm64/x86_64, manylinux_2_28 x86_64,
and Windows x64 wheels. All wheels are installed and smoke-tested before the
publish job starts.

Run the workflow manually and choose `build-only` to verify every build without
uploading. Then choose `testpypi`. After the package is verified there, either
run it again with `pypi`, or publish a tag:

```bash
git tag v0.0.1
git push origin v0.0.1
```

Tag pushes and manual `pypi` runs publish to the production index. PyPI
versions are immutable, so bump `bindings/python/kestrel_agent_core/_version.py`
for every release.

## Build

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m build
python -m twine check dist/*
```

`python -m build` creates:

- a platform wheel containing `kestrel_agent_core/_native/libkestrel_ffi.*`
- an sdist containing the Rust workspace, which can rebuild the native library
  on platforms without a prebuilt wheel

The wheel tag is platform-specific. Build one wheel per supported OS/CPU in
CI; do not publish a macOS wheel for Linux users.

## Test the artifacts

```bash
python3 -m venv /tmp/kestrel-wheel-test
/tmp/kestrel-wheel-test/bin/pip install dist/kestrel_agent_core-*.whl
/tmp/kestrel-wheel-test/bin/python -c \
  "import kestrel_agent_core as k; print(k.__version__, k.contract_version())"
```

## Upload to TestPyPI first

Create an API token in TestPyPI and export it only for the upload command:

```bash
export TWINE_USERNAME=__token__
read -rs TWINE_PASSWORD
export TWINE_PASSWORD
python -m twine upload \
  --repository-url https://test.pypi.org/legacy/ \
  dist/*
```

Install from TestPyPI in a clean environment:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  kestrel-agent-core
```

## Upload to PyPI

PyPI versions are immutable: bump `__version__` before every upload. Then:

```bash
python -m twine upload dist/*
```

Use an API token with the minimum required scope instead of your account
password. For recurring releases, configure PyPI Trusted Publishing and use
GitHub Actions OIDC rather than storing a long-lived token.
