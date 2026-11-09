# Validation

The selected package passed the following software checks on Windows with
Python 3.11.9, NumPy 2.2.6 and CPU-only PyTorch 2.11.0.

## Numerical checks

With the package source directory on PYTHONPATH:

~~~sh
python -B -m unittest discover -s tests -v
python -B examples/demo.py --out demo-output
~~~

All **11 CPU tests passed**. Independent NumPy array oracles cover coronal
slices, hemisphere extents, a 90-degree rotation and tilted AP coordinates.
Other checks cover chunking, out-of-volume padding and invalid options. The
decoder matches an exhaustive scalar search on a small sequence fixture;
section ordering, alternate sections and normal-angle conventions also pass.
This is small-fixture coverage, not an exhaustive proof for arbitrary inputs.

The generated example completed and wrote three plane summaries and three
sequence rows. It uses a procedural volume and supplied synthetic costs.

## Packaging and installed example

A wheel was built and installed in a disposable virtual environment. Existing
NumPy and CPU PyTorch packages were reused; build tools were installed only
in that environment. The final build used setuptools 81.0.0 and wheel 0.48.0:

~~~sh
python -m pip wheel --no-deps --no-build-isolation --no-index --wheel-dir wheels .
python -m pip install --force-reinstall --no-index --no-deps wheels/autoatlas_showcase-0.1.0-py3-none-any.whl
python -B examples/demo.py --out demo-output
~~~

The wheel build, installation and installed-package example all exited
successfully. The final example ran without PYTHONPATH. A targeted check
confirmed PyTorch's setuptools<82 requirement was satisfied. The initial host
build lacked bdist_wheel; an initial disposable build-tool selection exceeded
that setuptools constraint. The final build requirement corrects both issues.

## Provenance and sanitization

The sampler matches its fixed research-source bytes exactly. All four extracted
sequence functions match their original abstract syntax trees. The distribution
contains the selected code, documentation, tests and one specifically authorized
PNG; private research history, datasets, weights and internal execution records
are excluded. A local audit checks the exact file set and scans text and Git
history for credentials and private infrastructure identifiers. The authorized
PNG has no text or EXIF metadata. Audit receipts remain outside this repository.

## Not run

No real-subject registration, model training, learned inference, atlas-data
download, truth scoring, scientific benchmark, CUDA validation or multi-platform
test matrix was run for this package. Local checks do not establish scientific
acceptance or accuracy improvements.
