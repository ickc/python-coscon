SHELL = /usr/bin/env bash

_python ?= python
PYTESTPARALLEL ?= --workers auto
EXTRAS ?=
COVHTML ?= --cov-report html
# for bump2version, valid options are: major, minor, patch
PART ?= patch
N_MPI ?= 2

_pandoc = pandoc
pandocArgs = --toc -M date="`date "+%B %e, %Y"`" --filter=pantable --wrap=none
RSTs = CHANGELOG.rst README.rst

# Main Targets #################################################################

.PHONY: test test-mpi docs clean

docs: $(RSTs)
	$(MAKE) html
html: dist/docs/

test:
	$(_python) -m pytest -vv $(PYTESTPARALLEL) \
		--cov=src --cov-report term $(COVHTML) --no-cov-on-fail --cov-branch \
		tests

test-mpi:
	mpirun -n $(N_MPI) $(_python) -m pytest -vv --with-mpi \
		--capture=no \
		tests

clean:
	rm -f $(RSTs)

# docs #########################################################################

README.rst: docs/README.md docs/badges.csv
	printf \
		"%s\n\n" \
		".. This is auto-generated from \`$<\`. Do not edit this file directly." \
		> $@
	cd $(<D); \
	$(_pandoc) $(pandocArgs) $(<F) -V title='pantable Documentation' -s -t rst \
		>> ../$@

%.rst: %.md
	printf \
		"%s\n\n" \
		".. This is auto-generated from \`$<\`. Do not edit this file directly." \
		> $@
	$(_pandoc) $(pandocArgs) $< -s -t rst >> $@

dist/docs/:
	mkdir -p $@
	sphinx-build -E -b dirhtml docs $@
    # sphinx-build -b linkcheck docs dist/docs

# maintenance ##################################################################

.PHONY: pypi pypiManual gh-pages pep8 flake8 pylint
# Deploy to PyPI
## by CI, properly git tagged
pypi:
	git push origin v0.1.1
## Manually
pypiManual:
	rm -rf dist
	# tox -e check
	poetry build
	twine upload dist/*

gh-pages:
	ghp-import --no-jekyll --push dist/docs

# check python styles
pep8:
	pycodestyle . --ignore=E501
flake8:
	flake8 . --ignore=E501
pylint:
	pylint coscon

print-%:
	$(info $* = $($*))

# poetry #######################################################################

# since poetry doesn't support editable, we can build and extract the setup.py,
# temporary remove pyproject.toml and ask pip to install from setup.py instead.
editable:
	poetry build
	cd dist; tar -xf coscon-0.1.1.tar.gz coscon-0.1.1/setup.py
	mv dist/coscon-0.1.1/setup.py .
	rm -rf dist/coscon-0.1.1
	mv pyproject.toml .pyproject.toml
	$(_python) -m pip install -e .$(EXTRAS); mv .pyproject.toml pyproject.toml

# releasing ####################################################################

bump:
	bump2version $(PART)
	git push --follow-tags
