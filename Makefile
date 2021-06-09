all:
	python setup.py build_ext --inplace
	python setup_torch.py build_ext --inplace
