all: global_align

global_align:
	python setup_$@.py build_ext --inplace

clean:
	rm -rf build

distclean: clean
	rm -rf global_align.so global_align.c
