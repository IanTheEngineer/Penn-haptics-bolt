Python wrapper around the Global Alignment kernel code from M. Cuturi

Original code: http://www.iip.ist.i.kyoto-u.ac.jp/member/cuturi/GA.html

Written by Adrien Gaidon - INRIA - 2011
http://lear.inrialpes.fr/people/gaidon/

LICENSE: cf. logGAK.c

-------------------------------------------------------------------------------
DEPENDENCIES:

Python: tested with v2.7 but should work for version >= 2.4

Numpy:
  - The fundamental package needed for scientific computing with Python
  - Pre-installed on most *nix platforms
  - cf. http://numpy.scipy.org/ for installation
  - tested with v1.6

Cython:
  - Cython (http://cython.org/) is a language that makes writing (fast) C
    extensions for the Python language as easy as Python itself
  - the wrapper code (global_align.pyx) is written using Cython, which is
    required to generate global_align.c
  - to install Cython, the recommended way is to use the 'pip' python package
    installer (http://www.pip-installer.org/en/latest/index.html)
    and then do 'pip install Cython' in a terminal
  - tested with v0.15

-------------------------------------------------------------------------------
INSTALL

Just type 'make' in a terminal (cf. the Makefile)

Then add the directory where global_align.so is to your PYTHONPATH or move it
to one of the directories in your PYTHONPATH.

Type 'make distclean' to remove all generated files

-------------------------------------------------------------------------------
USAGE

cf. test.py:

In a terminal:

python -c "import test; test.test()"

T=0 	 exp(-tga_d)=0.27343
T=1 	 exp(-tga_d)=0.00000
T=10 	 exp(-tga_d)=0.00000
T=20 	 exp(-tga_d)=0.00000
T=21 	 exp(-tga_d)=0.06560
T=30 	 exp(-tga_d)=0.26787
T=40 	 exp(-tga_d)=0.27343

