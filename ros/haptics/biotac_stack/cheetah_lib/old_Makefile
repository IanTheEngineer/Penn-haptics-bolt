all: ROS_NOBUILD

#Check for 32 or 64 bit Kernel
#i686 if 32-bit
#x86_64 if 64-bit
OSTYPE := $(shell uname -m)
CHEETAH_VERSION = v3.05

TARBALL_NAME = cheetah-api-linux-$(OSTYPE)-$(CHEETAH_VERSION)
TARBALL = build/$(TARBALL_NAME).zip
TARBALL_URL = http://www.totalphase.com/download/zip/$(TARBALL_NAME).zip
SOURCE_DIR = build/$(TARBALL_NAME)
UNPACK_CMD = unzip
include $(shell rospack find mk)/download_unpack_build.mk

ROS_NOBUILD: $(SOURCE_DIR)/unpacked
	mkdir -p lib
	cp $(SOURCE_DIR)/c/cheetah.so lib
	touch ROS_NOBUILD
	echo "created by rosmake to mark as installed" >> ROS_NOBUILD


clean:
	-rm -f ROS_NOBUILD
	-rm -rf lib

wipe: clean
	-rm -rf build
