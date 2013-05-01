all: ROS_NOBUILD

#Check for 32 or 64 bit Kernel
#i686 if 32-bit
#x86_64 if 64-bit
CHEETAH_VERSION = v3.05

TARBALL_NAME = cheetah-library-$(CHEETAH_VERSION)
TARBALL = build/$(TARBALL_NAME).zip
TARBALL_URL = http://byte-haptics.no-ip.org:8010/downloads/biotac_resources/$(TARBALL_NAME).zip
SOURCE_DIR = build/$(TARBALL_NAME)
UNPACK_CMD = unzip
include $(shell rospack find mk)/download_unpack_build.mk

ROS_NOBUILD: $(SOURCE_DIR)/unpacked
	mkdir -p lib
	cp $(SOURCE_DIR)/cheetah.so lib
	touch ROS_NOBUILD
	echo "created by rosmake to mark as installed" >> ROS_NOBUILD


clean:
	-rm -f ROS_NOBUILD
	-rm -rf lib

wipe: clean
	-rm -rf build
