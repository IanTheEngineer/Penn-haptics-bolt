Penn-haptics-bolt
=================

Penn Haptics team's repository for BOLT related work


--Notes for getting started--

--Cheetah USB Driver--
- wget http://www.totalphase.com/download/zip/tp-usb-drivers-v2.10.zip
- unzip tp-usb-drivers-v2.10.zip
- sudo cp tp-usb-drivers-v2.10/linux/99-totalphase.rules /etc/udev/rules.d/

--Electrical Connections--
(Try for this order, but it doesn't really matter)
- Plug the biotac sensors into the Multi-BioTac board
- Plug the +5V nano-USB cable into the MBTB and your computer's USB
- Plug the Cheetah's 10 pin ribbon cable into the MBTB
- Plug the Cheetah's USB into your computer's USB port

--Software Setup--
- In your .bashrc.ros:
ROS_PACKAGE_PATH=~/your/path/to/biotac_stack:$ROS_PACKAGE_PATH

rosmake biotac_stack

--Running Software--
rosrun biotac_sensors biotac_pub

Check to make sure your sensors are working:
rostopic echo biotac_pub

Then to log some data in JSON form:
rosrun biotac_logger biotac_json_logger.py _filename:=trial_001.json
