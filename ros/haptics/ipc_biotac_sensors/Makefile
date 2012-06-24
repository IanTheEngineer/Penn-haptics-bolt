PKG= ipc_biotac_sensors
MEX_MSGS= BioTacData BioTacTime BioTacHand
ROS_MSGS= BioTacHand

MEX_FILES := $(foreach msg, $(MEX_MSGS), biotac_sensors_$(msg))
#ROS_FILES := $(foreach msg, $(ROS_MSGS), biotac_sensors_$(msg)_publisher)
ROS_FILES := $(foreach msg, $(ROS_MSGS), biotac_sensors_$(msg)_subscriber)

include $(shell rospack find ipc_rosgraph_msgs)/ipc_rosgraph_msgs.mk
include $(shell rospack find ipc_bridge_local)/ipc_bridge_local.mk
