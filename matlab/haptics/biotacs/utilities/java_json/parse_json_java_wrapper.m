function [ struct_object ] = parse_json_java_wrapper( line )
%parse_json Summary of this function goes here
%   Detailed explanation goes here

json_object = net.minidev.json.JSONValue.parse(line);

struct_object = struct();

struct_object = parse_json_java(struct_object, json_object);

end

