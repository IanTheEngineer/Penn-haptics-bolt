function [ struct_object ] = parse_json_java( struct_object, json_record )
%parse_json_java Given a json record, converts it into a matlab record

keys = json_record.keySet().toArray();

for i = 1:size(keys,1)
    field = keys(i);
    values = json_record.get(field);
    if isa(values, 'net.minidev.json.JSONObject')
        temp_struct = struct();
        values = parse_json_java(temp_struct, values);
    elseif isa(values, 'net.minidev.json.JSONArray')
        if (isjava(values.get(0)))
            array = [];
            for j = 0:size(values)-1
                json_ob = values.get(j);
                array_struct = struct();
                array_struct = parse_json_java(array_struct, json_ob);
                array = [array array_struct];
            end
            values = array;
        else
            values = cell2mat(values.toArray.cell);
        end
    end
    
    struct_object.(field) = values;
end

