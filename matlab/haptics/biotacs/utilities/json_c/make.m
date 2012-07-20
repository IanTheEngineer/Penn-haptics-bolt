mex -ljson -g -v -L/usr/local/lib fromjson.c
%%mex -ljson -lm -g tojson.c
%mex -I/home/vchu/Dropbox/git-repo/matlab/haptics/biWotacs/matlab-json/json -g testLibJSON.c