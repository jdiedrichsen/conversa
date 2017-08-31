function cnv_runPy(script)
% script refers to the full script name, e.g. test.py
system(horzcat('python ', script));
end