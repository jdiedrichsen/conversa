function D = cnv_checkLengthAll(dataID);
D=[]; 
for i=2:9 
    for c=1:2 
        dataID=sprintf('cam%di%d',c,i); 
        T.cam = c; 
        T.int = i; 
        T.length = cnv_checkLength(dataID); 
        D=addstruct(D,T); 
    end; 
end; 