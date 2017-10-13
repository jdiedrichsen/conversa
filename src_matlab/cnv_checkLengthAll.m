function D = cnv_checkLengthAll(dataID);
D=[]; 
color={'r','b','r','b'}; 
for i=22:26
    for c=3:4
        dataID=sprintf('cam%di%d',c,i); 
        T.cam = c; 
        T.int = i; 
        [T.length,R{c},x{c},Fs]= cnv_checkLength(dataID); 
        D=addstruct(D,T); 
        subplot(2,1,1); 
        plot(R{c}.timestamp,(R{c}.smile_l+R{c}.smile_r)/2,color{c}); 
        hold on; 
        subplot(2,1,2); 
        t=[1:length(x{c})]/Fs; 
        plot(t,abs(x{c}),color{c}); hold on; 
%         indx=find(abs(x{c})>0.01);
 %       drawline(t(indx)); 
%         subplot(2,1,1); 
%         drawline(t(indx)); 
    end; 
    subplot(2,1,1); 
    hold off; 
    subplot(2,1,2); 
    hold off; 
    keyboard; 
end; 