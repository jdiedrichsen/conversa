function D = cnv_checkLengthAll();
D=[];
color={'r','b','r','b'};
for i=31
    for c=3:4
        dataID=sprintf('cam%di%d',c,i);
        T.cam = c;
        T.int = i;
        [T.length,F{c},B{c},x{c},Fs]= cnv_checkLength(dataID);
        D=addstruct(D,T);
        subplot(3,1,1);
        plot(F{c}.timestamp,(F{c}.Smile_L+F{c}.Smile_L)/2,color{c});
        hold on;
        subplot(3,1,2);
        plot(B{c}.timestamp,B{c}.hand_L_tz,color{c});
        hold on;
        
        subplot(3,1,3);
        t=[1:length(x{c})]/Fs;
        plot(t,abs(x{c}),color{c}); hold on;
        [~,claps]=findpeaks(abs(x{c}),t,'MinPeakHeight',0.1,'MinPeakDistance',0.3);
        subplot(3,1,1);
        drawline(claps);
        subplot(3,1,2);
        drawline(claps);
    end;
    for i=1:3
        subplot(3,1,i);
        hold off;
    end;
    keyboard;
end;