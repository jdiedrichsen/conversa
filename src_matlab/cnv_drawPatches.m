function cnv_drawPatches(time,label,color);
startT = time(find(diff([0;label])==1));
endT   = time(find(diff([label;0])==-1));
YL = ylim;
for p=1:length(startT)
    patch('Faces', 1:4, 'Vertices', ...
        [startT(p) YL(1); endT(p) YL(1); endT(p) YL(2); startT(p) YL(2)], ...
        'FaceColor', color, 'FaceAlpha', 0.25, 'EdgeColor', 'none');
end;