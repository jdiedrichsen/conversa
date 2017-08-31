function out = cnv_loadData(trackingFile, labelFile)
out = cnv_applyLabels(cnv_loadTrackingData(trackingFile), cnv_loadLabelFile(labelFile));
end