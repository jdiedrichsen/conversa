function out = cnv_predictorsAndLabels(ld, behaviourNames, varargin)
% Returns a matrix of predictors and a matrix of labels given a labelled
% dataframe of behaviours and a list of behaviour names to use as labels
% By Shayaan Syed Ali
% Last updated 19-Jun-17

% TODO: Set up optionArgs for that some input or output features can be
% included/excluded, e.g. includefields, excludefields

% List of tracking fields
trackingFields = {'neckposx', 'neckposy', 'neckposz', 'neckrotx', 'neckroty', 'neckrotz', 'headposx', 'headposy', 'headposz', 'headrotx', 'headroty', 'headrotz', 'brow_up_l', 'brow_up_r', 'brow_down_l', 'brow_down_r', 'eye_closed_l', 'eye_closed_r', 'cheek_puffed_l', 'cheek_puffed_r', 'lips_pucker', 'lips_stretch_l', 'lips_stretch_r', 'lip_lower_down_l', 'lip_lower_down_r', 'smile_l', 'smile_r', 'frown_l', 'frown_r', 'jaw_l', 'jaw_r', 'jaw_open'};
nTrackingFields = length(trackingFields);
% Set predictor fields
nSamples = length(ld.timestamp);
out.predictors = zeros(nSamples, nTrackingFields);
for fieldNo = 1:nTrackingFields
	out.predictors(:,fieldNo) = ld.(trackingFields{fieldNo});
end;
% Set label fields
nBehaviours = length(behaviourNames);
out.labels = zeros(nSamples, nBehaviours);
for behavNo = 1:nBehaviours
	out.labels(:,behavNo) = ld.(behaviourNames{behavNo});
end;
end