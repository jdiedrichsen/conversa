function varargout = cnv_learn_svm(trainingData, behaviour, varargin)

% Training data is composed of kinematic data and labels

fieldMap = containers.Map({'kinematic', 'behaviour'}, {...
    {'neckposx', 'neckposy', 'neckposz', 'neckrotx', 'neckroty', 'neckrotz', 'headposx', 'headposy', 'headposz', 'headrotx', 'headroty', 'headrotz', 'brow_up_l', 'brow_up_r', 'brow_down_l', 'brow_down_r', 'eye_closed_l', 'eye_closed_r', 'cheek_puffed_l', 'cheek_puffed_r', 'lips_pucker', 'lips_stretch_l', 'lips_stretch_r', 'lip_lower_down_l', 'lip_lower_down_r', 'smile_l', 'smile_r', 'frown_l', 'frown_r', 'jaw_l', 'jaw_r', 'jaw_open'}, ...
    {} ...
    })

fieldmap('behaviour') = behaviour;
disp(horzcat('Learning with behaviour: ' + behaviour));



end