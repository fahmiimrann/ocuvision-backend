function ocu_main(modelPath, imagePath)
%OCU_MAIN  Standalone entry point for the OcuVision MATLAB pipeline.
%
%   ocu_main(modelPath, imagePath) loads the trained Classification
%   Learner export at MODELPATH, runs the OcuVision feature pipeline on
%   the image at IMAGEPATH, and prints the JSON-encoded result to stdout
%   between the markers
%
%       OCU_JSON_BEGIN { ... } OCU_JSON_END
%
%   The Node-side runner (matlab-runner.js) extracts that block from the
%   process stdout. Errors are reported as
%
%       OCU_ERROR:<message>
%
%   so the Node side can surface them as user-visible messages.
%
%   --- Compiling to a standalone executable ----------------------------
%   This file is also the MATLAB Compiler entry point. To build the
%   licence-free .exe / Linux binary, see compile_ocu.m.

    try
        if nargin < 2 || isempty(modelPath) || isempty(imagePath)
            error('OCU:Usage', 'Usage: ocu_main <modelPath> <imagePath>');
        end
        result = ocu_predict(char(modelPath), char(imagePath));
        fprintf('OCU_JSON_BEGIN%sOCU_JSON_END', jsonencode(result));
    catch err
        fprintf('OCU_ERROR:%s', err.message);
    end
end
