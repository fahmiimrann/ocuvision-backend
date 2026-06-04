% compile_ocu.m
% =========================================================================
% Build the OcuVision standalone executable with MATLAB Compiler.
%
% Run this once on the same machine that has MATLAB + Compiler installed
% (R2025b in your case). It produces:
%
%   ./dist/ocu_main.exe                  – the standalone executable
%   ./dist/requiredMCRProducts.txt       – which Runtime products are needed
%   ./dist/readme.txt                    – auto-generated usage info
%
% After this, the .exe runs WITHOUT a MATLAB licence on any machine that
% has the matching (free) MATLAB Runtime installed.
%
% Pre-flight check:
%   >> ver
%   Confirm "MATLAB Compiler" appears in the list. If not, request the
%   add-on from your university IT or buy it standalone.
%
% Usage:
%   >> cd 'C:\Users\USER\...\html\ocuvision'
%   >> compile_ocu
%
% After it finishes (~5-10 minutes), the new exe is at:
%   .\dist\ocu_main.exe
%
% Try it locally:
%   .\dist\ocu_main.exe ".\99_BaggedTreesModel.mat" "C:\path\to\fundus.jpg"
%
% You should see an OCU_FEATURES_BEGIN..OCU_FEATURES_END block followed by
% OCU_JSON_BEGIN..OCU_JSON_END with the predicted diagnosis.
% =========================================================================

clc;
fprintf('=== OcuVision MATLAB Compiler build ===\n');

%% 1. Pre-flight: do we have the Compiler? -------------------------------
v = ver;
hasCompiler = any(strcmpi({v.Name}, 'MATLAB Compiler'));
if ~hasCompiler
    error(['MATLAB Compiler toolbox is not installed on this MATLAB. ' ...
           'Run "ver" to confirm, and request the add-on from your ' ...
           'university IT or install it via Add-On Explorer.']);
end

%% 2. Output folder -------------------------------------------------------
outDir = fullfile(pwd, 'dist');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% 3. Compile -------------------------------------------------------------
% -m            : build a standalone application (.exe on Windows)
% -d            : output directory
% -o            : output basename
% -a            : additional file to bundle (the trained model)
% -R '-nojvm'   : faster startup (we don't need the JVM)
% ocu_predict.m is auto-discovered through ocu_main.m, but listing it
% explicitly avoids any chance of mcc missing it.
fprintf('Calling mcc -- this typically takes 5-10 minutes.\n');

mcc('-m', 'ocu_main.m', ...
    'ocu_predict.m', ...
    '-d', outDir, ...
    '-o', 'ocu_main', ...
    '-a', '99_BaggedTreesModel.mat', ...
    '-R', '-nojvm', ...
    '-R', '-nodisplay');

fprintf('\n');
fprintf('=== BUILD COMPLETE ===\n');
fprintf('Executable:    %s\n', fullfile(outDir, 'ocu_main.exe'));
fprintf('MCR products:  %s\n', fullfile(outDir, 'requiredMCRProducts.txt'));
fprintf('\n');
fprintf('To make matlab-runner.js use the .exe instead of full MATLAB:\n');
fprintf('  add this line to .env:\n');
fprintf('    OCU_COMPILED_EXE=%s\n', fullfile(outDir, 'ocu_main.exe'));
fprintf('\n');
fprintf('Done.\n');
