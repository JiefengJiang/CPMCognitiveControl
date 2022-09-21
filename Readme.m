%Last tested on Matlab R2022a and Windows 10
%running this script should replicate the TMT1->TMT2 analysis in Fig. 4E

load('ExampleData.mat');

[preY, nPredictor] = CPMPredictionExample(data.x, data.y, data.beta, data.p, 0.01, ones(1, size(data.x{1}, 2)), true);

scatter(data.y{2}, preY);
