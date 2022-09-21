%Sample code implementing the CPM algorithm
%inputs:
%
%x: 1x2 cell of lesion scores. x{1} is training data, x{2} is test data.
%each cell is a nSubject x nROIs matrix encoding lesion scores
%
%y: 1x2 cell of behevioaral scores. y{1} and y{2} represent training and
%test data respectively. Each cell is a nSubjectx1 vector.
%
%beta: regression coefficients for feature selection of CPM. beta iS a nROI
%x 2 matrix. Each column encodes the slope and intercept of each ROI.
%
%p: a 1xnROI vector encoding p values for each ROI's regression
%
%pThreshold: p-value threshold for CPM feature selection. We used 0.01 as
%in Rosenberg et al (2016).
%
%mask: Binary mask indicating which features will be used.
%
%posCoefOnly: whether to constrain feature selection on positive
%correlations only (i.e., more lesion related with worse performance)
%
%Outputs
%
%preY: CPM prediction of y{2}.
%
%nPredictor: how many features were selected when predicting each element
%in y{2}.


%preY will be in the original order of Y, so preY are comparable across
%different randomized assignments of cross validation folds
function [preY, nPredictor] = CPMPredictionExample(x, y, beta, p, pThreshold, mask, posCoefOnly)

if length(mask) ~= size(x{1}, 2) || length(mask) ~= size(x{2}, 2)
    disp('mask is a binary array indicating if each column in x is used');
else
    if length(posCoefOnly) == 1
        posCoefOnly = repmat(posCoefOnly, [1 size(x{2}, 2)]);
    end
  
    %separate training and test data
    trainingX = x{1};
    trainingY = y{1};
    testX = x{2};
    testY = y{2};


    %normalize test data based on training data. The betas were already
    %normalized
    meanPre = mean(trainingY);
    testX = (testX - repmat(mean(trainingX), [size(testX, 1), 1])) ./ repmat(std(trainingX), [size(testX, 1), 1]);

    curPreY = zeros(length(y{2}), 1);
    curN = 0;

    %CPM
    for k = 1 : size(testX, 2)
        if mask(k) > 0 && p(k) < pThreshold && (~posCoefOnly(k) || beta(k, 1) > 0)
            curPreY = curPreY + testX(:, k) * beta(k, 1) + beta(k, 2);
            curN = curN + 1;
        end
    end

    %if no feature passes threshold, then use mean
    if curN == 0
        curN = 1;
        curPreY(:) = meanPre;
    end

    preY = curPreY;
    nPredictor = curN;    
        
    
end
preY = preY';
nPredictor = nPredictor';
preY = preY ./ nPredictor;