%% load and transform the solution

% load raw solution of aggregation process (c_k(t)) as a martix AggrInfo
% first dimension is mass distribution (k)
% second dimension is temporal steps (t)

AggrInfo = readmatrix('SolutionData');

% data is modified with log-transformation with cut-off (\hat{c}_k(t))
% which simplifies and makes solution more distinguishable
% in terms of standard norms

DataX = zeros(size(AggrInfo));
DataX(AggrInfo < 10^(-7)) = 10^(-7);
DataX(AggrInfo >= 10^(-7)) = AggrInfo(AggrInfo >= 10^(-7));
DataX = log(DataX);

%% Get parameters of parametrizing NN_p for a sparse time grid

% here parameters of parametrizing network are retrieved
% parametrizing network plays the role of a parametric family (F(k, p(t)))
% and its two parameters (W and B) play the role of
% parameter vector (p(t))


m_precision = log(10^-7); % cut-off value used in transformation


W = zeros(size(1:20:4000));
B = zeros(size(1:20:4000));

step = 0;
tic
for t = 1:20:2000
    
    step = step+1;
    if (rem(step-1, 10) == 0)
        disp(step)
    end

    % solution for a fixed point t

    trainX = dlarray(1:10000, 'CT');
    trainY = dlarray(DataX(:,t)', 'CT');
    
    
    % establish small NN
    % two parameters are frozen
    % the other two are learnables
    layers = [
                sequenceInputLayer(1)
                fullyConnectedLayer(1, 'WeightLearnRateFactor', 0,  'Weights', -1)
                reluLayer
                fullyConnectedLayer(1, 'BiasLearnRateFactor', 0, 'Bias', m_precision)
            ];

    net = dlnetwork(layers);
    
    % train NN to get parameters
    initialLearnRate = 0.001;
    numEpoches = 500;
    averageGrad = [];
    averageSqGrad = [];

    for epoch = 1:numEpoches
        [loss,gradients] = dlfeval(@modelLoss,net,trainX,trainY);
        
        learnRate = initialLearnRate/(1);
        
        [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,epoch,learnRate);
        
        if (exit_condition)
            break
        end

    end
    
    % save parameters

    W(step) = double(net.Learnables.Value{3});
    B(step) = double(net.Learnables.Value{2});
    

end
toc
clear epoch gradients initialLearnRate layers learnRate loss m_precision ...
    momentum numEpoches step t vel

%% write parameters

writematrix([W; B], 'realParameters');