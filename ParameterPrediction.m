%% load and transform data


% two parameters are loaded and transformed (T(W) and T(B))
% next sections are to be executed for W_data and B_data separately

tempMat = readmatrix('realParameters');

W = tempMat(1,:);
B = tempMat(2,:);
TW_data = -(log(W) - log(W(1)));
TB_data = log(B) - log(B(1));


%% prepare datasets

% in this section we arrange all data sets needed for extrapolation
% sets trainX, trainY are used to ensure convergence of NN result to the
% training set; sets trainX_1 and trainY_1 are used to ensure conservation
% of observed properties on the target set

start_train = 1000;
end_train = 1900;
end_val = 2000;

dataX = (1:4000)./4000;
dataY = TW_data;

trainX = dlarray(dataX(start_train:end_train), 'CT');
trainY = dlarray(dataY(start_train:end_train), 'CT');

validX = dlarray(dataX(end_train:end_val),'CT');
validY = dlarray(dataY(end_train:end_val),'CT');

trainX_1 = dlarray(dataX(start_train:end), 'CT');
trainY_1 = dlarray(dataY(start_train:end), 'CT');

testX = dlarray(dataX((end_train+1):end), 'CT');
testY = dlarray(dataY((end_train+1):end), 'CT');
%% establish NN model

% extrapolating neural network NN_e

num_hidden_units = 5;
layers = [
    sequenceInputLayer(1)
    fullyConnectedLayer(num_hidden_units)
    sigmoidLayer
    fullyConnectedLayer(1)
    ];
 
net = dlnetwork(layers);

%% fix training parameters

numEpoches = 2000000;
initialLearnRate = 0.001;



%% train NN model

% model is trained with two steps per epoch: the first step ensures properties
% conservation on the target interval, while the second provides
% convergence to desired data on the training interval


figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on


velocity = [];
averageGrad = [];   
averageSqGrad = [];
iteration = 0;
loss_array = zeros(1,numEpoches);
valid_array = zeros(1,numEpoches);


start = tic;
min_valid = inf;
min_loss = inf;
min_index = 0;
for epoch = 1:numEpoches
    
    iteration = iteration + 1;

    % step to ensure output properties as a function

    [loss,gradients] = dlfeval(@modelLoss_1,net,trainX_1,trainY_1);
    learnRate = initialLearnRate/(1);
    [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iteration, learnRate);
    
    % step to maintain convergence

    [loss,gradients] = dlfeval(@modelLoss,net,trainX,trainY);
    learnRate = initialLearnRate/(1);
    [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iteration, learnRate);
    
    [valid_loss,gradients] = dlfeval(@modelLoss,net,validX,validY);

    D = duration(0,0,toc(start),Format="hh:mm:ss");

    loss = double(loss);
    valid_loss = double(valid_loss);
    addpoints(lineLossTrain,iteration,loss)
    title("Epoch: " + epoch + ", Valid loss = " + string(extractdata(valid_loss)) + ", Elapsed: " + string(D))
    axis([max([epoch-100,1]) epoch+10 0 2*extractdata(loss)])
    drawnow
    
    loss_array(epoch) = extractdata(loss);
    valid_array(epoch) = extractdata(valid_loss);
    
    
    if (loss_array(epoch) < min_loss)
        min_loss = loss_array(epoch);
        if ((valid_array(epoch) < min_valid))
            net_min = net;
            min_valid = valid_array(epoch);
            min_index = epoch;
        end
    end

    if (min_valid < 10^-6)
        break
    end
end

TW_pred = net(dlarray(dataX, 'CT'));

%% write predicted parameters

writematrix([TW_pred; TB_pred], 'predTParameters');