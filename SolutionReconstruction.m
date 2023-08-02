%% load data

% this script reconstructs predicted solution of aggregation process from
% predicted parameters of the parametrizing network and compares it to the
% real solution

% loading of predicted parameters

predParams = readmatrix('predTParameters');

TW_pred = predParams(1,:);
TB_pred = predParams(2,:);

% to use them in the parametrizing network we need to provide an inversed
% transformation, we will load real values of the parameters, as the
% inversed transormation requires first values
% inverse transformation for W and B

realParams = readmatrix('realParameters');

W = realParams(1,:);
B = realParams(2,:);

W_pred = exp(-TW_pred).*W(1);
B_pred = exp(TB_pred).*B(1);

% data for the whole solution is loaded for the following comparison

AggrInfo = readmatrix('SolutionData');

DataX = zeros(size(AggrInfo));
DataX(AggrInfo < 10^(-7)) = 10^(-7);
DataX(AggrInfo >= 10^(-7)) = AggrInfo(AggrInfo >= 10^(-7));
DataX = log(DataX);



%% establish NN_P for reconstruction of predicted solution

m_precision = log(10^-7);

layers = [
            sequenceInputLayer(1)
            fullyConnectedLayer(1, 'WeightLearnRateFactor', 0, 'Weights', -1)
            reluLayer
            fullyConnectedLayer(1, 'BiasLearnRateFactor', 0, 'Bias', m_precision)
        ];

net = dlnetwork(layers);

%% check prediction


k = 1:10000;
ind_t = 4000;

% predicted solution is given with parametrizing network with parameters
% corresponding to a relevant timestep

net.Learnables.Value{3} = dlarray(W_pred(ind_t));
net.Learnables.Value{2} = dlarray(B_pred(ind_t));

Solution_real = DataX(:,ind_t);
Solution_pred = forward(net,dlarray(k,'CT'));

figure();

hold on
plot(k, Solution_real, 'LineWidth', 2);
plot(k, Solution_pred , 'LineWidth', 2);

set(gca,'FontSize',16);
box on
grid on
xlabel('$k$','Interpreter','latex');
ylabel('$\mathcal{T}(\hat{c}_k(t))$','Interpreter','latex');
title(['t = ' num2str(ind_t.*0.01)],'Interpreter','latex');
legend('modified solution $\mathcal{T}(\hat{c}_k(t))$', 'knowledge-informed prediction','Interpreter','latex');

