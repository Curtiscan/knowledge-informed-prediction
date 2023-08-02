function [loss,gradients] = modelLoss_1(net, X)

    % forward data through the dlnetwork object.

    Y = forward(net,X);

    % compute gradients with respect to input
    
    % ensured that first derivative is positive
    grad_1 = dlgradient(sum(Y),X, 'EnableHigherDerivatives', true);
    grad_1_corr = mean(relu(-grad_1));
    
    % ensured that second derivative is negative
    grad_2 = dlgradient(sum(grad_1),X, 'EnableHigherDerivatives', true);
    grad_2_corr = mean(relu(grad_2));

    % ensured that third derivative is positive
    grad_3 = dlgradient(sum(grad_2),X, 'EnableHigherDerivatives', true);
    grad_3_corr = mean(relu(-grad_3));

    
    C = 100;
    loss = 10*C*grad_1_corr + C*grad_2_corr + 0.1*C*grad_3_corr;

    % Compute gradients
    gradients = dlgradient(loss, net.Learnables);

end