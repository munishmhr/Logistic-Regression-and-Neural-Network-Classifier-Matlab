
clear;
load('train.mat');

n_x_features = 512;
n_a1 = n_x_features;
n_a2 = 25 ;
n_a3 = 10;
INIT_EPSILON = 0.07;

theta1=rand(n_a2, n_a1 + 1) * ( 2 * INIT_EPSILON ) - INIT_EPSILON ;%rand(25,513)
theta2=rand(n_a3, n_a2 + 1) * ( 2 * INIT_EPSILON ) - INIT_EPSILON ;%rand(10,26)

trainSet = vertcat(X0,X1,X2,X3,X4,X5,X6,X7,X8,X9);
bais = ones(19978,1);
trainSet = horzcat(bais,trainSet);

targetx0 = [1,0,0,0,0,0,0,0,0,0];
targetx0 = repmat(targetx0,size(X0,1),1);

targetx1 = [0,1,0,0,0,0,0,0,0,0];
targetx1 = repmat(targetx1,size(X1,1),1);

targetx2 = [0,0,1,0,0,0,0,0,0,0];
targetx2 = repmat(targetx2,size(X2,1),1);

targetx3 = [0,0,0,1,0,0,0,0,0,0];
targetx3 = repmat(targetx3,size(X3,1),1);

targetx4 = [0,0,0,0,1,0,0,0,0,0];
targetx4 = repmat(targetx4,size(X4,1),1);

targetx5 = [0,0,0,0,0,1,0,0,0,0];
targetx5 = repmat(targetx5,size(X5,1),1);

targetx6 = [0,0,0,0,0,0,1,0,0,0];
targetx6 = repmat(targetx6,size(X6,1),1);

targetx7 = [0,0,0,0,0,0,0,1,0,0];
targetx7 = repmat(targetx7,size(X7,1),1);

targetx8 = [0,0,0,0,0,0,0,0,1,0];
targetx8 = repmat(targetx8,size(X8,1),1);

targetx9 = [0,0,0,0,0,0,0,0,0,1];
targetx9 = repmat(targetx9,size(X9,1),1);

targetVec = vertcat(targetx0,targetx1,targetx2,targetx3,targetx4,targetx5,targetx6,targetx7,targetx8,targetx9);
lambda= 0.01;
nncostplot=zeros(745,1);

for j = 1:745
    costF =0;
    Del1=0;
    Del2=0;
    
    for i = 1:19978
        % forward prob
        Z2 = theta1 * trainSet(i,:)';
        actL2 = vertcat(1, arrayfun(@sigmoid,Z2) );
        
        Z3 = theta2 * actL2;
        actL3 = arrayfun(@sigmoid,Z3);

        % Gradient computation
        error3 = (actL3 - targetVec(i,:)');      
        deri_a2 = actL2 .* ( 1 - actL2 );
        error2 = ( theta2' * error3 ) .* deri_a2 ;
        error2 = error2(2:end , :);
        
        Del1 = Del1 + error2 * trainSet(i,:); % out bais
        Del2 = Del2 + error3 * actL2';
        
        cost = nn_costCal( targetVec(i,:), actL3 );
        costF = costF + cost;        
    end
    
    reg = (lambda / 2 ) *  (sum( sum( theta1(:,2:end) .^ 2 )) + sum( sum(theta2(:,2:end) .^ 2 )) );
    costF =  -1 * (costF + reg) / 19978

    D1 = (Del1 / 19978 ) + lambda .* theta1;
    D2 = (Del2 / 19978 ) + lambda .* theta2;
    
    theta1 = theta1 - D1;
    theta2 = theta2 - D2;
    nncostplot(j)=costF;
    
    hold on 
    plot(j,costF)

   
end



