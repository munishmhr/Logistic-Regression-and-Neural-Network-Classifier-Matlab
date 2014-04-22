load('train.mat');
trainSet = vertcat(X0,X1,X2,X3,X4,X5,X6,X7,X8,X9);
%trainSetBais = bais(trainSet);
trainSetBais= horzcat(ones(19978,1),trainSet);


thetax1 = zeros(513,1);
thetax2 = zeros(513,1);
thetax3 = zeros(513,1);
thetax4 = zeros(513,1);
thetax5 = zeros(513,1);
thetax6 = zeros(513,1);
thetax7 = zeros(513,1);
thetax8 = zeros(513,1);
thetax9 = zeros(513,1);
thetax10 = zeros(513,1);


trainSetTargetVectorx1 = vertcat(ones(size(X0,1),1),zeros(size(X1,1),1),zeros(size(X2,1),1),zeros(size(X3,1),1),zeros(size(X4,1),1),zeros(size(X5,1),1),zeros(size(X6,1),1),zeros(size(X7,1),1),zeros(size(X8,1),1),zeros(size(X9,1),1));
trainSetTargetVectorx2 = vertcat(zeros(size(X0,1),1),ones(size(X1,1),1),zeros(size(X2,1),1),zeros(size(X3,1),1),zeros(size(X4,1),1),zeros(size(X5,1),1),zeros(size(X6,1),1),zeros(size(X7,1),1),zeros(size(X8,1),1),zeros(size(X9,1),1));
trainSetTargetVectorx3 = vertcat(zeros(size(X0,1),1),zeros(size(X1,1),1),ones(size(X2,1),1),zeros(size(X3,1),1),zeros(size(X4,1),1),zeros(size(X5,1),1),zeros(size(X6,1),1),zeros(size(X7,1),1),zeros(size(X8,1),1),zeros(size(X9,1),1));
trainSetTargetVectorx4 = vertcat(zeros(size(X0,1),1),zeros(size(X1,1),1),zeros(size(X2,1),1),ones(size(X3,1),1),zeros(size(X4,1),1),zeros(size(X5,1),1),zeros(size(X6,1),1),zeros(size(X7,1),1),zeros(size(X8,1),1),zeros(size(X9,1),1));
trainSetTargetVectorx5 = vertcat(zeros(size(X0,1),1),zeros(size(X1,1),1),zeros(size(X2,1),1),zeros(size(X3,1),1),ones(size(X4,1),1),zeros(size(X5,1),1),zeros(size(X6,1),1),zeros(size(X7,1),1),zeros(size(X8,1),1),zeros(size(X9,1),1));
trainSetTargetVectorx6 = vertcat(zeros(size(X0,1),1),zeros(size(X1,1),1),zeros(size(X2,1),1),zeros(size(X3,1),1),zeros(size(X4,1),1),ones(size(X5,1),1),zeros(size(X6,1),1),zeros(size(X7,1),1),zeros(size(X8,1),1),zeros(size(X9,1),1));
trainSetTargetVectorx7 = vertcat(zeros(size(X0,1),1),zeros(size(X1,1),1),zeros(size(X2,1),1),zeros(size(X3,1),1),zeros(size(X4,1),1),zeros(size(X5,1),1),ones(size(X6,1),1),zeros(size(X7,1),1),zeros(size(X8,1),1),zeros(size(X9,1),1));
trainSetTargetVectorx8 = vertcat(zeros(size(X0,1),1),zeros(size(X1,1),1),zeros(size(X2,1),1),zeros(size(X3,1),1),zeros(size(X4,1),1),zeros(size(X5,1),1),zeros(size(X6,1),1),ones(size(X7,1),1),zeros(size(X8,1),1),zeros(size(X9,1),1));
trainSetTargetVectorx9 = vertcat(zeros(size(X0,1),1),zeros(size(X1,1),1),zeros(size(X2,1),1),zeros(size(X3,1),1),zeros(size(X4,1),1),zeros(size(X5,1),1),zeros(size(X6,1),1),zeros(size(X7,1),1),ones(size(X8,1),1),zeros(size(X9,1),1));
trainSetTargetVectorx10 = vertcat(zeros(size(X0,1),1),zeros(size(X1,1),1),zeros(size(X2,1),1),zeros(size(X3,1),1),zeros(size(X4,1),1),zeros(size(X5,1),1),zeros(size(X6,1),1),zeros(size(X7,1),1),zeros(size(X8,1),1),ones(size(X9,1),1));

costX1=zeros(5000,1);
costX2=zeros(5000,1);
costX3=zeros(5000,1);
costX4=zeros(5000,1);
costX5=zeros(5000,1);
costX6=zeros(5000,1);
costX7=zeros(5000,1);
costX8=zeros(5000,1);
costX9=zeros(7000,1);
costX10=zeros(7000,1);
grad=0;
lambda=0.01;
tic

for i =1:6000
    
    costX1(i)=costCal(trainSetBais,trainSetTargetVectorx1,thetax1,lambda);
    [grad1,thetax1]=gradient(trainSetBais,trainSetTargetVectorx1,thetax1,lambda);
    thetax1=thetax1';  
    x1=i;
    x1
    
end



for i =1:6000

        costX2(i)=costCal(trainSetBais,trainSetTargetVectorx2,thetax2,lambda);
    [grad2,thetax2]=gradient(trainSetBais,trainSetTargetVectorx2,thetax2,lambda);
    thetax2=thetax2';
    x2=i;
    x2
end  


for i =1:7000
        
        costX3(i)=costCal(trainSetBais,trainSetTargetVectorx3,thetax3,lambda);
    [grad3,thetax3]=gradient(trainSetBais,trainSetTargetVectorx3,thetax3,lambda);
    thetax3=thetax3';
end


 for i =1:6000
     costX4(i)=costCal(trainSetBais,trainSetTargetVectorx4,thetax4,lambda);
    [grad4,thetax4]=gradient(trainSetBais,trainSetTargetVectorx4,thetax4,lambda);
    thetax4=thetax4';
    x4=i;
    x4
 end
    

   for i =1:6000
        
    costX5(i)=costCal(trainSetBais,trainSetTargetVectorx5,thetax5,lambda);
    [grad5,thetax5]=gradient(trainSetBais,trainSetTargetVectorx5,thetax5,lambda);
    thetax5=thetax5';
    x5=i;
    x5
   end
    
   


    for i =1:6000
        costX6(i)=costCal(trainSetBais,trainSetTargetVectorx6,thetax6,lambda);
    [grad6,thetax6]=gradient(trainSetBais,trainSetTargetVectorx6,thetax6,lambda);
    thetax6=thetax6';
    x6=i;
    x6
    end


    for i =1:3000
        costX7(i)=costCal(trainSetBais,trainSetTargetVectorx7,thetax7,lambda);
    [grad7,thetax7]=gradient(trainSetBais,trainSetTargetVectorx7,thetax7,lambda);
    thetax7=thetax7';
    x7=i;
    x7
    end
   

    
    for i =1:7000
        costX8(i)=costCal(trainSetBais,trainSetTargetVectorx8,thetax8,lambda);
    [grad8,thetax8]=gradient(trainSetBais,trainSetTargetVectorx8,thetax8,lambda);
    thetax8=thetax8';
    x8=i;
    end
    
   
    for i =1:7000
        costX9(i)=costCal(trainSetBais,trainSetTargetVectorx9,thetax9,lambda);
    [grad9,thetax9]=gradient(trainSetBais,trainSetTargetVectorx9,thetax9,lambda);
    thetax9=thetax9';
    x9=i;
    end

    for i =1:7000
    costX10(i)=costCal(trainSetBais,trainSetTargetVectorx10,thetax10,lambda);
    [grad10,thetax10]=gradient(trainSetBais,trainSetTargetVectorx10,thetax10,lambda);
    thetax10=thetax10';
    x10=i;
    x10
    end
    
 toc
