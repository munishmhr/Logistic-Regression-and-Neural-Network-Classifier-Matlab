function [grad1,thetaN] = gradient(trainSetBais,trainSetTargetVector,theta,lambda)
dataSetSize = size(trainSetBais,1);
sig=zeros(dataSetSize,1);
grad=0;
thetaN=zeros(1,513);





for i = 1:dataSetSize
        sig(i) = (sigmoid(trainSetBais(i,:)*theta)-trainSetTargetVector(i,:));
end

for j=1:513
        for i=1:dataSetSize
            grad=grad+(sig(i)*trainSetBais(i,j))/dataSetSize;
            grad1=grad;
            
        end
        regularize=(lambda/dataSetSize)*sum(theta(j));
        if(j==1)
            regularize=0;
        end
        thetaN(j) = theta(j) - 0.2*(grad+regularize);
        grad=0;
end

end