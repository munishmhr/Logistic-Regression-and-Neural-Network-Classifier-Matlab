function c=cost(dataset,target,theta,lambda)

dataSetSize = size(dataset,1);
c=0;
l1=0;
l2=0;

for i=1:dataSetSize
    l1=target(i)*log(sigmoid(dataset(i,:)*theta));
    l2=(1-target(i))*log(1-sigmoid(dataset(i,:)*theta));
    c=c-(l1+l2);
end
c=(c/dataSetSize)+(lambda/(2*dataSetSize))*sum(theta(2:end))^2;





%{
tic
sig1=dataset*theta;
sig1o=arrayfun(@sigmoid,sig1);
log1 = arrayfun(@log,sig1o);
l1 = arrayfun(@costmulti,target,log1);
l1=l1';

sig2=dataset*theta;
sig11=arrayfun(@sigmoid,sig2);
log2 = arrayfun(@log,(1-sig11));
l2 = arrayfun(@costmulti,(1-target),log2);

c=l2(dataSetSize);

toc

tic

for i=1:dataSetSize
     c1=(1-target(i))*log(1-sigmoid(dataset(i,:)*theta));
end
toc

%}


end