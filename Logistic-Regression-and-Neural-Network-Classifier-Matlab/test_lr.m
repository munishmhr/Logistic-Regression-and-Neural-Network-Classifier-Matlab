load test.mat;
error_lr=0;
%test = bais(test);
test = horzcat(ones(1500,1),test);

yo = test*thetax1;
probClass0 = arrayfun(@sigmoid, yo);
probClass0= probClass0(1:150,:);

y1 = test*thetax2;
probClass1 = arrayfun(@sigmoid, y1);
probClass1= probClass1(151:300,:);

y2 = test*thetax3;
probClass2 = arrayfun(@sigmoid, y2);
probClass2= probClass2(301:450,:);

y3 = test*thetax4;
probClass3= arrayfun(@sigmoid, y3);
probClass3= probClass3(451:600,:);

y4 = test*thetax5;
probClass4 = arrayfun(@sigmoid, y4);
probClass4= probClass4(601:750,:);

y5 = test*thetax6;
probClass5 = arrayfun(@sigmoid, y5);
probClass5= probClass5(751:900,:);

y6 = test*thetax7;
probClass6 = arrayfun(@sigmoid, y6);
probClass6= probClass6(901:1050,:);

y7 = test*thetax8;
probClass7 = arrayfun(@sigmoid, y7);
probClass7= probClass7(1051:1200,:);

y8 = test*thetax9;
probClass8 = arrayfun(@sigmoid, y8);
probClass8= probClass8(1201:1350,:);

y9 = test*thetax10;
probClass9 = arrayfun(@sigmoid, y9);
probClass9= probClass9(1351:1500,:);

prob = horzcat(probClass0,probClass1,probClass2,probClass3,probClass4,probClass5,probClass6,probClass7,probClass8,probClass9);

for i =1:1500
    if(prob(i)<.5)
    error_lr=error_lr+1;
    end
end 

dlmwrite('classes lr.txt', [probClass0,probClass1,probClass2,probClass3,probClass4,probClass5,probClass6,probClass7,probClass8,probClass9],'delimiter', '\t','delimiter','\t');

error_lr=(error_lr/1500)*100



%set(gca,'XTick', [0 150 300 450 600 750 900 1050 1200 1350 1500])

%Wrong predicted 29

% 2 3 15 9 3 8 3 13 11 9