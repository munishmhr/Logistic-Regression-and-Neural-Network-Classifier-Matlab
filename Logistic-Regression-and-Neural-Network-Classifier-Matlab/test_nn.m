load('test.mat');
load nn_final.mat

bais = ones(1500,1);
test = horzcat(bais,test);
prob=zeros(1500,10);

y=zeros(1500);
wpclass0=0;
wpclass1=0;
wpclass2=0;
wpclass3=0;
wpclass4=0;
wpclass5=0;
wpclass6=0;
wpclass7=0;
wpclass8=0;
wpclass9=0;
    for i = 1:1500
        % forward prob
        Z2 = theta1 * test(i,:)';
        actL2 = vertcat(1, arrayfun(@sigmoid,Z2) );
        Z3 = theta2 * actL2;
        actL3 = arrayfun(@sigmoid,Z3); 
        prob(i,1:10)=actL3(:);        
    end
    p0 = prob(1:150,1);p1 = prob(151:300,2);p2 = prob(301:450,3);p3 = prob(451:600,4);p4 = prob(601:750,5);
    p5 = prob(751:900,6);p6 = prob(901:1050,7);p7 = prob(1051:1200,8);p8 = prob(1201:1350,9);p9 = prob(1351:1500,10);
    probTestSet = vertcat(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
   plot(probTestSet);
    for i=1:150
       if(prob(i,1)<.5)
          wpclass0=wpclass0+1 ;
       end
       
       
    end
    
     for i=151:300
       if(prob(i,2)<.5)
          wpclass1=wpclass1+1 ;
       end
     end
    
      for i=301:450
       if(prob(i,3)<.5)
          wpclass2=wpclass2+1 ;
       end
      end
    
       for i=451:600
       if(prob(i,4)<.5)
          wpclass3=wpclass3+1 ;
       end
       end
    
        for i=601:750
       if(prob(i,5)<.5)
          wpclass4=wpclass4+1 ;
       end
        end
     for i=751:900
       if(prob(i,6)<.5)
          wpclass5=wpclass5+1 ;
       end
     end
     for i=901:1050
       if(prob(i,7)<.5)
          wpclass6=wpclass7+1 ;
       end
     end
     for i=1051:1200
       if(prob(i,8)<.5)
          wpclass7=wpclass7+1 ;
       end
     end
    for i=1201:1350
       if(prob(i,9)<.5)
          wpclass8=wpclass8+1 ;
       end
    end
    for i=1351:1500
       if(prob(i,10)<.5)
          wpclass9=wpclass9+1 ;
       end
    end
    
    % error
    
    dlmwrite('classes_nn.txt', prob);
    error = wpclass0 + wpclass1 +wpclass2 +wpclass3 +wpclass4 +wpclass5 + wpclass6 +wpclass7 +wpclass8 +wpclass9 ; 
    error = (error/1500)*100
   

