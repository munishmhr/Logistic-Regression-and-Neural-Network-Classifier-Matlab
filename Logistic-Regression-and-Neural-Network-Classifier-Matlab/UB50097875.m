x = 50097875;
 
M=rand(3,4);

N=rand(4,3);

Sum_M=0;Sum_M=sum(M(:));
Sum_M ;

Sum_N=0;Sum_N=sum(N(:));

Sum_N ;

M_=[];
N_=[];
M_=transpose(M);
M_;
N_=transpose(N)
N_;



Product_MN=[];Product_MN=M*N;
Product_MN;

Element_Product_MN=[];Element_Product_MN=M.*N_;
Element_Product_MN;

M=rand(3,4);
M;


Largest_M=max(M(:));
Largest_M;

[row,col] = find(M==Largest_M);
 row;
 col;

Eig_Product_MN=eig(Product_MN);
Eig_Product_MN;


L=[1:1000];
Sum1=sum(L(1,:));
Sum1;



Sum2=0;a=1;
while a<1001
Sum2=Sum2+L(1,a);
a=a+1;
end


Sum3=0;
for a=1:1000
Sum3=Sum3+L(1,a);
end

t= -6:0.1:6;
y = sin(t);
plot(t,y,'r');title('A Sin(t) Function');
