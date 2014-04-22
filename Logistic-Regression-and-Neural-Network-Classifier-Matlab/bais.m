function [bais1]=bais(X)
 	baisVector = ones(size(X, 1), 1);
	bais1 = horzcat(baisVector,X);	
end


 