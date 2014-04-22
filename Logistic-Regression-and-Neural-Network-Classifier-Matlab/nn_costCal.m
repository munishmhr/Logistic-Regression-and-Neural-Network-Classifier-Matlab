function lcost = nn_costCal(target,actL4)
%xxx = target
%yyy = actL4
lcost  = 0;
for k=1:10
    lcost = lcost + ( target(k)*log(actL4(k)) + (1-target(k))*(log(1-actL4(k))) );
end

end