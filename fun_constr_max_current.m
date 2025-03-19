function [noneq,ceq] = fun_constr_max_current(x,I_max)

    noneq=[0];
    ceq=[x(1)^2+x(2)^2-I_max^2];

end
