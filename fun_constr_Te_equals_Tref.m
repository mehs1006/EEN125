function [c,ceq] = fun_constr_Te_equals_Tref(x,fun_Trq,T_ref)

    I_d = x(1);
    I_q = x(2);

    Trq = fun_Trq(I_d, I_q);
    
    c = 0;
    ceq = [Trq-T_ref];

end
