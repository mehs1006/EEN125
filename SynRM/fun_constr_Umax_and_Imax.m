function [noneq,ceq] = fun_constr_Umax_and_Imax(x, omega0, fun_L_d, fun_L_q, R_s, U_max, I_max)

    I_d = x(1);
    I_q = x(2);

    L_d = fun_L_d(I_d, I_q);
    L_q = fun_L_q(I_d, I_q);

    %omega0 = n_ref/60*2*pi;
    
    U_d = R_s*I_d-omega0*L_q*I_q;
    U_q = R_s*I_q+omega0*L_d*I_d;    
    
    
    noneq = [I_d^2+I_q^2-I_max^2];
    
    ceq = [U_d^2+U_q^2-U_max^2; 0];

end
