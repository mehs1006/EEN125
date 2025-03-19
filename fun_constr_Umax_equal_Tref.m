function [noneq,ceq] = fun_constr_Umax_equal_Tref(x, fun_Psi_m, fun_L_d, fun_L_q, N_pp, omega0, R_s, U_max, I_max, T_ref)

    I_d = x(1);
    I_q = x(2);

    Psi_m = fun_Psi_m(I_q);
    L_d = fun_L_d(I_d, I_q);
    L_q = fun_L_q(I_d, I_q);

    Trq = 3*N_pp/2*(Psi_m*I_q+(L_d-L_q)*I_d*I_q);
    U_d = R_s*I_d-omega0*L_q*I_q;
    U_q = R_s*I_q+omega0*(Psi_m+L_d*I_d);    

    noneq = [I_d^2+I_q^2-I_max^2; U_d^2+U_q^2-U_max^2];
    
    ceq = [Trq-T_ref;0];

end
