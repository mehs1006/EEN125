function [out] = fun_max_Trq(x, fun_L_d, fun_L_q, N_pp)

    I_d = x(1);
    I_q = x(2);

    L_d = fun_L_d(I_d, I_q);
    L_q = fun_L_q(I_d, I_q);
   
    Trq=3*N_pp/2*(L_d-L_q)*I_d*I_q;

    out=[-Trq];

end

