function [out] = fun_max_Trq(x, fun_Psi_m, fun_L_d, fun_L_q, N_pp)

    I_d = x(1);
    I_q = x(2);

    Psi_m = fun_Psi_m(I_q);
    L_d = fun_L_d(I_d, I_q);
    L_q = fun_L_q(I_d, I_q);
   
    Trq=3*N_pp/2*(Psi_m*I_q+(L_d-L_q)*I_d*I_q);
    %Trq=interp2(Id_mat,Iq_mat,Trq_Maxwell_mat,isd,isq);
    out=[-Trq];

end

