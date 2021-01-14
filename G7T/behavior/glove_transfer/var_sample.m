sub_n=6;
dat=load_sessions(sub_n,6);
varR=cell(1,6);
varL=cell(1,6);
varT=cell(1,6);
for session_n=1:6
    session_dat=dat{session_n};
    Rdat=session_dat.learnR.HitTarget;
    Ldat=session_dat.learnL.HitTarget;
    varianceR=nan(1,19);
    varianceL=nan(1,19);
    variance_total=nan(1,39);
    for i = 1:19
        varianceR(i)=abs(Rdat(i+1)-Rdat(i));
        varianceL(i)=abs(Ldat(i+1)-Ldat(i));
        variance_total(2*i-1)=abs(Ldat(i)-Rdat(i));
        variance_total(2*i)=abs(Rdat(i+1)-Rdat(i));
    end
    variance_total(39)=abs(Ldat(end)-Rdat(end));
    varR{session_n}=varianceR;
    varL{session_n}=varianceL;
    varT{session_n}=variance_total;
end

R_var=cell2mat(varR);
L_var=cell2mat(varL);
T_var=cell2mat(varT);
