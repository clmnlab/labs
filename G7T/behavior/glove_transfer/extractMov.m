function total_mov = extractMov(subj_idn,total_n)
    total_movL0=cell(1,total_n);
    total_movR0=cell(1,total_n);
    total_movL0_hit=cell(1,total_n);
    total_movR0_hit=cell(1,total_n);
    for session_n = 1:total_n
        subj=load_Psession(subj_idn,total_n);
        left=subj{session_n}.learnL;
        right=subj{session_n}.learnR;
        session_movL=nan(1,length(left.data));
        session_movR=nan(1,length(right.data));
        session_movL_hit=nan(1,length(left.data));
        session_movR_hit=nan(1,length(right.data));
        for trial_n = 1:length(session_movL)
            dataL=left.data{trial_n};
            dataR=right.data{trial_n};
            movL_data=nan(13,length(dataL)-1);
            movR_data=nan(13,length(dataR)-1);
            for i = 2:length(dataL)
                movL_data(:,i-1)=abs(dataL(:,i)-dataL(:,i-1));
                movR_data(:,i-1)=abs(dataR(:,i)-dataR(:,i-1));
            end
            session_movL(trial_n)=sum(mean(movL_data,2));
            session_movR(trial_n)=sum(mean(movR_data,2));
            session_movL_hit(trial_n)=sum(mean(movL_data,2))/(left.HitTarget(trial_n)+1);
            session_movR_hit(trial_n)=sum(mean(movR_data,2))/(right.HitTarget(trial_n)+1);
        end
        total_movL0{session_n}=session_movL;
        total_movR0{session_n}=session_movR;
        total_movL0_hit{session_n}=session_movL_hit;
        total_movR0_hit{session_n}=session_movR_hit;
        total_mov.L=cell2mat(total_movL0);
        total_mov.R=cell2mat(total_movR0);
        total_mov.L_hit=cell2mat(total_movL0_hit);
        total_mov.R_hit=cell2mat(total_movR0_hit);
    end
end