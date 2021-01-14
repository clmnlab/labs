function [rewardRT_mat,rewardRTyoked_mat,onset_time,onset_time_yoked,inputStim,inputStim_yoked]=extractRegressor(ID_n,total_n,session_n)
    folname='main_ex';
    subID=strcat(folname,'/GT_Main_',num2str(ID_n),'/GT_Main_',num2str(ID_n));
    sessions=load_session(subID,total_n);
    inputsession=sessions{session_n};
    start_time=inputsession.start_wait;
    sessionRT=inputsession.learn.rewardRT;
    sessionRT_yoked=inputsession.yoked.rewardRT;
    inputStim=inputsession.learn.stimtime;
    inputStim_yoked=inputsession.yoked.stimtime;
    inputFix=inputsession.learn.fixtime;
    inputFix_yoked=inputsession.yoked.fixtime;

    reward_time=cell(1,length(sessionRT));
    reward_time_yoked=cell(1,length(sessionRT_yoked));
    onset_time = nan(1,length(sessionRT));
    onset_time_yoked=nan(1,length(sessionRT));
    trial_start=start_time;
    trial_start_yoked=start_time+inputStim(1)+inputFix(1);
    for trial_n = 1: length(inputStim)
        ex_time=trial_start;
        ex_yoked_time=trial_start_yoked;
        onset_time(trial_n)=trial_start;
        onset_time_yoked(trial_n)=trial_start_yoked;
        inputRT=rmmissing(sessionRT{trial_n});
        inputRT_yoked=rmmissing(sessionRT_yoked{trial_n});
        trial_time=nan(1,length(inputRT));
        trial_time_yoked=nan(1,length(inputRT));
        for i = 1:length(inputRT)
            trial_time(i)=ex_time+inputRT(i);
            trial_time_yoked(i)=ex_yoked_time+inputRT_yoked(i);
            ex_time=ex_time+inputRT(i);
            ex_yoked_time=ex_yoked_time+inputRT_yoked(i);
        end
        reward_time{trial_n}=trial_time;
        reward_time_yoked{trial_n}=trial_time_yoked;
        trial_start=trial_start+inputStim(trial_n)+inputFix(trial_n)+inputStim_yoked(trial_n)+inputFix_yoked(trial_n);
        if trial_n ~= length(inputStim)
            trial_start_yoked=trial_start_yoked+inputStim_yoked(trial_n)+inputFix_yoked(trial_n)+inputStim(trial_n+1)+inputFix(trial_n+1);
        end
    end
    rewardRT_mat=cell2mat(reward_time);
    rewardRTyoked_mat=cell2mat(reward_time_yoked);
    
end