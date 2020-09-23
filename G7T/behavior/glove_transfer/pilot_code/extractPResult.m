function sub00=extractPResult(sub_idn,total_n)
%     ID_name = strcat('GT_','Pilot','_',num2str(sub_idn));
%     result_dir='pilot_ex';
%     load_ID=strcat(result_dir,'/',ID_name,'/',ID_name);
    sessions=load_Psession(sub_idn,total_n);

    total_rewardL=cell(1,length(sessions));
    total_rt_L=cell(1,length(sessions));
    total_rmrt_L=cell(1,length(sessions));
    total_rewardR=cell(1,length(sessions));
    total_rt_R=cell(1,length(sessions));
    total_rmrt_R=cell(1,length(sessions));
    for session_n = 1: length(sessions)
        [total_rewardL{session_n},total_rt_L{session_n},total_rmrt_L{session_n},total_rewardR{session_n},total_rt_R{session_n},total_rmrt_R{session_n}] = extract_rewardRT(sessions,session_n);
    end
    resultL.reward=cell2mat(total_rewardL);
    resultL.rt=cell2mat(total_rt_L);
    resultL.rmrt=cell2mat(total_rmrt_L);
    resultL.meanReward=evenMean(resultL.reward);
    resultL.meanRT=evenMean(resultL.rmrt);
    resultR.reward=cell2mat(total_rewardR);
    resultR.rt=cell2mat(total_rt_R);
    resultR.rmrt=cell2mat(total_rmrt_R);
    resultR.meanReward=evenMean(resultR.reward);
    resultR.meanRT=evenMean(resultR.rmrt);
    sub00.left=resultL;
    sub00.right=resultR;
end

function [rewardL,rt_L,rmrt_L,rewardR,rt_R,rmrt_R]= extract_rewardRT(sessions,session_n)
    inputsession=sessions{session_n};
    input_learnL=inputsession.learnL;
    input_learnR=inputsession.learnR;
    rewardL=input_learnL.HitTarget;
    rewardR=input_learnR.HitTarget;
    [rt_L,rmrt_L]=cellmean(input_learnL.rewardRT);
    [rt_R,rmrt_R]=cellmean(input_learnR.rewardRT);
end


