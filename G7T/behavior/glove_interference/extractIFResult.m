function sub00=extractIFResult(sub_idn,day_n,exdays,total_n)
    subj_data=load_IFsession(sub_idn,total_n,exdays);
    
    total_reward1st=cell(1,total_n);
    total_rt_1st=cell(1,total_n);
    total_rmrt_1st=cell(1,total_n);
    total_reward2nd=cell(1,total_n);
    total_rt_2nd=cell(1,total_n);
    total_rmrt_2nd=cell(1,total_n);
    
    for sessions = 1: total_n
        [total_reward1st{sessions},total_rt_1st{sessions},total_rmrt_1st{sessions},total_reward2nd{sessions},total_rt_2nd{sessions},total_rmrt_2nd{sessions}] = extract_rewardRT(subj_data{day_n},sessions);
    end
    
    result1st.reward=cell2mat(total_reward1st);
    result1st.rt=cell2mat(total_rt_1st);
    result1st.rmrt=cell2mat(total_rmrt_1st);
    result1st.meanReward=evenMean(result1st.reward);
    result1st.meanRT=evenMean(result1st.rmrt);
    result2nd.reward=cell2mat(total_reward2nd);
    result2nd.rt=cell2mat(total_rt_2nd);
    result2nd.rmrt=cell2mat(total_rmrt_2nd);
    result2nd.meanReward=evenMean(result2nd.reward);
    result2nd.meanRT=evenMean(result2nd.rmrt);
    
    sub00.First=result1st;
    sub00.Second=result2nd;
end

function [reward1st,rt_1st,rmrt_1st,reward2nd,rt_2nd,rmrt_2nd]= extract_rewardRT(sessions,session_n)
    input_session=sessions{session_n};
    input_learn1st=input_session.learn1st;
    input_learn2nd=input_session.learn2nd;
    reward1st=input_learn1st.HitTarget;
    reward2nd=input_learn2nd.HitTarget;
    [rt_1st,rmrt_1st]=cellmean(input_learn1st.rewardRT);
    [rt_2nd,rmrt_2nd]=cellmean(input_learn2nd.rewardRT);
end



