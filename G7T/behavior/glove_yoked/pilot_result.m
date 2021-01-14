function results=pilot_result(id_n)
folname='main_ex';
subID=strcat(folname,'/GT_Main_',num2str(id_n),'/GT_Main_',num2str(id_n));
session1=load(strcat(subID,'_1.mat'));
session2=load(strcat(subID,'_2.mat'));
session3=load(strcat(subID,'_3.mat'));
session4=load(strcat(subID,'_4.mat'));
session5=load(strcat(subID,'_5.mat'));
session6=load(strcat(subID,'_6.mat'));
total_reward=cell(1,6);
total_rt=cell(1,6);
for session_n=1:6
    switch session_n
        case 1
            inputsession=session1;
        case 2
            inputsession=session2;
        case 3
            inputsession=session3;
        case 4
            inputsession=session4;
        case 5
            inputsession=session5;
        case 6
            inputsession=session6;
    end
    rewardRT=inputsession.learn.rewardRT;
    session_Reward=nan(1,20);
    session_RT=nan(1,20);
    for i = 1:length(rewardRT)
        session_Reward(i)=length(rmmissing(rewardRT{i}));
        session_RT(i)=mean(rmmissing(rewardRT{i}));
    end
    total_reward{session_n}=session_Reward;
    total_rt{session_n}=session_RT;
end
reward_result=cell2mat(total_reward);
rt_result=cell2mat(total_rt);
rt_result_naomit=nan(1,120);

for rtn=1:120
    if isnan(rt_result(rtn))
        rt_result_naomit(rtn)=10;
    else
        rt_result_naomit(rtn)=rt_result(rtn);
    end
end

mean_reward_result=nan(1,60);
mean_rt_result=nan(1,60);
k2=1;
for k=1:120
    if rem(k,2)==0
        mean_reward_result(k2)=mean([reward_result(k-1),reward_result(k)]);
        if isnan(rt_result(k-1)) && isnan(rt_result(k))
            mean_rt_result(k2)=mean([10,10]);
        elseif isnan(rt_result(k-1))
            mean_rt_result(k2)=mean([10,rt_result(k)]);
        elseif isnan(rt_result(k))
            mean_rt_result(k2)=mean([10,rt_result(k-1)]);
        end
        k2=k2+1;
    end
end

mean_rt_result_naomit=nan(1,60);
mean_k2=1;
for mean_k=1:120
    if rem(mean_k,2)==0
        mean_rt_result_naomit(mean_k2)=mean([rt_result_naomit(mean_k-1),rt_result_naomit(mean_k)]);
        mean_k2=mean_k2+1;
    end
end
results.reward.mean=mean_reward_result;
results.rt.mean=mean_rt_result;
results.rtomit.mean=mean_rt_result_naomit;
results.reward.total=reward_result;
results.rt.total=rt_result;
results.rtomit.total=rt_result_naomit;
end
        