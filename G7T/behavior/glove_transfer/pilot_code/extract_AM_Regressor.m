function regressor = extract_AM_Regressor(sub_idn,session_n,total_n)
    sessions=load_Psession(sub_idn,total_n);
    inputsession=sessions{session_n};
    real_time=inputsession.start_wait;
    input_Left=inputsession.learnL;
    input_Right=inputsession.learnR;

    input_LFixtime=input_Left.fixtime;
    input_RFixtime=input_Right.fixtime;
    rightTime=input_Right.stimtime;
    leftTime=input_Left.stimtime;

    rightRT=add1stRT(input_Right);
    leftRT=add1stRT(input_Left);
    rightHit=input_Right.HitTarget;
    leftHit=input_Left.HitTarget;

    RP1Reward=nan(1,length(rightTime)/2);
    RP2Reward=nan(1,length(rightTime)/2);
    LP1Reward=nan(1,length(leftTime)/2);
    LP2Reward=nan(1,length(leftTime)/2);
    real_RP1Time=nan(1,length(rightTime)/2);
    real_RP2Time=nan(1,length(rightTime)/2);
    real_LP1Time=nan(1,length(rightTime)/2);
    real_LP2Time=nan(1,length(rightTime)/2);

    RP1Reward_3s=cell(1,length(rightTime)/2);
    RP2Reward_3s=cell(1,length(rightTime)/2);
    LP1Reward_3s=cell(1,length(rightTime)/2);
    LP2Reward_3s=cell(1,length(rightTime)/2);
    real_RP1Time_3s=cell(1,length(rightTime)/2);
    real_RP2Time_3s=cell(1,length(rightTime)/2);
    real_LP1Time_3s=cell(1,length(rightTime)/2);
    real_LP2Time_3s=cell(1,length(rightTime)/2);

    RP1_n=1;
    RP2_n=1;
    LP1_n=1;
    LP2_n=1;

    for real_i = 1:2:length(rightTime)
        real_RP1Time(RP1_n)=real_time;
        RP1Reward(RP1_n)= rightHit(real_i);
        [rp1reward_3s,rp1time_3s]=reward3s(real_time,real_i,rightRT);
        RP1Reward_3s{RP1_n}=rp1reward_3s;
        real_RP1Time_3s{RP1_n}=rp1time_3s;
        RP1_n=RP1_n+1;
        real_time=real_time+input_Right.stimtime(real_i)+input_RFixtime(real_i);

        real_LP1Time(LP1_n)=real_time;
        LP1Reward(LP1_n)= leftHit(real_i);
        [lp1reward_3s,lp1time_3s]=reward3s(real_time,real_i,leftRT);
        LP1Reward_3s{LP1_n}=lp1reward_3s;
        real_LP1Time_3s{LP1_n}=lp1time_3s;
        LP1_n=LP1_n+1;
        real_time=real_time+input_Left.stimtime(real_i)+input_LFixtime(real_i);

        real_RP2Time(RP2_n)=real_time;
        RP2Reward(RP2_n)= rightHit(real_i+1);
        [rp2reward_3s,rp2time_3s]=reward3s(real_time,real_i+1,rightRT);
        RP2Reward_3s{RP2_n}=rp2reward_3s;
        real_RP2Time_3s{RP2_n}=rp2time_3s;
        RP2_n=RP2_n+1;
        real_time=real_time+input_Right.stimtime(real_i+1)+input_RFixtime(real_i+1);

        real_LP2Time(LP2_n)=real_time;
        LP2Reward(LP2_n)= leftHit(real_i+1);
        [lp2reward_3s,lp2time_3s]=reward3s(real_time,real_i+1,leftRT);
        LP2Reward_3s{LP2_n}=lp2reward_3s;
        real_LP2Time_3s{LP2_n}=lp2time_3s;
        LP2_n=LP2_n+1;
        real_time=real_time+input_Left.stimtime(real_i+1)+input_LFixtime(real_i+1);

    end
    regressor.RP1_Reward=RP1Reward;
    regressor.RP2_Reward=RP2Reward;
    regressor.LP1_Reward=LP1Reward;
    regressor.LP2_Reward=LP2Reward;
    regressor.RP1_Reward_3s=cell2mat(RP1Reward_3s);
    regressor.RP2_Reward_3s=cell2mat(RP2Reward_3s);
    regressor.LP1_Reward_3s=cell2mat(LP1Reward_3s);
    regressor.LP2_Reward_3s=cell2mat(LP2Reward_3s);
    regressor.RealRP1=real_RP1Time;
    regressor.RealRP2=real_RP2Time;
    regressor.RealLP1=real_LP1Time;
    regressor.RealLP2=real_LP2Time;
    regressor.RealRP1_3s=cell2mat(real_RP1Time_3s);
    regressor.RealRP2_3s=cell2mat(real_RP2Time_3s);
    regressor.RealLP1_3s=cell2mat(real_LP1Time_3s);
    regressor.RealLP2_3s=cell2mat(real_LP2Time_3s);
end

function [reward3s,times3s]=reward3s(start_time,trial_n,inputRT)
    try 
        reward3s=nan(1,3);
        times3s=nan(1,3);
        RT=inputRT{trial_n};
        times3s(1)=start_time;
        times3s(2)=start_time+3;
        times3s(3)=start_time+6;
        Hit_n=1;
        if isempty(RT)
            reward3s=zeros(1,3);
        else
            while (sum(RT(1:Hit_n))<3)&& Hit_n<length(RT)
                Hit_n=Hit_n+1;
            end
            reward3s(1)=Hit_n-1;
            while sum(RT(1:Hit_n))<6 && Hit_n<length(RT) 
                Hit_n=Hit_n+1;
            end
            reward3s(2)=Hit_n-reward3s(1);
            reward3s(3)=length(RT)-Hit_n+1;
        end
    catch
        disp('error')
        disp(RT)
        disp(Hit_n)
        disp(reward3s)
    end
end
function totalRTs=add1stRT(inputhand)
    totalRTs=cell(1,length(inputhand.stimtime_start));
    for i =1:length(inputhand.stimtime_start)
        startTime=inputhand.stimtime_start(i);
        rewardTime=rmmissing(inputhand.rewardtime{i});
        inputRT=rmmissing(inputhand.rewardRT{i});
        if ~isempty(rewardTime) && ~isempty(inputRT)
            startHit=rewardTime(1)-startTime-inputRT(1);
            totalRTs{i}=[startHit inputRT];
        elseif (length(rewardTime)==1) && isempty(inputRT)
            startHit=rewardTime(1)-startTime;
            totalRTs{i}=[startHit inputRT];
        end
    end
end
