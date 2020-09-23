function regressor=extract_AM_regressor_IF(sub_idn,day_n,session_n)
    subjdata=load_IFdata(sub_idn);
    dayta=subjdata{day_n};
    inputsession=dayta{session_n};
    real_time=inputsession.start_wait;
    input_M2=inputsession.learnL;
    input_M1=inputsession.learnR;

    input_M2Fixtime=input_M2.fixtime;
    input_M1Fixtime=input_M1.fixtime;
    M1Time=input_M1.stimtime;
    M2Time=input_M2.stimtime;

    M1RT=add1stRT(input_M1);
    M2RT=add1stRT(input_M2);
    M1Hit=input_M1.HitTarget;
    M2Hit=input_M2.HitTarget;

    M1P1Reward=nan(1,length(M1Time)/2);
    M1P2Reward=nan(1,length(M1Time)/2);
    M2P1Reward=nan(1,length(M2Time)/2);
    M2P2Reward=nan(1,length(M2Time)/2);
    real_M1P1Time=nan(1,length(M1Time)/2);
    real_M1P2Time=nan(1,length(M1Time)/2);
    real_M2P1Time=nan(1,length(M1Time)/2);
    real_M2P2Time=nan(1,length(M1Time)/2);

    M1P1Reward_3s=cell(1,length(M1Time)/2);
    M1P2Reward_3s=cell(1,length(M1Time)/2);
    M2P1Reward_3s=cell(1,length(M1Time)/2);
    M2P2Reward_3s=cell(1,length(M1Time)/2);
    real_M1P1Time_3s=cell(1,length(M1Time)/2);
    real_M1P2Time_3s=cell(1,length(M1Time)/2);
    real_M2P1Time_3s=cell(1,length(M1Time)/2);
    real_M2P2Time_3s=cell(1,length(M1Time)/2);

    M1P1_n=1;
    M1P2_n=1;
    M2P1_n=1;
    M2P2_n=1;

    for real_i = 1:2:length(M1Time)
        real_M1P1Time(M1P1_n)=real_time;
        M1P1Reward(M1P1_n)= M1Hit(real_i);
        [rp1reward_3s,rp1time_3s]=reward3s(real_time,real_i,M1RT);
        M1P1Reward_3s{M1P1_n}=rp1reward_3s;
        real_M1P1Time_3s{M1P1_n}=rp1time_3s;
        M1P1_n=M1P1_n+1;
        real_time=real_time+input_M1.stimtime(real_i)+input_M1Fixtime(real_i);

        real_M2P1Time(M2P1_n)=real_time;
        M2P1Reward(M2P1_n)= M2Hit(real_i);
        [lp1reward_3s,lp1time_3s]=reward3s(real_time,real_i,M2RT);
        M2P1Reward_3s{M2P1_n}=lp1reward_3s;
        real_M2P1Time_3s{M2P1_n}=lp1time_3s;
        M2P1_n=M2P1_n+1;
        real_time=real_time+input_M2.stimtime(real_i)+input_M2Fixtime(real_i);

        real_M1P2Time(M1P2_n)=real_time;
        M1P2Reward(M1P2_n)= M1Hit(real_i+1);
        [rp2reward_3s,rp2time_3s]=reward3s(real_time,real_i+1,M1RT);
        M1P2Reward_3s{M1P2_n}=rp2reward_3s;
        real_M1P2Time_3s{M1P2_n}=rp2time_3s;
        M1P2_n=M1P2_n+1;
        real_time=real_time+input_M1.stimtime(real_i+1)+input_M1Fixtime(real_i+1);

        real_M2P2Time(M2P2_n)=real_time;
        M2P2Reward(M2P2_n)= M2Hit(real_i+1);
        [lp2reward_3s,lp2time_3s]=reward3s(real_time,real_i+1,M2RT);
        M2P2Reward_3s{M2P2_n}=lp2reward_3s;
        real_M2P2Time_3s{M2P2_n}=lp2time_3s;
        M2P2_n=M2P2_n+1;
        real_time=real_time+input_M2.stimtime(real_i+1)+input_M2Fixtime(real_i+1);

    end
    regressor.M1P1_Reward=M1P1Reward;
    regressor.M1P2_Reward=M1P2Reward;
    regressor.M2P1_Reward=M2P1Reward;
    regressor.M2P2_Reward=M2P2Reward;
    regressor.M1P1_Reward_3s=cell2mat(M1P1Reward_3s);
    regressor.M1P2_Reward_3s=cell2mat(M1P2Reward_3s);
    regressor.M2P1_Reward_3s=cell2mat(M2P1Reward_3s);
    regressor.M2P2_Reward_3s=cell2mat(M2P2Reward_3s);
    regressor.RealM1P1=real_M1P1Time;
    regressor.RealM1P2=real_M1P2Time;
    regressor.RealM2P1=real_M2P1Time;
    regressor.RealM2P2=real_M2P2Time;
    regressor.RealM1P1_3s=cell2mat(real_M1P1Time_3s);
    regressor.RealM1P2_3s=cell2mat(real_M1P2Time_3s);
    regressor.RealM2P1_3s=cell2mat(real_M2P1Time_3s);
    regressor.RealM2P2_3s=cell2mat(real_M2P2Time_3s);
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