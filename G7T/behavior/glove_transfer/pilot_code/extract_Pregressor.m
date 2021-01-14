function regressor=extract_Pregressor(sub_idn,session_n,total_n)
    sessions=load_Psession(sub_idn,total_n);
    inputsession=sessions{session_n};
    real_time=inputsession.start_wait;
    input_Left=inputsession.learnL;
    input_Right=inputsession.learnR;
    input_LFixtime=input_Left.fixtime;
    input_RFixtime=input_Right.fixtime;
    
    RightTime=input_Right.stimtime;
    LeftTime=input_Left.stimtime;
    LSSTime=nan(1,length(RightTime)+length(LeftTime));
    RP1Time=nan(1,length(RightTime)/2);
    RP2Time=nan(1,length(RightTime)/2);
    LP1Time=nan(1,length(RightTime)/2);
    LP2Time=nan(1,length(RightTime)/2);
    real_RP1Time=nan(1,length(RightTime)/2);
    real_RP2Time=nan(1,length(RightTime)/2);
    real_LP1Time=nan(1,length(RightTime)/2);
    real_LP2Time=nan(1,length(RightTime)/2);
    
    RP1_n=1;
    RP2_n=1;
    LP1_n=1;
    LP2_n=1;
    LSS_n=1;

    for real_i = 1:2:length(RightTime)
        real_RP1Time(RP1_n)=real_time;
        LSSTime(LSS_n)=real_time;
        RP1Time(RP1_n)= RightTime(real_i);
        RP1_n=RP1_n+1;
        LSS_n=LSS_n+1;
        real_time=real_time+input_Right.stimtime(real_i)+input_RFixtime(real_i);
        
        real_LP1Time(LP1_n)=real_time;
        LSSTime(LSS_n)=real_time;
        LP1Time(LP1_n)= LeftTime(real_i);
        LP1_n=LP1_n+1;
        LSS_n=LSS_n+1;
        real_time=real_time+input_Left.stimtime(real_i)+input_LFixtime(real_i);
        
        real_RP2Time(RP2_n)=real_time;
        LSSTime(LSS_n)=real_time;
        RP2Time(RP2_n)= RightTime(real_i+1);
        RP2_n=RP2_n+1;
        LSS_n=LSS_n+1;
        real_time=real_time+input_Right.stimtime(real_i+1)+input_RFixtime(real_i+1);
        
        real_LP2Time(LP2_n)=real_time;
        LSSTime(LSS_n)=real_time;
        LP2Time(LP2_n)= LeftTime(real_i+1);
        LP2_n=LP2_n+1;
        LSS_n=LSS_n+1;
        real_time=real_time+input_Left.stimtime(real_i+1)+input_LFixtime(real_i+1);
        
    end
    regressor.RP1_time=RP1Time;
    regressor.RP2_time=RP2Time;
    regressor.LP1_time=LP1Time;
    regressor.LP2_time=LP2Time;
    regressor.RealRP1=real_RP1Time;
    regressor.RealRP2=real_RP2Time;
    regressor.RealLP1=real_LP1Time;
    regressor.RealLP2=real_LP2Time;
    regressor.LSS=LSSTime;
end