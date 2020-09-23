function regressor=extract_Pregressor2(sub_idn,session_n,total_n)
    sessions=load_Psession(sub_idn,total_n);
    inputsession=sessions{session_n};
    real_time=inputsession.start_wait;
    input_Left=inputsession.learnL;
    input_Right=inputsession.learnR;
    input_LFixtime=input_Left.fixtime;
    input_RFixtime=input_Right.fixtime;
    
    RightTime=input_Right.stimtime;
    LeftTime=input_Left.stimtime;
    real_RightTime=nan(1,length(RightTime));
    real_LeftTime=nan(1,length(LeftTime));
    Pos1Time=nan(1,length(RightTime));
    Pos2Time=nan(1,length(RightTime));
    real_Pos1Time=nan(1,length(RightTime));
    real_Pos2Time=nan(1,length(RightTime));
    pos1_i=1;
    pos2_i=1;

    for real_i = 1:length(RightTime)
        real_RightTime(real_i)=real_time;
        if rem(real_i,2)==1
            real_Pos1Time(pos1_i)=real_time;
            Pos1Time(pos1_i)= RightTime(real_i);
            pos1_i=pos1_i+1;
        elseif rem(real_i,2)==0
            real_Pos2Time(pos2_i)=real_time;
            Pos2Time(pos2_i)= RightTime(real_i);
            pos2_i=pos2_i+1;
        end
        real_time=real_time+input_Right.stimtime(real_i)+input_RFixtime(real_i);
        real_LeftTime(real_i)=real_time;
        if rem(real_i,2)==1
            real_Pos1Time(pos1_i)=real_time;
            Pos1Time(pos1_i)= LeftTime(real_i);
            pos1_i=pos1_i+1;
        elseif rem(real_i,2)==0
            real_Pos2Time(pos2_i)=real_time;
            Pos2Time(pos2_i)= LeftTime(real_i);
            pos2_i=pos2_i+1;
        end
        real_time=real_time+input_Left.stimtime(real_i)+input_LFixtime(real_i);
    end
    regressor.Right=RightTime;
    regressor.Left=LeftTime;
    regressor.RealR=real_RightTime;
    regressor.RealL=real_LeftTime;
    regressor.Pos1=Pos1Time;
    regressor.Pos2=Pos2Time;
    regressor.RealP1=real_Pos1Time;
    regressor.RealP2=real_Pos2Time;
end