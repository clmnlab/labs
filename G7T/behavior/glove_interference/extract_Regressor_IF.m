function regressor=extract_Regressor_IF(subj_idn,day_n,session_n)
    subj_data=load_IFdata(subj_idn);
    dayta=subj_data{day_n};
    inputsession=dayta{session_n};

    real_time=inputsession.start_wait;
    input_1st=inputsession.learnR;
    input_2nd=inputsession.learnL;
    input_1stFixtime=input_1st.fixtime;
    input_2ndFixtime=input_2nd.fixtime;

    time1st=input_1st.stimtime;
    time2nd=input_2nd.stimtime;
    LSSTime=nan(1,length(time2nd)+length(time1st));

    P1Time1st=nan(1,length(time1st)/2);
    P2Time1st=nan(1,length(time1st)/2);
    P1Time2nd=nan(1,length(time2nd)/2);
    P2Time2nd=nan(1,length(time2nd)/2);

    real_P1Time1st=nan(1,length(time1st)/2);
    real_P2Time1st=nan(1,length(time1st)/2);
    real_P1Time2nd=nan(1,length(time2nd)/2);
    real_P2Time2nd=nan(1,length(time2nd)/2);

    P1_1stn=1;
    P2_1stn=1;
    P1_2ndn=1;
    P2_2ndn=1;
    LSS_n=1;

    for real_i = 1:2:length(time2nd)
        real_P1Time1st(P1_1stn)=real_time;
        LSSTime(LSS_n)=real_time;
        P1Time1st(P1_1stn)= time1st(real_i);
        P1_1stn=P1_1stn+1;
        LSS_n=LSS_n+1;
        real_time=real_time+input_1st.stimtime(real_i)+input_1stFixtime(real_i);

        real_P1Time2nd(P1_2ndn)=real_time;
        LSSTime(LSS_n)=real_time;
        P1Time2nd(P1_2ndn)= time2nd(real_i);
        P1_2ndn=P1_2ndn+1;
        LSS_n=LSS_n+1;
        real_time=real_time+input_2nd.stimtime(real_i)+input_2ndFixtime(real_i);

        real_P2Time1st(P2_1stn)=real_time;
        LSSTime(LSS_n)=real_time;
        P2Time1st(P2_1stn)= time1st(real_i+1);
        P2_1stn=P2_1stn+1;
        LSS_n=LSS_n+1;
        real_time=real_time+input_1st.stimtime(real_i+1)+input_1stFixtime(real_i+1);

        real_P2Time2nd(P2_2ndn)=real_time;
        LSSTime(LSS_n)=real_time;
        P2Time2nd(P2_2ndn)= time2nd(real_i+1);
        P2_2ndn=P2_2ndn+1;
        LSS_n=LSS_n+1;
        real_time=real_time+input_2nd.stimtime(real_i+1)+input_2ndFixtime(real_i+1);

    end

    regressor.P1_time1st=P1Time1st;
    regressor.P2_time1st=P2Time1st;
    regressor.P1_time2nd=P1Time2nd;
    regressor.P2_time2nd=P2Time2nd;
    regressor.RealP1_1st=real_P1Time1st;
    regressor.RealP2_1st=real_P2Time1st;
    regressor.RealP1_2nd=real_P1Time2nd;
    regressor.RealP2_2nd=real_P2Time2nd;
    regressor.LSS=LSSTime;
end