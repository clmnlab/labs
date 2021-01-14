function plot_IF_Hit(subj_idn)
    total_n=6;
    exdays=7;
    subj_data=load_IFdata(subj_idn);
    t = tiledlayout(1,exdays);

    for day_n=1:exdays
        day_1stHit0=cell(1,total_n);
        day_2ndHit0=cell(1,total_n);
        for session_n = 1:total_n
            sessiondata=subj_data{day_n}{session_n};
            if isfield(sessiondata,'learnR')
                day_1stHit0{session_n}=sessiondata.learnR.HitTarget;
                day_2ndHit0{session_n}=sessiondata.learnL.HitTarget;
            else
                day_1stHit0{session_n}=sessiondata.learn1st.HitTarget;
                day_2ndHit0{session_n}=sessiondata.learn2nd.HitTarget;
            end
        end
        day_1stHit=evenMean(cell2mat(day_1stHit0));
        day_2ndHit=evenMean(cell2mat(day_2ndHit0));
        nexttile
        plot(day_1stHit)
        hold on
        plot(day_2ndHit,'r')
        title(strcat('DAY ',num2str(day_n)))
        axis([0 60 0 12])

    end
    legend('Mapping1','Mapping2')
    t.Padding = 'none';
    t.TileSpacing = 'none';
end