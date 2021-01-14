function days=load_IFdata(id_n)
    id_N=Num22(id_n);
    exsetting=strcat('ex_data/G7T_IFE_',id_N,'/G7T_IFE_');
    total_matn=length(dir(strcat(exsetting,id_N,'*.mat')))/6;
    days=cell(1,total_matn);
    for day_n = 1:total_matn
        sessions=cell(1,6);
        for trial_n = 1:6
            ex_ID=strcat(exsetting,id_N,'_',exType((day_n-1)*6+trial_n),'_',Num22((day_n-1)*6+trial_n),'.mat');
            sessions{trial_n}=load(ex_ID);
            disp(ex_ID)
        end
        days{day_n}=sessions;
    end
end
function extype=exType(ex_num)
    if ex_num<7 || ex_num>36
        extype='fMRI';
    else
        extype='Behavior';
    end
end
function num22=Num22(num)
    if num<10
        num22=strcat('0',num2str(num));
    else
        num22=num2str(num);
    end
end