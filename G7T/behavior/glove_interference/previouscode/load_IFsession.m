function dayData=load_IFsession(sub_idn,total_n,day_n,EL)
    if sub_idn<10
        subIDn=strcat('0',num2str(sub_idn));
    elseif sub_idn>=10
        subIDn=num2str(sub_idn);
    end
    ID_name = strcat('G7T_IF',EL,'_*_',subIDn);
    result_dir='ex_data';
    dayData=cell(1,day_n);
    for dayN = 1:day_n
        load_ID=strcat(result_dir,'/',ID_name,'/day',num2str(dayN),'/',ID_name);
        dayData{dayN}=load_session(load_ID,total_n);
    end
end

function sessions=load_session(subID,total_n)
    sessions=cell(1,total_n);
    for i = 1:total_n
        sessions{i}=load(strcat(subID,'_',num2str(i),'.mat'));
    end
end