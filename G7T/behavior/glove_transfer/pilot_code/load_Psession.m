function sessions=load_Psession(sub_idn,total_n)
    ID_name = strcat('GT_','Pilot','_',num2str(sub_idn));
    result_dir='pilot_ex';
    load_ID=strcat(result_dir,'/',ID_name,'/',ID_name);
    sessions=load_session(load_ID,total_n);
end

function sessions=load_session(subID,total_n)
    sessions=cell(1,total_n);
    for i = 1:total_n
        sessions{i}=load(strcat(subID,'_',num2str(i),'.mat'));
    end
end