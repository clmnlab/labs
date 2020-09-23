function sessions=load_sessions(sub_idn,total_n)
    if sub_idn>9
        sID=num2str(sub_idn);
    elseif sub_idn<=9
        sID=strcat('0',num2str(sub_idn));
    end
    ID_name = strcat('G7T_','TR','_',sID);
    result_dir='ex_data';
    load_ID=strcat(result_dir,'/',ID_name,'/',ID_name);
    if sub_idn<=7
        sessions=load_session(load_ID,total_n);
    else
        sessions=load_fMRI_session(load_ID,total_n);
    end
end

function sessions=load_session(subID,total_n)
    sessions=cell(1,total_n);
    for i = 1:total_n
        sessions{i}=load(strcat(subID,'_',num2str(i),'.mat'));
    end
end

function sessions=load_fMRI_session(subID,total_n)
    sessions=cell(1,total_n);
    for i = 1:total_n
        sessions{i}=load(strcat(subID,'_fMRI_0',num2str(i),'.mat'));
    end
end