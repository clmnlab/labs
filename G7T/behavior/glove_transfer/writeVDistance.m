function writeVDistance(subj_idn,total_n)
    if subj_idn < 10 
        if exist(strcat('regressor/G7T_TRP0',num2str(subj_idn),'/Vdistance'),'dir')==0
            mkdir(strcat('regressor/G7T_TRP0',num2str(subj_idn),'/Vdistance'))
        end
        result_dir=strcat('regressor/G7T_TRP0',num2str(subj_idn),'/Vdistance');
    else
        if exist(strcat('regressor/G7T_TRP',num2str(subj_idn),'/Vdistance'),'dir')==0
            mkdir(strcat('regressor/G7T_TRP',num2str(subj_idn),'/Vdistance'))
        end
        result_dir=strcat('regressor/G7T_TRP',num2str(subj_idn),'/Vdistance');
    end
    
    Distance=extractVdistance(subj_idn,total_n);
    for session_n = 1: total_n
        leftVdistance_file_name=strcat(result_dir,'/leftVdistance_run0',num2str(session_n),'.txt');
        rightVdistance_file_name=strcat(result_dir,'/rightVdistance_run0',num2str(session_n),'.txt');
        totalVdistance_file_name=strcat(result_dir,'/totalVdistance_run0',num2str(session_n),'.txt');
        leftVdistance=Distance.left(20*(session_n-1)+1:20*session_n);
        rightVdistance=Distance.right(20*(session_n-1)+1:20*session_n);
        totalVdistance=Distance.total(40*(session_n-1)+1:40*session_n);
        writeData(leftVdistance_file_name,leftVdistance);
        writeData(rightVdistance_file_name,rightVdistance);
        writeData(totalVdistance_file_name,totalVdistance);
    end
end

function writeData(filename,writeData)
    wrtieFile=fopen(filename,'w');
    fprintf(wrtieFile,'%d' ,writeData(1));
    fclose(wrtieFile);
    wrtieFile=fopen(filename,'a');
    fprintf(wrtieFile,' %d' ,writeData(2:end));
    fclose(wrtieFile);
end