function write_Pregressor(subj_idn,total_n)
    if subj_idn < 10 
        if exist(strcat('regressor/G7T_TRP0',num2str(subj_idn)),'dir')==0
            mkdir(strcat('regressor/G7T_TRP0',num2str(subj_idn)))
        end
        result_dir=strcat('regressor/G7T_TRP0',num2str(subj_idn));
    else
        if exist(strcat('regressor/G7T_TRP',num2str(subj_idn)),'dir')==0
            mkdir(strcat('regressor/G7T_TRP',num2str(subj_idn)))
        end
        result_dir=strcat('regressor/G7T_TRP',num2str(subj_idn));
    end
    regressor=extract_Pregressor(subj_idn,1,total_n);
    leftP1_File = fopen(strcat(result_dir,'/leftP1time.txt'),'w');
    rightP1_File = fopen(strcat(result_dir,'/rightP1time.txt'),'w');
    leftP2_File = fopen(strcat(result_dir,'/leftP2time.txt'),'w');
    rightP2_File = fopen(strcat(result_dir,'/rightP2time.txt'),'w');
    
    fprintf(leftP1_File,'%f:%f',regressor.RealLP1(1),regressor.LP1_time(1));
    fprintf(rightP1_File,'%f:%f',regressor.RealRP1(1),regressor.RP1_time(1));
    fprintf(leftP2_File,'%f:%f',regressor.RealLP2(1),regressor.LP2_time(1));
    fprintf(rightP2_File,'%f:%f',regressor.RealRP2(1),regressor.RP2_time(1));
    
    for i = 2:length(regressor.RealLP1)
        fprintf(leftP1_File,' %f:%f',regressor.RealLP1(i),regressor.LP1_time(i));
        fprintf(rightP1_File,' %f:%f',regressor.RealRP1(i),regressor.RP1_time(i));
        fprintf(leftP2_File,' %f:%f',regressor.RealLP2(i),regressor.LP2_time(i));
        fprintf(rightP2_File,' %f:%f',regressor.RealRP2(i),regressor.RP2_time(i));
    end
    
    fclose(leftP1_File);
    fclose(rightP1_File);
    fclose(leftP2_File);
    fclose(rightP2_File);
    
    for session_n = 2:total_n
        regressor=extract_Pregressor(subj_idn,session_n,total_n);
        leftP1_File = fopen(strcat(result_dir,'/leftP1time.txt'),'a');
        rightP1_File = fopen(strcat(result_dir,'/rightP1time.txt'),'a');
        leftP2_File = fopen(strcat(result_dir,'/leftP2time.txt'),'a');
        rightP2_File = fopen(strcat(result_dir,'/rightP2time.txt'),'a');
        
        fprintf(leftP1_File,'\n%f:%f',regressor.RealLP1(1),regressor.LP1_time(1));
        fprintf(rightP1_File,'\n%f:%f',regressor.RealRP1(1),regressor.RP1_time(1));
        fprintf(leftP2_File,'\n%f:%f',regressor.RealLP2(1),regressor.LP2_time(1));
        fprintf(rightP2_File,'\n%f:%f',regressor.RealRP2(1),regressor.RP2_time(1));
        
        for j = 2:length(regressor.RealRP1)
            fprintf(leftP1_File,' %f:%f',regressor.RealLP1(j),regressor.LP1_time(j));
            fprintf(rightP1_File,' %f:%f',regressor.RealRP1(j),regressor.RP1_time(j));
            fprintf(leftP2_File,' %f:%f',regressor.RealLP2(j),regressor.LP2_time(j));
            fprintf(rightP2_File,' %f:%f',regressor.RealRP2(j),regressor.RP2_time(j));
        end
        fclose(leftP1_File);
        fclose(rightP1_File);
        fclose(leftP2_File);
        fclose(rightP2_File);
    end
    
    for run = 1 : total_n
        regressor=extract_Pregressor(subj_idn,run,total_n);
        leftP1_File_run = fopen(strcat(result_dir,'/leftP1time_run0',num2str(run),'.txt'),'w');
        rightP1_File_run = fopen(strcat(result_dir,'/rightP1time_run0',num2str(run),'.txt'),'w');
        leftP2_File_run = fopen(strcat(result_dir,'/leftP2time_run0',num2str(run),'.txt'),'w');
        rightP2_File_run = fopen(strcat(result_dir,'/rightP2time_run0',num2str(run),'.txt'),'w');
        
        LSS_File_run = fopen(strcat(result_dir,'/LSStime_run0',num2str(run),'.txt'),'w');

        fprintf(leftP1_File_run,'%f:%f',regressor.RealLP1(1),regressor.LP1_time(1));
        fprintf(rightP1_File_run,'%f:%f',regressor.RealRP1(1),regressor.RP1_time(1));
        fprintf(leftP2_File_run,'%f:%f',regressor.RealLP2(1),regressor.LP2_time(1));
        fprintf(rightP2_File_run,'%f:%f',regressor.RealRP2(1),regressor.RP2_time(1));
        
        fprintf(LSS_File_run,'%f',regressor.LSS(1));

        for k = 2:length(regressor.RealRP1)
            fprintf(leftP1_File_run,' %f:%f',regressor.RealLP1(k),regressor.LP1_time(k));
            fprintf(rightP1_File_run,' %f:%f',regressor.RealRP1(k),regressor.RP1_time(k));
            fprintf(leftP2_File_run,' %f:%f',regressor.RealLP2(k),regressor.LP2_time(k));
            fprintf(rightP2_File_run,' %f:%f',regressor.RealRP2(k),regressor.RP2_time(k));
        end
        for k2 = 2:length(regressor.LSS)
            fprintf(LSS_File_run,' %f',regressor.LSS(k2));
        end
        fclose(leftP1_File_run);
        fclose(rightP1_File_run);
        fclose(leftP2_File_run);
        fclose(rightP2_File_run);
        
        fclose(LSS_File_run);
    end
end