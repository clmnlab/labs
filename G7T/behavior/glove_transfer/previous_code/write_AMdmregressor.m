function write_AMdmregressor(subj_idn,total_n)
    if subj_idn < 10 
        if exist(strcat('regressor/G7T_TRP0',num2str(subj_idn),'/AM_dm'),'dir')==0
            mkdir(strcat('regressor/G7T_TRP0',num2str(subj_idn),'/AM_dm'))
        end
        result_dir=strcat('regressor/G7T_TRP0',num2str(subj_idn),'/AM_dm');
    else
        if exist(strcat('regressor/G7T_TRP',num2str(subj_idn),'/AM_dm'),'dir')==0
            mkdir(strcat('regressor/G7T_TRP',num2str(subj_idn),'/AM_dm'))
        end
        result_dir=strcat('regressor/G7T_TRP',num2str(subj_idn),'/AM_dm');
    end
    regressor=extract_AM_Regressor(subj_idn,1,total_n);
    leftP1_File = fopen(strcat(result_dir,'/leftP1Reward.txt'),'w');
    rightP1_File = fopen(strcat(result_dir,'/rightP1Reward.txt'),'w');
    leftP2_File = fopen(strcat(result_dir,'/leftP2Reward.txt'),'w');
    rightP2_File = fopen(strcat(result_dir,'/rightP2Reward.txt'),'w');
    
    leftP1_3sFile = fopen(strcat(result_dir,'/leftP1Reward_3s.txt'),'w');
    rightP1_3sFile = fopen(strcat(result_dir,'/rightP1Reward_3s.txt'),'w');
    leftP2_3sFile = fopen(strcat(result_dir,'/leftP2Reward_3s.txt'),'w');
    rightP2_3sFile = fopen(strcat(result_dir,'/rightP2Reward_3s.txt'),'w');
    
    fprintf(leftP1_File,'%f*%d:%d',regressor.RealLP1(1),regressor.LP1_Reward(1),9);
    fprintf(rightP1_File,'%f*%d:%d',regressor.RealRP1(1),regressor.RP1_Reward(1),9);
    fprintf(leftP2_File,'%f*%d:%d',regressor.RealLP2(1),regressor.LP2_Reward(1),9);
    fprintf(rightP2_File,'%f*%d:%d',regressor.RealRP2(1),regressor.RP2_Reward(1),9);
    
    fprintf(leftP1_3sFile,'%f*%d:%d',regressor.RealLP1_3s(1),regressor.LP1_Reward_3s(1),3);
    fprintf(rightP1_3sFile,'%f*%d:%d',regressor.RealRP1_3s(1),regressor.RP1_Reward_3s(1),3);
    fprintf(leftP2_3sFile,'%f*%d:%d',regressor.RealLP2_3s(1),regressor.LP2_Reward_3s(1),3);
    fprintf(rightP2_3sFile,'%f*%d:%d',regressor.RealRP2_3s(1),regressor.RP2_Reward_3s(1),3);
    

    
    for i = 2:length(regressor.RealLP1)
        fprintf(leftP1_File,' %f*%d:%d',regressor.RealLP1(i),regressor.LP1_Reward(i),9);
        fprintf(rightP1_File,' %f*%d:%d',regressor.RealRP1(i),regressor.RP1_Reward(i),9);
        fprintf(leftP2_File,' %f*%d:%d',regressor.RealLP2(i),regressor.LP2_Reward(i),9);
        fprintf(rightP2_File,' %f*%d:%d',regressor.RealRP2(i),regressor.RP2_Reward(i),9);
    end
    
    fclose(leftP1_File);
    fclose(rightP1_File);
    fclose(leftP2_File);
    fclose(rightP2_File);
    
    for i = 2:length(regressor.RealLP1_3s)
        fprintf(leftP1_3sFile,'%f*%d:%d',regressor.RealLP1_3s(i),regressor.LP1_Reward_3s(i),3);
        fprintf(rightP1_3sFile,'%f*%d:%d',regressor.RealRP1_3s(i),regressor.RP1_Reward_3s(i),3);
        fprintf(leftP2_3sFile,'%f*%d:%d',regressor.RealLP2_3s(i),regressor.LP2_Reward_3s(i),3);
        fprintf(rightP2_3sFile,'%f*%d:%d',regressor.RealRP2_3s(i),regressor.RP2_Reward_3s(i),3);
    end
    
    fclose(leftP1_3sFile);
    fclose(rightP1_3sFile);
    fclose(leftP2_3sFile);
    fclose(rightP2_3sFile);
    
    for session_n = 2:total_n
        regressor=extract_AM_Regressor(subj_idn,session_n,total_n);
        leftP1_File = fopen(strcat(result_dir,'/leftP1Reward.txt'),'a');
        rightP1_File = fopen(strcat(result_dir,'/rightP1Reward.txt'),'a');
        leftP2_File = fopen(strcat(result_dir,'/leftP2Reward.txt'),'a');
        rightP2_File = fopen(strcat(result_dir,'/rightP2Reward.txt'),'a');
        
        leftP1_3sFile = fopen(strcat(result_dir,'/leftP1Reward_3s.txt'),'a');
        rightP1_3sFile = fopen(strcat(result_dir,'/rightP1Reward_3s.txt'),'a');
        leftP2_3sFile = fopen(strcat(result_dir,'/leftP2Reward_3s.txt'),'a');
        rightP2_3sFile = fopen(strcat(result_dir,'/rightP2Reward_3s.txt'),'a');
        
        fprintf(leftP1_File,'\n%f*%d:%d',regressor.RealLP1(1),regressor.LP1_Reward(1),9);
        fprintf(rightP1_File,'\n%f*%d:%d',regressor.RealRP1(1),regressor.RP1_Reward(1),9);
        fprintf(leftP2_File,'\n%f*%d:%d',regressor.RealLP2(1),regressor.LP2_Reward(1),9);
        fprintf(rightP2_File,'\n%f*%d:%d',regressor.RealRP2(1),regressor.RP2_Reward(1),9);
        
        fprintf(leftP1_3sFile,'%f*%d:%d',regressor.RealLP1_3s(1),regressor.LP1_Reward_3s(1),3);
        fprintf(rightP1_3sFile,'%f*%d:%d',regressor.RealRP1_3s(1),regressor.RP1_Reward_3s(1),3);
        fprintf(leftP2_3sFile,'%f*%d:%d',regressor.RealLP2_3s(1),regressor.LP2_Reward_3s(1),3);
        fprintf(rightP2_3sFile,'%f*%d:%d',regressor.RealRP2_3s(1),regressor.RP2_Reward_3s(1),3);
        
        for j = 2:length(regressor.RealRP1)
            fprintf(leftP1_File,' %f*%d:%d',regressor.RealLP1(j),regressor.LP1_Reward(j),9);
            fprintf(rightP1_File,' %f*%d:%d',regressor.RealRP1(j),regressor.RP1_Reward(j),9);
            fprintf(leftP2_File,' %f*%d:%d',regressor.RealLP2(j),regressor.LP2_Reward(j),9);
            fprintf(rightP2_File,' %f*%d:%d',regressor.RealRP2(j),regressor.RP2_Reward(j),9);
        end
        fclose(leftP1_File);
        fclose(rightP1_File);
        fclose(leftP2_File);
        fclose(rightP2_File);
        
        for j = 2:length(regressor.RealLP1_3s)
            fprintf(leftP1_3sFile,'%f*%d:%d',regressor.RealLP1_3s(j),regressor.LP1_Reward_3s(j),3);
            fprintf(rightP1_3sFile,'%f*%d:%d',regressor.RealRP1_3s(j),regressor.RP1_Reward_3s(j),3);
            fprintf(leftP2_3sFile,'%f*%d:%d',regressor.RealLP2_3s(j),regressor.LP2_Reward_3s(j),3);
            fprintf(rightP2_3sFile,'%f*%d:%d',regressor.RealRP2_3s(j),regressor.RP2_Reward_3s(j),3);
        end
        
        fclose(leftP1_3sFile);
        fclose(rightP1_3sFile);
        fclose(leftP2_3sFile);
        fclose(rightP2_3sFile);
    end
    
    for run = 1 : total_n
        regressor=extract_AM_Regressor(subj_idn,run,total_n);
        leftP1_File_run = fopen(strcat(result_dir,'/leftP1Reward_run0',num2str(run),'.txt'),'w');
        rightP1_File_run = fopen(strcat(result_dir,'/rightP1Reward_run0',num2str(run),'.txt'),'w');
        leftP2_File_run = fopen(strcat(result_dir,'/leftP2Reward_run0',num2str(run),'.txt'),'w');
        rightP2_File_run = fopen(strcat(result_dir,'/rightP2Reward_run0',num2str(run),'.txt'),'w');
        
        leftP1_3sFile_run = fopen(strcat(result_dir,'/leftP1Reward3s_run0',num2str(run),'.txt'),'w');
        rightP1_3sFile_run = fopen(strcat(result_dir,'/rightP1Reward3s_run0',num2str(run),'.txt'),'w');
        leftP2_3sFile_run = fopen(strcat(result_dir,'/leftP2Reward3s_run0',num2str(run),'.txt'),'w');
        rightP2_3sFile_run = fopen(strcat(result_dir,'/rightP2Reward3s_run0',num2str(run),'.txt'),'w');
        
        fprintf(leftP1_File_run,'%f*%d:%d',regressor.RealLP1(1),regressor.LP1_Reward(1),9);
        fprintf(rightP1_File_run,'%f*%d:%d',regressor.RealRP1(1),regressor.RP1_Reward(1),9);
        fprintf(leftP2_File_run,'%f*%d:%d',regressor.RealLP2(1),regressor.LP2_Reward(1),9);
        fprintf(rightP2_File_run,'%f*%d:%d',regressor.RealRP2(1),regressor.RP2_Reward(1),9);
        
        fprintf(leftP1_3sFile_run,'%f*%d:%d',regressor.RealLP1_3s(1),regressor.LP1_Reward_3s(1),3);
        fprintf(rightP1_3sFile_run,'%f*%d:%d',regressor.RealRP1_3s(1),regressor.RP1_Reward_3s(1),3);
        fprintf(leftP2_3sFile_run,'%f*%d:%d',regressor.RealLP2_3s(1),regressor.LP2_Reward_3s(1),3);
        fprintf(rightP2_3sFile_run,'%f*%d:%d',regressor.RealRP2_3s(1),regressor.RP2_Reward_3s(1),3);
        

        for k = 2:length(regressor.RealRP1)
            fprintf(leftP1_File_run,' %f*%d:%d',regressor.RealLP1(k),regressor.LP1_Reward(k),9);
            fprintf(rightP1_File_run,' %f*%d:%d',regressor.RealRP1(k),regressor.RP1_Reward(k),9);
            fprintf(leftP2_File_run,' %f*%d:%d',regressor.RealLP2(k),regressor.LP2_Reward(k),9);
            fprintf(rightP2_File_run,' %f*%d:%d',regressor.RealRP2(k),regressor.RP2_Reward(k),9);
        end
        
        for k = 2:length(regressor.RealRP1_3s)
            fprintf(leftP1_3sFile_run,' %f*%d:%d',regressor.RealLP1_3s(k),regressor.LP1_Reward_3s(k),3);
            fprintf(rightP1_3sFile_run,' %f*%d:%d',regressor.RealRP1_3s(k),regressor.RP1_Reward_3s(k),3);
            fprintf(leftP2_3sFile_run,' %f*%d:%d',regressor.RealLP2_3s(k),regressor.LP2_Reward_3s(k),3);
            fprintf(rightP2_3sFile_run,' %f*%d:%d',regressor.RealRP2_3s(k),regressor.RP2_Reward_3s(k),3);
        end
        
        fclose(leftP1_File_run);
        fclose(rightP1_File_run);
        fclose(leftP2_File_run);
        fclose(rightP2_File_run);
        
        fclose(leftP1_3sFile_run);
        fclose(rightP1_3sFile_run);
        fclose(leftP2_3sFile_run);
        fclose(rightP2_3sFile_run);
    end
end