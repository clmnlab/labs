function write_AMregressor(subj_idn,total_n)
    if subj_idn < 10 
        if exist(strcat('regressor/G7T_TR0',num2str(subj_idn),'/AM_hit'),'dir')==0
            mkdir(strcat('regressor/G7T_TR0',num2str(subj_idn),'/AM_hit'))
        end
        result_dir=strcat('regressor/G7T_TR0',num2str(subj_idn),'/AM_hit');
    else
        if exist(strcat('regressor/G7T_TR',num2str(subj_idn),'/AM_hit'),'dir')==0
            mkdir(strcat('regressor/G7T_TR',num2str(subj_idn),'/AM_hit'))
        end
        result_dir=strcat('regressor/G7T_TR',num2str(subj_idn),'/AM_hit');
    end
    regressor=extract_AMRegressor(subj_idn,1,total_n);
    leftP1_File = fopen(strcat(result_dir,'/leftP1Hit.txt'),'w');
    rightP1_File = fopen(strcat(result_dir,'/rightP1Hit.txt'),'w');
    leftP2_File = fopen(strcat(result_dir,'/leftP2Hit.txt'),'w');
    rightP2_File = fopen(strcat(result_dir,'/rightP2Hit.txt'),'w');
    
    leftP1_3sFile = fopen(strcat(result_dir,'/leftP1Hit_3s.txt'),'w');
    rightP1_3sFile = fopen(strcat(result_dir,'/rightP1Hit_3s.txt'),'w');
    leftP2_3sFile = fopen(strcat(result_dir,'/leftP2Hit_3s.txt'),'w');
    rightP2_3sFile = fopen(strcat(result_dir,'/rightP2Hit_3s.txt'),'w');
    
    fprintf(leftP1_File,'%f*%d',regressor.RealLP1(1),regressor.LP1_Reward(1));
    fprintf(rightP1_File,'%f*%d',regressor.RealRP1(1),regressor.RP1_Reward(1));
    fprintf(leftP2_File,'%f*%d',regressor.RealLP2(1),regressor.LP2_Reward(1));
    fprintf(rightP2_File,'%f*%d',regressor.RealRP2(1),regressor.RP2_Reward(1));
    
    fprintf(leftP1_3sFile,'%f*%d',regressor.RealLP1_3s(1),regressor.LP1_Reward_3s(1));
    fprintf(rightP1_3sFile,'%f*%d',regressor.RealRP1_3s(1),regressor.RP1_Reward_3s(1));
    fprintf(leftP2_3sFile,'%f*%d',regressor.RealLP2_3s(1),regressor.LP2_Reward_3s(1));
    fprintf(rightP2_3sFile,'%f*%d',regressor.RealRP2_3s(1),regressor.RP2_Reward_3s(1));
    

    
    for i = 2:length(regressor.RealLP1)
        fprintf(leftP1_File,' %f*%d',regressor.RealLP1(i),regressor.LP1_Reward(i));
        fprintf(rightP1_File,' %f*%d',regressor.RealRP1(i),regressor.RP1_Reward(i));
        fprintf(leftP2_File,' %f*%d',regressor.RealLP2(i),regressor.LP2_Reward(i));
        fprintf(rightP2_File,' %f*%d',regressor.RealRP2(i),regressor.RP2_Reward(i));
    end
    
    fclose(leftP1_File);
    fclose(rightP1_File);
    fclose(leftP2_File);
    fclose(rightP2_File);
    
    for i = 2:length(regressor.RealLP1_3s)
        fprintf(leftP1_3sFile,' %f*%d',regressor.RealLP1_3s(i),regressor.LP1_Reward_3s(i));
        fprintf(rightP1_3sFile,' %f*%d',regressor.RealRP1_3s(i),regressor.RP1_Reward_3s(i));
        fprintf(leftP2_3sFile,' %f*%d',regressor.RealLP2_3s(i),regressor.LP2_Reward_3s(i));
        fprintf(rightP2_3sFile,' %f*%d',regressor.RealRP2_3s(i),regressor.RP2_Reward_3s(i));
    end
    
    fclose(leftP1_3sFile);
    fclose(rightP1_3sFile);
    fclose(leftP2_3sFile);
    fclose(rightP2_3sFile);
    
    for session_n = 2:total_n
        regressor=extract_AMRegressor(subj_idn,session_n,total_n);
        leftP1_File = fopen(strcat(result_dir,'/leftP1Hit.txt'),'a');
        rightP1_File = fopen(strcat(result_dir,'/rightP1Hit.txt'),'a');
        leftP2_File = fopen(strcat(result_dir,'/leftP2Hit.txt'),'a');
        rightP2_File = fopen(strcat(result_dir,'/rightP2Hit.txt'),'a');
        
        leftP1_3sFile = fopen(strcat(result_dir,'/leftP1Hit_3s.txt'),'a');
        rightP1_3sFile = fopen(strcat(result_dir,'/rightP1Hit_3s.txt'),'a');
        leftP2_3sFile = fopen(strcat(result_dir,'/leftP2Hit_3s.txt'),'a');
        rightP2_3sFile = fopen(strcat(result_dir,'/rightP2Hit_3s.txt'),'a');
        
        fprintf(leftP1_File,'\n%f*%d',regressor.RealLP1(1),regressor.LP1_Reward(1));
        fprintf(rightP1_File,'\n%f*%d',regressor.RealRP1(1),regressor.RP1_Reward(1));
        fprintf(leftP2_File,'\n%f*%d',regressor.RealLP2(1),regressor.LP2_Reward(1));
        fprintf(rightP2_File,'\n%f*%d',regressor.RealRP2(1),regressor.RP2_Reward(1));
        
        fprintf(leftP1_3sFile,'\n%f*%d',regressor.RealLP1_3s(1),regressor.LP1_Reward_3s(1));
        fprintf(rightP1_3sFile,'\n%f*%d',regressor.RealRP1_3s(1),regressor.RP1_Reward_3s(1));
        fprintf(leftP2_3sFile,'\n%f*%d',regressor.RealLP2_3s(1),regressor.LP2_Reward_3s(1));
        fprintf(rightP2_3sFile,'\n%f*%d',regressor.RealRP2_3s(1),regressor.RP2_Reward_3s(1));
        
        for j = 2:length(regressor.RealRP1)
            fprintf(leftP1_File,' %f*%d',regressor.RealLP1(j),regressor.LP1_Reward(j));
            fprintf(rightP1_File,' %f*%d',regressor.RealRP1(j),regressor.RP1_Reward(j));
            fprintf(leftP2_File,' %f*%d',regressor.RealLP2(j),regressor.LP2_Reward(j));
            fprintf(rightP2_File,' %f*%d',regressor.RealRP2(j),regressor.RP2_Reward(j));
        end
        fclose(leftP1_File);
        fclose(rightP1_File);
        fclose(leftP2_File);
        fclose(rightP2_File);
        
        for j = 2:length(regressor.RealLP1_3s)
            fprintf(leftP1_3sFile,' %f*%d',regressor.RealLP1_3s(j),regressor.LP1_Reward_3s(j));
            fprintf(rightP1_3sFile,' %f*%d',regressor.RealRP1_3s(j),regressor.RP1_Reward_3s(j));
            fprintf(leftP2_3sFile,' %f*%d',regressor.RealLP2_3s(j),regressor.LP2_Reward_3s(j));
            fprintf(rightP2_3sFile,' %f*%d',regressor.RealRP2_3s(j),regressor.RP2_Reward_3s(j));
        end
        
        fclose(leftP1_3sFile);
        fclose(rightP1_3sFile);
        fclose(leftP2_3sFile);
        fclose(rightP2_3sFile);
    end
    
    for run = 1 : total_n
        regressor=extract_AMRegressor(subj_idn,run,total_n);
        leftP1_File_run = fopen(strcat(result_dir,'/leftP1Hit_run0',num2str(run),'.txt'),'w');
        rightP1_File_run = fopen(strcat(result_dir,'/rightP1Hit_run0',num2str(run),'.txt'),'w');
        leftP2_File_run = fopen(strcat(result_dir,'/leftP2Hit_run0',num2str(run),'.txt'),'w');
        rightP2_File_run = fopen(strcat(result_dir,'/rightP2Hit_run0',num2str(run),'.txt'),'w');
        
        leftP1_3sFile_run = fopen(strcat(result_dir,'/leftP1Hit3s_run0',num2str(run),'.txt'),'w');
        rightP1_3sFile_run = fopen(strcat(result_dir,'/rightP1Hit3s_run0',num2str(run),'.txt'),'w');
        leftP2_3sFile_run = fopen(strcat(result_dir,'/leftP2Hit3s_run0',num2str(run),'.txt'),'w');
        rightP2_3sFile_run = fopen(strcat(result_dir,'/rightP2Hit3s_run0',num2str(run),'.txt'),'w');
        
        fprintf(leftP1_File_run,'%f*%d',regressor.RealLP1(1),regressor.LP1_Reward(1));
        fprintf(rightP1_File_run,'%f*%d',regressor.RealRP1(1),regressor.RP1_Reward(1));
        fprintf(leftP2_File_run,'%f*%d',regressor.RealLP2(1),regressor.LP2_Reward(1));
        fprintf(rightP2_File_run,'%f*%d',regressor.RealRP2(1),regressor.RP2_Reward(1));
        
        fprintf(leftP1_3sFile_run,'%f*%d',regressor.RealLP1_3s(1),regressor.LP1_Reward_3s(1));
        fprintf(rightP1_3sFile_run,'%f*%d',regressor.RealRP1_3s(1),regressor.RP1_Reward_3s(1));
        fprintf(leftP2_3sFile_run,'%f*%d',regressor.RealLP2_3s(1),regressor.LP2_Reward_3s(1));
        fprintf(rightP2_3sFile_run,'%f*%d',regressor.RealRP2_3s(1),regressor.RP2_Reward_3s(1));
        

        for k = 2:length(regressor.RealRP1)
            fprintf(leftP1_File_run,' %f*%d',regressor.RealLP1(k),regressor.LP1_Reward(k));
            fprintf(rightP1_File_run,' %f*%d',regressor.RealRP1(k),regressor.RP1_Reward(k));
            fprintf(leftP2_File_run,' %f*%d',regressor.RealLP2(k),regressor.LP2_Reward(k));
            fprintf(rightP2_File_run,' %f*%d',regressor.RealRP2(k),regressor.RP2_Reward(k));
        end
        
        for k = 2:length(regressor.RealRP1_3s)
            fprintf(leftP1_3sFile_run,' %f*%d',regressor.RealLP1_3s(k),regressor.LP1_Reward_3s(k));
            fprintf(rightP1_3sFile_run,' %f*%d',regressor.RealRP1_3s(k),regressor.RP1_Reward_3s(k));
            fprintf(leftP2_3sFile_run,' %f*%d',regressor.RealLP2_3s(k),regressor.LP2_Reward_3s(k));
            fprintf(rightP2_3sFile_run,' %f*%d',regressor.RealRP2_3s(k),regressor.RP2_Reward_3s(k));
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